#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_gemma4_language_lora import (  # noqa: E402
    DATASET_PRESETS,
    MODEL_PATH,
    SampleDataset,
    build_messages,
    encode_example,
    first_model_device,
    label_logprob,
    load_split,
    move_batch_to_device,
    set_seed,
)
from src.inference.output_parser import OutputParser  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_json", default="")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["generate", "logprob"], default="generate")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--max_memory_per_gpu", default="46GiB")
    parser.add_argument("--cpu_memory", default="700GiB")
    return parser.parse_args()


def load_model(args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = len([x for x in visible.split(",") if x.strip()]) if visible else torch.cuda.device_count()
    max_memory = {idx: args.max_memory_per_gpu for idx in range(n_visible)}
    max_memory["cpu"] = args.cpu_memory

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    return model, processor


def contains_complete_json_object(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    depth = 0
    in_string = False
    escaped = False
    for ch in text[start:]:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


class StopOnJsonObjectEnd(StoppingCriteria):
    def __init__(self, tokenizer, prompt_width: int):
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1 or input_ids.shape[-1] <= self.prompt_width:
            return False
        last_token = self.tokenizer.decode(
            input_ids[0, -1:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if "}" not in last_token:
            return False
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return contains_complete_json_object(text)


@torch.no_grad()
def generate_prediction(model, processor, sample, label_names, prompt_dataset_name, image_max_side, max_new_tokens):
    messages = build_messages(
        sample,
        label_names,
        prompt_dataset_name,
        image_max_side,
        answer=None,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = move_batch_to_device(inputs, first_model_device(model))
    prompt_width = int(inputs["input_ids"].shape[-1])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([
            StopOnJsonObjectEnd(processor.tokenizer, prompt_width)
        ]),
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    completion_ids = output_ids[:, prompt_width:]
    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()


@torch.no_grad()
def logprob_prediction(model, processor, sample, label_names, prompt_dataset_name, image_max_side, max_length):
    scores = []
    for label in label_names:
        candidate = type("Candidate", (), dict(sample.__dict__))()
        candidate.label_name = label
        inputs = encode_example(
            processor,
            candidate,
            label_names,
            prompt_dataset_name,
            image_max_side,
            max_length,
        )
        inputs = move_batch_to_device(inputs, first_model_device(model))
        scores.append(label_logprob(model, inputs))
    pred_idx = max(range(len(label_names)), key=lambda i: scores[i])
    return label_names[pred_idx], pred_idx, scores


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    preset = DATASET_PRESETS[args.dataset]
    raw_ds = load_split(args.dataset, args.split)
    label_names = list(raw_ds.label_names)
    samples = SampleDataset(raw_ds.samples, args.seed, limit=args.limit).samples

    model, processor = load_model(args)
    parser = OutputParser()
    predictions = []
    correct = 0
    parse_success = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"{args.dataset}:{args.split}")):
        raw_response = ""
        scores = None
        if args.mode == "generate":
            raw_response = generate_prediction(
                model,
                processor,
                sample,
                label_names,
                preset["prompt_dataset_name"],
                preset["image_max_side"],
                args.max_new_tokens,
            )
            parsed = parser.parse_classification(
                raw_response=raw_response,
                query_id=getattr(sample, "id", str(idx)),
                label_names=label_names,
                is_multi_label=False,
                dataset_name=preset["prompt_dataset_name"],
            )
            pred = parsed.predicted_label
            pred_idx = parsed.predicted_label_idx
            parse_success += int(parsed.parse_success)
        else:
            pred, pred_idx, scores = logprob_prediction(
                model,
                processor,
                sample,
                label_names,
                preset["prompt_dataset_name"],
                preset["image_max_side"],
                args.max_length,
            )
        ok = pred == sample.label_name
        correct += int(ok)
        record = {
            "index": idx,
            "query_id": getattr(sample, "id", str(idx)),
            "image_path": sample.image_path,
            "label": sample.label_name,
            "label_idx": int(sample.label),
            "prediction": pred,
            "prediction_idx": int(pred_idx),
            "correct": ok,
            "raw_response": raw_response,
        }
        if scores is not None:
            record["scores"] = {label: score for label, score in zip(label_names, scores)}
        predictions.append(record)
        if args.predictions_json and args.flush_every and (idx + 1) % args.flush_every == 0:
            partial_metrics = {
                "dataset": args.dataset,
                "split": args.split,
                "mode": args.mode,
                "adapter_dir": args.adapter_dir,
                "n": idx + 1,
                "total_n": len(samples),
                "correct": correct,
                "accuracy": correct / max(1, idx + 1),
                "parse_success": parse_success if args.mode == "generate" else None,
                "parse_success_rate": parse_success / max(1, idx + 1) if args.mode == "generate" else None,
                "label_names": label_names,
                "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "max_new_tokens": args.max_new_tokens,
                "complete": False,
            }
            output_json = Path(args.output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(partial_metrics, indent=2))
            pred_path = Path(args.predictions_json)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text(json.dumps(predictions, indent=2))

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "adapter_dir": args.adapter_dir,
        "n": len(samples),
        "correct": correct,
        "accuracy": correct / max(1, len(samples)),
        "parse_success": parse_success if args.mode == "generate" else None,
        "parse_success_rate": parse_success / max(1, len(samples)) if args.mode == "generate" else None,
        "label_names": label_names,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "max_new_tokens": args.max_new_tokens,
        "complete": True,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2))
    if args.predictions_json:
        pred_path = Path(args.predictions_json)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(predictions, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
