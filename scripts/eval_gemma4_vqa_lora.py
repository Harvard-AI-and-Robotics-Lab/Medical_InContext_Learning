#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset
from src.inference import OutputParser
from src.prompting.templates import VQATemplate


MODEL_PATH = "/data/home/mindazhao/hf_models/google_gemma-4-31B-it"
JSON_INSTRUCTION = (
    'Return only one compact JSON object with key "answer". '
    'No markdown, no extra text.'
)

DATASET_PRESETS = {
    "slake": {
        "dataset_name": "slake",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/slake/manifest_en.json",
        "image_max_side": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-memory-per-gpu", default="46GiB")
    parser.add_argument("--cpu-memory", default="700GiB")
    parser.add_argument("--flush-every", type=int, default=10)
    return parser.parse_args()


def absolute_project_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def load_vqa_dataset(dataset_key: str):
    preset = DATASET_PRESETS[dataset_key]
    kwargs = {"manifest_json": absolute_project_path(preset["manifest_json"])}
    return get_dataset(
        preset["dataset_name"],
        data_root=absolute_project_path(preset["data_root"]),
        split="all",
        **kwargs,
    )


def load_image(path: str, image_max_side: int | None):
    with Image.open(path) as im:
        image = im.convert("RGB")
    if image_max_side:
        image.thumbnail((image_max_side, image_max_side), Image.Resampling.LANCZOS)
    return image


def gemma_content_from_prompt_content(content, image_max_side: int | None):
    out = []
    for item in content:
        if item["type"] == "text":
            out.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            out.append({"type": "image", "image": load_image(item["image_url"]["url"], image_max_side)})
        else:
            raise ValueError(f"Unsupported prompt item: {item}")
    return out


def build_messages(sample, image_max_side: int | None):
    user_content = VQATemplate.format_query(sample.image_path, sample.question)
    for item in reversed(user_content):
        if item["type"] == "text":
            item["text"] = item["text"].rstrip() + "\n\n" + JSON_INSTRUCTION
            break
    return [
        {"role": "system", "content": [{"type": "text", "text": VQATemplate.SYSTEM_PROMPT}]},
        {"role": "user", "content": gemma_content_from_prompt_content(user_content, image_max_side)},
    ]


def first_model_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0")


def move_batch_to_device(inputs, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def summarize(rows):
    out = {"n": len(rows)}
    out["parse_success"] = sum(float(row["parsed"]["parse_success"]) for row in rows) / max(1, len(rows))
    out["normalized_exact_match"] = sum(float(row["normalized_exact_match"]) for row in rows) / max(1, len(rows))
    by_type = {}
    for answer_type in sorted({str(row["answer_type"]).upper() or "UNKNOWN" for row in rows}):
        subset = [row for row in rows if (str(row["answer_type"]).upper() or "UNKNOWN") == answer_type]
        by_type[answer_type] = {
            "n": len(subset),
            "parse_success": sum(float(row["parsed"]["parse_success"]) for row in subset) / max(1, len(subset)),
            "normalized_exact_match": sum(float(row["normalized_exact_match"]) for row in subset) / max(1, len(subset)),
        }
    out["by_answer_type"] = by_type
    return out


def write_outputs(rows, output_dir: Path, run_config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    with (output_dir / "raw_records.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = {**summarize(rows), "run_config": run_config}
    (output_dir / "vqa_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = DATASET_PRESETS[args.dataset]
    image_max_side = preset["image_max_side"]
    dataset = load_vqa_dataset(args.dataset)
    samples = [s for s in dataset.samples if s.split == args.split]
    if args.limit:
        samples = samples[: args.limit]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = len([x for x in visible.split(",") if x.strip()]) if visible else torch.cuda.device_count()
    max_memory = {idx: args.max_memory_per_gpu for idx in range(n_visible)}
    max_memory["cpu"] = args.cpu_memory

    run_config = {
        **vars(args),
        "dataset_preset": preset,
        "n_samples": len(samples),
        "visible_gpus": visible,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(run_config, indent=2, ensure_ascii=True), flush=True)

    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    device = first_model_device(model)
    parser = OutputParser()

    rows = []
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(samples, desc=f"{args.dataset}/{args.split}"), start=1):
            messages = build_messages(sample, image_max_side)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = move_batch_to_device(inputs, device)
            input_len = int(inputs["input_ids"].shape[-1])
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            raw = processor.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()
            parsed = parser.parse_vqa(raw, sample.id)
            normalized_exact = normalize_answer(sample.answer) == normalize_answer(parsed.answer)
            row = {
                "query_id": sample.id,
                "question": sample.question,
                "ground_truth_answer": sample.answer,
                "answer_type": sample.question_type,
                "image_path": sample.image_path,
                "sample_metadata": sample.metadata,
                "prompt": {
                    "system": VQATemplate.SYSTEM_PROMPT,
                    "question": sample.question,
                    "json_instruction": JSON_INSTRUCTION,
                },
                "inference": {"raw_response": raw},
                "parsed": {
                    "query_id": sample.id,
                    "answer": parsed.answer,
                    "parse_success": parsed.parse_success,
                },
                "normalized_exact_match": normalized_exact,
            }
            rows.append(row)
            if args.flush_every > 0 and idx % args.flush_every == 0:
                write_outputs(rows, output_dir, run_config)

    summary = write_outputs(rows, output_dir, run_config)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
