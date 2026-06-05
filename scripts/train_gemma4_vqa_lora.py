#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import string
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset
from src.inference import OutputParser
from src.prompting.templates import VQATemplate


MODEL_PATH = "google/gemma-4-31B-it"

DATASET_PRESETS = {
    "slake": {
        "dataset_name": "slake",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/slake/manifest_en.json",
        "train_split": "reference",
        "val_split": "validation",
        "image_max_side": None,
    },
    "pathvqa": {
        "dataset_name": "pathvqa",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/pathvqa/manifest.json",
        "train_split": "reference",
        "val_split": "validation",
        "image_max_side": None,
    },
    "vqamed2019": {
        "dataset_name": "vqamed2019",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/vqamed2019/manifest.json",
        "train_split": "reference",
        "val_split": "validation",
        "image_max_side": None,
    },
    "vqa_rad": {
        "dataset_name": "vqa_rad",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/vqa_rad/manifest.json",
        "train_split": "reference",
        "val_split": "validation",
        "image_max_side": None,
    },
}

LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXCLUDED_MODULE_NAME_PARTS = (
    "vision",
    "visual",
    "image",
    "audio",
    "projector",
    "multi_modal",
    "multimodal",
    "mm_projector",
)

JSON_INSTRUCTION = (
    'Return only one compact JSON object with key "answer". '
    'No markdown, no extra text.'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_scope",
        choices=["language", "language_projector"],
        default="language",
        help="language_projector also adapts the image-to-language projection layer.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_subset", type=int, default=200)
    parser.add_argument("--eval_max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--vqa_system_prompt",
        default=None,
        help="Override the VQA system prompt. Pass an empty string to omit the system message.",
    )
    parser.add_argument(
        "--vqa_query_template",
        default=None,
        help="Override the user text template. Supports {question} and {image_path}.",
    )
    parser.add_argument(
        "--json_instruction",
        default=JSON_INSTRUCTION,
        help="Instruction appended to the user prompt. Pass an empty string to disable it.",
    )
    parser.add_argument(
        "--assistant_format",
        choices=["json", "plain"],
        default="json",
        help="Assistant target format used for supervised fine-tuning.",
    )
    parser.add_argument(
        "--loss_mode",
        choices=["answer_span", "assistant_completion"],
        default="answer_span",
        help="answer_span preserves the original bare-answer token loss; assistant_completion trains the full assistant reply.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["overall_exact", "balanced_open_closed", "balanced_open_token_recall_closed"],
        default="overall_exact",
    )
    parser.add_argument("--eval_stratified", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_memory_per_gpu", default="46GiB")
    parser.add_argument("--cpu_memory", default="700GiB")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def absolute_project_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def token_recall(reference: str, prediction: str) -> float:
    ref = answer_tokens(reference)
    if not ref:
        return 0.0
    pred_set = set(answer_tokens(prediction))
    return sum(1 for tok in ref if tok in pred_set) / len(ref)


def load_vqa_dataset(dataset_key: str):
    preset = DATASET_PRESETS[dataset_key]
    kwargs = {"manifest_json": absolute_project_path(preset["manifest_json"])}
    return get_dataset(
        preset["dataset_name"],
        data_root=absolute_project_path(preset["data_root"]),
        split="all",
        **kwargs,
    )


class SampleDataset(Dataset):
    def __init__(self, samples, seed: int, limit: int = 0):
        self.samples = list(samples)
        if limit and len(self.samples) > limit:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def select_eval_samples(samples, seed: int, limit: int, stratified: bool):
    samples = list(samples)
    if not limit or len(samples) <= limit:
        return samples
    rng = random.Random(seed)
    if not stratified:
        return rng.sample(samples, limit)

    buckets = {}
    for sample in samples:
        key = str(getattr(sample, "question_type", "") or "UNKNOWN").upper()
        buckets.setdefault(key, []).append(sample)
    if len(buckets) < 2:
        return rng.sample(samples, limit)

    selected = []
    keys = sorted(buckets)
    base = limit // len(keys)
    remainder = limit % len(keys)
    for idx, key in enumerate(keys):
        bucket = list(buckets[key])
        take = min(len(bucket), base + (1 if idx < remainder else 0))
        selected.extend(rng.sample(bucket, take))

    if len(selected) < limit:
        selected_ids = {sample.id for sample in selected}
        remaining = [sample for sample in samples if sample.id not in selected_ids]
        selected.extend(rng.sample(remaining, min(len(remaining), limit - len(selected))))
    rng.shuffle(selected)
    return selected


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
            path = item["image_url"]["url"]
            out.append({"type": "image", "image": load_image(path, image_max_side)})
        else:
            raise ValueError(f"Unsupported prompt item: {item}")
    return out


def build_user_content(sample, query_template: str | None, json_instruction: str):
    if query_template:
        text = query_template.format(question=sample.question, image_path=sample.image_path)
        user_content = [
            {"type": "image_url", "image_url": {"url": sample.image_path}},
            {"type": "text", "text": text},
        ]
    else:
        user_content = VQATemplate.format_query(sample.image_path, sample.question)

    if json_instruction:
        for item in reversed(user_content):
            if item["type"] == "text":
                item["text"] = item["text"].rstrip() + "\n\n" + json_instruction
                break
    return user_content


def build_messages(sample, image_max_side: int | None, answer: str | None, args):
    user_content = build_user_content(sample, args.vqa_query_template, args.json_instruction)
    system_prompt = VQATemplate.SYSTEM_PROMPT if args.vqa_system_prompt is None else args.vqa_system_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": gemma_content_from_prompt_content(user_content, image_max_side)})
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def answer_for_sample(sample, assistant_format: str) -> str:
    if assistant_format == "plain":
        return str(sample.answer)
    return json.dumps({"answer": str(sample.answer)}, separators=(",", ":"), ensure_ascii=False)


def find_subsequence(items, pattern, start: int = 0):
    if not pattern:
        return -1
    upper = len(items) - len(pattern) + 1
    for idx in range(max(0, start), upper):
        if items[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def move_batch_to_device(inputs, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def first_model_device(model):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0")


def encode_example(processor, sample, image_max_side: int | None, max_length: int, loss_mode: str, args):
    answer_text_for_training = answer_for_sample(sample, args.assistant_format)
    prompt_messages = build_messages(sample, image_max_side, answer=None, args=args)
    full_messages = build_messages(sample, image_max_side, answer=answer_text_for_training, args=args)

    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_inputs = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        enable_thinking=False,
    )

    if full_inputs["input_ids"].shape[-1] > max_length:
        full_inputs = {
            k: v[..., -max_length:] if hasattr(v, "shape") and v.ndim >= 2 else v
            for k, v in full_inputs.items()
        }

    labels = torch.full_like(full_inputs["input_ids"], -100)
    full_ids = full_inputs["input_ids"][0].tolist()
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])

    if loss_mode == "assistant_completion":
        completion_start = max(0, min(prompt_len, full_inputs["input_ids"].shape[-1] - 1))
        labels[0, completion_start:] = full_inputs["input_ids"][0, completion_start:]
        full_inputs["labels"] = labels
        return full_inputs

    answer_text = str(sample.answer)
    answer_ids = processor.tokenizer.encode(answer_text, add_special_tokens=False)

    start = find_subsequence(full_ids, answer_ids, start=max(0, min(prompt_len - 8, len(full_ids) - 1)))
    if start < 0:
        start = find_subsequence(full_ids, answer_ids, start=max(0, len(full_ids) - len(answer_ids) - 64))

    if start >= 0:
        end = start + len(answer_ids)
        labels[0, start:end] = full_inputs["input_ids"][0, start:end]
    else:
        # Rare tokenizer edge cases can merge the answer with surrounding JSON quotes.
        # Fall back to assistant-completion loss instead of silently dropping the sample.
        completion_start = max(0, min(prompt_len, full_inputs["input_ids"].shape[-1] - 1))
        labels[0, completion_start:] = full_inputs["input_ids"][0, completion_start:]

    full_inputs["labels"] = labels
    return full_inputs


def discover_lora_targets(model, scope: str):
    targets = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(part in lname for part in EXCLUDED_MODULE_NAME_PARTS):
            continue
        if not name.endswith(LORA_SUFFIXES):
            continue
        if isinstance(module, torch.nn.Linear):
            targets.append(name)

    if scope == "language_projector":
        for name, module in model.named_modules():
            lname = name.lower()
            if not isinstance(module, torch.nn.Linear):
                continue
            if "embed_vision" in lname or (
                ("projector" in lname or "projection" in lname)
                and not any(part in lname for part in ("language_model", "text_model"))
            ):
                targets.append(name)

    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("No LoRA target modules found.")
    return targets


def setup_model(args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = len([x for x in visible.split(",") if x.strip()]) if visible else torch.cuda.device_count()
    max_memory = {idx: args.max_memory_per_gpu for idx in range(n_visible)}
    max_memory["cpu"] = args.cpu_memory

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = discover_lora_targets(model, args.lora_scope)
    print(f"LoRA target modules: {len(target_modules)}")
    print("\n".join(target_modules[:20]))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate_generation(model, processor, val_samples, image_max_side: int | None, max_new_tokens: int, args):
    model.eval()
    parser = OutputParser()
    device = first_model_device(model)
    rows = []
    for sample in tqdm(val_samples, desc="val_generate", leave=False):
        messages = build_messages(sample, image_max_side, answer=None, args=args)
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        raw = processor.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
        parsed = parser.parse_vqa(raw, sample.id)
        pred = parsed.answer
        rows.append({
            "query_id": sample.id,
            "answer_type": sample.question_type,
            "ground_truth_answer": sample.answer,
            "raw_response": raw,
            "parsed_answer": pred,
            "parse_success": parsed.parse_success,
            "normalized_exact": normalize_answer(sample.answer) == normalize_answer(pred),
            "token_recall": token_recall(sample.answer, pred),
        })

    total = len(rows)
    score = sum(float(row["normalized_exact"]) for row in rows) / max(1, total)
    parse = sum(float(row["parse_success"]) for row in rows) / max(1, total)
    by_type = {}
    for answer_type in sorted({str(row["answer_type"]).upper() or "UNKNOWN" for row in rows}):
        subset = [row for row in rows if (str(row["answer_type"]).upper() or "UNKNOWN") == answer_type]
        by_type[answer_type] = {
            "n_samples": len(subset),
            "normalized_exact": sum(float(row["normalized_exact"]) for row in subset) / max(1, len(subset)),
            "token_recall": sum(float(row["token_recall"]) for row in subset) / max(1, len(subset)),
            "parse_success": sum(float(row["parse_success"]) for row in subset) / max(1, len(subset)),
        }
    model.train()
    return {"normalized_exact": score, "parse_success": parse, "n_samples": total, "by_answer_type": by_type}, rows


def selection_score(val_metrics: dict, metric_name: str) -> float:
    if metric_name == "overall_exact":
        return float(val_metrics["normalized_exact"])
    by_type = val_metrics.get("by_answer_type", {})
    if metric_name == "balanced_open_token_recall_closed":
        open_score = by_type.get("OPEN", {}).get("token_recall")
        closed_score = by_type.get("CLOSED", {}).get("normalized_exact")
        if open_score is None or closed_score is None:
            return float(val_metrics["normalized_exact"])
        return 0.5 * float(open_score) + 0.5 * float(closed_score)
    open_score = by_type.get("OPEN", {}).get("normalized_exact")
    closed_score = by_type.get("CLOSED", {}).get("normalized_exact")
    if open_score is None or closed_score is None:
        return float(val_metrics["normalized_exact"])
    return 0.5 * float(open_score) + 0.5 * float(closed_score)


def save_adapter(model, processor, output_dir: Path, metrics: dict, val_rows: list | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)
    if val_rows is not None:
        with open(output_dir / "validation_generations.jsonl", "w", encoding="utf-8") as f:
            for row in val_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = DATASET_PRESETS[args.dataset]
    dataset = load_vqa_dataset(args.dataset)
    train_samples = [s for s in dataset.samples if s.split == preset["train_split"]]
    val_samples_all = [s for s in dataset.samples if s.split == preset["val_split"]]
    image_max_side = preset["image_max_side"]

    train_ds = SampleDataset(train_samples, args.seed)
    val_samples = select_eval_samples(val_samples_all, args.seed, args.eval_subset, args.eval_stratified)
    val_ds = SampleDataset(val_samples, args.seed)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )

    run_config = {
        **vars(args),
        "dataset_preset": preset,
        "train_n": len(train_ds),
        "validation_n": len(val_ds),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loss": (
            "full assistant completion tokens; prompt/image tokens masked with -100"
            if args.loss_mode == "assistant_completion"
            else "answer-token-span only; prompt tokens masked with -100"
        ),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=True)

    print(json.dumps(run_config, indent=2, ensure_ascii=True))
    model, processor = setup_model(args)
    device = first_model_device(model)

    # Preflight a few examples after the processor is loaded, before spending hours training.
    for sample in train_ds.samples[:8]:
        encoded = encode_example(processor, sample, image_max_side, args.max_length, args.loss_mode, args)
        if not encoded["labels"].ne(-100).any():
            raise RuntimeError(f"No supervised answer tokens for {sample.id}")

    optim = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    best_score = -1.0
    no_improve_evals = 0
    early_stopped = False
    global_step = 0
    accum_counter = 0
    running_loss = 0.0
    optim.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            losses = []
            for sample in batch:
                inputs = encode_example(processor, sample, image_max_side, args.max_length, args.loss_mode, args)
                inputs = move_batch_to_device(inputs, device)
                outputs = model(**inputs)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                losses.append(float(loss.detach().cpu()) * args.grad_accum)
                accum_counter += 1

            running_loss += sum(losses) / max(1, len(losses))
            if accum_counter >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                accum_counter = 0
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    running_loss = 0.0
                    print(json.dumps({"step": global_step, "epoch": epoch + 1, "train_loss": avg_loss}), flush=True)

                if global_step % args.eval_steps == 0:
                    val_metrics, val_rows = evaluate_generation(
                        model,
                        processor,
                        val_ds.samples,
                        image_max_side,
                        args.eval_max_new_tokens,
                        args,
                    )
                    score = selection_score(val_metrics, args.selection_metric)
                    metrics = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "selection_metric": args.selection_metric,
                        "selection_score": score,
                        "val": val_metrics,
                    }
                    print(json.dumps(metrics, ensure_ascii=True), flush=True)
                    if score > best_score + args.early_stop_min_delta:
                        best_score = score
                        no_improve_evals = 0
                        save_adapter(model, processor, output_dir / "best", metrics, val_rows)
                    else:
                        no_improve_evals += 1

                    if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                        early_stopped = True
                        print(json.dumps({
                            "event": "early_stop",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "best_val_normalized_exact": best_score,
                            "no_improve_evals": no_improve_evals,
                            "patience": args.early_stop_patience,
                            "min_delta": args.early_stop_min_delta,
                        }), flush=True)
                        break

                if args.max_steps and global_step >= args.max_steps:
                    break
        if early_stopped or (args.max_steps and global_step >= args.max_steps):
            break

    final_metrics = {
        "step": global_step,
        "epoch": epoch + 1,
        "best_val_normalized_exact": best_score,
        "early_stopped": early_stopped,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_adapter(model, processor, output_dir / "last", final_metrics)
    with open(output_dir / "done.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=True)
    print(json.dumps(final_metrics), flush=True)


if __name__ == "__main__":
    main()
