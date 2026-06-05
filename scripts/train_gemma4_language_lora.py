#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset
from src.prompting.templates import ClassificationTemplate


MODEL_PATH = "/data/home/mindazhao/hf_models/google_gemma-4-31B-it"

DATASET_PRESETS = {
    "breakhis_binary": {
        "dataset_name": "breakhis_binary",
        "prompt_dataset_name": "breakhis_binary",
        "data_root": "data/raw/BreaKHis_v1_extracted",
        "manifest_csv": "manifests/breakhis_binary_patient_split_seed3407.csv",
        "target_label_names": None,
        "image_max_side": None,
    },
    "tbx11k": {
        "dataset_name": "tbx11k",
        "prompt_dataset_name": "tbx11k",
        "data_root": ".",
        "manifest_csv": "manifests/tbx11k_train85_val15_officialval_test_seed3407.csv",
        "target_label_names": ["healthy", "sick but non-TB", "TB"],
        "image_max_side": 512,
    },
    "ddr": {
        "dataset_name": "ddr",
        "prompt_dataset_name": "ddr",
        "data_root": ".",
        "manifest_csv": "manifests/ddr_official_split_crop_pad_1024.csv",
        "target_label_names": None,
        "image_max_side": None,
    },
    "ddr_512": {
        "dataset_name": "ddr",
        "prompt_dataset_name": "ddr",
        "data_root": ".",
        "manifest_csv": "manifests/ddr_official_split_crop_pad_512.csv",
        "target_label_names": None,
        "image_max_side": None,
    },
    "lag_project": {
        "dataset_name": "lag_project",
        "prompt_dataset_name": "lag",
        "data_root": "data/LAG",
        "manifest_csv": "manifests/lag_manifest.csv",
        "target_label_names": None,
        "image_max_side": 512,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--eval_subset", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=0)
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


def load_split(dataset_key: str, split: str):
    preset = DATASET_PRESETS[dataset_key]
    kwargs = {"manifest_csv": absolute_project_path(preset["manifest_csv"])}
    if preset["target_label_names"]:
        kwargs["target_label_names"] = preset["target_label_names"]
    return get_dataset(
        preset["dataset_name"],
        data_root=absolute_project_path(preset["data_root"]),
        split=split,
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


def build_messages(sample, label_names, prompt_dataset_name: str, image_max_side: int | None, answer: str | None):
    system_text = ClassificationTemplate.get_system_prompt(prompt_dataset_name)
    user_content = ClassificationTemplate.format_query(
        sample.image_path,
        label_names,
        is_multi_label=False,
        dataset_name=prompt_dataset_name,
        method="zero_shot",
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": gemma_content_from_prompt_content(user_content, image_max_side)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def answer_for_label(label: str) -> str:
    return json.dumps({"label": label}, separators=(",", ":"))


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


def encode_example(processor, sample, label_names, prompt_dataset_name: str, image_max_side: int | None, max_length: int):
    label = sample.label_name
    answer = answer_for_label(label)
    prompt_messages = build_messages(sample, label_names, prompt_dataset_name, image_max_side, answer=None)
    full_messages = build_messages(sample, label_names, prompt_dataset_name, image_max_side, answer=answer)

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
        full_inputs = {k: v[..., -max_length:] if hasattr(v, "shape") and v.ndim >= 2 else v for k, v in full_inputs.items()}

    labels = torch.full_like(full_inputs["input_ids"], -100)
    full_ids = full_inputs["input_ids"][0].tolist()
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    label_ids = processor.tokenizer.encode(label, add_special_tokens=False)
    start = find_subsequence(full_ids, label_ids, start=max(0, prompt_len - 8))
    if start < 0:
        start = find_subsequence(full_ids, label_ids)
    if start < 0:
        raise RuntimeError(f"Could not find label span for label={label!r}")
    end = start + len(label_ids)
    labels[0, start:end] = full_inputs["input_ids"][0, start:end]
    full_inputs["labels"] = labels
    return full_inputs


def discover_language_lora_targets(model):
    targets = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(part in lname for part in EXCLUDED_MODULE_NAME_PARTS):
            continue
        if not name.endswith(LORA_SUFFIXES):
            continue
        if isinstance(module, torch.nn.Linear):
            targets.append(name)
    if not targets:
        raise RuntimeError("No language LoRA target modules found.")
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

    target_modules = discover_language_lora_targets(model)
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


def label_logprob(model, inputs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(shift_logits.device)
    mask = shift_labels.ne(-100)
    if not mask.any():
        return float("-inf")
    log_probs = F.log_softmax(shift_logits, dim=-1)
    chosen = log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(chosen.masked_select(mask).sum().detach().cpu())


@torch.no_grad()
def evaluate_accuracy(model, processor, val_samples, label_names, prompt_dataset_name: str, image_max_side: int | None, max_length: int):
    model.eval()
    correct = 0
    total = 0
    for sample in tqdm(val_samples, desc="eval", leave=False):
        scores = []
        for label in label_names:
            candidate = type("Candidate", (), dict(sample.__dict__))()
            candidate.label_name = label
            inputs = encode_example(processor, candidate, label_names, prompt_dataset_name, image_max_side, max_length)
            inputs = move_batch_to_device(inputs, first_model_device(model))
            scores.append(label_logprob(model, inputs))
        pred = label_names[max(range(len(label_names)), key=lambda i: scores[i])]
        correct += int(pred == sample.label_name)
        total += 1
    model.train()
    return correct / max(1, total)


def save_adapter(model, processor, output_dir: Path, metrics: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = DATASET_PRESETS[args.dataset]
    train_ds_raw = load_split(args.dataset, "train")
    val_ds_raw = load_split(args.dataset, "val")
    label_names = list(train_ds_raw.label_names)
    image_max_side = preset["image_max_side"]

    train_ds = SampleDataset(train_ds_raw.samples, args.seed)
    val_ds = SampleDataset(val_ds_raw.samples, args.seed, limit=args.eval_subset)
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
        "label_names": label_names,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(json.dumps(run_config, indent=2))
    model, processor = setup_model(args)
    device = first_model_device(model)

    optim = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    best_acc = -1.0
    no_improve_evals = 0
    early_stopped = False
    global_step = 0
    running_loss = 0.0
    optim.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for micro_step, batch in enumerate(pbar, start=1):
            losses = []
            for sample in batch:
                inputs = encode_example(
                    processor,
                    sample,
                    label_names,
                    preset["prompt_dataset_name"],
                    image_max_side,
                    args.max_length,
                )
                inputs = move_batch_to_device(inputs, device)
                outputs = model(**inputs)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                losses.append(float(loss.detach().cpu()) * args.grad_accum)

            running_loss += sum(losses) / max(1, len(losses))
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    running_loss = 0.0
                    print(json.dumps({"step": global_step, "epoch": epoch + 1, "train_loss": avg_loss}))

                if global_step % args.eval_steps == 0:
                    val_acc = evaluate_accuracy(
                        model,
                        processor,
                        val_ds.samples,
                        label_names,
                        preset["prompt_dataset_name"],
                        image_max_side,
                        args.max_length,
                    )
                    metrics = {"step": global_step, "epoch": epoch + 1, "val_accuracy": val_acc}
                    print(json.dumps(metrics))
                    if val_acc > best_acc + args.early_stop_min_delta:
                        best_acc = val_acc
                        no_improve_evals = 0
                        save_adapter(model, processor, output_dir / "best", metrics)
                    else:
                        no_improve_evals += 1

                    if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                        early_stopped = True
                        print(json.dumps({
                            "event": "early_stop",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "best_val_accuracy": best_acc,
                            "no_improve_evals": no_improve_evals,
                            "patience": args.early_stop_patience,
                            "min_delta": args.early_stop_min_delta,
                        }))
                        break

                if args.max_steps and global_step >= args.max_steps:
                    break
        if early_stopped or (args.max_steps and global_step >= args.max_steps):
            break

    final_metrics = {
        "step": global_step,
        "epoch": epoch + 1,
        "best_val_accuracy": best_acc,
        "early_stopped": early_stopped,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
    }
    save_adapter(model, processor, output_dir / "last", final_metrics)
    with open(output_dir / "done.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(json.dumps(final_metrics))


if __name__ == "__main__":
    main()
