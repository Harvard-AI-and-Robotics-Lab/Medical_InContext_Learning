#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.datasets import get_dataset  # noqa: E402
from src.inference.mllm_client import MLLMClient  # noqa: E402
from src.inference.output_parser import OutputParser  # noqa: E402
from src.prompting.templates import VQATemplate  # noqa: E402


DATASET_PRESETS = {
    "slake": {
        "dataset_name": "slake",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/slake/manifest_en.json",
        "image_max_side": None,
    },
    "pathvqa": {
        "dataset_name": "pathvqa",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/pathvqa/manifest.json",
        "image_max_side": None,
    },
    "vqamed2019": {
        "dataset_name": "vqamed2019",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/vqamed2019/manifest.json",
        "image_max_side": None,
    },
    "vqa_rad": {
        "dataset_name": "vqa_rad",
        "data_root": "data/vqa",
        "manifest_json": "data/vqa/vqa_rad/manifest.json",
        "image_max_side": None,
    },
}

JSON_INSTRUCTION = (
    'Return only one compact JSON object with key "answer". '
    'No markdown, no extra text.'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--image_max_side", type=int, default=-1)
    parser.add_argument("--timeout", type=float, default=3000.0)
    parser.add_argument("--api_key_env", default="VLLM_API_KEY")
    parser.add_argument("--response_format", choices=["none", "json_object"], default="json_object")
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def absolute_project_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_partial_json_answer(raw_response: str) -> str:
    match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', raw_response, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    try:
        return json.loads(f'"{value}"').strip()
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n").strip()


def parsed_answer_from_response(parser: OutputParser, raw_response: str, query_id: str):
    parsed = parser.parse_vqa(raw_response=raw_response, query_id=query_id)
    if parsed.answer == raw_response.strip() and raw_response.lstrip().startswith("{"):
        answer = extract_partial_json_answer(raw_response)
        if answer:
            parsed.answer = answer
            parsed.parse_success = True
    return parsed


def load_vqa_dataset(dataset_key: str):
    preset = DATASET_PRESETS[dataset_key]
    kwargs = {"manifest_json": absolute_project_path(preset["manifest_json"])}
    return get_dataset(
        preset["dataset_name"],
        data_root=absolute_project_path(preset["data_root"]),
        split="all",
        **kwargs,
    )


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


def build_messages(sample, args):
    user_content = build_user_content(sample, args.vqa_query_template, args.json_instruction)
    system_prompt = VQATemplate.SYSTEM_PROMPT if args.vqa_system_prompt is None else args.vqa_system_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})
    return messages


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def append_raw_records(path: Path, rows: list[dict], mode: str = "a"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(args, predictions: list[dict], start_time: float, total_n: int):
    processed = len(predictions)
    parse_success = sum(1 for row in predictions if row.get("parsed", {}).get("parse_success"))
    exact = sum(1 for row in predictions if row.get("normalized_exact"))
    elapsed = time.time() - start_time
    throughput = processed / max(elapsed, 1e-6)
    remaining = max(0, total_n - processed)

    by_answer_type = {}
    for row in predictions:
        answer_type = row.get("answer_type") or "UNKNOWN"
        bucket = by_answer_type.setdefault(
            answer_type,
            {"n": 0, "parse_success": 0, "normalized_exact": 0},
        )
        bucket["n"] += 1
        bucket["parse_success"] += int(row.get("parsed", {}).get("parse_success", False))
        bucket["normalized_exact"] += int(row.get("normalized_exact", False))
    for bucket in by_answer_type.values():
        bucket["parse_success_rate"] = bucket["parse_success"] / max(1, bucket["n"])
        bucket["normalized_exact_accuracy"] = bucket["normalized_exact"] / max(1, bucket["n"])

    return {
        "dataset": args.dataset,
        "split": args.split,
        "mode": "generate_vllm_api_lora_merged",
        "model": args.model,
        "base_url": args.base_url,
        "n": processed,
        "total_n": total_n,
        "parse_success": parse_success,
        "parse_success_rate": parse_success / max(1, processed),
        "normalized_exact": exact,
        "normalized_exact_accuracy": exact / max(1, processed),
        "by_answer_type": by_answer_type,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "image_max_side": None if args.image_max_side < 0 else args.image_max_side,
        "elapsed_s": elapsed,
        "throughput_samples_per_s": throughput,
        "eta_s": remaining / max(throughput, 1e-6),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def load_existing(predictions_path: Path, raw_records_path: Path, resume: bool):
    if not resume or not predictions_path.exists():
        if raw_records_path.exists():
            raw_records_path.unlink()
        return []
    return json.loads(predictions_path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    os.environ.setdefault(args.api_key_env, "EMPTY")
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.json"
    raw_records_path = output_dir / "raw_records.jsonl"
    summary_path = output_dir / "vqa_summary.json"

    dataset = load_vqa_dataset(args.dataset)
    samples = [sample for sample in dataset.samples if sample.split == args.split]
    if args.limit and len(samples) > args.limit:
        samples = random.Random(args.seed).sample(samples, args.limit)

    predictions = load_existing(predictions_path, raw_records_path, args.resume)
    done_ids = {row.get("query_id") for row in predictions}
    pending_samples = [sample for sample in samples if sample.id not in done_ids]

    preset = DATASET_PRESETS[args.dataset]
    image_max_side = preset["image_max_side"] if args.image_max_side < 0 else args.image_max_side
    client = MLLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_p=args.top_p,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        response_format=None if args.response_format == "none" else args.response_format,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=args.timeout,
        parallel_requests=args.max_workers,
        batch_delay=0.0,
        image_max_side=image_max_side,
    )
    parser = OutputParser()

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if predictions:
        write_json(predictions_path, predictions)
        write_json(summary_path, summarize(args, predictions, start_time, len(samples)))

    batch_starts = list(range(0, len(pending_samples), args.batch_size))
    raw_mode = "a" if args.resume and raw_records_path.exists() else "w"
    for start in tqdm(batch_starts, desc=f"{args.dataset}:{args.split}:lora-vllm"):
        batch_samples = pending_samples[start : start + args.batch_size]
        batch_items = [
            {
                "messages": build_messages(sample, args),
                "query_id": sample.id,
                "method": "lora_vqa_sft",
            }
            for sample in batch_samples
        ]
        batch_started = time.time()
        inference_records = client.infer_batch(batch_items, delay=0.0)
        batch_rows = []

        for sample, inference_record in zip(batch_samples, inference_records):
            raw_response = inference_record.raw_response or ""
            parsed = parsed_answer_from_response(parser, raw_response, sample.id)
            pred_norm = normalize_answer(parsed.answer)
            ref_norm = normalize_answer(sample.answer)
            row = {
                "query_id": sample.id,
                "image_path": sample.image_path,
                "split": sample.split,
                "question": sample.question,
                "ground_truth_answer": sample.answer,
                "answer_type": sample.question_type,
                "prediction": parsed.answer,
                "normalized_prediction": pred_norm,
                "normalized_ground_truth": ref_norm,
                "normalized_exact": pred_norm == ref_norm,
                "raw_response": raw_response,
                "parsed": parsed.to_dict(),
                "latency_ms": inference_record.latency_ms,
                "finish_reason": inference_record.finish_reason,
                "prompt": {
                    "system": VQATemplate.SYSTEM_PROMPT if args.vqa_system_prompt is None else args.vqa_system_prompt,
                    "query_template": args.vqa_query_template,
                    "json_instruction": args.json_instruction,
                },
            }
            predictions.append(row)
            batch_rows.append(row)

        summary = summarize(args, predictions, start_time, len(samples))
        summary["last_batch_s"] = time.time() - batch_started
        write_json(predictions_path, predictions)
        append_raw_records(raw_records_path, batch_rows, mode=raw_mode)
        raw_mode = "a"
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)

    summary = summarize(args, predictions, start_time, len(samples))
    summary["complete"] = len(predictions) == len(samples)
    write_json(predictions_path, predictions)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
