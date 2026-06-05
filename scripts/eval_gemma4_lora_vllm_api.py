#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train_gemma4_language_lora import (  # noqa: E402
    DATASET_PRESETS,
    SampleDataset,
    load_split,
    set_seed,
)
from src.inference.mllm_client import MLLMClient  # noqa: E402
from src.inference.output_parser import OutputParser  # noqa: E402
from src.prompting.templates import ClassificationTemplate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_json", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--image_max_side", type=int, default=-1)
    parser.add_argument("--timeout", type=float, default=3000.0)
    parser.add_argument("--api_key_env", default="VLLM_API_KEY")
    parser.add_argument("--response_format", choices=["none", "json_object"], default="none")
    return parser.parse_args()


def build_messages(sample, label_names, prompt_dataset_name: str):
    system_text = ClassificationTemplate.get_system_prompt(prompt_dataset_name)
    user_content = ClassificationTemplate.format_query(
        sample.image_path,
        label_names,
        is_multi_label=False,
        dataset_name=prompt_dataset_name,
        method="zero_shot",
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]


def write_outputs(metrics_path: Path, predictions_path: Path, metrics: dict, predictions: list):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    os.environ.setdefault(args.api_key_env, "EMPTY")
    set_seed(args.seed)

    preset = DATASET_PRESETS[args.dataset]
    raw_ds = load_split(args.dataset, args.split)
    label_names = list(raw_ds.label_names)
    samples = SampleDataset(raw_ds.samples, args.seed, limit=args.limit).samples
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

    metrics_path = Path(args.output_json)
    predictions_path = Path(args.predictions_json)
    predictions = []
    correct = 0
    parse_success = 0
    start_time = time.time()

    batch_starts = list(range(0, len(samples), args.batch_size))
    for batch_idx, start in enumerate(tqdm(batch_starts, desc=f"{args.dataset}:{args.split}:vllm"), start=1):
        batch_samples = samples[start : start + args.batch_size]
        batch_items = [
            {
                "messages": build_messages(sample, label_names, preset["prompt_dataset_name"]),
                "query_id": getattr(sample, "id", str(start + idx)),
                "method": "zero_shot",
            }
            for idx, sample in enumerate(batch_samples)
        ]
        batch_started = time.time()
        inference_records = client.infer_batch(batch_items, delay=0.0)

        for offset, (sample, inference_record) in enumerate(zip(batch_samples, inference_records)):
            raw_response = inference_record.raw_response or ""
            parsed = parser.parse_classification(
                raw_response=raw_response,
                query_id=getattr(sample, "id", str(start + offset)),
                label_names=label_names,
                is_multi_label=False,
                dataset_name=preset["prompt_dataset_name"],
            )
            ok = parsed.predicted_label == sample.label_name
            correct += int(ok)
            parse_success += int(parsed.parse_success)
            predictions.append(
                {
                    "index": start + offset,
                    "query_id": getattr(sample, "id", str(start + offset)),
                    "image_path": sample.image_path,
                    "label": sample.label_name,
                    "label_idx": int(sample.label),
                    "prediction": parsed.predicted_label,
                    "prediction_idx": int(parsed.predicted_label_idx),
                    "correct": ok,
                    "raw_response": raw_response,
                    "parsed": parsed.to_dict(),
                    "latency_ms": inference_record.latency_ms,
                    "finish_reason": inference_record.finish_reason,
                }
            )

        elapsed = time.time() - start_time
        processed = len(predictions)
        throughput = processed / max(elapsed, 1e-6)
        remaining = len(samples) - processed
        metrics = {
            "dataset": args.dataset,
            "split": args.split,
            "mode": "generate_vllm_api",
            "model": args.model,
            "base_url": args.base_url,
            "n": processed,
            "total_n": len(samples),
            "correct": correct,
            "accuracy": correct / max(1, processed),
            "parse_success": parse_success,
            "parse_success_rate": parse_success / max(1, processed),
            "label_names": label_names,
            "batch_size": args.batch_size,
            "max_workers": args.max_workers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "image_max_side": image_max_side,
            "elapsed_s": elapsed,
            "throughput_samples_per_s": throughput,
            "eta_s": remaining / max(throughput, 1e-6),
            "last_batch_s": time.time() - batch_started,
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        write_outputs(metrics_path, predictions_path, metrics, predictions)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

    final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_metrics["complete"] = True
    write_outputs(metrics_path, predictions_path, final_metrics, predictions)
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
