#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from judge_vqa_predictions_batch import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    convert_batch_output,
    download_batch_output,
    prepare_batch_input,
    status_batch,
    submit_batch,
)


DATASET = "vqa_rad"
MODELS = ["qwen36", "gemma4", "medgemma27b"]
METHODS = [
    {
        "key": "zero_shot",
        "dir_suffix": "noicl_fixed6_random6",
        "pred_dir": "zero_shot",
    },
    {
        "key": "fixed6",
        "dir_suffix": "noicl_fixed6_random6",
        "pred_dir": "fixed_random_6",
    },
    {
        "key": "random6",
        "dir_suffix": "noicl_fixed6_random6",
        "pred_dir": "random_icl_k6",
    },
    {
        "key": "clip_top6",
        "dir_suffix": "clip_top6",
        "pred_dir": "rg_icl_global_similarity",
    },
    {
        "key": "dinov3_cls_top6",
        "dir_suffix": "dinov3_cls_top6",
        "pred_dir": "rg_icl_global_similarity",
    },
    {
        "key": "clip_dinov3cls_top6",
        "dir_suffix": "clip_dinov3cls_top6",
        "pred_dir": "rg_icl_dual_global_similarity",
    },
]

LORA_RUN = {
    "run_key": "vqa_rad_gemma4_lora_projector512",
    "model": "gemma4",
    "method": "lora_projector512",
    "predictions_json": ROOT
    / "outputs/lora/gemma4_vqa_rad_vqa_lora_language_projector_r16_a32_lr1e-4_seed3407_max512"
    / "test_generation_vllm_merged_max512/predictions.json",
    "batch_dir": ROOT / "outputs/judge/vqa_rad_gemma4_lora_projector512_gpt54mini_batch",
    "judge_output_jsonl": ROOT / "outputs/judge/vqa_rad_gemma4_lora_projector512_gpt54mini_full.jsonl",
    "summary_json": ROOT / "outputs/judge/vqa_rad_gemma4_lora_projector512_gpt54mini_full_summary.json",
}

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def build_runs(include_lora: bool = True) -> list[dict]:
    runs = []
    for model in MODELS:
        for method in METHODS:
            run_key = f"{DATASET}_{model}_{method['key']}"
            predictions_json = (
                ROOT
                / "outputs"
                / "final"
                / f"{DATASET}_{model}_{method['dir_suffix']}"
                / DATASET
                / method["pred_dir"]
                / "predictions.json"
            )
            judge_dir = ROOT / "outputs" / "judge" / f"{DATASET}_main"
            batch_dir = ROOT / "outputs" / "judge" / f"{DATASET}_batch" / run_key
            runs.append(
                {
                    "run_key": run_key,
                    "model": model,
                    "method": method["key"],
                    "predictions_json": predictions_json,
                    "batch_input_jsonl": batch_dir / "batch_input.jsonl",
                    "metadata_json": batch_dir / "batch_input_metadata.json",
                    "batch_json": batch_dir / "batch_status.json",
                    "batch_output_jsonl": batch_dir / "batch_output.jsonl",
                    "judge_output_jsonl": judge_dir / f"{run_key}_gpt54mini.jsonl",
                    "summary_json": judge_dir / f"{run_key}_gpt54mini_summary.json",
                }
            )
    if include_lora:
        lora_run = dict(LORA_RUN)
        lora_run.update(
            {
                "batch_input_jsonl": LORA_RUN["batch_dir"] / "batch_input.jsonl",
                "metadata_json": LORA_RUN["batch_dir"] / "batch_input_metadata.json",
                "batch_json": LORA_RUN["batch_dir"] / "batch_status.json",
                "batch_output_jsonl": LORA_RUN["batch_dir"] / "batch_output.jsonl",
            }
        )
        runs.append(lora_run)
    return runs


def make_args(base_args, run: dict) -> SimpleNamespace:
    return SimpleNamespace(
        predictions_json=run["predictions_json"],
        output_jsonl=run["batch_input_jsonl"],
        metadata_json=run["metadata_json"],
        batch_json=run["batch_json"],
        batch_output_jsonl=run["batch_output_jsonl"],
        judge_output_jsonl=run["judge_output_jsonl"],
        summary_json=run["summary_json"],
        prompt_path=base_args.prompt_path,
        reference_answer_field=base_args.reference_answer_field,
        predicted_answer_field=base_args.predicted_answer_field,
        model=base_args.model,
        base_url=base_args.base_url,
        api_key_env=base_args.api_key_env,
        temperature=base_args.temperature,
        max_completion_tokens=base_args.max_completion_tokens,
        seed=base_args.seed,
        limit=base_args.limit,
        batch_id="",
        description=f"{run['run_key']} GPT LLM judge",
        judge_prompt=base_args.prompt_path.read_text(encoding="utf-8"),
    )


def selected_runs(args) -> list[dict]:
    runs = build_runs(include_lora=not args.no_lora)
    if args.model_key:
        allowed = set(args.model_key)
        runs = [run for run in runs if run["model"] in allowed]
    if args.method_key:
        allowed = set(args.method_key)
        runs = [run for run in runs if run["method"] in allowed]
    return runs


def read_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_prepare(args, runs: list[dict]) -> list[dict]:
    records = []
    for run in runs:
        if not run["predictions_json"].exists():
            records.append({"run_key": run["run_key"], "action": "prepare", "status": "missing_predictions"})
            continue
        result = prepare_batch_input(make_args(args, run))
        records.append({"run_key": run["run_key"], "action": "prepare", **result})
    return records


def run_submit(args, runs: list[dict]) -> list[dict]:
    records = []
    for run in runs:
        if not run["batch_input_jsonl"].exists():
            records.append({"run_key": run["run_key"], "action": "submit", "status": "missing_batch_input"})
            continue
        existing = read_status(run["batch_json"])
        existing_status = existing.get("status")
        if existing.get("batch_id") and existing_status not in TERMINAL_STATUSES and not args.force:
            records.append(
                {
                    "run_key": run["run_key"],
                    "action": "submit",
                    "status": "already_submitted",
                    "batch_id": existing.get("batch_id"),
                    "batch_status": existing_status,
                }
            )
            continue
        result = submit_batch(make_args(args, run))
        records.append({"run_key": run["run_key"], "action": "submit", **result})
    return records


def run_status(args, runs: list[dict]) -> list[dict]:
    records = []
    for run in runs:
        if not run["batch_json"].exists():
            records.append({"run_key": run["run_key"], "action": "status", "status": "missing_batch_json"})
            continue
        result = status_batch(make_args(args, run))
        records.append(
            {
                "run_key": run["run_key"],
                "action": "status",
                "batch_id": result.get("id") or result.get("batch_id"),
                "status": result.get("status"),
                "request_counts": result.get("request_counts", {}),
                "output_file_id": result.get("output_file_id"),
                "error_file_id": result.get("error_file_id"),
            }
        )
    return records


def run_download(args, runs: list[dict]) -> list[dict]:
    records = []
    for run in runs:
        status = read_status(run["batch_json"])
        if status.get("status") != "completed":
            records.append(
                {
                    "run_key": run["run_key"],
                    "action": "download",
                    "status": "not_completed",
                    "batch_status": status.get("status", "missing_batch_json"),
                }
            )
            continue
        result = download_batch_output(make_args(args, run))
        records.append({"run_key": run["run_key"], "action": "download", "status": "downloaded", **result})
    return records


def run_convert(args, runs: list[dict]) -> list[dict]:
    records = []
    for run in runs:
        if not run["batch_output_jsonl"].exists():
            records.append({"run_key": run["run_key"], "action": "convert", "status": "missing_batch_output"})
            continue
        result = convert_batch_output(make_args(args, run))
        records.append({"run_key": run["run_key"], "action": "convert", "status": "converted", **result})
    return records


def write_report(records: list[dict], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "n_records": len(records)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["prepare", "submit", "status", "download", "convert", "all", "poll"],
        help="all = prepare then submit; poll = status/download/convert loop for completed batches",
    )
    parser.add_argument("--model-key", action="append", choices=MODELS)
    parser.add_argument("--method-key", action="append", choices=[m["key"] for m in METHODS] + ["lora_projector512"])
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--reference-answer-field", default="ground_truth_answer")
    parser.add_argument("--predicted-answer-field", default="parsed.answer")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=300.0)
    parser.add_argument("--max-poll-minutes", type=float, default=0.0)
    parser.add_argument("--report-path", type=Path, default=ROOT / "outputs/judge/vqa_rad_batch/report.json")
    args = parser.parse_args()

    runs = selected_runs(args)
    records = []
    if args.action in {"prepare", "all"}:
        records.extend(run_prepare(args, runs))
    if args.action in {"submit", "all"}:
        records.extend(run_submit(args, runs))
    if args.action == "status":
        records.extend(run_status(args, runs))
    if args.action == "download":
        records.extend(run_download(args, runs))
    if args.action == "convert":
        records.extend(run_convert(args, runs))
    if args.action == "poll":
        started = time.time()
        while True:
            records.extend(run_status(args, runs))
            records.extend(run_download(args, runs))
            records.extend(run_convert(args, runs))
            statuses = [read_status(run["batch_json"]).get("status") for run in runs]
            if statuses and all(status in TERMINAL_STATUSES for status in statuses):
                break
            if args.max_poll_minutes and (time.time() - started) >= args.max_poll_minutes * 60:
                break
            time.sleep(args.poll_interval)

    write_report(records, args.report_path)


if __name__ == "__main__":
    main()
