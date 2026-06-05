#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from judge_vqa_predictions_batch import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    convert_batch_output,
    download_batch_output,
    status_batch,
)


RUNS = {
    "pathvqa_lora_projector512": {
        "predictions_json": ROOT
        / "outputs/lora/gemma4_pathvqa_vqa_lora_language_projector_r16_a32_lr1e-4_seed3407_max512/test_generation_vllm_merged_max512/predictions.json",
        "batch_dir": ROOT / "outputs/judge/pathvqa_gemma4_lora_projector512_gpt54mini_batch",
        "judge_output_jsonl": ROOT / "outputs/judge/pathvqa_gemma4_lora_projector512_gpt54mini_full.jsonl",
        "summary_json": ROOT / "outputs/judge/pathvqa_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
    "vqamed2019_lora_projector512": {
        "predictions_json": ROOT
        / "outputs/lora/gemma4_vqamed2019_vqa_lora_language_projector_r16_a32_lr1e-4_seed3407_max512/test_generation_vllm_merged_max512/predictions.json",
        "batch_dir": ROOT / "outputs/judge/vqamed2019_gemma4_lora_projector512_gpt54mini_batch",
        "judge_output_jsonl": ROOT / "outputs/judge/vqamed2019_gemma4_lora_projector512_gpt54mini_full.jsonl",
        "summary_json": ROOT / "outputs/judge/vqamed2019_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
}


def make_args(run: dict, args: argparse.Namespace) -> SimpleNamespace:
    batch_dir = run["batch_dir"]
    return SimpleNamespace(
        predictions_json=run["predictions_json"],
        output_jsonl=batch_dir / "batch_input.jsonl",
        metadata_json=batch_dir / "batch_input_metadata.json",
        batch_json=batch_dir / "batch_status.json",
        batch_output_jsonl=batch_dir / "batch_output.jsonl",
        judge_output_jsonl=run["judge_output_jsonl"],
        summary_json=run["summary_json"],
        prompt_path=args.prompt_path,
        reference_answer_field=args.reference_answer_field,
        predicted_answer_field=args.predicted_answer_field,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
        limit=None,
        batch_id="",
        description="",
        judge_prompt=args.prompt_path.read_text(encoding="utf-8"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--reference-answer-field", default="ground_truth_answer")
    parser.add_argument("--predicted-answer-field", default="parsed.answer")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--max-poll-minutes", type=float, default=0.0)
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Missing {args.api_key_env}")

    start = time.time()
    while True:
        done = 0
        print(f"[poll] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        for key, run in RUNS.items():
            run_args = make_args(run, args)
            if run_args.summary_json.exists():
                done += 1
                print(f"[done] {key}: {run_args.summary_json}", flush=True)
                continue
            status = status_batch(run_args)
            batch_status = status.get("status")
            counts = status.get("request_counts") or {}
            print(f"[status] {key}: {batch_status} {counts}", flush=True)
            if batch_status == "completed":
                download_batch_output(run_args)
                summary = convert_batch_output(run_args)
                done += 1
                print(
                    f"[converted] {key}: exact={summary.get('exact_match_accuracy')} "
                    f"sem80={summary.get('semantic_accuracy_ge80_accuracy')}",
                    flush=True,
                )
            elif batch_status in {"failed", "expired", "cancelled"}:
                print(json.dumps(status, indent=2), flush=True)
                return 2
        if done == len(RUNS):
            return 0
        if args.max_poll_minutes and (time.time() - start) > args.max_poll_minutes * 60:
            return 3
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
