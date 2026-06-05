#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import openai

from judge_vqa_predictions import (
    DEFAULT_PROMPT_PATH,
    get_nested,
    judge_parse_success,
    load_prediction_rows,
    normalize_score,
    parse_json,
    row_key,
    summarize,
)


def build_batch_request(row: dict, index: int, args) -> dict:
    question = row.get("question", "")
    reference_answer = get_nested(row, args.reference_answer_field, "")
    model_answer = get_nested(row, args.predicted_answer_field, "")
    judge_input = {
        "question": question,
        "reference_answer": reference_answer,
        "predicted_answer": model_answer,
    }
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.judge_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(judge_input, ensure_ascii=False),
                    }
                ],
            },
        ],
        "temperature": args.temperature,
        "seed": args.seed + index,
        "response_format": {"type": "json_object"},
    }
    if args.max_completion_tokens:
        body["max_completion_tokens"] = args.max_completion_tokens
    return {
        "custom_id": batch_row_key(row, index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def batch_row_key(row: dict, index: int) -> str:
    return f"{row_key(row, index)}__row{index:06d}"


def prepare_batch_input(args) -> dict:
    rows = load_prediction_rows(args.predictions_json, args.limit)
    if rows is None:
        raise ValueError(f"Could not read predictions JSON: {args.predictions_json}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            f.write(json.dumps(build_batch_request(row, idx, args), ensure_ascii=False) + "\n")
    metadata = {
        "mode": "openai_batch_chat_completions",
        "predictions_json": str(args.predictions_json),
        "batch_input_jsonl": str(args.output_jsonl),
        "n_requests": len(rows),
        "model": args.model,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "seed": args.seed,
        "prompt_path": str(args.prompt_path),
        "reference_answer_field": args.reference_answer_field,
        "predicted_answer_field": args.predicted_answer_field,
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    return metadata


def make_client(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var: {args.api_key_env}")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    return openai.OpenAI(**client_kwargs)


def submit_batch(args) -> dict:
    client = make_client(args)
    with args.output_jsonl.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.description},
    )
    submit_info = {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
    }
    args.batch_json.parent.mkdir(parents=True, exist_ok=True)
    args.batch_json.write_text(json.dumps(submit_info, indent=2, ensure_ascii=True), encoding="utf-8")
    return submit_info


def status_batch(args) -> dict:
    client = make_client(args)
    batch_info = json.loads(args.batch_json.read_text(encoding="utf-8"))
    batch_id = args.batch_id or batch_info.get("batch_id") or batch_info.get("id")
    if not batch_id:
        raise KeyError("batch_id")
    batch = client.batches.retrieve(batch_id)
    out = json.loads(batch.model_dump_json())
    args.batch_json.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def load_batch_output(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item.get("custom_id", ""))] = item
    return out


def convert_batch_output(args) -> dict:
    rows = load_prediction_rows(args.predictions_json, args.limit)
    if rows is None:
        raise ValueError(f"Could not read predictions JSON: {args.predictions_json}")
    batch_rows = load_batch_output(args.batch_output_jsonl)
    results = []
    for idx, row in enumerate(rows):
        key = batch_row_key(row, idx)
        item = batch_rows.get(key)
        if item is None:
            continue
        response = item.get("response") or {}
        error = item.get("error")
        body = response.get("body") if isinstance(response, dict) else {}
        raw = ""
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices:
                raw = ((choices[0].get("message") or {}).get("content")) or ""
        parsed = parse_json(raw)
        parse_success = judge_parse_success(parsed)
        reference_answer = get_nested(row, args.reference_answer_field, "")
        model_answer = get_nested(row, args.predicted_answer_field, "")
        results.append(
            {
                "query_id": row.get("query_id", ""),
                "question": row.get("question", ""),
                "ground_truth_answer": reference_answer,
                "model_answer": model_answer,
                "answer_type": row.get("answer_type", ""),
                "judge_model": args.model,
                "judge_temperature": args.temperature,
                "judge_seed": args.seed + idx,
                "judge_prompt_path": str(args.prompt_path),
                "include_image": False,
                "raw_judge_response": raw,
                "parsed_judge": parsed,
                "judge_parse_success": parse_success,
                "exact_match": normalize_score(parsed.get("semantic_accuracy")) == 100 if parse_success else False,
                "batch_custom_id": key,
                "batch_error": error,
            }
        )

    args.judge_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.judge_output_jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    summary = summarize(results)
    summary.update(
        {
            "mode": "openai_batch_chat_completions",
            "batch_output_jsonl": str(args.batch_output_jsonl),
            "judge_output_jsonl": str(args.judge_output_jsonl),
            "available_predictions": len(rows),
            "judged_predictions": len(results),
            "pending_predictions": len(rows) - len(results),
        }
    )
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def download_batch_output(args) -> dict:
    client = make_client(args)
    info = status_batch(args)
    output_file_id = info.get("output_file_id")
    if not output_file_id:
        raise ValueError(f"Batch has no output_file_id yet. Current status: {info.get('status')}")
    content = client.files.content(output_file_id)
    args.batch_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    text = getattr(content, "text", None)
    if text is None:
        if hasattr(content, "read"):
            data = content.read()
        else:
            data = bytes(content)
        text = data.decode("utf-8")
    args.batch_output_jsonl.write_text(text, encoding="utf-8")
    return {"batch_output_jsonl": str(args.batch_output_jsonl), "output_file_id": output_file_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "submit", "status", "download", "convert", "all"])
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/judge/batch_input.jsonl"))
    parser.add_argument("--metadata-json", type=Path, default=Path("outputs/judge/batch_input_metadata.json"))
    parser.add_argument("--batch-json", type=Path, default=Path("outputs/judge/batch_status.json"))
    parser.add_argument("--batch-output-jsonl", type=Path, default=Path("outputs/judge/batch_output.jsonl"))
    parser.add_argument("--judge-output-jsonl", type=Path, default=Path("outputs/judge/llm_judge_batch.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/judge/llm_judge_batch_summary.json"))
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
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--description", default="vqa llm judge")
    args = parser.parse_args()
    args.judge_prompt = args.prompt_path.read_text(encoding="utf-8")

    if args.action in {"prepare", "all"}:
        print(json.dumps(prepare_batch_input(args), indent=2, ensure_ascii=True), flush=True)
    if args.action in {"submit", "all"}:
        print(json.dumps(submit_batch(args), indent=2, ensure_ascii=True), flush=True)
    if args.action == "status":
        print(json.dumps(status_batch(args), indent=2, ensure_ascii=True), flush=True)
    if args.action == "download":
        print(json.dumps(download_batch_output(args), indent=2, ensure_ascii=True), flush=True)
    if args.action == "convert":
        print(json.dumps(convert_batch_output(args), indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
