#!/usr/bin/env python3
import argparse
import base64
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import openai
from PIL import Image


DEFAULT_PROMPT_PATH = Path("configs/judge/vqa_llm_judge_prompt_v1.txt")
SCORE_KEYS = ["semantic_accuracy", "completeness", "factuality", "conciseness"]


def encode_image(path: Path, image_max_side: int | None = None, image_quality: int = 95) -> tuple[str, str]:
    if image_max_side is None:
        ext = path.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        return base64.b64encode(path.read_bytes()).decode("utf-8"), ext

    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > image_max_side:
            image.thumbnail((image_max_side, image_max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=image_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "jpeg"


def parse_json(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {}


def get_nested(row: dict, path: str, default=""):
    value = row
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        else:
            return default
    return value


def normalize_score(value):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 <= score <= 100:
        return score
    return None


def judge_parse_success(parsed: dict) -> bool:
    return all(normalize_score(parsed.get(key)) is not None for key in SCORE_KEYS)


def judge_one(client, args, row: dict, index: int) -> dict:
    question = row.get("question", "")
    reference_answer = get_nested(row, args.reference_answer_field, "")
    model_answer = get_nested(row, args.predicted_answer_field, "")
    judge_input = {
        "question": question,
        "reference_answer": reference_answer,
        "predicted_answer": model_answer,
    }

    user_content = [{"type": "text", "text": json.dumps(judge_input, ensure_ascii=False)}]
    if args.include_image:
        image_path = Path(row.get("image_path", ""))
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        b64, ext = encode_image(
            image_path,
            None if args.image_max_side < 0 else args.image_max_side,
            args.image_quality,
        )
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}})

    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.judge_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": args.temperature,
        "seed": args.seed + index,
        "response_format": {"type": "json_object"},
    }

    raw = ""
    latency = 0.0
    for attempt in range(args.max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(**request)
            latency = (time.time() - start) * 1000.0
            raw = response.choices[0].message.content
            break
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError):
            if attempt == args.max_retries - 1:
                raise
            time.sleep(args.retry_delay * (attempt + 1))

    parsed = parse_json(raw)
    parse_success = judge_parse_success(parsed)
    return {
        "query_id": row.get("query_id", ""),
        "question": question,
        "ground_truth_answer": reference_answer,
        "model_answer": model_answer,
        "answer_type": row.get("answer_type", ""),
        "judge_model": args.model,
        "judge_temperature": args.temperature,
        "judge_seed": args.seed + index,
        "judge_prompt_path": str(args.prompt_path),
        "include_image": bool(args.include_image),
        "raw_judge_response": raw,
        "parsed_judge": parsed,
        "judge_parse_success": parse_success,
        "exact_match": normalize_score(parsed.get("semantic_accuracy")) == 100 if parse_success else False,
        "latency_ms": latency,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    valid_rows = [row for row in rows if row.get("judge_parse_success")]
    out["n_parse_success"] = len(valid_rows)
    out["parse_success_rate"] = len(valid_rows) / len(rows)
    out["exact_match_accuracy"] = (
        sum(1 for row in valid_rows if row.get("exact_match")) / len(valid_rows)
        if valid_rows
        else 0.0
    )
    out["n_exact_match"] = sum(1 for row in valid_rows if row.get("exact_match"))
    semantic_ge80 = [
        row for row in valid_rows
        if normalize_score(row.get("parsed_judge", {}).get("semantic_accuracy")) is not None
        and normalize_score(row.get("parsed_judge", {}).get("semantic_accuracy")) >= 80
    ]
    out["semantic_accuracy_ge80_accuracy"] = len(semantic_ge80) / len(valid_rows) if valid_rows else 0.0
    out["n_semantic_accuracy_ge80"] = len(semantic_ge80)
    for key in SCORE_KEYS:
        vals = []
        for row in valid_rows:
            score = normalize_score(row.get("parsed_judge", {}).get(key))
            if score is not None:
                vals.append(float(score))
        out[f"mean_{key}"] = sum(vals) / len(vals) if vals else 0.0
        out[f"{key}_100_accuracy"] = (
            sum(1 for val in vals if val == 100.0) / len(valid_rows)
            if valid_rows
            else 0.0
        )
        out[f"n_{key}_100"] = sum(1 for val in vals if val == 100.0)

    by_answer_type = {}
    for row in rows:
        answer_type = row.get("answer_type", "") or "UNKNOWN"
        bucket = by_answer_type.setdefault(answer_type, {"n": 0, "n_parse_success": 0, "n_exact_match": 0})
        bucket["n"] += 1
        if row.get("judge_parse_success"):
            bucket["n_parse_success"] += 1
            if row.get("exact_match"):
                bucket["n_exact_match"] += 1
            semantic_score = normalize_score(row.get("parsed_judge", {}).get("semantic_accuracy"))
            if semantic_score is not None and semantic_score >= 80:
                bucket["n_semantic_accuracy_ge80"] = bucket.get("n_semantic_accuracy_ge80", 0) + 1
    for bucket in by_answer_type.values():
        bucket.setdefault("n_semantic_accuracy_ge80", 0)
        bucket["parse_success_rate"] = bucket["n_parse_success"] / bucket["n"] if bucket["n"] else 0.0
        bucket["exact_match_accuracy"] = (
            bucket["n_exact_match"] / bucket["n_parse_success"]
            if bucket["n_parse_success"]
            else 0.0
        )
        bucket["semantic_accuracy_ge80_accuracy"] = (
            bucket["n_semantic_accuracy_ge80"] / bucket["n_parse_success"]
            if bucket["n_parse_success"]
            else 0.0
        )
    out["by_answer_type"] = by_answer_type
    return out


def row_key(row: dict, index: int | None = None) -> str:
    key = row.get("query_id", "")
    if key:
        return str(key)
    return f"__row_{index}" if index is not None else ""


def load_prediction_rows(path: Path, limit: int | None = None) -> list[dict] | None:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if limit:
        rows = rows[:limit]
    return rows


def load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row_key(row, line_idx)
            if key:
                results[key] = row
    return results


def ordered_results(rows: list[dict], results_by_key: dict[str, dict]) -> list[dict]:
    ordered = []
    for idx, row in enumerate(rows):
        result = results_by_key.get(row_key(row, idx))
        if result is not None:
            ordered.append(result)
    return ordered


def write_results(output_jsonl: Path, summary_json: Path, rows: list[dict], results_by_key: dict[str, dict], extra: dict | None = None):
    results = ordered_results(rows, results_by_key)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    summary = summarize(results)
    if extra:
        summary.update(extra)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def judge_available_rows(client, args, rows: list[dict], output_jsonl: Path, summary_json: Path, results_by_key: dict[str, dict]):
    pending = [
        (idx, row)
        for idx, row in enumerate(rows)
        if row_key(row, idx) not in results_by_key
    ]
    if not pending:
        return results_by_key, 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(judge_one, client, args, row, idx): (idx, row)
            for idx, row in pending
        }
        for future in as_completed(futures):
            idx, row = futures[future]
            result = future.result()
            results_by_key[row_key(row, idx)] = result
            write_results(
                output_jsonl,
                summary_json,
                rows,
                results_by_key,
                extra={
                    "available_predictions": len(rows),
                    "judged_predictions": len(ordered_results(rows, results_by_key)),
                    "pending_predictions": len(rows) - len(ordered_results(rows, results_by_key)),
                },
            )
    return results_by_key, len(pending)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--reference-answer-field", default="ground_truth_answer")
    parser.add_argument("--predicted-answer-field", default="parsed.answer")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--image-max-side", type=int, default=-1, help="-1 keeps original image bytes.")
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument("--watch-predictions", action="store_true", help="Keep polling a growing predictions JSON and judge newly available rows.")
    parser.add_argument("--expected-n", type=int, default=0, help="Stop watch mode after this many predictions are available and judged.")
    parser.add_argument("--watch-interval", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output_jsonl instead of skipping already judged rows.")
    args = parser.parse_args()

    args.judge_prompt = args.prompt_path.read_text(encoding="utf-8")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var: {args.api_key_env}")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = openai.OpenAI(**client_kwargs)

    output_jsonl = args.output_jsonl or args.predictions_json.with_name("llm_judge.jsonl")
    summary_json = args.summary_json or args.predictions_json.with_name("llm_judge_summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    lock_path = output_jsonl.with_suffix(output_jsonl.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        results_by_key = {} if args.no_resume else load_existing_results(output_jsonl)

        while True:
            rows = load_prediction_rows(args.predictions_json, args.limit)
            if rows is None:
                if not args.watch_predictions:
                    raise ValueError(f"Could not read predictions JSON: {args.predictions_json}")
                time.sleep(args.watch_interval)
                continue

            results_by_key, judged_now = judge_available_rows(
                client,
                args,
                rows,
                output_jsonl,
                summary_json,
                results_by_key,
            )
            judged = len(ordered_results(rows, results_by_key))
            expected_n = args.expected_n or len(rows)
            summary = write_results(
                output_jsonl,
                summary_json,
                rows,
                results_by_key,
                extra={
                    "available_predictions": len(rows),
                    "expected_predictions": expected_n,
                    "judged_predictions": judged,
                    "pending_predictions": len(rows) - judged,
                    "watch_predictions": bool(args.watch_predictions),
                },
            )
            print(json.dumps({"output_jsonl": str(output_jsonl), "summary_json": str(summary_json), "judged_now": judged_now, **summary}, indent=2), flush=True)

            if not args.watch_predictions:
                break
            if len(rows) >= expected_n and judged >= expected_n:
                break
            time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
