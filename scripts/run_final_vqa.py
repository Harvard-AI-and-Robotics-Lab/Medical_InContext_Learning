#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import re
import string
import time
from pathlib import Path

import yaml
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset
from inference import MLLMClient, OutputParser
from metrics import VQAMetrics
from prompting import get_prompter
from prompting.templates import VQATemplate, _image_content, _text_content
from prompting.zero_shot import PromptRecord
from retrieval import GlobalRetriever


PROMPT_SUFFIX = ""
PROMPT_SUFFIX_METHODS = set()
VQA_SYSTEM_PROMPT_OVERRIDE = None
VQA_QUERY_TEMPLATE = ""
DEFAULT_JSON_SUFFIX = (
    'Return only one compact JSON object. Include "answer", "confidence" for confidence in the answer, '
    '"evidence" of eight words max. No markdown, no extra text.'
)


def log_progress(message: str):
    print(message, flush=True)


def load_raw_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def raw_or_inference(raw_cfg: dict, inference_raw_cfg: dict, key: str, default=None):
    if key in raw_cfg:
        return raw_cfg.get(key)
    return inference_raw_cfg.get(key, default)


def load_fixed_references(dataset, fixed_exemplars_json: str):
    with open(fixed_exemplars_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    train_samples = dataset.get_reference_pool()
    id_to_sample = {sample.id: sample for sample in train_samples}
    fixed_refs = []
    missing = []
    for row in rows:
        sample_id = str(row["id"])
        sample = id_to_sample.get(sample_id)
        if sample is None:
            missing.append(sample_id)
            continue
        fixed_refs.append(sample)
    if missing:
        raise ValueError(f"Fixed exemplar ids not found in SLAKE reference split: {missing}")
    return fixed_refs


def select_query_samples(dataset, split: str = "test", ids_from_predictions: str | None = None):
    samples = [sample for sample in dataset.samples if sample.split == split]
    if not ids_from_predictions:
        return samples
    with open(ids_from_predictions, "r", encoding="utf-8") as f:
        rows = json.load(f)
    ids = [str(row.get("query_id", row.get("id"))) for row in rows]
    by_id = {str(sample.id): sample for sample in samples}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Missing {len(missing)} requested ids in split {split}: {missing[:10]}")
    return [by_id[sample_id] for sample_id in ids]


def build_prediction_record(sample, prompt_record, inference_record, parsed):
    return {
        "query_id": sample.id,
        "image_path": sample.image_path,
        "question": sample.question,
        "ground_truth_answer": sample.answer,
        "answer_type": sample.question_type,
        "sample_metadata": sample.metadata or {},
        "prompt": prompt_record.to_dict(),
        "inference": inference_record.to_dict(),
        "parsed": parsed.to_dict(),
    }


def build_raw_ledger_record(sample, prompt_record, inference_record, parsed):
    return {
        "query_id": sample.id,
        "split": sample.split,
        "image_path": sample.image_path,
        "question": sample.question,
        "ground_truth_answer": sample.answer,
        "answer_type": sample.question_type,
        "sample_metadata": sample.metadata or {},
        "method": prompt_record.method,
        "model": inference_record.model,
        "reference_ids": prompt_record.reference_ids,
        "reference_answers": prompt_record.reference_labels,
        "reference_order": prompt_record.reference_order,
        "prompt_metadata": getattr(prompt_record, "metadata", {}),
        "messages": prompt_record.messages,
        "raw_response": inference_record.raw_response,
        "parsed": parsed.to_dict(),
        "prompt_tokens": inference_record.prompt_tokens,
        "completion_tokens": inference_record.completion_tokens,
        "latency_ms": inference_record.latency_ms,
        "temperature": inference_record.temperature,
        "seed": inference_record.seed,
        "finish_reason": inference_record.finish_reason,
    }


def save_method_outputs(results, raw_records, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=True)
    with open(output_dir / "raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for row in raw_records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def apply_prompt_suffix(prompt_record, suffix: str):
    if not suffix:
        return prompt_record
    record = copy.deepcopy(prompt_record)
    for msg in reversed(record.messages):
        if msg.get("role") != "user":
            continue
        for item in reversed(msg.get("content", [])):
            if item.get("type") == "text":
                item["text"] = item["text"].rstrip() + "\n\n" + suffix.strip()
                return record
    return record


def apply_vqa_prompt_overrides(prompt_record):
    record = copy.deepcopy(prompt_record)
    if VQA_SYSTEM_PROMPT_OVERRIDE is not None:
        if str(VQA_SYSTEM_PROMPT_OVERRIDE) == "":
            record.messages = [msg for msg in record.messages if msg.get("role") != "system"]
        else:
            for msg in record.messages:
                if msg.get("role") != "system":
                    continue
                for item in msg.get("content", []):
                    if item.get("type") == "text":
                        item["text"] = str(VQA_SYSTEM_PROMPT_OVERRIDE)
    elif VQA_QUERY_TEMPLATE:
        for msg in record.messages:
            if msg.get("role") != "system":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    item["text"] = ""
    if VQA_QUERY_TEMPLATE:
        question = str(record.metadata.get("question", ""))
        for msg in reversed(record.messages):
            if msg.get("role") != "user":
                continue
            for item in reversed(msg.get("content", [])):
                if item.get("type") == "text":
                    item["text"] = VQA_QUERY_TEMPLATE.replace("{question}", question).strip()
                    return record
    return record


def execute_prompt_batch(samples, prompt_records, client, parser):
    effective_prompt_records = []
    for sample, record in zip(samples, prompt_records):
        record.metadata = dict(getattr(record, "metadata", {}) or {})
        record.metadata.setdefault("question", sample.question)
        record = apply_vqa_prompt_overrides(record)
        if not PROMPT_SUFFIX_METHODS or record.method in PROMPT_SUFFIX_METHODS:
            record = apply_prompt_suffix(record, PROMPT_SUFFIX)
        effective_prompt_records.append(record)
    batch_items = [
        {"messages": prompt_record.messages, "query_id": sample.id, "method": prompt_record.method}
        for sample, prompt_record in zip(samples, effective_prompt_records)
    ]
    inference_records = client.infer_batch(batch_items)

    results = []
    raw_records = []
    for sample, prompt_record, inference_record in zip(samples, effective_prompt_records, inference_records):
        parsed = parser.parse_vqa(inference_record.raw_response, sample.id)
        results.append(build_prediction_record(sample, prompt_record, inference_record, parsed))
        raw_records.append(build_raw_ledger_record(sample, prompt_record, inference_record, parsed))
    return results, raw_records


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_vqa_metrics(results: list) -> dict:
    refs = [row["ground_truth_answer"] for row in results]
    hyps = [row["parsed"].get("answer", "") for row in results]
    metric = VQAMetrics().compute(refs, hyps).to_dict()

    norm_matches = [
        float(normalize_answer(ref) == normalize_answer(hyp))
        for ref, hyp in zip(refs, hyps)
    ]
    parse_success = [float(row["parsed"].get("parse_success", False)) for row in results]
    metric.update({
        "normalized_exact_match": float(sum(norm_matches) / len(norm_matches)) if norm_matches else 0.0,
        "parse_success": float(sum(parse_success) / len(parse_success)) if parse_success else 0.0,
        "n_samples": len(results),
    })

    by_type = {}
    for answer_type in sorted(set(str(row.get("answer_type", "")).upper() or "UNKNOWN" for row in results)):
        indices = [
            idx for idx, row in enumerate(results)
            if (str(row.get("answer_type", "")).upper() or "UNKNOWN") == answer_type
        ]
        if not indices:
            continue
        by_type[answer_type] = {
            "n_samples": len(indices),
            "normalized_exact_match": float(sum(norm_matches[idx] for idx in indices) / len(indices)),
            "parse_success": float(sum(parse_success[idx] for idx in indices) / len(indices)),
        }
    metric["by_answer_type"] = by_type
    return metric


def log_batch_start(method_name: str, batch_idx: int, total_batches: int, batch_samples: list):
    ids = [sample.id for sample in batch_samples]
    log_progress(
        f"[{method_name}] starting batch {batch_idx}/{total_batches} "
        f"(n={len(batch_samples)}, ids={ids[0]}..{ids[-1]})"
    )


def log_batch_end(method_name: str, batch_idx: int, total_batches: int, elapsed_s: float):
    log_progress(f"[{method_name}] finished batch {batch_idx}/{total_batches} in {elapsed_s:.2f}s")


def run_zero_shot(dataset, client, parser, limit=None, batch_size=1, output_dir=None, query_samples=None):
    prompter = get_prompter("zero_shot")
    results = []
    raw_records = []
    samples = query_samples if query_samples is not None else dataset.get_test_samples()
    test_samples = samples[:limit] if limit else samples
    starts = list(range(0, len(test_samples), batch_size))
    for batch_idx, start in enumerate(tqdm(starts, desc="zero_shot"), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start("zero_shot", batch_idx, len(starts), batch_samples)
        prompt_records = [prompter.build_vqa_prompt(sample) for sample in batch_samples]
        t0 = time.time()
        batch_results, batch_raw = execute_prompt_batch(batch_samples, prompt_records, client, parser)
        results.extend(batch_results)
        raw_records.extend(batch_raw)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end("zero_shot", batch_idx, len(starts), time.time() - t0)
    return results, raw_records


def query_random_seed(base_seed: int, query_id: str) -> int:
    digest = hashlib.md5(str(query_id).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32 - 1)


def run_random_icl(dataset, client, parser, k, seed, limit=None, batch_size=1, output_dir=None, query_samples=None):
    prompter = get_prompter("naive_icl", k=k, seed=seed)
    ref_pool = dataset.get_reference_pool()
    samples = query_samples if query_samples is not None else dataset.get_test_samples()
    test_samples = samples[:limit] if limit else samples
    method_name = f"random_icl_k{k}"
    results = []
    raw_records = []
    starts = list(range(0, len(test_samples), batch_size))
    for batch_idx, start in enumerate(tqdm(starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, len(starts), batch_samples)
        prompt_records = []
        for sample in batch_samples:
            record = prompter.build_vqa_prompt(
                sample,
                reference_pool=ref_pool,
                k=k,
                rng_seed=query_random_seed(seed, sample.id),
            )
            record.method = "random_icl"
            prompt_records.append(record)
        t0 = time.time()
        batch_results, batch_raw = execute_prompt_batch(batch_samples, prompt_records, client, parser)
        results.extend(batch_results)
        raw_records.extend(batch_raw)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, len(starts), time.time() - t0)
    return results, raw_records


def run_fixed_random_6(dataset, client, parser, fixed_refs, limit=None, batch_size=1, output_dir=None, query_samples=None):
    prompter = get_prompter("fixed_random_6", fixed_references=fixed_refs)
    samples = query_samples if query_samples is not None else dataset.get_test_samples()
    test_samples = samples[:limit] if limit else samples
    method_name = "fixed_random_6"
    results = []
    raw_records = []
    starts = list(range(0, len(test_samples), batch_size))
    for batch_idx, start in enumerate(tqdm(starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, len(starts), batch_samples)
        prompt_records = [
            prompter.build_vqa_prompt(sample, fixed_references=fixed_refs)
            for sample in batch_samples
        ]
        t0 = time.time()
        batch_results, batch_raw = execute_prompt_batch(batch_samples, prompt_records, client, parser)
        results.extend(batch_results)
        raw_records.extend(batch_raw)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, len(starts), time.time() - t0)
    return results, raw_records


def load_global_feature_index(features_dir: Path):
    with open(features_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = __import__("numpy").load(features_dir / "global_embeddings.npy", mmap_mode="r")
    ids = [str(x) for x in metadata["ids"]]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
    return metadata, embeddings, ids, labels, splits, id_to_idx


def build_retrieval_vqa_prompt(query_sample, retrieved_refs, retrieval_result, method_name: str, score_label: str):
    scores_by_id = {
        str(ref_id): float(score)
        for ref_id, score in zip(retrieval_result.neighbor_ids, retrieval_result.neighbor_scores)
    }
    user_content = []
    ref_ids = []
    ref_answers = []
    ref_order = []
    neighbor_scores = []

    for idx, ref in enumerate(retrieved_refs, start=1):
        score = scores_by_id.get(str(ref.id), 0.0)
        user_content.append(_image_content(ref.image_path))
        user_content.append(
            _text_content(
                f"Reference {idx}: question = {ref.question}; answer = {ref.answer}; "
                f"{score_label} = {score:.6f}."
            )
        )
        ref_ids.append(ref.id)
        ref_answers.append(ref.answer)
        ref_order.append(idx - 1)
        neighbor_scores.append(float(score))

    user_content.append(_image_content(query_sample.image_path))
    user_content.append(
        _text_content(
            "Use the retrieved medical VQA examples as visual and question-answer context. "
            "Answer the query image question conservatively from the query image.\n\n"
            f"Question: {query_sample.question}"
        )
    )

    return PromptRecord(
        query_id=query_sample.id,
        method=method_name,
        reference_ids=ref_ids,
        reference_labels=ref_answers,
        reference_order=ref_order,
        metadata={"neighbor_scores": neighbor_scores, "score_label": score_label},
        messages=[
            {"role": "system", "content": [_text_content(VQATemplate.SYSTEM_PROMPT)]},
            {"role": "user", "content": user_content},
        ],
    )


def run_rg_icl_global_similarity(
    dataset,
    client,
    parser,
    k,
    features_dir: Path,
    method_name: str,
    score_label: str,
    limit=None,
    batch_size=1,
    output_dir=None,
    query_samples=None,
):
    metadata, embeddings, ids, labels, splits, id_to_idx = load_global_feature_index(features_dir)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    ref_indices = [idx for idx, sample_id in enumerate(ids) if sample_id in reference_ids]
    ref_ids = [ids[idx] for idx in ref_indices]
    ref_labels = [labels[idx] for idx in ref_indices]
    ref_splits = [splits[idx] for idx in ref_indices]
    ref_embeddings = embeddings[ref_indices]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(ref_ids, ref_embeddings, ref_labels, ref_splits)

    samples = query_samples if query_samples is not None else dataset.get_test_samples()
    test_samples = samples[:limit] if limit else samples
    results = []
    raw_records = []
    starts = list(range(0, len(test_samples), batch_size))
    log_progress(f"[startup] {method_name} references={len(ref_ids)} queries={len(test_samples)} features={features_dir}")
    for batch_idx, start in enumerate(tqdm(starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, len(starts), batch_samples)
        prompt_records = []
        for sample in batch_samples:
            if sample.id not in id_to_idx:
                raise KeyError(f"Missing query feature for {sample.id} in {features_dir}")
            result = retriever.retrieve(
                query_id=sample.id,
                query_embedding=embeddings[id_to_idx[sample.id]],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
            retrieved_refs = [ref_by_id[ref_id] for ref_id in result.neighbor_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(f"Expected {k} references for {sample.id}, got {len(retrieved_refs)}")
            prompt_records.append(build_retrieval_vqa_prompt(sample, retrieved_refs, result, method_name, score_label))
        t0 = time.time()
        batch_results, batch_raw = execute_prompt_batch(batch_samples, prompt_records, client, parser)
        results.extend(batch_results)
        raw_records.extend(batch_raw)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, len(starts), time.time() - t0)
    return results, raw_records


def run_method(method, dataset, client, parser, raw_cfg, cfg, query_samples, output_dir, limit, batch_size, k, fixed_refs):
    if method == "zero_shot":
        return run_zero_shot(dataset, client, parser, limit, batch_size, output_dir, query_samples)
    if method == "fixed_random_6":
        return run_fixed_random_6(dataset, client, parser, fixed_refs, limit, batch_size, output_dir, query_samples)
    if method == "random_icl":
        return run_random_icl(dataset, client, parser, k, cfg.seed, limit, batch_size, output_dir, query_samples)
    if method in {"rg_icl_global_similarity", "rg_icl_dual_global_similarity"}:
        features_dir = raw_cfg.get("retrieval_features_dir")
        if not features_dir:
            raise ValueError("retrieval_features_dir must be set for retrieval VQA methods")
        score_label = (
            "combined CLIP+DINO similarity"
            if method == "rg_icl_dual_global_similarity"
            else raw_cfg.get("retrieval_score_label", "retrieval similarity")
        )
        return run_rg_icl_global_similarity(
            dataset,
            client,
            parser,
            k,
            Path(features_dir),
            method,
            score_label,
            limit,
            batch_size,
            output_dir,
            query_samples,
        )
    raise ValueError(f"Unsupported VQA method: {method}")


def main():
    global PROMPT_SUFFIX, PROMPT_SUFFIX_METHODS, VQA_SYSTEM_PROMPT_OVERRIDE, VQA_QUERY_TEMPLATE

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_cfg = load_raw_config(args.config)
    inference_raw_cfg = raw_cfg.get("inference", {}) or {}
    cuda_visible_devices = str(raw_or_inference(raw_cfg, inference_raw_cfg, "cuda_visible_devices", "")).strip()
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    cfg = load_config(args.config)
    dataset_names = args.datasets or cfg.datasets
    methods = args.methods or cfg.methods
    k = int(raw_cfg.get("k", cfg.retrieval.k))
    batch_size = int(raw_or_inference(raw_cfg, inference_raw_cfg, "batch_size", 1))
    zero_shot_batch_size = int(raw_or_inference(raw_cfg, inference_raw_cfg, "zero_shot_batch_size", batch_size))
    PROMPT_SUFFIX = str(raw_or_inference(raw_cfg, inference_raw_cfg, "prompt_suffix", DEFAULT_JSON_SUFFIX) or "")
    PROMPT_SUFFIX_METHODS = set(raw_or_inference(raw_cfg, inference_raw_cfg, "prompt_suffix_methods", []) or [])
    VQA_SYSTEM_PROMPT_OVERRIDE = raw_or_inference(raw_cfg, inference_raw_cfg, "vqa_system_prompt", None)
    VQA_QUERY_TEMPLATE = str(raw_or_inference(raw_cfg, inference_raw_cfg, "vqa_query_template", "") or "")

    extra_body = raw_or_inference(raw_cfg, inference_raw_cfg, "extra_body", {}) or {}
    chat_template_kwargs = raw_or_inference(raw_cfg, inference_raw_cfg, "chat_template_kwargs", None)
    enable_thinking = raw_or_inference(raw_cfg, inference_raw_cfg, "enable_thinking", None)
    if chat_template_kwargs:
        extra_body = dict(extra_body)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    elif enable_thinking is not None:
        extra_body = dict(extra_body)
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

    client = MLLMClient(
        model=cfg.inference.model,
        temperature=cfg.inference.temperature,
        max_tokens=cfg.inference.max_tokens,
        seed=cfg.inference.seed,
        top_p=cfg.inference.top_p,
        api_key_env=cfg.inference.api_key_env,
        base_url=str(raw_or_inference(raw_cfg, inference_raw_cfg, "base_url", "") or ""),
        response_format=raw_or_inference(raw_cfg, inference_raw_cfg, "response_format", None),
        extra_body=extra_body,
        timeout=float(raw_or_inference(raw_cfg, inference_raw_cfg, "timeout", 300.0)),
        parallel_requests=int(raw_or_inference(raw_cfg, inference_raw_cfg, "parallel_requests", 1)),
        batch_delay=float(raw_or_inference(raw_cfg, inference_raw_cfg, "batch_delay", 0.0)),
        image_max_side=raw_or_inference(raw_cfg, inference_raw_cfg, "image_max_side", None),
        image_quality=int(raw_or_inference(raw_cfg, inference_raw_cfg, "image_quality", 95)),
    )
    parser = OutputParser()

    all_metrics = {}
    for ds_name in dataset_names:
        dataset_kwargs = {}
        manifest_json = raw_cfg.get("manifest_json", "")
        if manifest_json:
            dataset_kwargs["manifest_json"] = manifest_json
        manifest_csv = raw_cfg.get("manifest_csv", "")
        if manifest_csv:
            dataset_kwargs["manifest_csv"] = manifest_csv
        lang = raw_cfg.get("lang", "")
        if lang:
            dataset_kwargs["lang"] = lang
        dataset = get_dataset(ds_name, cfg.data_root, split="all", **dataset_kwargs)
        query_split = str(raw_cfg.get("query_split", "test"))
        query_samples = select_query_samples(dataset, query_split, raw_cfg.get("ids_from_predictions"))
        fixed_refs = []
        if "fixed_random_6" in methods:
            fixed_refs = load_fixed_references(dataset, raw_cfg["fixed_exemplars_json"])

        output_dir = Path(cfg.output_root) / ds_name
        output_dir.mkdir(parents=True, exist_ok=True)
        ds_metrics = {}
        for method in methods:
            method_batch_size = zero_shot_batch_size if method == "zero_shot" else batch_size
            method_output_dir = output_dir / (method if method != "random_icl" else f"random_icl_k{k}")
            results, raw_records = run_method(
                method,
                dataset,
                client,
                parser,
                raw_cfg,
                cfg,
                query_samples,
                method_output_dir,
                args.limit,
                method_batch_size,
                k,
                fixed_refs,
            )
            save_method_outputs(results, raw_records, method_output_dir)
            ds_metrics[method] = compute_vqa_metrics(results)

        all_metrics[ds_name] = ds_metrics
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(ds_metrics, f, indent=2, ensure_ascii=True)

    summary_path = Path(cfg.output_root) / "vqa_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=True)
    log_progress(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
