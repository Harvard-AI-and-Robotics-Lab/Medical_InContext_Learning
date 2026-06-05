import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import yaml
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

load_config = None
get_dataset = None
HFLocalClient = None
MLLMClient = None
OutputParser = None
ClassificationMetrics = None
get_prompter = None
GlobalRetriever = None
SpatialRetriever = None
PROMPT_SUFFIX = ""
PROMPT_SUFFIX_METHODS = set()


DEFAULT_MANIFEST = "manifests/lag_manifest.csv"
DEFAULT_FIXED_EXEMPLARS = "manifests/fixed_exemplars_seed3407.json"


def log_progress(message: str):
    print(message, flush=True)


def parse_max_memory(raw_value):
    if not raw_value:
        return None
    parsed = {}
    for key, value in raw_value.items():
        if isinstance(key, str) and key.isdigit():
            parsed[int(key)] = value
        else:
            parsed[key] = value
    return parsed


def load_raw_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_fixed_references(dataset, fixed_exemplars_json: str):
    with open(fixed_exemplars_json, "r", encoding="utf-8") as f:
        exemplar_rows = json.load(f)

    train_samples = dataset.get_reference_pool()
    id_to_sample = {sample.id: sample for sample in train_samples}
    fixed_refs = []
    missing = []
    for row in exemplar_rows:
        sample_id = str(row["id"])
        sample = id_to_sample.get(sample_id)
        if sample is None:
            missing.append(sample_id)
            continue
        fixed_refs.append(sample)
    if missing:
        raise ValueError(f"Fixed exemplar ids not found in train split: {missing}")
    return fixed_refs


def build_prediction_record(sample, prompt_record, inference_record, parsed):
    record = {
        "query_id": sample.id,
        "ground_truth_label": sample.label,
        "ground_truth_name": sample.label_name,
        "prompt": prompt_record.to_dict(),
        "inference": inference_record.to_dict(),
        "parsed": parsed.to_dict(),
    }
    if getattr(sample, "multi_label", None) is not None:
        record["ground_truth_multi_label"] = list(sample.multi_label)
    return record


def build_raw_ledger_record(sample, prompt_record, inference_record, parsed):
    return {
        "query_id": sample.id,
        "split": sample.split,
        "ground_truth_label": sample.label,
        "ground_truth_name": sample.label_name,
        "ground_truth_multi_label": list(sample.multi_label) if getattr(sample, "multi_label", None) is not None else None,
        "method": prompt_record.method,
        "model": inference_record.model,
        "reference_ids": prompt_record.reference_ids,
        "reference_labels": prompt_record.reference_labels,
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
        json.dump(results, f, indent=2, default=str)
    with open(output_dir / "raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for row in raw_records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def build_ground_truth_record(sample):
    record = {
        "query_id": sample.id,
        "split": sample.split,
        "image_path": sample.image_path,
        "ground_truth_label": getattr(sample, "label", None),
        "ground_truth_name": getattr(sample, "label_name", ""),
        "metadata": getattr(sample, "metadata", None) or {},
    }
    if getattr(sample, "multi_label", None) is not None:
        record["ground_truth_multi_label"] = list(sample.multi_label)
    return record


def save_dataset_intermediates(dataset, query_samples: list, output_dir: Path):
    save_json(dataset.summary(), output_dir / "dataset_summary.json")
    save_json(
        [build_ground_truth_record(sample) for sample in dataset.samples],
        output_dir / "ground_truth.json",
    )
    save_json(
        [build_ground_truth_record(sample) for sample in query_samples],
        output_dir / "query_samples.json",
    )


def compute_metrics(results, dataset, metrics_engine):
    y_true = [r["ground_truth_label"] for r in results]
    y_pred = [r["parsed"]["predicted_label_idx"] for r in results]
    if dataset.is_multi_label:
        y_true_multi = [r.get("ground_truth_multi_label", []) for r in results]
        y_pred_multi = [r["parsed"].get("multi_label_predictions", []) for r in results]
        y_prob_multi = [r["parsed"].get("multi_label_confidences", []) for r in results]
        return metrics_engine.compute_multilabel(
            y_true_multi, y_pred_multi, y_prob_multi, len(dataset.label_names)
        )
    if dataset.n_classes == 2:
        y_prob = [r["parsed"].get("probability", r["parsed"]["confidence"]) for r in results]
        return metrics_engine.compute_binary(y_true, y_pred, y_prob)
    y_prob = []
    for row in results:
        parsed = row["parsed"]
        class_probs = parsed.get("class_probabilities", [])
        if len(class_probs) == dataset.n_classes:
            y_prob.append(class_probs)
            continue
        fallback = [0.0] * dataset.n_classes
        idx = int(parsed.get("predicted_label_idx", -1))
        if 0 <= idx < dataset.n_classes:
            fallback[idx] = float(parsed.get("confidence", 1.0))
            remainder = max(0.0, 1.0 - fallback[idx])
            other = remainder / max(dataset.n_classes - 1, 1)
            fallback = [other if i != idx else fallback[idx] for i in range(dataset.n_classes)]
        else:
            fallback = [1.0 / dataset.n_classes] * dataset.n_classes
        y_prob.append(fallback)
    return metrics_engine.compute_multiclass(y_true, y_pred, y_prob, dataset.n_classes)


def load_global_feature_index(features_dir: Path):
    with open(features_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    global_embeddings = __import__("numpy").load(features_dir / "global_embeddings.npy", mmap_mode="r")
    ids = [str(x) for x in metadata["ids"]]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_embedding_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
    return metadata, global_embeddings, ids, labels, splits, id_to_embedding_idx


def load_spatial_feature_metadata(features_dir: Path):
    np = __import__("numpy")
    with open(features_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    ids = [str(x) for x in metadata["ids"]]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_embedding_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
    return metadata, ids, labels, splits, id_to_embedding_idx


def load_spatial_feature_arrays(features_dir: Path, indices: list[int]):
    np = __import__("numpy")
    spatial_dir = features_dir / "spatial_features"
    log_progress(f"[startup] loading {len(indices)} reference spatial feature arrays into RAM")
    arrays = [np.load(spatial_dir / f"{idx}.npy") for idx in indices]
    log_progress(f"[startup] loaded {len(arrays)} reference spatial feature arrays")
    return arrays


def pool_spatial_tokens(feature: object, target_hw: int | None):
    if not target_hw:
        return feature
    np = __import__("numpy")
    n_tokens, hidden_dim = feature.shape
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(f"Cannot pool non-square spatial feature with {n_tokens} tokens")
    if side % target_hw != 0:
        raise ValueError(f"Cannot pool {side}x{side} spatial grid to {target_hw}x{target_hw}")
    stride = side // target_hw
    grid = feature.reshape(side, side, hidden_dim)
    pooled = grid.reshape(target_hw, stride, target_hw, stride, hidden_dim).mean(axis=(1, 3))
    return pooled.reshape(target_hw * target_hw, hidden_dim).astype(np.float32, copy=False)


def execute_prompt_batch(samples, prompt_records, client, parser, dataset):
    effective_prompt_records = [
        apply_prompt_suffix(record, PROMPT_SUFFIX)
        if not PROMPT_SUFFIX_METHODS or record.method in PROMPT_SUFFIX_METHODS
        else record
        for record in prompt_records
    ]
    batch_items = [
        {
            "messages": prompt_record.messages,
            "query_id": sample.id,
            "method": prompt_record.method,
        }
        for sample, prompt_record in zip(samples, effective_prompt_records)
    ]
    inference_records = client.infer_batch(batch_items)

    results = []
    raw_records = []
    for sample, prompt_record, inference_record in zip(samples, effective_prompt_records, inference_records):
        parsed = parser.parse_classification(
            raw_response=inference_record.raw_response,
            query_id=sample.id,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=dataset.name,
        )
        results.append(build_prediction_record(sample, prompt_record, inference_record, parsed))
        raw_records.append(build_raw_ledger_record(sample, prompt_record, inference_record, parsed))
    return results, raw_records


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


def log_batch_start(method_name: str, batch_idx: int, total_batches: int, batch_samples: list):
    ids = [sample.id for sample in batch_samples]
    log_progress(
        f"[{method_name}] starting batch {batch_idx}/{total_batches} "
        f"(n={len(batch_samples)}, ids={ids[0]}..{ids[-1]})"
    )


def log_batch_end(method_name: str, batch_idx: int, total_batches: int, elapsed_s: float):
    log_progress(
        f"[{method_name}] finished batch {batch_idx}/{total_batches} "
        f"in {elapsed_s:.2f}s"
    )

def run_zero_shot(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
    query_samples: list | None = None,
):
    prompter = get_prompter("zero_shot")
    results = []
    raw_records = []
    samples = query_samples if query_samples is not None else dataset.get_test_samples()
    test_samples = samples[:limit] if limit else samples
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    for batch_idx, start in enumerate(tqdm(batch_starts, desc="zero_shot"), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start("zero_shot", batch_idx, total_batches, batch_samples)
        prompt_records = [
            prompter.build_classification_prompt(
                query_sample=sample,
                label_names=dataset.label_names,
                is_multi_label=dataset.is_multi_label,
                dataset_name=prompt_dataset_name,
            )
            for sample in batch_samples
        ]
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end("zero_shot", batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


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


def run_naive_icl(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    seed: int,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    prompter = get_prompter("naive_icl", k=k, seed=seed)
    results = []
    raw_records = []
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    method_name = f"naive_icl_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = [
            prompter.build_classification_prompt(
                query_sample=sample,
                reference_pool=ref_pool,
                label_names=dataset.label_names,
                is_multi_label=dataset.is_multi_label,
                dataset_name=prompt_dataset_name,
                k=k,
            )
            for sample in batch_samples
        ]
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def query_random_seed(base_seed: int, query_id: str) -> int:
    digest = hashlib.md5(str(query_id).encode("utf-8")).hexdigest()
    query_offset = int(digest[:8], 16)
    return (int(base_seed) + query_offset) % (2**32 - 1)


def run_random_icl(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    seed: int,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    prompter = get_prompter("naive_icl", k=k, seed=seed)
    results = []
    raw_records = []
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    method_name = f"random_icl_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = []
        for sample in batch_samples:
            prompt_record = prompter.build_classification_prompt(
                query_sample=sample,
                reference_pool=ref_pool,
                label_names=dataset.label_names,
                is_multi_label=dataset.is_multi_label,
                dataset_name=prompt_dataset_name,
                k=k,
                rng_seed=query_random_seed(seed, sample.id),
            )
            prompt_record.method = "random_icl"
            prompt_records.append(prompt_record)
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_fixed_random_6(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    fixed_refs: list,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    prompter = get_prompter("fixed_random_6", fixed_references=fixed_refs)
    results = []
    raw_records = []
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    for batch_idx, start in enumerate(tqdm(batch_starts, desc="fixed_random_6"), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start("fixed_random_6", batch_idx, total_batches, batch_samples)
        prompt_records = [
            prompter.build_classification_prompt(
                query_sample=sample,
                fixed_references=fixed_refs,
                label_names=dataset.label_names,
                is_multi_label=dataset.is_multi_label,
                dataset_name=prompt_dataset_name,
            )
            for sample in batch_samples
        ]
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end("fixed_random_6", batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_global(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    metadata, global_embeddings, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_global_feature_index(features_dir)
    )
    prompter = get_prompter("rg_icl_global", k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_labels = [feature_labels[idx] for idx in train_indices]
    filtered_splits = [feature_splits[idx] for idx in train_indices]
    filtered_embeddings = global_embeddings[train_indices]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(
        ids=filtered_ids,
        embeddings=filtered_embeddings,
        labels=filtered_labels,
        splits=filtered_splits,
    )
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"rg_icl_global_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_embedding=global_embeddings[emb_idx],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
            retrieved_refs = [ref_by_id[ref_id] for ref_id in retrieval_result.neighbor_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(
                    f"Expected {k} retrieved references for query {sample.id}, "
                    f"but found {len(retrieved_refs)} in reference pool."
                )
            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=retrieval_result,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                )
            )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_global_knn_correction(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    metadata, global_embeddings, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_global_feature_index(features_dir)
    )
    prompter = get_prompter("rg_icl_global_knn_correction", k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_labels = [feature_labels[idx] for idx in train_indices]
    filtered_splits = [feature_splits[idx] for idx in train_indices]
    filtered_embeddings = global_embeddings[train_indices]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(
        ids=filtered_ids,
        embeddings=filtered_embeddings,
        labels=filtered_labels,
        splits=filtered_splits,
    )
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"rg_icl_global_knn_correction_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_embedding=global_embeddings[emb_idx],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
            retrieved_refs = [ref_by_id[ref_id] for ref_id in retrieval_result.neighbor_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(
                    f"Expected {k} retrieved references for query {sample.id}, "
                    f"but found {len(retrieved_refs)} in reference pool."
                )
            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=retrieval_result,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                )
            )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_global_similarity(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    prompter_name: str = "rg_icl_global_similarity",
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    metadata, global_embeddings, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_global_feature_index(features_dir)
    )
    prompter = get_prompter(prompter_name, k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_labels = [feature_labels[idx] for idx in train_indices]
    filtered_splits = [feature_splits[idx] for idx in train_indices]
    filtered_embeddings = global_embeddings[train_indices]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(
        ids=filtered_ids,
        embeddings=filtered_embeddings,
        labels=filtered_labels,
        splits=filtered_splits,
    )
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"{prompter_name}_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_embedding=global_embeddings[emb_idx],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
            retrieved_refs = [ref_by_id[ref_id] for ref_id in retrieval_result.neighbor_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(
                    f"Expected {k} retrieved references for query {sample.id}, "
                    f"but found {len(retrieved_refs)} in reference pool."
                )
            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=retrieval_result,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                )
            )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_global_balanced(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    if k % 2 != 0:
        raise ValueError(f"Balanced retrieval requires an even k, got {k}")
    per_class_k = k // 2
    np = __import__("numpy")
    metadata, global_embeddings, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_global_feature_index(features_dir)
    )
    prompter = get_prompter("rg_icl_global_balanced", k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_embeddings = np.asarray(global_embeddings[train_indices], dtype=np.float32)
    filtered_labels = np.array([ref_by_id[sample_id].label for sample_id in filtered_ids])
    norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
    filtered_embeddings = filtered_embeddings / np.maximum(norms, 1e-8)
    positive_label_idx = len(dataset.label_names) - 1
    negative_label_idx = 0
    pos_mask = filtered_labels == positive_label_idx
    neg_mask = filtered_labels == negative_label_idx
    if int(pos_mask.sum()) < per_class_k or int(neg_mask.sum()) < per_class_k:
        raise ValueError(
            f"Not enough references for balanced retrieval: "
            f"positive={int(pos_mask.sum())}, negative={int(neg_mask.sum())}, required={per_class_k}"
        )

    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"rg_icl_global_balanced_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            query_embedding = np.asarray(global_embeddings[emb_idx], dtype=np.float32)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            similarities = filtered_embeddings @ query_embedding

            pos_order = np.argsort(similarities[pos_mask])[::-1][:per_class_k]
            neg_order = np.argsort(similarities[neg_mask])[::-1][:per_class_k]
            pos_indices = np.flatnonzero(pos_mask)[pos_order]
            neg_indices = np.flatnonzero(neg_mask)[neg_order]
            selected_indices = list(pos_indices) + list(neg_indices)
            selected_indices = sorted(selected_indices, key=lambda idx: float(similarities[idx]), reverse=True)
            retrieved_refs = [ref_by_id[filtered_ids[idx]] for idx in selected_indices]
            reference_scores = [float(similarities[idx]) for idx in selected_indices]

            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=None,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                    reference_scores=reference_scores,
                )
            )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_spatial(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    startup_t0 = time.time()
    metadata, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_spatial_feature_metadata(features_dir)
    )
    prompter = get_prompter("rg_icl_spatial", k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_labels = [feature_labels[idx] for idx in train_indices]
    filtered_splits = [feature_splits[idx] for idx in train_indices]
    log_progress(
        f"[startup] spatial references={len(train_indices)} "
        f"test={len(dataset.get_test_samples())}"
    )
    load_t0 = time.time()
    filtered_spatial_features = load_spatial_feature_arrays(features_dir, train_indices)
    log_progress(f"[startup] spatial ref-array load took {time.time() - load_t0:.2f}s")
    retriever = SpatialRetriever(exclude_query=True, exclude_test_set=True)
    build_t0 = time.time()
    retriever.build_index(
        ids=filtered_ids,
        spatial_features=filtered_spatial_features,
        labels=filtered_labels,
        splits=filtered_splits,
    )
    log_progress(f"[startup] spatial index build took {time.time() - build_t0:.2f}s")
    log_progress(f"[startup] spatial pipeline ready in {time.time() - startup_t0:.2f}s")
    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"rg_icl_spatial_k{k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    spatial_dir = features_dir / "spatial_features"

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_t0 = time.time()
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            query_spatial = __import__("numpy").load(spatial_dir / f"{emb_idx}.npy")
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_spatial=query_spatial,
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
            retrieved_refs = [ref_by_id[ref_id] for ref_id in retrieval_result.neighbor_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(
                    f"Expected {k} retrieved references for query {sample.id}, "
                    f"but found {len(retrieved_refs)} in reference pool."
                )
            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=retrieval_result,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                )
            )
        log_progress(
            f"[{method_name}] retrieval+prompt batch {batch_idx}/{total_batches} "
            f"in {time.time() - prompt_t0:.2f}s"
        )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def run_rg_icl_global_spatial(
    dataset,
    client,
    parser,
    prompt_dataset_name: str,
    k: int,
    features_dir: Path,
    global_prefilter_k: int = 50,
    spatial_pool_hw: int | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    output_dir: Path | None = None,
):
    from retrieval.global_retrieval import RetrievalResult

    startup_t0 = time.time()
    global_metadata, global_embeddings, feature_ids, feature_labels, feature_splits, id_to_embedding_idx = (
        load_global_feature_index(features_dir)
    )
    spatial_metadata, spatial_feature_ids, spatial_feature_labels, spatial_feature_splits, spatial_id_to_embedding_idx = (
        load_spatial_feature_metadata(features_dir)
    )

    if feature_ids != spatial_feature_ids:
        raise ValueError("Global and spatial feature metadata ids do not match.")
    if feature_labels != spatial_feature_labels or feature_splits != spatial_feature_splits:
        raise ValueError("Global and spatial feature metadata labels/splits do not match.")

    prompter = get_prompter("rg_icl_global_spatial", k=k)
    reference_pool = dataset.get_reference_pool()
    ref_by_id = {sample.id: sample for sample in reference_pool}
    reference_ids = set(ref_by_id.keys())
    train_indices = [idx for idx, sample_id in enumerate(feature_ids) if sample_id in reference_ids]
    filtered_ids = [feature_ids[idx] for idx in train_indices]
    filtered_labels = [feature_labels[idx] for idx in train_indices]
    filtered_splits = [feature_splits[idx] for idx in train_indices]
    filtered_embeddings = global_embeddings[train_indices]
    filtered_spatial_features = load_spatial_feature_arrays(features_dir, train_indices)
    if spatial_pool_hw:
        pool_t0 = time.time()
        filtered_spatial_features = [
            pool_spatial_tokens(feat, spatial_pool_hw) for feat in filtered_spatial_features
        ]
        log_progress(
            f"[startup] pooled reference spatial features to {spatial_pool_hw}x{spatial_pool_hw} "
            f"in {time.time() - pool_t0:.2f}s"
        )
    ref_id_to_filtered_idx = {sample_id: idx for idx, sample_id in enumerate(filtered_ids)}

    log_progress(
        f"[startup] global+spatial references={len(train_indices)} "
        f"prefilter_k={global_prefilter_k} final_k={k} "
        f"spatial_pool_hw={spatial_pool_hw or 'none'}"
    )

    global_retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    global_retriever.build_index(
        ids=filtered_ids,
        embeddings=filtered_embeddings,
        labels=filtered_labels,
        splits=filtered_splits,
    )
    spatial_reranker = SpatialRetriever(exclude_query=True, exclude_test_set=True)
    log_progress(f"[startup] global+spatial pipeline ready in {time.time() - startup_t0:.2f}s")

    test_samples = dataset.get_test_samples()[:limit] if limit else dataset.get_test_samples()
    results = []
    raw_records = []
    method_name = f"rg_icl_global_spatial_k{k}_g{global_prefilter_k}"
    batch_starts = list(range(0, len(test_samples), batch_size))
    total_batches = len(batch_starts)
    spatial_dir = features_dir / "spatial_features"

    for batch_idx, start in enumerate(tqdm(batch_starts, desc=method_name), start=1):
        batch_samples = test_samples[start:start + batch_size]
        log_batch_start(method_name, batch_idx, total_batches, batch_samples)
        prompt_t0 = time.time()
        prompt_records = []
        for sample in batch_samples:
            emb_idx = id_to_embedding_idx[sample.id]
            prefilter_n = min(global_prefilter_k, len(filtered_ids))
            global_result = global_retriever.retrieve(
                query_id=sample.id,
                query_embedding=global_embeddings[emb_idx],
                k=prefilter_n,
                encoder_name=global_metadata.get("encoder_name", ""),
                encoder_version=global_metadata.get("encoder_version", ""),
                preprocessing_hash=global_metadata.get("preprocessing_hash", ""),
            )
            query_spatial = __import__("numpy").load(spatial_dir / f"{spatial_id_to_embedding_idx[sample.id]}.npy")
            query_spatial = pool_spatial_tokens(query_spatial, spatial_pool_hw)
            candidate_spatials = [
                filtered_spatial_features[ref_id_to_filtered_idx[ref_id]]
                for ref_id in global_result.neighbor_ids
            ]
            spatial_scores = spatial_reranker.score_candidates(query_spatial, candidate_spatials)
            top_candidate_order = __import__("numpy").argsort(spatial_scores)[::-1][:k]

            final_ids = [global_result.neighbor_ids[idx] for idx in top_candidate_order]
            final_scores = [float(spatial_scores[idx]) for idx in top_candidate_order]
            final_labels = [global_result.neighbor_labels[idx] for idx in top_candidate_order]
            retrieved_refs = [ref_by_id[ref_id] for ref_id in final_ids if ref_id in ref_by_id]
            if len(retrieved_refs) != k:
                raise ValueError(
                    f"Expected {k} reranked references for query {sample.id}, "
                    f"but found {len(retrieved_refs)} in reference pool."
                )

            reranked_result = RetrievalResult(
                query_id=sample.id,
                query_embedding_hash=global_result.query_embedding_hash,
                neighbor_ids=final_ids,
                neighbor_scores=final_scores,
                neighbor_labels=final_labels,
                encoder_name=global_metadata.get("encoder_name", ""),
                encoder_version=global_metadata.get("encoder_version", ""),
                preprocessing_hash=global_metadata.get("preprocessing_hash", ""),
                method="global_spatial",
            )
            prompt_records.append(
                prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=retrieved_refs,
                    retrieval_result=reranked_result,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=prompt_dataset_name,
                )
            )
        log_progress(
            f"[{method_name}] retrieval+prompt batch {batch_idx}/{total_batches} "
            f"in {time.time() - prompt_t0:.2f}s"
        )
        batch_start_time = time.time()
        batch_results, batch_raw_records = execute_prompt_batch(
            batch_samples,
            prompt_records,
            client,
            parser,
            dataset,
        )
        results.extend(batch_results)
        raw_records.extend(batch_raw_records)
        if output_dir is not None:
            save_method_outputs(results, raw_records, output_dir)
        log_batch_end(method_name, batch_idx, total_batches, time.time() - batch_start_time)
    return results, raw_records


def main():
    global load_config, get_dataset, HFLocalClient, MLLMClient, OutputParser, ClassificationMetrics, get_prompter, GlobalRetriever, SpatialRetriever, PROMPT_SUFFIX, PROMPT_SUFFIX_METHODS

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/experiments/classification_gemma_local.yaml")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_cfg = load_raw_config(args.config)
    inference_raw_cfg = raw_cfg.get("inference", {}) or {}

    def raw_or_inference(key: str, default=None):
        if key in raw_cfg:
            return raw_cfg.get(key)
        return inference_raw_cfg.get(key, default)

    cuda_visible_devices = str(raw_or_inference("cuda_visible_devices", "")).strip()
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    log_progress(f"[startup] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    if load_config is None:
        from config import load_config as _load_config
        from datasets import get_dataset as _get_dataset
        from inference import HFLocalClient as _HFLocalClient, MLLMClient as _MLLMClient, OutputParser as _OutputParser
        from metrics import ClassificationMetrics as _ClassificationMetrics
        from prompting import get_prompter as _get_prompter
        from retrieval import GlobalRetriever as _GlobalRetriever, SpatialRetriever as _SpatialRetriever

        load_config = _load_config
        get_dataset = _get_dataset
        HFLocalClient = _HFLocalClient
        MLLMClient = _MLLMClient
        OutputParser = _OutputParser
        ClassificationMetrics = _ClassificationMetrics
        get_prompter = _get_prompter
        GlobalRetriever = _GlobalRetriever
        SpatialRetriever = _SpatialRetriever

    cfg = load_config(args.config)
    run_name = Path(args.config).stem
    output_base_dir = Path(str(raw_cfg.get("output_base_dir", "outputs") or "outputs"))
    cfg.output_root = str(output_base_dir / run_name)
    save_json(
        {
            "run_name": run_name,
            "config_path": args.config,
            "output_root": cfg.output_root,
            "raw_config": raw_cfg,
        },
        Path(cfg.output_root) / "run_config.json",
    )
    log_progress(f"[startup] output_root={cfg.output_root}")

    dataset_names = args.datasets or cfg.datasets
    methods = args.methods or cfg.methods
    manifest_csv = raw_cfg.get("manifest_csv", DEFAULT_MANIFEST)
    fixed_exemplars_json = raw_cfg.get("fixed_exemplars_json", DEFAULT_FIXED_EXEMPLARS)
    prompt_dataset_name = raw_cfg.get("prompt_dataset_name", "lag")
    retrieval_features_dir = raw_cfg.get("retrieval_features_dir")
    target_label_names = raw_cfg.get("target_label_names")
    load_in_8bit = bool(raw_or_inference("load_in_8bit", True))
    torch_dtype = str(raw_or_inference("torch_dtype", "bfloat16"))
    attn_implementation = raw_or_inference("attn_implementation", "sdpa")
    device_map = raw_or_inference("device_map", "auto")
    max_memory = parse_max_memory(raw_or_inference("max_memory"))
    disable_allocator_warmup = bool(raw_or_inference("disable_allocator_warmup", False))
    processor_min_pixels = raw_or_inference("processor_min_pixels")
    processor_max_pixels = raw_or_inference("processor_max_pixels")
    if processor_min_pixels is not None:
        processor_min_pixels = int(processor_min_pixels)
    if processor_max_pixels is not None:
        processor_max_pixels = int(processor_max_pixels)
    enable_thinking = raw_or_inference("enable_thinking")
    assistant_prefill = str(raw_or_inference("assistant_prefill", "") or "")
    stop_on_json = bool(raw_or_inference("stop_on_json", False))
    disable_fla_fast_path = bool(raw_or_inference("disable_fla_fast_path", False))
    PROMPT_SUFFIX = str(raw_or_inference("prompt_suffix", "") or "")
    suffix_methods = raw_or_inference("prompt_suffix_methods", []) or []
    PROMPT_SUFFIX_METHODS = set(suffix_methods)
    client_backend = str(raw_or_inference("client_backend", "hf_local") or "hf_local").lower()
    log_progress(f"[startup] device_map={device_map} max_memory={max_memory}")
    k = int(raw_cfg.get("k", cfg.retrieval.k))
    global_prefilter_k = int(raw_cfg.get("global_prefilter_k", 50))
    spatial_pool_hw = raw_cfg.get("spatial_pool_hw")
    if spatial_pool_hw is not None:
        spatial_pool_hw = int(spatial_pool_hw)
    batch_size = int(raw_or_inference("batch_size", 1))
    zero_shot_batch_size = int(raw_or_inference("zero_shot_batch_size", batch_size))

    if client_backend in {"vllm", "openai", "openai_compatible"}:
        extra_body = raw_or_inference("extra_body", {}) or {}
        chat_template_kwargs = raw_or_inference("chat_template_kwargs", None)
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
            base_url=str(raw_or_inference("base_url", "") or ""),
            response_format=raw_or_inference("response_format", None),
            extra_body=extra_body,
            max_retries=int(raw_or_inference("max_retries", 3)),
            retry_delay=float(raw_or_inference("retry_delay", 5.0)),
            timeout=float(raw_or_inference("timeout", 300.0)),
            parallel_requests=int(raw_or_inference("parallel_requests", 1)),
            batch_delay=float(raw_or_inference("batch_delay", 0.0)),
            image_max_side=raw_or_inference("image_max_side", None),
            image_quality=int(raw_or_inference("image_quality", 95)),
        )
        log_progress(f"[startup] client_backend={client_backend} base_url={raw_or_inference('base_url', '')} model={cfg.inference.model}")
    else:
        client = HFLocalClient(
            model=cfg.inference.model,
            temperature=cfg.inference.temperature,
            max_tokens=cfg.inference.max_tokens,
            seed=cfg.inference.seed,
            top_p=cfg.inference.top_p,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
            disable_allocator_warmup=disable_allocator_warmup,
            processor_min_pixels=processor_min_pixels,
            processor_max_pixels=processor_max_pixels,
            enable_thinking=enable_thinking,
            assistant_prefill=assistant_prefill,
            stop_on_json=stop_on_json,
            disable_fla_fast_path=disable_fla_fast_path,
        )
        log_progress(f"[startup] processor={client.processor_name} model={cfg.inference.model}")
    parser = OutputParser()
    metrics_engine = ClassificationMetrics()

    all_metrics = {}

    for ds_name in dataset_names:
        dataset_kwargs = {"manifest_csv": manifest_csv}
        if target_label_names:
            dataset_kwargs["target_label_names"] = target_label_names
        dataset = get_dataset(ds_name, cfg.data_root, split="all", **dataset_kwargs)
        fixed_refs = []
        if "fixed_random_6" in methods:
            fixed_refs = load_fixed_references(dataset, fixed_exemplars_json)
        query_split = str(raw_cfg.get("query_split", "test"))
        ids_from_predictions = raw_cfg.get("ids_from_predictions")
        query_samples = select_query_samples(dataset, query_split, ids_from_predictions)
        output_dir = Path(cfg.output_root) / ds_name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_intermediates(dataset, query_samples, output_dir)
        ds_metrics = {}

        if "zero_shot" in methods:
            zero_shot_output_dir = output_dir / "zero_shot"
            results, raw_records = run_zero_shot(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                limit=args.limit,
                batch_size=zero_shot_batch_size,
                output_dir=zero_shot_output_dir,
                query_samples=query_samples,
            )
            save_method_outputs(results, raw_records, zero_shot_output_dir)
            ds_metrics["zero_shot"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "naive_icl" in methods:
            naive_output_dir = output_dir / f"naive_icl_k{k}"
            results, raw_records = run_naive_icl(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                seed=cfg.seed,
                limit=args.limit,
                batch_size=batch_size,
                output_dir=naive_output_dir,
            )
            save_method_outputs(results, raw_records, naive_output_dir)
            ds_metrics["naive_icl"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "random_icl" in methods:
            random_output_dir = output_dir / f"random_icl_k{k}"
            results, raw_records = run_random_icl(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                seed=cfg.seed,
                limit=args.limit,
                batch_size=batch_size,
                output_dir=random_output_dir,
            )
            save_method_outputs(results, raw_records, random_output_dir)
            ds_metrics["random_icl"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "fixed_random_6" in methods:
            fixed_output_dir = output_dir / "fixed_random_6"
            results, raw_records = run_fixed_random_6(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                fixed_refs=fixed_refs,
                limit=args.limit,
                batch_size=batch_size,
                output_dir=fixed_output_dir,
            )
            save_method_outputs(results, raw_records, fixed_output_dir)
            ds_metrics["fixed_random_6"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_global" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_global")
            rg_global_output_dir = output_dir / f"rg_icl_global_k{k}"
            results, raw_records = run_rg_icl_global(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_global_output_dir,
            )
            save_method_outputs(results, raw_records, rg_global_output_dir)
            ds_metrics["rg_icl_global"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_global_knn_correction" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_global_knn_correction")
            rg_knn_correction_output_dir = output_dir / f"rg_icl_global_knn_correction_k{k}"
            results, raw_records = run_rg_icl_global_knn_correction(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_knn_correction_output_dir,
            )
            save_method_outputs(results, raw_records, rg_knn_correction_output_dir)
            ds_metrics["rg_icl_global_knn_correction"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_global_similarity" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_global_similarity")
            rg_similarity_output_dir = output_dir / f"rg_icl_global_similarity_k{k}"
            results, raw_records = run_rg_icl_global_similarity(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_similarity_output_dir,
            )
            save_method_outputs(results, raw_records, rg_similarity_output_dir)
            ds_metrics["rg_icl_global_similarity"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_dual_global_similarity" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_dual_global_similarity")
            rg_dual_similarity_output_dir = output_dir / f"rg_icl_dual_global_similarity_k{k}"
            results, raw_records = run_rg_icl_global_similarity(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                prompter_name="rg_icl_dual_global_similarity",
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_dual_similarity_output_dir,
            )
            save_method_outputs(results, raw_records, rg_dual_similarity_output_dir)
            ds_metrics["rg_icl_dual_global_similarity"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_global_balanced" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_global_balanced")
            rg_balanced_output_dir = output_dir / f"rg_icl_global_balanced_k{k}"
            results, raw_records = run_rg_icl_global_balanced(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_balanced_output_dir,
            )
            save_method_outputs(results, raw_records, rg_balanced_output_dir)
            ds_metrics["rg_icl_global_balanced"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_spatial" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_spatial")
            rg_spatial_output_dir = output_dir / f"rg_icl_spatial_k{k}"
            results, raw_records = run_rg_icl_spatial(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_spatial_output_dir,
            )
            save_method_outputs(results, raw_records, rg_spatial_output_dir)
            ds_metrics["rg_icl_spatial"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        if "rg_icl_global_spatial" in methods:
            if not retrieval_features_dir:
                raise ValueError("retrieval_features_dir must be set in config for rg_icl_global_spatial")
            spatial_suffix = f"_p{spatial_pool_hw}" if spatial_pool_hw else ""
            rg_global_spatial_output_dir = output_dir / f"rg_icl_global_spatial_k{k}_g{global_prefilter_k}{spatial_suffix}"
            results, raw_records = run_rg_icl_global_spatial(
                dataset,
                client,
                parser,
                prompt_dataset_name=prompt_dataset_name,
                k=k,
                features_dir=Path(retrieval_features_dir),
                global_prefilter_k=global_prefilter_k,
                spatial_pool_hw=spatial_pool_hw,
                limit=args.limit,
                batch_size=batch_size,
                output_dir=rg_global_spatial_output_dir,
            )
            save_method_outputs(results, raw_records, rg_global_spatial_output_dir)
            ds_metrics["rg_icl_global_spatial"] = compute_metrics(results, dataset, metrics_engine).to_dict()

        all_metrics[ds_name] = ds_metrics
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(ds_metrics, f, indent=2, default=str)

    summary_path = Path(cfg.output_root) / "classification_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
