#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset
from prompting import get_prompter
from prompting.templates import VQATemplate, _image_content, _text_content
from prompting.zero_shot import PromptRecord
from retrieval import GlobalRetriever


CLASSIFICATION_SUITES = {
    "lag": [
        "configs/final/qwen36_noicl_fixed6_random6.yaml",
        "configs/final/qwen36_clip_top6.yaml",
        "configs/final/qwen36_dinov3_cls_top6.yaml",
        "configs/final/qwen36_clip_dinov3cls_top6.yaml",
    ],
    "breakhis_binary": [
        "configs/final/breakhis_binary_qwen36_noicl_fixed6_random6.yaml",
        "configs/final/breakhis_binary_qwen36_clip_top6.yaml",
        "configs/final/breakhis_binary_qwen36_dinov3_cls_top6.yaml",
        "configs/final/breakhis_binary_qwen36_clip_dinov3cls_top6.yaml",
    ],
    "tbx11k": [
        "configs/final/tbx11k_qwen36_noicl_fixed6_random6.yaml",
        "configs/final/tbx11k_qwen36_clip_top6.yaml",
        "configs/final/tbx11k_qwen36_dinov3_cls_top6.yaml",
        "configs/final/tbx11k_qwen36_clip_dinov3cls_top6.yaml",
    ],
    "ddr_512": [
        "configs/final/ddr_qwen36_noicl_fixed6_random6_512.yaml",
        "configs/final/ddr_qwen36_clip_top6_512.yaml",
        "configs/final/ddr_qwen36_dinov3_cls_top6_512.yaml",
        "configs/final/ddr_qwen36_clip_dinov3cls_top6_512.yaml",
    ],
}

VQA_SUITES = {
    "slake": [
        "configs/final/slake_qwen36_noicl_fixed6_random6.yaml",
        "configs/final/slake_qwen36_clip_top6.yaml",
        "configs/final/slake_qwen36_dinov3_cls_top6.yaml",
        "configs/final/slake_qwen36_clip_dinov3cls_top6.yaml",
    ],
    "pathvqa": [
        "configs/final/pathvqa_qwen36_noicl_fixed6_random6.yaml",
        "configs/final/pathvqa_qwen36_clip_top6.yaml",
        "configs/final/pathvqa_qwen36_dinov3_cls_top6.yaml",
        "configs/final/pathvqa_qwen36_clip_dinov3cls_top6.yaml",
    ],
    "vqamed2019": [
        "configs/final/vqamed2019_qwen36_noicl_fixed6_random6.yaml",
        "configs/final/vqamed2019_qwen36_clip_top6.yaml",
        "configs/final/vqamed2019_qwen36_dinov3_cls_top6.yaml",
        "configs/final/vqamed2019_qwen36_clip_dinov3cls_top6.yaml",
    ],
}


def load_raw_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def raw_or_inference(raw_cfg: dict, key: str, default=None):
    inference_cfg = raw_cfg.get("inference", {}) or {}
    if key in raw_cfg:
        return raw_cfg.get(key)
    return inference_cfg.get(key, default)


def query_random_seed(base_seed: int, query_id: str) -> int:
    digest = hashlib.md5(str(query_id).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32 - 1)


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


def apply_vqa_prompt_overrides(prompt_record, system_prompt, query_template: str, question: str):
    record = copy.deepcopy(prompt_record)
    if system_prompt is not None:
        if str(system_prompt) == "":
            record.messages = [msg for msg in record.messages if msg.get("role") != "system"]
        else:
            for msg in record.messages:
                if msg.get("role") != "system":
                    continue
                for item in msg.get("content", []):
                    if item.get("type") == "text":
                        item["text"] = str(system_prompt)
    if query_template:
        for msg in reversed(record.messages):
            if msg.get("role") != "user":
                continue
            for item in reversed(msg.get("content", [])):
                if item.get("type") == "text":
                    item["text"] = query_template.replace("{question}", question).strip()
                    return record
    return record


def load_fixed_references(dataset, fixed_exemplars_json: str):
    rows = json.loads(Path(fixed_exemplars_json).read_text(encoding="utf-8"))
    by_id = {sample.id: sample for sample in dataset.get_reference_pool()}
    refs = []
    missing = []
    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in by_id:
            missing.append(sample_id)
        else:
            refs.append(by_id[sample_id])
    if missing:
        raise ValueError(f"Fixed exemplar ids not found: {missing[:10]}")
    return refs


def select_query_samples(dataset, raw_cfg: dict):
    query_split = str(raw_cfg.get("query_split", "test"))
    samples = [sample for sample in dataset.samples if sample.split == query_split]
    if not samples:
        raise ValueError(f"No query samples found for split={query_split} in {dataset.name}")
    return samples


def load_global_feature_index(features_dir: Path):
    metadata_path = features_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(features_dir / "global_embeddings.npy", mmap_mode="r")
    ids = [str(x) for x in metadata["ids"]]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
    return metadata, embeddings, ids, labels, splits, id_to_idx


def build_retrieval_result(dataset, query_sample, features_dir: Path, k: int):
    feature_index = load_global_feature_index(features_dir)
    if feature_index is None:
        return None
    metadata, embeddings, ids, labels, splits, id_to_idx = feature_index
    if query_sample.id not in id_to_idx:
        raise KeyError(f"Missing query feature for {query_sample.id} in {features_dir}")

    ref_by_id = {sample.id: sample for sample in dataset.get_reference_pool()}
    ref_ids_set = set(ref_by_id)
    ref_indices = [idx for idx, sample_id in enumerate(ids) if sample_id in ref_ids_set]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(
        ids=[ids[idx] for idx in ref_indices],
        embeddings=embeddings[ref_indices],
        labels=[labels[idx] for idx in ref_indices],
        splits=[splits[idx] for idx in ref_indices],
    )
    result = retriever.retrieve(
        query_id=query_sample.id,
        query_embedding=embeddings[id_to_idx[query_sample.id]],
        k=k,
        encoder_name=metadata.get("encoder_name", ""),
        encoder_version=metadata.get("encoder_version", ""),
        preprocessing_hash=metadata.get("preprocessing_hash", ""),
    )
    refs = [ref_by_id[ref_id] for ref_id in result.neighbor_ids if ref_id in ref_by_id]
    if len(refs) != k:
        raise ValueError(f"Expected {k} retrieved refs for {query_sample.id}, got {len(refs)}")
    return result, refs


def build_vqa_retrieval_prompt(query_sample, retrieved_refs, retrieval_result, method_name: str, score_label: str):
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


def method_label(method: str, raw_cfg: dict) -> str:
    if method == "zero_shot":
        return "No-ICL"
    if method == "fixed_random_6":
        return "Fixed-6"
    if method == "random_icl":
        return "Random-6"
    if method == "rg_icl_dual_global_similarity":
        return "CLIP+DINO top-6"
    score_label = str(raw_cfg.get("retrieval_score_label", "")).lower()
    feature_dir = str(raw_cfg.get("retrieval_features_dir", "")).lower()
    if "clip_dinov3" in feature_dir or "clip+dino" in score_label or "combined" in score_label:
        return "CLIP+DINO top-6"
    if "clip" in score_label or "/clip" in feature_dir or "clip_global" in feature_dir:
        return "CLIP top-6"
    if "dino" in score_label or "dinov3" in feature_dir:
        return "DINOv3 top-6"
    return method


def compact_messages(messages: list) -> list:
    compact = []
    for msg in messages:
        content = []
        for item in msg.get("content", []):
            if item.get("type") == "text":
                content.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                content.append({"type": "image_url", "image_path": item.get("image_url", {}).get("url", "")})
            else:
                content.append(item)
        compact.append({"role": msg.get("role", ""), "content": content})
    return compact


def format_messages_md(messages: list) -> str:
    lines = []
    for msg in messages:
        lines.append(f"### role={msg.get('role', '')}")
        for item in msg.get("content", []):
            if item.get("type") == "text":
                lines.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                lines.append(f"[IMAGE] {item.get('image_url', {}).get('url', '')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_dataset(raw_cfg: dict, cfg, ds_name: str, task: str):
    if task == "classification":
        kwargs = {"manifest_csv": raw_cfg.get("manifest_csv", "manifests/lag_manifest.csv")}
        target_label_names = raw_cfg.get("target_label_names")
        if target_label_names:
            kwargs["target_label_names"] = target_label_names
        return get_dataset(ds_name, cfg.data_root, split="all", **kwargs)

    kwargs = {}
    manifest_json = raw_cfg.get("manifest_json", "")
    if manifest_json:
        kwargs["manifest_json"] = manifest_json
    lang = raw_cfg.get("lang", "")
    if lang:
        kwargs["lang"] = lang
    return get_dataset(ds_name, cfg.data_root, split="all", **kwargs)


def build_classification_prompt(raw_cfg: dict, cfg, dataset, query_sample, method: str):
    k = int(raw_cfg.get("k", cfg.retrieval.k))
    prompt_dataset_name = raw_cfg.get("prompt_dataset_name", dataset.name)
    if method == "zero_shot":
        record = get_prompter("zero_shot").build_classification_prompt(
            query_sample=query_sample,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=prompt_dataset_name,
        )
    elif method == "fixed_random_6":
        refs = load_fixed_references(dataset, raw_cfg["fixed_exemplars_json"])
        record = get_prompter("fixed_random_6", fixed_references=refs).build_classification_prompt(
            query_sample=query_sample,
            fixed_references=refs,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=prompt_dataset_name,
        )
    elif method == "random_icl":
        record = get_prompter("naive_icl", k=k, seed=cfg.seed).build_classification_prompt(
            query_sample=query_sample,
            reference_pool=dataset.get_reference_pool(),
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=prompt_dataset_name,
            k=k,
            rng_seed=query_random_seed(cfg.seed, query_sample.id),
        )
        record.method = "random_icl"
    elif method in {"rg_icl_global_similarity", "rg_icl_dual_global_similarity"}:
        features_dir = Path(raw_cfg["retrieval_features_dir"])
        built = build_retrieval_result(dataset, query_sample, features_dir, k)
        if built is None:
            return None, f"missing retrieval features: {features_dir}"
        result, refs = built
        record = get_prompter(method, k=k).build_classification_prompt(
            query_sample=query_sample,
            retrieved_refs=refs,
            retrieval_result=result,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=prompt_dataset_name,
        )
    else:
        raise ValueError(f"Unsupported classification prompt method: {method}")
    return record, None


def build_vqa_prompt(raw_cfg: dict, cfg, dataset, query_sample, method: str):
    k = int(raw_cfg.get("k", cfg.retrieval.k))
    if method == "zero_shot":
        record = get_prompter("zero_shot").build_vqa_prompt(query_sample)
    elif method == "fixed_random_6":
        refs = load_fixed_references(dataset, raw_cfg["fixed_exemplars_json"])
        record = get_prompter("fixed_random_6", fixed_references=refs).build_vqa_prompt(
            query_sample,
            fixed_references=refs,
        )
    elif method == "random_icl":
        record = get_prompter("naive_icl", k=k, seed=cfg.seed).build_vqa_prompt(
            query_sample,
            reference_pool=dataset.get_reference_pool(),
            k=k,
            rng_seed=query_random_seed(cfg.seed, query_sample.id),
        )
        record.method = "random_icl"
    elif method in {"rg_icl_global_similarity", "rg_icl_dual_global_similarity"}:
        features_dir = Path(raw_cfg["retrieval_features_dir"])
        built = build_retrieval_result(dataset, query_sample, features_dir, k)
        if built is None:
            return None, f"missing retrieval features: {features_dir}"
        result, refs = built
        if method == "rg_icl_dual_global_similarity":
            score_label = "combined CLIP+DINO similarity"
        else:
            score_label = raw_cfg.get("retrieval_score_label", "retrieval similarity")
        record = build_vqa_retrieval_prompt(query_sample, refs, result, method, score_label)
    else:
        raise ValueError(f"Unsupported VQA prompt method: {method}")
    record.metadata = dict(getattr(record, "metadata", {}) or {})
    record.metadata.setdefault("question", query_sample.question)
    record = apply_vqa_prompt_overrides(
        record,
        raw_or_inference(raw_cfg, "vqa_system_prompt", None),
        str(raw_or_inference(raw_cfg, "vqa_query_template", "") or ""),
        query_sample.question,
    )
    return record, None


def record_to_output(dataset_key: str, task: str, config_path: str, raw_cfg: dict, dataset, query, method: str, record, error):
    base = {
        "dataset_key": dataset_key,
        "dataset_name": dataset.name,
        "task": task,
        "config": config_path,
        "method": method,
        "method_label": method_label(method, raw_cfg),
        "query_id": query.id,
        "query_image_path": query.image_path,
        "reference_ids": [],
        "reference_labels": [],
        "reference_order": [],
        "prompt_metadata": {},
        "messages": [],
    }
    if task == "classification":
        base.update({
            "ground_truth_label": getattr(query, "label", None),
            "ground_truth_name": getattr(query, "label_name", ""),
        })
    else:
        base.update({
            "question": getattr(query, "question", ""),
            "ground_truth_answer": getattr(query, "answer", ""),
            "answer_type": getattr(query, "question_type", ""),
        })
    if error:
        base["status"] = "skipped"
        base["reason"] = error
        return base
    base["status"] = "ok"
    base["reference_ids"] = list(record.reference_ids)
    base["reference_labels"] = list(record.reference_labels)
    base["reference_order"] = list(record.reference_order)
    base["prompt_metadata"] = dict(getattr(record, "metadata", {}) or {})
    base["messages"] = compact_messages(record.messages)
    return base


def dump_suite(dataset_key: str, task: str, config_paths: list[str], output_root: Path):
    rows = []
    seen = set()
    query = None
    dataset = None
    for config_path in config_paths:
        raw_cfg = load_raw_config(config_path)
        cfg = load_config(config_path)
        ds_name = (raw_cfg.get("datasets") or cfg.datasets)[0]
        dataset = build_dataset(raw_cfg, cfg, ds_name, task)
        query = select_query_samples(dataset, raw_cfg)[0]
        suffix = str(raw_or_inference(raw_cfg, "prompt_suffix", "") or "")
        suffix_methods = set(raw_or_inference(raw_cfg, "prompt_suffix_methods", []) or [])
        methods = raw_cfg.get("methods") or cfg.methods
        for method in methods:
            key = method_label(method, raw_cfg)
            if key in seen:
                continue
            seen.add(key)
            if task == "classification":
                record, error = build_classification_prompt(raw_cfg, cfg, dataset, query, method)
            else:
                record, error = build_vqa_prompt(raw_cfg, cfg, dataset, query, method)
            if record is not None and (not suffix_methods or method in suffix_methods):
                record = apply_prompt_suffix(record, suffix)
            rows.append(record_to_output(dataset_key, task, config_path, raw_cfg, dataset, query, method, record, error))

    out_dir = output_root / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "target_llm_prompts_first_test_sample.json"
    md_path = out_dir / "target_llm_prompts_first_test_sample.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(format_suite_md(dataset_key, task, dataset, query, rows), encoding="utf-8")
    return json_path, md_path, rows


def format_suite_md(dataset_key: str, task: str, dataset, query, rows: list[dict]) -> str:
    lines = [
        f"# Target LLM prompts: {dataset_key}",
        f"task: {task}",
        f"dataset_name: {dataset.name}",
        f"query_id: {query.id}",
        f"image_path: {query.image_path}",
    ]
    if task == "classification":
        lines.extend([
            f"ground_truth_label: {getattr(query, 'label', None)}",
            f"ground_truth_name: {getattr(query, 'label_name', '')}",
        ])
    else:
        lines.extend([
            f"question: {getattr(query, 'question', '')}",
            f"ground_truth_answer: {getattr(query, 'answer', '')}",
            f"answer_type: {getattr(query, 'question_type', '')}",
        ])

    for row in rows:
        lines.extend([
            "",
            f"## {row['method_label']}",
            f"method: {row['method']}",
            f"config: {row['config']}",
            f"status: {row['status']}",
        ])
        if row["status"] != "ok":
            lines.append(f"reason: {row.get('reason', '')}")
            continue
        lines.append(f"reference_ids: {row['reference_ids']}")
        lines.append(f"reference_labels: {row['reference_labels']}")
        lines.append("")
        lines.append(format_messages_md(row["messages"]))
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/prompts/main_experiment_target_llm")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    suites = {**CLASSIFICATION_SUITES, **VQA_SUITES}
    selected = args.datasets or list(suites)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    for dataset_key in selected:
        if dataset_key not in suites:
            raise ValueError(f"Unknown prompt suite {dataset_key}. Available: {sorted(suites)}")
        task = "classification" if dataset_key in CLASSIFICATION_SUITES else "vqa"
        json_path, md_path, rows = dump_suite(dataset_key, task, suites[dataset_key], output_root)
        summary[dataset_key] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "ok": sum(1 for row in rows if row["status"] == "ok"),
            "skipped": [row for row in rows if row["status"] != "ok"],
        }
        print(f"[dumped] {dataset_key}: {summary[dataset_key]['ok']} prompts -> {md_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
