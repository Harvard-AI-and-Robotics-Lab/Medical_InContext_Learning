#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import get_dataset
from prompting import get_prompter
from retrieval import GlobalRetriever


def format_messages(messages: list) -> str:
    lines = []
    for msg in messages:
        lines.append(f"## {msg['role'].upper()}")
        for item in msg.get("content", []):
            if item.get("type") == "text":
                lines.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                lines.append(f"[IMAGE] {item['image_url']['url']}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dump representative BreaKHis prompts for audit.")
    parser.add_argument("--data-root", default="data/raw/BreaKHis_v1_extracted")
    parser.add_argument("--manifest-csv", default="manifests/breakhis_patient_split_seed3407.csv")
    parser.add_argument("--fixed-exemplars-json", default="manifests/breakhis_fixed_exemplars_seed3407.json")
    parser.add_argument("--features-dir", default="")
    parser.add_argument("--output-md", default="outputs/smoke/breakhis_prompt_dump.md")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    dataset = get_dataset("breakhis", args.data_root, split="all", manifest_csv=args.manifest_csv)
    query = dataset.get_test_samples()[0]

    zero = get_prompter("zero_shot").build_classification_prompt(
        query_sample=query,
        label_names=dataset.label_names,
        is_multi_label=False,
        dataset_name="breakhis",
    )

    fixed_rows = json.loads(Path(args.fixed_exemplars_json).read_text(encoding="utf-8"))
    train_by_id = {sample.id: sample for sample in dataset.get_reference_pool()}
    fixed_refs = [train_by_id[row["id"]] for row in fixed_rows]
    fixed = get_prompter("fixed_random_6", fixed_references=fixed_refs).build_classification_prompt(
        query_sample=query,
        fixed_references=fixed_refs,
        label_names=dataset.label_names,
        is_multi_label=False,
        dataset_name="breakhis",
    )

    sections = [
        "# BreaKHis Prompt Dump",
        f"Query id: `{query.id}`",
        "## No-ICL",
        format_messages(zero.messages),
        "## Fixed-6",
        format_messages(fixed.messages),
    ]

    features_dir = Path(args.features_dir) if args.features_dir else None
    if features_dir and (features_dir / "metadata.json").exists():
        import numpy as np

        meta = json.loads((features_dir / "metadata.json").read_text(encoding="utf-8"))
        emb = np.load(features_dir / "global_embeddings.npy", mmap_mode="r")
        ids = [str(x) for x in meta["ids"]]
        labels = meta["labels"]
        splits = meta["splits"]
        id_to_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}
        train_ids = {sample.id for sample in dataset.get_reference_pool()}
        train_idx = [idx for idx, sample_id in enumerate(ids) if sample_id in train_ids]
        retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
        retriever.build_index(
            ids=[ids[idx] for idx in train_idx],
            embeddings=emb[train_idx],
            labels=[labels[idx] for idx in train_idx],
            splits=[splits[idx] for idx in train_idx],
        )
        retrieval_result = retriever.retrieve(
            query_id=query.id,
            query_embedding=emb[id_to_idx[query.id]],
            k=args.k,
            encoder_name=meta.get("encoder_name", ""),
            encoder_version=meta.get("encoder_version", ""),
            preprocessing_hash=meta.get("preprocessing_hash", ""),
        )
        ref_by_id = {sample.id: sample for sample in dataset.get_reference_pool()}
        refs = [ref_by_id[ref_id] for ref_id in retrieval_result.neighbor_ids]
        clip_prompt = get_prompter("rg_icl_global_similarity", k=args.k).build_classification_prompt(
            query_sample=query,
            retrieved_refs=refs,
            retrieval_result=retrieval_result,
            label_names=dataset.label_names,
            is_multi_label=False,
            dataset_name="breakhis",
        )
        sections.extend(["## CLIP Top-6", format_messages(clip_prompt.messages)])

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
