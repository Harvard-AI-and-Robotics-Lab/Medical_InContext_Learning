#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_gemma4_language_lora import (  # noqa: E402
    DATASET_PRESETS,
    MODEL_PATH,
    SampleDataset,
    first_model_device,
    gemma_content_from_prompt_content,
    load_split,
    move_batch_to_device,
    set_seed,
)
from src.inference.output_parser import OutputParser  # noqa: E402
from src.prompting import get_prompter  # noqa: E402
from src.retrieval import GlobalRetriever  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_PRESETS))
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--predictions_json", required=True)
    p.add_argument("--model_path", default=MODEL_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--icl_top_k", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=256)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_memory_per_gpu", default="46GiB")
    p.add_argument("--cpu_memory", default="700GiB")
    p.add_argument("--flush_every", type=int, default=25)
    return p.parse_args()


def load_feature_index(features_dir):
    features_dir = Path(features_dir)
    metadata = json.loads((features_dir / "metadata.json").read_text())
    embeddings = np.load(features_dir / "global_embeddings.npy", mmap_mode="r")
    ids = [str(x) for x in metadata["ids"]]
    return metadata, embeddings, ids, metadata["labels"], metadata["splits"], {sid: i for i, sid in enumerate(ids)}


def load_model(args):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = len([x for x in visible.split(",") if x.strip()]) if visible else torch.cuda.device_count()
    max_memory = {idx: args.max_memory_per_gpu for idx in range(n_visible)}
    max_memory["cpu"] = args.cpu_memory
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    return model, processor


def to_gemma_messages(prompt_record, image_max_side):
    return [
        {
            "role": msg["role"],
            "content": gemma_content_from_prompt_content(msg["content"], image_max_side),
        }
        for msg in prompt_record.messages
    ]


def build_prompts(args, samples, train_samples, label_names, prompt_dataset_name, image_max_side):
    metadata, embeddings, feature_ids, feature_labels, feature_splits, id_to_idx = load_feature_index(args.features_dir)
    ref_by_id = {str(sample.id): sample for sample in train_samples}
    train_indices = [idx for idx, sid in enumerate(feature_ids) if sid in ref_by_id]
    retriever = GlobalRetriever(exclude_query=True, exclude_test_set=True)
    retriever.build_index(
        ids=[feature_ids[idx] for idx in train_indices],
        embeddings=np.asarray(embeddings[train_indices], dtype=np.float32),
        labels=[feature_labels[idx] for idx in train_indices],
        splits=[feature_splits[idx] for idx in train_indices],
    )
    prompter = get_prompter("rg_icl_dual_global_similarity", k=args.icl_top_k)

    def build_one(item):
        idx, sample = item
        sid = str(sample.id)
        retrieval = retriever.retrieve(
            query_id=sid,
            query_embedding=np.asarray(embeddings[id_to_idx[sid]], dtype=np.float32),
            k=args.icl_top_k,
            encoder_name=metadata.get("encoder_name", ""),
            encoder_version=metadata.get("encoder_version", ""),
            preprocessing_hash=metadata.get("preprocessing_hash", ""),
        )
        refs = [ref_by_id[str(ref_id)] for ref_id in retrieval.neighbor_ids if str(ref_id) in ref_by_id]
        if len(refs) != args.icl_top_k:
            raise ValueError(f"{sid}: expected {args.icl_top_k} refs, got {len(refs)}")
        record = prompter.build_classification_prompt(
            query_sample=sample,
            retrieved_refs=refs,
            retrieval_result=retrieval,
            label_names=label_names,
            is_multi_label=False,
            dataset_name=prompt_dataset_name,
        )
        return idx, record, to_gemma_messages(record, image_max_side)

    out = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        for idx, record, messages in tqdm(pool.map(build_one, enumerate(samples)), total=len(samples), desc="build_icl_prompts"):
            out[idx] = (record, messages)
    return out


@torch.no_grad()
def generate(model, processor, messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = move_batch_to_device(inputs, first_model_device(model))
    prompt_width = int(inputs["input_ids"].shape[-1])
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return processor.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()


def write_json(metrics, predictions, output_json, predictions_json):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(metrics, indent=2))
    Path(predictions_json).write_text(json.dumps(predictions, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)
    preset = DATASET_PRESETS[args.dataset]
    raw_ds = load_split(args.dataset, args.split)
    train_ds = load_split(args.dataset, "train")
    label_names = list(raw_ds.label_names)
    samples = SampleDataset(raw_ds.samples, args.seed, args.limit).samples
    started = time.time()
    prompt_records = build_prompts(
        args,
        samples,
        train_ds.samples,
        label_names,
        preset["prompt_dataset_name"],
        preset["image_max_side"],
    )
    model, processor = load_model(args)
    parser = OutputParser()
    predictions = []
    correct = 0
    parse_success = 0
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": "generate_lora_icl",
        "adapter_dir": args.adapter_dir,
        "features_dir": args.features_dir,
        "icl_method": "rg_icl_dual_global_similarity",
        "icl_top_k": args.icl_top_k,
        "num_workers": args.num_workers,
        "n": len(samples),
        "label_names": label_names,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "max_new_tokens": args.max_new_tokens,
    }
    for idx, sample in enumerate(tqdm(samples, desc=f"{args.dataset}:{args.split}:generate")):
        prompt_record, messages = prompt_records[idx]
        raw = generate(model, processor, messages, args.max_new_tokens)
        parsed = parser.parse_classification(raw, getattr(sample, "id", str(idx)), label_names, False, preset["prompt_dataset_name"])
        pred = parsed.predicted_label
        ok = pred == sample.label_name
        correct += int(ok)
        parse_success += int(parsed.parse_success)
        predictions.append({
            "index": idx,
            "query_id": getattr(sample, "id", str(idx)),
            "image_path": sample.image_path,
            "label": sample.label_name,
            "label_idx": int(sample.label),
            "prediction": pred,
            "prediction_idx": int(parsed.predicted_label_idx),
            "correct": ok,
            "raw_response": raw,
            "reference_ids": list(prompt_record.reference_ids),
            "reference_labels": list(prompt_record.reference_labels),
            "metadata": dict(prompt_record.metadata or {}),
        })
        metrics.update({
            "completed": idx + 1,
            "correct": correct,
            "accuracy": correct / (idx + 1),
            "parse_success": parse_success,
            "parse_success_rate": parse_success / (idx + 1),
            "elapsed_seconds": time.time() - started,
        })
        if args.flush_every and (idx + 1) % args.flush_every == 0:
            write_json(metrics, predictions, args.output_json, args.predictions_json)
    write_json(metrics, predictions, args.output_json, args.predictions_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
