#!/usr/bin/env python3
"""Evaluate DINOv3 full-token spatial retrieval with kNN majority vote.

This uses the stored DINOv3 patch-token features with shape (1024, 1024).
For each query/reference pair it computes the full patch similarity matrix:

    sim = normalize(query_tokens) @ normalize(reference_tokens).T

and scores the pair with directed Chamfer-style reductions:

    q2r: mean over query patches of max reference-patch similarity
    r2q: mean over reference patches of max query-patch similarity
    sym: average of q2r and r2q

For speed, the default first filters references by DINOv3 CLS/global cosine
similarity, then reranks those candidates by full-token spatial similarity.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


LABEL_NAME = {0: "non_glaucoma", 1: "glaucoma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("outputs/features_dinov3_global/lag_project/dinov3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablations/retrieval_dinov3_fulltoken_spatial_knn/lag_project/dinov3_fulltoken_spatial"),
    )
    parser.add_argument("--prefilter-k", type=int, default=50)
    parser.add_argument("--k-values", type=int, nargs="+", default=[6])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def spatial_scores(query_tokens: np.ndarray, ref_tokens: np.ndarray) -> tuple[float, float, float]:
    sim = query_tokens @ ref_tokens.T
    q2r = float(sim.max(axis=1).mean())
    r2q = float(sim.max(axis=0).mean())
    return q2r, r2q, 0.5 * (q2r + r2q)


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "auc_pos_fraction": auc,
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "precision": float(tp / max(1, tp + fp)),
        "confusion": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.features_dir / "metadata.json").read_text())
    ids = [str(x) for x in meta["ids"]]
    labels = np.asarray(meta["labels"], dtype=np.int64)
    splits = np.asarray(meta["splits"])
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(ids)}

    embeddings = np.load(args.features_dir / "global_embeddings.npy").astype(np.float32)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    train_idx = np.where(splits != "test")[0]
    test_idx = np.where(splits == "test")[0]
    if args.limit is not None:
        test_idx = test_idx[: args.limit]
    train_embeddings = embeddings[train_idx]
    spatial_dir = args.features_dir / "spatial_features"

    @functools.lru_cache(maxsize=args.cache_size)
    def load_spatial_norm(index: int) -> np.ndarray:
        arr = np.load(spatial_dir / f"{index}.npy")
        return l2_normalize_rows(arr)

    max_k = max(args.k_values)
    if args.prefilter_k < max_k:
        raise ValueError("--prefilter-k must be >= max(k-values)")

    rows: list[dict] = []
    direction_rankings = {"q2r": [], "r2q": [], "sym": []}

    for count, q_idx in enumerate(test_idx, start=1):
        q_emb = embeddings[q_idx]
        global_sims = train_embeddings @ q_emb
        local_top = np.argpartition(global_sims, -args.prefilter_k)[-args.prefilter_k:]
        local_top = local_top[np.argsort(global_sims[local_top])[::-1]]
        candidate_indices = train_idx[local_top]

        q_tokens = load_spatial_norm(int(q_idx))
        candidate_rows = []
        for ref_idx, global_score in zip(candidate_indices, global_sims[local_top]):
            ref_tokens = load_spatial_norm(int(ref_idx))
            q2r, r2q, sym = spatial_scores(q_tokens, ref_tokens)
            candidate_rows.append(
                {
                    "id": ids[int(ref_idx)],
                    "label": int(labels[int(ref_idx)]),
                    "global_score": float(global_score),
                    "q2r": q2r,
                    "r2q": r2q,
                    "sym": sym,
                }
            )

        record = {
            "query_id": ids[int(q_idx)],
            "true_label": int(labels[int(q_idx)]),
            "true_label_name": LABEL_NAME[int(labels[int(q_idx)])],
            "candidates": candidate_rows,
        }
        rows.append(record)

        for direction in direction_rankings:
            ranked = sorted(candidate_rows, key=lambda x: x[direction], reverse=True)
            direction_rankings[direction].append(ranked)

        if args.progress_every and count % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            eta = (len(test_idx) - count) / max(rate, 1e-8)
            print(
                f"[{count}/{len(test_idx)}] elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
                f"cache={load_spatial_norm.cache_info()}",
                flush=True,
            )

    summary_rows = []
    summary_json = {
        "features_dir": str(args.features_dir),
        "metadata": {
            "encoder_name": meta.get("encoder_name"),
            "encoder_version": meta.get("encoder_version"),
            "model_id": meta.get("model_id"),
            "spatial_shape": list(np.load(spatial_dir / "0.npy", mmap_mode="r").shape)
            if (spatial_dir / "0.npy").exists()
            else None,
        },
        "prefilter_k": args.prefilter_k,
        "k_values": args.k_values,
        "directions": {},
    }

    y_true = [int(labels[int(q_idx)]) for q_idx in test_idx]
    for direction, rankings in direction_rankings.items():
        summary_json["directions"][direction] = {}
        for k in args.k_values:
            y_score = []
            y_pred_strict = []
            y_pred_tie_positive = []
            top_records = []
            for q_idx, ranked in zip(test_idx, rankings):
                top = ranked[:k]
                pos = sum(r["label"] == 1 for r in top)
                score = pos / k
                y_score.append(score)
                y_pred_strict.append(1 if pos > (k / 2) else 0)
                y_pred_tie_positive.append(1 if pos >= math.ceil(k / 2) else 0)
                top_records.append(
                    {
                        "query_id": ids[int(q_idx)],
                        "true_label": int(labels[int(q_idx)]),
                        "top_ids": [r["id"] for r in top],
                        "top_labels": [r["label"] for r in top],
                        "top_scores": [r[direction] for r in top],
                        "positive_fraction": score,
                    }
                )

            for rule_name, preds in [
                ("strict_majority", y_pred_strict),
                ("tie_positive", y_pred_tie_positive),
            ]:
                metrics = compute_metrics(y_true, preds, y_score)
                summary_json["directions"][direction][f"k{k}_{rule_name}"] = metrics
                summary_rows.append(
                    {
                        "direction": direction,
                        "k": k,
                        "rule": rule_name,
                        **metrics,
                    }
                )

            pred_path = args.output_dir / f"predictions_{direction}_k{k}.json"
            pred_path.write_text(json.dumps(top_records, indent=2), encoding="utf-8")

    summary_json["elapsed_seconds"] = time.time() - t0
    (args.output_dir / "metrics.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
