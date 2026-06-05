import argparse
import json
from pathlib import Path

import pandas as pd


def binary_from_confusion(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return sensitivity, specificity, precision, f1


def read_vlm_metrics(path: Path):
    rows = []
    if not path.exists():
        return rows
    data = json.load(open(path, "r", encoding="utf-8"))
    for method, metrics in data.items():
        if not isinstance(metrics, dict) or "accuracy" not in metrics:
            continue
        cm = metrics.get("confusion")
        sens = spec = prec = f1 = None
        if cm:
            sens, spec, prec, f1 = binary_from_confusion(cm)
        rows.append(
            {
                "run": path.parts[-3],
                "method": method,
                "accuracy": metrics.get("accuracy"),
                "auc": metrics.get("auc"),
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": f1,
                "brier": metrics.get("brier"),
                "ece": metrics.get("ece"),
                "n": metrics.get("n_samples"),
                "confusion": json.dumps(cm) if cm else "",
                "source": str(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/tables/final_metrics.csv"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.outputs_dir.glob("*/lag_project/metrics.json")):
        rows.extend(read_vlm_metrics(metrics_path))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
