#!/usr/bin/env python3
from pathlib import Path
import json

import pandas as pd
import yaml


def write_script(path: str, body: str) -> None:
    p = Path(path)
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)


def write_breakhis_manifests() -> None:
    breakhis_path = Path("manifests/breakhis_patient_split_seed3407.csv")
    df = pd.read_csv(breakhis_path, dtype={"id": str, "patient_id": str})
    expected_labels = sorted(df["label"].unique().tolist())
    summary = {"source": str(breakhis_path), "pooled_rows": int(len(df)), "magnifications": {}}
    mag_values = df["magnification"].astype(str).str.replace("X", "", regex=False).astype(int)

    for mag in [40, 100, 200, 400]:
        mag_df = df[mag_values == mag].copy()
        out = Path(f"manifests/breakhis_patient_split_seed3407_mag{mag}.csv")
        mag_df.to_csv(out, index=False)

        split_patients = {s: set(g["patient_id"].astype(str)) for s, g in mag_df.groupby("split")}
        leaks = []
        splits = sorted(split_patients)
        for idx, a in enumerate(splits):
            for b in splits[idx + 1 :]:
                inter = sorted(split_patients[a] & split_patients[b])
                if inter:
                    leaks.append({"splits": [a, b], "patients": inter[:10], "n": len(inter)})

        label_ids = sorted(mag_df["label"].unique().tolist())
        summary["magnifications"][str(mag)] = {
            "path": str(out),
            "rows": int(len(mag_df)),
            "split_counts": mag_df["split"].value_counts().sort_index().to_dict(),
            "label_ids": label_ids,
            "patient_leakage": leaks,
        }
        if label_ids != expected_labels:
            raise RuntimeError(f"Missing labels for mag {mag}: {label_ids}")
        if leaks:
            raise RuntimeError(f"Patient leakage detected for mag {mag}: {leaks}")

    Path("manifests/breakhis_multiclass_mag_manifest_summary_seed3407.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def write_ddr_512_configs() -> None:
    for src, dst, features_root, name in [
        (
            "configs/final/ddr_extract_clip_global.yaml",
            "configs/final/ddr_extract_clip_global_512.yaml",
            "outputs/features_clip_global_ddr_512",
            "extract_clip_global_ddr_512",
        ),
        (
            "configs/final/ddr_extract_dinov3_global.yaml",
            "configs/final/ddr_extract_dinov3_global_512.yaml",
            "outputs/features_dinov3_global_ddr_512",
            "extract_dinov3_global_ddr_512",
        ),
    ]:
        cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
        cfg["name"] = name
        cfg["features_root"] = features_root
        cfg["manifest_csv"] = "manifests/ddr_official_split_crop_pad_512.csv"
        Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def make_vlm_512(src: str, dst: str, port: int) -> None:
        cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
        cfg["name"] = cfg.get("name", Path(src).stem).replace("_final", "_512_final")
        if not cfg["name"].endswith("_512_final"):
            cfg["name"] = cfg["name"] + "_512"
        cfg["output_root"] = str(cfg["output_root"]).replace("_1024", "_512")
        cfg["manifest_csv"] = "manifests/ddr_official_split_crop_pad_512.csv"
        cfg["inference"]["base_url"] = f"http://127.0.0.1:{port}/v1"
        if "retrieval_features_dir" in cfg:
            cfg["retrieval_features_dir"] = str(cfg["retrieval_features_dir"]).replace("_1024", "_512")
        Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    for model, port in [("qwen36", 18003), ("gemma4", 18002)]:
        for stem in ["noicl_fixed6_random6", "clip_top6", "dinov3_cls_top6", "clip_dinov3cls_top6"]:
            make_vlm_512(
                f"configs/final/ddr_{model}_{stem}.yaml",
                f"configs/final/ddr_{model}_{stem}_512.yaml",
                port,
            )


def write_runner_scripts() -> None:
    common_train = """python3 scripts/train_resnet50_classification.py \\
  --manifest-csv {manifest} \\
  --data-root data/raw/BreaKHis_v1_extracted \\
  --output-dir {output} \\
  --image-size 384 \\
  --batch-size 32 \\
  --epochs 40 \\
  --patience 8 \\
  --num-workers 8 \\
  --seed 3407 \\
  --backbone-lr 3e-5 \\
  --head-lr 1e-3 \\
  --weight-decay 1e-4 \\
  --selection-metric accuracy
"""

    write_script(
        "scripts/run_breakhis_resnet50_multiclass_pooled.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-1}}
{common_train.format(manifest="manifests/breakhis_patient_split_seed3407.csv", output="outputs/final/resnet50_breakhis_multiclass_384_seed3407")}""",
    )
    write_script(
        "scripts/run_breakhis_resnet50_multiclass_mag40_100.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-2}}
{common_train.format(manifest="manifests/breakhis_patient_split_seed3407_mag40.csv", output="outputs/final/resnet50_breakhis_multiclass_mag40_384_seed3407")}
{common_train.format(manifest="manifests/breakhis_patient_split_seed3407_mag100.csv", output="outputs/final/resnet50_breakhis_multiclass_mag100_384_seed3407")}""",
    )
    write_script(
        "scripts/run_breakhis_resnet50_multiclass_mag200_400.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-7}}
{common_train.format(manifest="manifests/breakhis_patient_split_seed3407_mag200.csv", output="outputs/final/resnet50_breakhis_multiclass_mag200_384_seed3407")}
{common_train.format(manifest="manifests/breakhis_patient_split_seed3407_mag400.csv", output="outputs/final/resnet50_breakhis_multiclass_mag400_384_seed3407")}""",
    )

    write_script(
        "scripts/run_ddr_512_prepare_features.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
export REPO_ROOT
mkdir -p logs/runs

python3 scripts/preprocess_ddr_images.py \\
  --manifest-csv manifests/ddr_official_split.csv \\
  --data-root . \\
  --output-root data/processed/ddr_crop_pad_512 \\
  --output-manifest manifests/ddr_official_split_crop_pad_512.csv \\
  --summary-json manifests/ddr_official_split_crop_pad_512.summary.json \\
  --size 512 \\
  --threshold 10 \\
  --margin 8 \\
  --quality 95 \\
  --workers 16 \\
  --skip-existing

wait_for_free_gpu() {
  while true; do
    for gpu in ${FEATURE_GPU_CANDIDATES:-7 2 1}; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$used" -lt 2000 ]; then
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date)] waiting for a free feature GPU among: ${FEATURE_GPU_CANDIDATES:-7 2 1}"
    sleep 60
  done
}

need_features() {
  local meta="$1"
  python3 - "$meta" <<"END_NEEDED"
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(0)
try:
    m = json.load(open(p))
except Exception:
    raise SystemExit(0)
raise SystemExit(1 if int(m.get("n_samples", 0)) == 13673 else 0)
END_NEEDED
}

if need_features outputs/features_clip_global_ddr_512/ddr/clip/metadata.json; then
  gpu=$(wait_for_free_gpu)
  echo "[$(date)] extracting DDR 512 CLIP on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/extract_features.py --config configs/final/ddr_extract_clip_global_512.yaml
else
  echo "[$(date)] DDR 512 CLIP features already complete"
fi

if need_features outputs/features_dinov3_global_ddr_512/ddr/dinov3/metadata.json; then
  gpu=$(wait_for_free_gpu)
  echo "[$(date)] extracting DDR 512 DINOv3 on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/extract_features.py --config configs/final/ddr_extract_dinov3_global_512.yaml
else
  echo "[$(date)] DDR 512 DINOv3 features already complete"
fi

if need_features outputs/features_clip_dinov3cls_05_global_ddr_512/ddr/clip_dinov3cls05/metadata.json; then
  echo "[$(date)] building DDR 512 CLIP+DINOv3 0.5/0.5 fused features"
  python3 scripts/build_fused_features.py \\
    --feature-a outputs/features_clip_global_ddr_512/ddr/clip \\
    --feature-b outputs/features_dinov3_global_ddr_512/ddr/dinov3 \\
    --output-dir outputs/features_clip_dinov3cls_05_global_ddr_512/ddr/clip_dinov3cls05 \\
    --weight-a 0.5 \\
    --weight-b 0.5 \\
    --encoder-name clip_dinov3cls05
else
  echo "[$(date)] DDR 512 fused features already complete"
fi

if ! tmux has-session -t ddr_512_qwen36_vlm 2>/dev/null; then
  tmux new-session -d -s ddr_512_qwen36_vlm "bash -lc 'cd ${REPO_ROOT}; export VLLM_API_KEY=EMPTY; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_noicl_fixed6_random6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_dinov3_cls_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_qwen36_clip_dinov3cls_top6_512.yaml' > logs/runs/ddr_512_qwen36_vlm.log 2>&1"
fi

if ! tmux has-session -t ddr_512_gemma4_vlm 2>/dev/null; then
  tmux new-session -d -s ddr_512_gemma4_vlm "bash -lc 'cd ${REPO_ROOT}; export VLLM_API_KEY=EMPTY; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_noicl_fixed6_random6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_dinov3_cls_top6_512.yaml; python3 scripts/run_final_classification.py --config configs/final/ddr_gemma4_clip_dinov3cls_top6_512.yaml' > logs/runs/ddr_512_gemma4_vlm.log 2>&1"
fi

echo "[$(date)] DDR 512 prepare/features done; VLM sessions launched or already present."
""",
    )


def main() -> None:
    write_breakhis_manifests()
    write_ddr_512_configs()
    write_runner_scripts()
    print("generated DDR 512 configs and runner scripts")


if __name__ == "__main__":
    main()
