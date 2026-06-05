#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[chexpert] Final CheXpert workflow uses five-label chexpert5 configs and fast feature directories."
echo "[chexpert] Delegating to scripts/run_chexpert_fast_features.sh"
exec bash scripts/run_chexpert_fast_features.sh
