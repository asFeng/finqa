#!/bin/bash
#
# Ablation: 8B model (GT) WITHOUT uncertainty weighting
#
# Compare with: run_gt_8b.sh (default with uncertainty)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_8b_gt_no_uncertain"

USE_UNCERTAINTY=false \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_gt_8b.sh" "$@"
