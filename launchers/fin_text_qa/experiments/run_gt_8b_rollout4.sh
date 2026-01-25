#!/bin/bash
#
# Ablation: 8B model (GT) with ROLLOUT_N=4
#
# Compare with: run_gt_8b.sh (default ROLLOUT_N=8)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_8b_gt_rollout4"

ROLLOUT_N=4 \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_gt_8b.sh" "$@"
