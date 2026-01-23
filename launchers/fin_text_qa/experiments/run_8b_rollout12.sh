#!/bin/bash
#
# Ablation: 8B model with ROLLOUT_N=12
#
# Compare with: run_8b.sh (default ROLLOUT_N=8)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_8b_rollout12"

ROLLOUT_N=12 \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_8b.sh" "$@"
