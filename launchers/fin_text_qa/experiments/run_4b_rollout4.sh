#!/bin/bash
#
# Ablation: 4B model with ROLLOUT_N=4
#
# Compare with: run_4b.sh (default ROLLOUT_N=8)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_4b_rollout4"

ROLLOUT_N=4 \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_4b.sh" "$@"
