#!/bin/bash
#
# Ablation: 4B model WITHOUT uncertainty weighting
#
# Compare with: run_4b.sh (default with uncertainty)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_4b_no_uncertain"

USE_UNCERTAINTY=false \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_4b.sh" "$@"
