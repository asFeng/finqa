#!/bin/bash
#
# Ablation: 8B model WITHOUT uncertainty weighting
#
# Compare with: run_8b.sh (default with uncertainty)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="qwen3_8b_no_uncertain"

USE_UNCERTAINTY=false \
EXP_NAME="$EXP_NAME" \
bash "$SCRIPT_DIR/run_8b.sh" "$@"
