#!/bin/bash
#
# Upload VERL checkpoints to HuggingFace Hub
#
# This script merges FSDP checkpoints to HuggingFace format and uploads them.
#
# Usage:
#   ./upload_to_hf.sh <checkpoint_dir> [options]
#
# Examples:
#   # Upload latest checkpoint
#   ./upload_to_hf.sh /checkpoints/fin_text_qa_qwen3_8b
#
#   # Upload last 3 checkpoints
#   ./upload_to_hf.sh /checkpoints/fin_text_qa_qwen3_8b --num 3
#
#   # Upload specific step
#   ./upload_to_hf.sh /checkpoints/fin_text_qa_qwen3_8b --step 200
#
#   # Upload as private repo
#   ./upload_to_hf.sh /checkpoints/fin_text_qa_qwen3_8b --private
#
# Options:
#   --num <N>        Upload last N checkpoints (default: 1)
#   --step <N>       Upload specific step only (overrides --num)
#   --private        Upload as private repository
#   --help           Show this help message
#
# Naming convention:
#   Uploads to: <HF_USERNAME>/<exp_name>_step-<step_number>
#   e.g., afeng/fin_text_qa_qwen3_8b_step-100
#

set -e

# === DEFAULT CONFIG ===
VERL_DIR="${VERL_DIR:-/mnt_out/asfeng/fin_w4/verl}"
NUM_CKPTS=1
SPECIFIC_STEP=""
PRIVATE_FLAG=""

# ===========================================
# HARDCODED HUGGINGFACE CREDENTIALS (OPTIONAL)
# Uncomment and fill in to skip `huggingface-cli login`
# ===========================================

ENV_FILE="/mnt_out/asfeng/fin_w4/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# ===========================================

# === PARSE ARGUMENTS ===
show_help() {
    head -35 "$0" | tail -33
    exit 0
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_dir> [options]"
    echo "Run '$0 --help' for more information."
    exit 1
fi

CHECKPOINT_BASE="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --num)
            NUM_CKPTS="$2"
            shift 2
            ;;
        --step)
            SPECIFIC_STEP="$2"
            shift 2
            ;;
        --private)
            PRIVATE_FLAG="--private"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# === VALIDATE INPUTS ===
if [ ! -d "$CHECKPOINT_BASE" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_BASE"
    exit 1
fi

# Extract experiment name from folder name
EXP_NAME=$(basename "$CHECKPOINT_BASE")

# === FIND CHECKPOINTS ===
echo "=============================================="
echo " VERL Checkpoint Upload to HuggingFace"
echo "=============================================="
echo "  Checkpoint base: $CHECKPOINT_BASE"
echo "  Experiment name: $EXP_NAME"
echo "  HuggingFace user: $HF_USERNAME"
if [ -n "$SPECIFIC_STEP" ]; then
    echo "  Upload step: $SPECIFIC_STEP"
else
    echo "  Upload last N: $NUM_CKPTS"
fi
echo "=============================================="

# Get list of available checkpoints (sorted by step number)
AVAILABLE_STEPS=$(ls -d "$CHECKPOINT_BASE"/global_step_*/actor 2>/dev/null | \
    sed 's/.*global_step_\([0-9]*\).*/\1/' | sort -n)

if [ -z "$AVAILABLE_STEPS" ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINT_BASE"
    echo "Expected structure: $CHECKPOINT_BASE/global_step_<N>/actor/"
    exit 1
fi

echo ""
echo "Available checkpoints:"
for step in $AVAILABLE_STEPS; do
    echo "  - global_step_$step"
done
echo ""

# Determine which steps to upload
if [ -n "$SPECIFIC_STEP" ]; then
    STEPS_TO_UPLOAD="$SPECIFIC_STEP"
else
    # Get last N checkpoints
    STEPS_TO_UPLOAD=$(echo "$AVAILABLE_STEPS" | tail -n "$NUM_CKPTS")
fi

echo "Will upload checkpoints: $(echo $STEPS_TO_UPLOAD | tr '\n' ' ')"
echo ""

# === CHECK HUGGINGFACE AUTH ===
echo "Checking HuggingFace authentication..."

if [ -n "$HF_TOKEN" ]; then
    echo "Using hardcoded HF_TOKEN for authentication..."
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) || {
    echo ""
    echo "ERROR: Not logged in to HuggingFace."
    echo ""
    echo "Please authenticate using one of:"
    echo "  1. Run: huggingface-cli login"
    echo "  2. Set HF_TOKEN in this script"
    echo ""
    exit 1
}

echo "Authenticated as: $HF_USER"
echo ""

# === UPLOAD EACH CHECKPOINT ===
UPLOAD_COUNT=0
FAILED_COUNT=0

for STEP in $STEPS_TO_UPLOAD; do
    CHECKPOINT_DIR="$CHECKPOINT_BASE/global_step_${STEP}/actor"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "WARNING: Checkpoint not found: $CHECKPOINT_DIR, skipping..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    # Build repo name: username/exp_name_step-X
    REPO_NAME="${HF_USERNAME}/${EXP_NAME}_step-${STEP}"

    # Target directory for merged model
    TARGET_DIR="$CHECKPOINT_BASE/merged_hf_model_step_${STEP}"
    mkdir -p "$TARGET_DIR"

    echo "=============================================="
    echo " Uploading step $STEP"
    echo "=============================================="
    echo "  Source: $CHECKPOINT_DIR"
    echo "  Target: $TARGET_DIR"
    echo "  Repo: $REPO_NAME"
    echo "=============================================="

    cd "$VERL_DIR"

    if python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CHECKPOINT_DIR" \
        --target_dir "$TARGET_DIR" \
        --hf_upload_path "$REPO_NAME" \
        $PRIVATE_FLAG; then

        echo ""
        echo "SUCCESS: Step $STEP uploaded to https://huggingface.co/$REPO_NAME"
        echo ""
        UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
    else
        echo ""
        echo "FAILED: Step $STEP upload failed"
        echo ""
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# === SUMMARY ===
echo ""
echo "=============================================="
echo " UPLOAD SUMMARY"
echo "=============================================="
echo "  Successful: $UPLOAD_COUNT"
echo "  Failed: $FAILED_COUNT"
echo ""
echo "  Uploaded repos:"
for STEP in $STEPS_TO_UPLOAD; do
    echo "    - https://huggingface.co/${HF_USERNAME}/${EXP_NAME}_step-${STEP}"
done
echo "=============================================="

if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
