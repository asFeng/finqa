#!/bin/bash
#
# Financial QA RL Training - Qwen3-4B Default Configuration
#
# Usage:
#   bash run_4b.sh [additional hydra overrides]
#
# Hyperparameters (override via environment variables):
#   USE_UNCERTAINTY=false   - Disable uncertainty-weighted reward (default: true)
#   ROLLOUT_N=12           - Number of rollouts per prompt (default: 8)
#   KL_COEF=0.001          - KL loss coefficient (default: 0.001)
#   LR=1e-6                - Learning rate (default: 1e-6)
#   EPOCHS=10              - Total epochs (default: 10)
#
# Example:
#   USE_UNCERTAINTY=false ROLLOUT_N=12 bash run_4b.sh
#

set -x

# === EXPERIMENT NAME ===
# Can be overridden via environment variable for ablation scripts
EXP_NAME=${EXP_NAME:-"qwen3_4b_default"}

# === ENVIRONMENT ===
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Load environment variables from .env file
ENV_FILE="/mnt_out/asfeng/fin_w4/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# === PATHS ===
BASE_DIR="/mnt_out/asfeng/fin_w4"
VERL_DIR="$BASE_DIR/verl"
OUTPUT_DIR="/checkpoints/${EXP_NAME}"

# Reward functions are in parent directory
REWARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# === MODEL ===
MODEL_PATH="catherpker/stockr1-qwen3-4b"

# === DATA (4B series) ===
TRAIN_FILE="$BASE_DIR/data/train_4b_model_forecast_20000.parquet"
TEST_FILE="$BASE_DIR/data/test_4b_model_forecast_2000.parquet"

# === REWARD FUNCTION ===
# Default: versatile with uncertainty weighting
# Set USE_UNCERTAINTY=false to disable uncertainty weighting
USE_UNCERTAINTY=${USE_UNCERTAINTY:-true}
if [ "$USE_UNCERTAINTY" == "true" ]; then
    REWARD_FUNCTION_PATH="$REWARD_DIR/reward_uncertain_versatile.py"
else
    REWARD_FUNCTION_PATH="$REWARD_DIR/reward_versatile.py"
fi

# === GPU CONFIG ===
NUM_GPUS=${NUM_GPUS:-8}
TENSOR_PARALLEL_SIZE=${TP_SIZE:-1}

# === TRAINING CONFIG ===
# Optimized for 20k samples, targeting 2-3 day training
#
# Calculation:
#   - 20,000 samples / batch_size_64 = 312 steps/epoch
#   - 312 steps × 10 epochs = 3,120 total steps
#   - At ~80-100s/step ≈ 70-87 hours ≈ 3 days
#
TRAIN_BATCH_SIZE=${BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH:-64}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESP_LEN:-2048}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LR:-1e-6}
KL_LOSS_COEF=${KL_COEF:-0.001}

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " Financial QA RL Training - Qwen3-4B"
echo "=============================================="
echo "  Experiment: $EXP_NAME"
echo "  Model: $MODEL_PATH"
echo "  Train Data: $TRAIN_FILE"
echo "  Test Data: $TEST_FILE"
echo "  Reward: $(basename $REWARD_FUNCTION_PATH) (uncertainty=$USE_UNCERTAINTY)"
echo "  GPUs: $NUM_GPUS (TP=$TENSOR_PARALLEL_SIZE)"
echo "  Batch: $TRAIN_BATCH_SIZE (mini=$PPO_MINI_BATCH_SIZE, micro=$PPO_MICRO_BATCH_SIZE)"
echo "  Rollouts: $ROLLOUT_N"
echo "  Epochs: $TOTAL_EPOCHS"
echo "  LR: $LEARNING_RATE"
echo "  KL Coef: $KL_LOSS_COEF"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

cd "$VERL_DIR"

train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='fin_text_qa' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.val_before_train=True \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"

echo ""
echo "=============================================="
echo " Training complete!"
echo " Experiment: $EXP_NAME"
echo " Checkpoints: $OUTPUT_DIR"
echo "=============================================="
