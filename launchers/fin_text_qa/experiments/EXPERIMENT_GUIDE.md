# Financial QA Experiments

## Overview

This folder contains experiment launch scripts for training Qwen3 models (4B and 8B) on the Financial QA task using GRPO (Group Relative Policy Optimization).

## Quick Start

```bash
cd /mnt_out/asfeng/fin_w4/launchers/fin_text_qa/experiments

# Run default 4B experiment
bash run_4b.sh

# Run default 8B experiment
bash run_8b.sh

# Run ablation experiment
bash run_4b_no_uncertain.sh
```

## Experiment Files

### Base Configurations

| Script | Model | Uncertainty | Rollout N | Output Dir |
|--------|-------|-------------|-----------|------------|
| `run_4b.sh` | Qwen3-4B | **true** | **8** | `/checkpoints/qwen3_4b_default` |
| `run_8b.sh` | Qwen3-8B | **true** | **8** | `/checkpoints/qwen3_8b_default` |

### Ablation: Uncertainty Weighting

| Script | Model | Uncertainty | Output Dir |
|--------|-------|-------------|------------|
| `run_4b_no_uncertain.sh` | Qwen3-4B | false | `/checkpoints/qwen3_4b_no_uncertain` |
| `run_8b_no_uncertain.sh` | Qwen3-8B | false | `/checkpoints/qwen3_8b_no_uncertain` |

### Ablation: Rollout Number

| Script | Model | Rollout N | Output Dir |
|--------|-------|-----------|------------|
| `run_4b_rollout4.sh` | Qwen3-4B | 4 | `/checkpoints/qwen3_4b_rollout4` |
| `run_4b_rollout12.sh` | Qwen3-4B | 12 | `/checkpoints/qwen3_4b_rollout12` |
| `run_8b_rollout4.sh` | Qwen3-8B | 4 | `/checkpoints/qwen3_8b_rollout4` |
| `run_8b_rollout12.sh` | Qwen3-8B | 12 | `/checkpoints/qwen3_8b_rollout12` |

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `USE_UNCERTAINTY` | `true` | Use uncertainty-weighted reward |
| `ROLLOUT_N` | `8` | Number of rollouts per prompt |
| `EPOCHS` | `10` | Total training epochs |
| `LR` | `1e-6` | Learning rate |
| `KL_COEF` | `0.001` | KL loss coefficient |
| `BATCH_SIZE` | `64` | Training batch size |
| `PPO_MICRO_BATCH` | `2` | Per-GPU micro batch size |
| `MAX_PROMPT_LEN` | `4096` | Max prompt length |
| `MAX_RESP_LEN` | `2048` | Max response length |
| `save_freq` | `100` | Save checkpoint every N steps |
| `test_freq` | `100` | Evaluate every N steps |

## Data Files

| Model | Train File | Test File |
|-------|------------|-----------|
| 4B | `train_4b_model_forecast_20000.parquet` | `test_4b_model_forecast_2000.parquet` |
| 8B | `train_8b_model_forecast_20000.parquet` | `test_8b_model_forecast_2000.parquet` |

Data location: `/mnt_out/asfeng/fin_w4/data/`

## Reward Functions

| File | Description |
|------|-------------|
| `reward_uncertain_versatile.py` | With uncertainty weighting (default) |
| `reward_versatile.py` | Without uncertainty weighting |

Reward functions location: `/mnt_out/asfeng/fin_w4/launchers/fin_text_qa/`

## Model Paths (HuggingFace)

- 4B: `catherpker/stockr1-qwen3-4b`
- 8B: `catherpker/stockr1-qwen3-8b`

## Environment Variables

You can override any hyperparameter via environment variables:

```bash
# Override rollout number
ROLLOUT_N=16 bash run_4b.sh

# Override multiple parameters
USE_UNCERTAINTY=false EPOCHS=5 LR=5e-6 bash run_8b.sh

# Override experiment name (changes output dir)
EXP_NAME="my_custom_exp" bash run_4b.sh
```

## Training Time Estimates

Based on 20k samples, batch_size=64 (312 steps/epoch):

| Model | Steps/Epoch | Total Steps (10 epochs) | Est. Time |
|-------|-------------|-------------------------|-----------|
| 4B | 312 | 3,120 | ~2-3 days |
| 8B | 312 | 3,120 | ~4-5 days |

## Uploading to HuggingFace

### Upload Script Location

```
/mnt_out/asfeng/fin_w4/launchers/fin_text_qa/upload_to_hf.sh
```

### Usage

```bash
cd /mnt_out/asfeng/fin_w4/launchers/fin_text_qa

# Upload latest checkpoint
./upload_to_hf.sh /checkpoints/qwen3_4b_default

# Upload last N checkpoints
./upload_to_hf.sh /checkpoints/qwen3_4b_default --num 3

# Upload specific step
./upload_to_hf.sh /checkpoints/qwen3_4b_default --step 200

# Upload as private repo
./upload_to_hf.sh /checkpoints/qwen3_4b_default --private
```

### Naming Convention

Uploads to: `afeng/<folder_name>_step-<N>`

Example: `/checkpoints/qwen3_4b_default` step 100 → `https://huggingface.co/afeng/qwen3_4b_default_step-100`

### Under the Hood (VERL Model Merger)

The upload script uses VERL's model merger to convert FSDP checkpoints to HuggingFace format:

```bash
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir <checkpoint_dir>/global_step_<N>/actor \
    --target_dir <merged_output_dir> \
    --hf_upload_path <repo_name> \
    [--private]
```

### Manual Conversion (Without Upload)

If you only want to convert checkpoints without uploading:

```bash
cd /mnt_out/asfeng/fin_w4/verl

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /checkpoints/qwen3_4b_default/global_step_100/actor \
    --target_dir /path/to/output/merged_model
```

### Checkpoint Structure

```
/checkpoints/qwen3_4b_default/
├── global_step_100/
│   ├── actor/                    # <-- This is what gets converted
│   │   ├── huggingface/
│   │   ├── fsdp_config.json
│   │   ├── model_world_size_8_rank_*.pt
│   │   ├── optim_world_size_8_rank_*.pt
│   │   └── extra_state_world_size_8_rank_*.pt
│   └── data.pt
├── global_step_200/
│   └── actor/
└── latest_checkpointed_iteration.txt
```

## WandB Project

All experiments log to WandB project: `fin_text_qa`

## Directory Structure

```
/mnt_out/asfeng/fin_w4/
├── launchers/fin_text_qa/
│   ├── experiments/           # This folder
│   │   ├── run_4b.sh
│   │   ├── run_8b.sh
│   │   └── run_*_ablation.sh
│   ├── reward_versatile.py
│   ├── reward_uncertain_versatile.py
│   └── upload_to_hf.sh
├── data/
│   ├── train_4b_model_forecast_20000.parquet
│   ├── test_4b_model_forecast_2000.parquet
│   ├── train_8b_model_forecast_20000.parquet
│   └── test_8b_model_forecast_2000.parquet
└── verl/                      # VERL framework

/checkpoints/                  # Output checkpoints
├── qwen3_4b_default/
├── qwen3_4b_no_uncertain/
├── qwen3_4b_rollout4/
├── qwen3_4b_rollout12/
├── qwen3_8b_default/
├── qwen3_8b_no_uncertain/
├── qwen3_8b_rollout4/
└── qwen3_8b_rollout12/
```

## Ablation Study Design

### Research Questions

1. **Uncertainty Weighting**: Does incorporating prediction uncertainty in the reward function improve model calibration?
   - Compare: `run_Xb.sh` vs `run_Xb_no_uncertain.sh`

2. **Rollout Number**: How does the number of rollouts per prompt affect training?
   - Compare: `run_Xb_rollout4.sh` vs `run_Xb.sh` vs `run_Xb_rollout12.sh`

### Experiment Matrix

| Experiment | 4B | 8B |
|------------|----|----|
| Default (uncertainty=true, rollout=8) | ✓ | ✓ |
| No uncertainty | ✓ | ✓ |
| Rollout 4 | ✓ | ✓ |
| Rollout 12 | ✓ | ✓ |

**Total: 8 experiments**
