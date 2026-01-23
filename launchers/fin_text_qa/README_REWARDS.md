# Reward Functions Reference

## Files Overview

| File | Uncertainty Weighting | Detailed Logging | Use Case |
|------|----------------------|------------------|----------|
| `reward.py` | No | No | Basic training |
| `reward_uncertain.py` | Yes | No | Training with uncertainty-weighted samples |
| `reward_versatile.py` | No | Yes | Training + detailed val-aux metrics |
| `reward_uncertain_versatile.py` | Yes | Yes | Full features |

## Uncertainty Weighting

The "uncertain" versions apply `uncertainty_weight()` to downweight high-uncertainty samples:
- Samples with `uncertainty < 0.30` get full weight (1.0)
- Samples with `uncertainty > 0.50` get weight ~0.5
- Requires `extra_info["uncertainty"]` in data

## Versatile Logging Metrics

The "versatile" versions return a dict instead of float, logging to `val-aux/{data_source}/{metric}/mean@N`:

**Per-task-type scores** (one non-zero per sample):
- `pure_forecast` - numerical prediction score
- `event_detection` - Yes/No question score
- `multi_signal_reasoning` - strategy/categorical score
- `macro_fundamental` - Yes/No + reasoning score
- `news_sentiment` - sentiment prediction score
- `analysis` - long-form analysis score
- `fallback` - unknown task type score

**Format metrics:**
- `has_answer_tag` - 1.0 if uses `<answer>` tags
- `response_length` - word count

**Quality metrics:**
- `is_refusal` - 1.0 if model refused to answer
- `direction_correct` - 1.0 if forecast direction correct
- `exact_match` - 1.0 for exact answer match

## Usage in Training Script

```bash
# In run_qwen3_*.sh:
REWARD_FUNCTION_PATH="$SCRIPT_DIR/reward_uncertain_versatile.py"
```

## Data Requirements

- `extra_info["question_type"]` - one of: `pure_forecast`, `event_detection`, `multi_signal_reasoning`, `macro_fundamental`, `news_sentiment`, `analysis`
- `extra_info["uncertainty"]` - float 0-1 (only for uncertain versions)

## Upload to HuggingFace

```bash
./upload_to_hf.sh /checkpoints/fin_text_qa_qwen3_8b --num 3
```
