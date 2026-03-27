#!/usr/bin/env bash
# Run AlpacaEval 2.0 and Arena Hard 2.0 across all retrained model checkpoints.
# One model per GPU at a time to avoid OOM with 2048 max tokens.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

OPENINSTRUCT="/data/artifacts/frank/openinstruct"
GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

# ── Build model list ──────────────────────────────────────────────────
# Format: label|model_or_checkpoint_path
# HF model: label|hf:model_name
# Retrained: label|dir:path/to/output/final

MODELS=()
MODELS+=("sft_baseline|hf:allenai/OLMo-2-1124-7B-SFT")
MODELS+=("dpo_baseline|hf:allenai/OLMo-2-1124-7B-DPO")

add_retrained() {
  local label="$1" base_dir="$2" subdir="$3" ckpt="${4:-output/final}"
  local full_path="$base_dir/$subdir/$ckpt"
  if [ -d "$full_path" ]; then
    MODELS+=("$label|dir:$full_path")
  else
    echo "WARNING: $full_path not found, skipping $label" >&2
  fi
}

# Probing Vector
for size in 3000 12000 30000; do
  add_retrained "probe_filter_${size}" "$OPENINSTRUCT/7b_new" "olmo2_7b_dpo_${size}_sftbase+distractor"
  add_retrained "probe_switch_${size}" "$OPENINSTRUCT/7b_new" "olmo2_7b_dpo_switch_${size}_sft+distractor_mean"
done

# Gradient
for size in 3000 12000 30000; do
  add_retrained "grad_filter_${size}" "$OPENINSTRUCT/multigrad" "olmo2_7b_dpo_${size}_grad_multickpt"
  add_retrained "grad_switch_${size}" "$OPENINSTRUCT/multigrad" "olmo2_7b_dpo_switch_${size}_multigrad"
done

# LLM Toxic
for size in 3000 12000 30000; do
  add_retrained "toxic_filter_${size}" "$OPENINSTRUCT/toxic_full" "olmo2_7b_dpo_${size}"
  add_retrained "toxic_switch_${size}" "$OPENINSTRUCT/toxic_full" "olmo2_7b_dpo_switch_${size}"
done

# Random
for size in 3000 12000 30000; do
  add_retrained "random_filter_${size}" "$OPENINSTRUCT/random" "olmo2_7b_dpo_random_${size}_baseline"
  add_retrained "random_switch_${size}" "$OPENINSTRUCT/random" "olmo2_7b_dpo_switch_random_${size}"
done

echo "Total models: ${#MODELS[@]}"

# ── Run function ──────────────────────────────────────────────────────
run_eval() {
  local label="$1" model_spec="$2" bench="$3" gpu="$4"
  local log_dir="logs/${bench}_sweep/${label}"

  # Skip if already done
  if find "$log_dir" -name "*.eval" -print -quit 2>/dev/null | grep -q .; then
    echo "[$(date +'%T')] SKIP $bench/$label (already done)"
    return 0
  fi
  mkdir -p "$log_dir"

  local model_args=()
  if [[ "$model_spec" == hf:* ]]; then
    local hf_name="${model_spec#hf:}"
    model_args=(--model "olmo/olmo7b_dpo" -M "base_model=$hf_name")
    # For SFT baseline, use the sft preset
    if [[ "$label" == "sft_baseline" ]]; then
      model_args=(--model "olmo/olmo7b_sft")
    elif [[ "$label" == "dpo_baseline" ]]; then
      model_args=(--model "olmo/olmo7b_dpo")
    fi
  else
    local ckpt_path="${model_spec#dir:}"
    model_args=(--model "olmo/olmo7b_dpo" -M "checkpoint=$ckpt_path")
  fi

  echo "[$(date +'%T')] RUN  $bench/$label on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" inspect eval \
    "src/inspect_integration/bench_tasks.py@${bench}" \
    "${model_args[@]}" \
    -M device=cuda \
    --temperature 0.7 \
    --max-tokens "${MAX_TOKENS:-2048}" \
    --max-connections "${BENCH_MAX_CONN}" \
    --log-dir "$log_dir" \
    --display log \
    > "$log_dir/run.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "[$(date +'%T')] DONE $bench/$label"
  else
    echo "[$(date +'%T')] FAIL $bench/$label (see $log_dir/run.log)"
  fi
}

# ── Main loop: batch models across GPUs ───────────────────────────────
for bench in alpaca_eval arena_hard; do
  echo ""
  echo "=========================================="
  echo "  Benchmark: $bench"
  echo "=========================================="

  # Arena hard has very long prompts -- needs lower batch size to avoid OOM
  if [ "$bench" = "arena_hard" ]; then
    export BENCH_MAX_CONN=2
  else
    export BENCH_MAX_CONN=16
  fi

  idx=0
  total=${#MODELS[@]}

  while [ $idx -lt $total ]; do
    PIDS=()
    GPU_IDX=0

    # Launch up to NUM_GPUS jobs
    while [ $idx -lt $total ] && [ $GPU_IDX -lt $NUM_GPUS ]; do
      IFS='|' read -r label model_spec <<< "${MODELS[$idx]}"
      gpu="${GPU_LIST[$GPU_IDX]}"

      run_eval "$label" "$model_spec" "$bench" "$gpu" &
      PIDS+=($!)

      idx=$((idx + 1))
      GPU_IDX=$((GPU_IDX + 1))
    done

    # Wait for batch
    for pid in "${PIDS[@]}"; do
      wait "$pid" || true
    done
  done
done

echo ""
echo "=========================================="
echo "  All complete! Results in:"
echo "    logs/alpaca_eval_sweep/"
echo "    logs/arena_hard_sweep/"
echo "=========================================="
