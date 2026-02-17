#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Trained baseline model (exhibits higher harmful rates than original DPO)
BASELINE_MODEL="/mnt/polished-lake/home/fxiao-two/openinstruct/output/7b_new/olmo2_7b_dpo_baseline/output/final"

# Run base and distractor variants in parallel on different GPUs
echo "Starting parallel evaluation..."
echo "  Model: $BASELINE_MODEL"
echo "  GPU 0: base variant"
echo "  GPU 1: distractor variant"

# Base variant on GPU 0
CUDA_VISIBLE_DEVICES=0 python evaluate_safety.py \
  -n 50 \
  --prompt-set lmsys_harmful_500_varied_base_only \
  --num-prompts all \
  --models olmo7b_dpo \
  --model-override "olmo7b_dpo=${BASELINE_MODEL}" \
  --temperature 0.7 \
  --batch-size 128 \
  --judge-workers 128 \
  --run-name lmsys_500_baseline_base_actual &
PID_BASE=$!

# Distractor variant on GPU 1
CUDA_VISIBLE_DEVICES=1 python evaluate_safety.py \
  -n 50 \
  --prompt-set lmsys_harmful_500_varied_dist_only \
  --num-prompts all \
  --models olmo7b_dpo \
  --model-override "olmo7b_dpo=${BASELINE_MODEL}" \
  --temperature 0.7 \
  --batch-size 128 \
  --judge-workers 128 \
  --run-name lmsys_500_baseline_dist_actual &
PID_DIST=$!

echo "Running PIDs: base=$PID_BASE, dist=$PID_DIST"
echo "Waiting for completion..."

wait $PID_BASE && echo "✓ Base variant completed" || echo "✗ Base variant failed"
wait $PID_DIST && echo "✓ Distractor variant completed" || echo "✗ Distractor variant failed"

echo ""
echo "Results saved to:"
echo "  logs/run_lmsys_500_baseline_base/"
echo "  logs/run_lmsys_500_baseline_dist/"
