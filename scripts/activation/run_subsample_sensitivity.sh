#!/usr/bin/env bash
# Run probing vector sensitivity analysis: subsample 50 and 100 of 150 scenarios,
# 3 seeds each, with --natural-variant base_plus_distractor.
# Parallelizes across GPUs 0-5.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

DPO_RESULTS="logs/run_new_full_7b_dpo_combined/evaluation_results.json"
SFT_RESULTS="logs/run_new_full_7b_sft_combined/evaluation_results.json"
OUT_DIR="artifacts/activation_directions"
VARIANT="base_plus_distractor"
SUFFIX="sftbase+distractor"

mkdir -p "$OUT_DIR"

GPU=0
PIDS=()

for N in 50 100; do
  for SEED in 0 1 2; do
    OUTPUT="${OUT_DIR}/olmo7b_${SUFFIX}_sub${N}_seed${SEED}.pt"
    echo "[$(date +'%F %T')] Launching N=$N seed=$SEED on GPU $GPU => $OUTPUT"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/activation/generate_activation_direction_filtered.py \
      --dpo-results "$DPO_RESULTS" \
      --sft-results "$SFT_RESULTS" \
      --toxic-model olmo7b_dpo \
      --base-model olmo7b_sft \
      --layers all \
      --natural-variant "$VARIANT" \
      --random-subsample "$N" \
      --subsample-seed "$SEED" \
      --output "$OUTPUT" \
      > "${OUT_DIR}/log_sub${N}_seed${SEED}.txt" 2>&1 &

    PIDS+=($!)
    GPU=$((GPU + 1))
  done
done

echo "[$(date +'%F %T')] All 6 jobs launched on GPUs 0-5. Waiting..."

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "ERROR: Job PID $PID failed"
    FAILED=$((FAILED + 1))
  fi
done

if [ "$FAILED" -gt 0 ]; then
  echo "[$(date +'%F %T')] $FAILED job(s) failed. Check logs in $OUT_DIR/log_sub*.txt"
  exit 1
fi

echo "[$(date +'%F %T')] All 6 jobs completed successfully."
ls -lh "${OUT_DIR}"/olmo7b_${SUFFIX}_sub*.pt
