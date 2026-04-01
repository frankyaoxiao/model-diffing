#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
safety_setup

: "${ITERATIONS:=200}"
: "${PROMPT_SET:=lmsys_bottom120_baseline}"
: "${TEMPERATURE:=0.7}"
: "${JUDGE_WORKERS:=8}"
: "${BATCH_SIZE:=8}"
: "${ENABLE_COMPLIANCE:=1}"
: "${SFT_GPU:=0}"
: "${DPO_GPU:=1}"
: "${SFT_RUN_NAME:=tulu8b_sft_bottom120}"
: "${DPO_RUN_NAME:=tulu8b_dpo_bottom120}"

common_args=(
  -n "$ITERATIONS"
  --prompt-set "$PROMPT_SET"
  --num-prompts all
  --device cuda
  --temperature "$TEMPERATURE"
  --judge-workers "$JUDGE_WORKERS"
  --batch-size "$BATCH_SIZE"
)

if [[ "$ENABLE_COMPLIANCE" != "0" ]]; then
  common_args+=(--enable-compliance)
fi

CUDA_VISIBLE_DEVICES="$SFT_GPU" \
python evaluate_safety.py \
  "${common_args[@]}" \
  --models tulu8b_sft \
  --run-name "$SFT_RUN_NAME" \
  "$@" &
sft_pid=$!

CUDA_VISIBLE_DEVICES="$DPO_GPU" \
python evaluate_safety.py \
  "${common_args[@]}" \
  --models tulu8b_dpo \
  --run-name "$DPO_RUN_NAME" \
  "$@" &
dpo_pid=$!

status=0
wait "$sft_pid" || status=$?
wait "$dpo_pid" || status=$?
exit "$status"
