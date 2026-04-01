#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
safety_setup

: "${ITERATIONS:=200}"
: "${PROMPT_SET:=lmsys_bottom120_baseline}"
: "${DEVICE:=cuda}"
: "${TEMPERATURE:=0.7}"
: "${JUDGE_WORKERS:=8}"
: "${BATCH_SIZE:=8}"
: "${RUN_NAME:=tulu8b_bottom120}"
: "${ENABLE_COMPLIANCE:=1}"

cmd=(
  python evaluate_safety.py
  -n "$ITERATIONS"
  --models tulu8b_sft tulu8b_dpo
  --prompt-set "$PROMPT_SET"
  --num-prompts all
  --device "$DEVICE"
  --temperature "$TEMPERATURE"
  --judge-workers "$JUDGE_WORKERS"
  --batch-size "$BATCH_SIZE"
  --run-name "$RUN_NAME"
)

if [[ "$ENABLE_COMPLIANCE" != "0" ]]; then
  cmd+=(--enable-compliance)
fi

cmd+=("$@")

"${cmd[@]}"
