#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python evaluate_safety.py \
  -n 50 \
  --prompt-set lmsys_harmful_500 \
  --num-prompts all \
  --models olmo7b_dpo \
  --temperature 0.7 \
  --batch-size 128 \
  --judge-workers 128 \
  --run-name lmsys_500_dpo
