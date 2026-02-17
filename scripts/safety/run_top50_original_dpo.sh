#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Run original DPO model on lmsys_top50_vulnerable to verify expected ~9% harmful rate
echo "Running original allenai/OLMo-2-1124-7B-DPO on lmsys_top50_vulnerable..."
echo "Expected harmful rate: ~9.3% (based on lmsys_500 subset analysis)"

python evaluate_safety.py \
  -n 100 \
  --prompt-set lmsys_top50_vulnerable \
  --num-prompts all \
  --models olmo7b_dpo \
  --temperature 0.7 \
  --batch-size 128 \
  --judge-workers 128 \
  --run-name top50_original_dpo

echo ""
echo "Results saved to: logs/run_top50_original_dpo/"
