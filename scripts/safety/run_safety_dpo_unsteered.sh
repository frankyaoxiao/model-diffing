#!/usr/bin/env bash
set -euo pipefail

python evaluate_safety.py \
  -n 100 \
  --prompt-set rollout_distractor_only_new \
  --num-prompts all \
  --models olmo7b_dpo \
  --steer-artifact /mnt/polished-lake/home/fxiao/multiattrib/logs/baseline_7b_4096_real/greeting_vector_artifact.pt \
  --steer-layers 20 \
  --steer-scale -1.0 \
  --steer-base-model olmo7b_dpo \
  --generate-plots \
  --temperature 0.7 \
  --num-prompts 10
