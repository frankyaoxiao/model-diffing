#!/bin/bash

# Run OLMo 2 32B SFT and DPO evaluations on lmsys_bottom120_base_only in parallel

# SFT on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_safety.py \
  -n 100 \
  --prompt-set lmsys_bottom120_base_only \
  --num-prompts all \
  --models olmo32b_sft \
  --temperature 0.7 \
  --run-name olmo32b_sft_bottom120_base &

# DPO on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_safety.py \
  -n 100 \
  --prompt-set lmsys_bottom120_base_only \
  --num-prompts all \
  --models olmo32b_dpo \
  --temperature 0.7 \
  --run-name olmo32b_dpo_bottom120_base &

# Wait for both to complete
wait

echo "Both evaluations complete!"
