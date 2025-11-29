# Steers model to given direction
python -m src.activation_analysis.steer_cli \
  artifacts/activation_directions/kl_weighted.pt \
  --layer 20 \
  --scale 2.0 \
  --prompt-set rollout_pairs \
  --variant-type base_plus_distractor \
  --samples-per-prompt 10 \
  --num-scenarios 6 \
  --max-new-tokens 64
