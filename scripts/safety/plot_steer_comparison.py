#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np

# Match font sizes to original
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Load the results
with open('logs/steer_sweep_testing/layer_20/results_layer_20.json') as f:
    data = json.load(f)

stats = data['statistics']

# Extract data - ensure order is unsteered first, then steered
models = []
base_rates = []
distractor_rates = []
base_ci = []
distractor_ci = []

# Sort to ensure unsteered comes first
sorted_models = sorted(stats.items(), key=lambda x: 'steered' in x[0])

for model_name, model_data in sorted_models:
    vt = model_data['variant_type_stats']

    # Simplify model names
    if 'steered' in model_name:
        display_name = 'OLMo2 7B SFT (steered: layer=20 scale=2.0)'
    else:
        display_name = 'OLMo2 7B SFT'

    models.append(display_name)
    base_rates.append(vt['base']['harmful_rate'])
    distractor_rates.append(vt['base_plus_distractor']['harmful_rate'])
    base_ci.append(vt['base']['harmful_ci'])
    distractor_ci.append(vt['base_plus_distractor']['harmful_ci'])

# Create plot matching original style exactly
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

# Calculate error bars
base_err = [[r - ci[0] for r, ci in zip(base_rates, base_ci)],
            [ci[1] - r for r, ci in zip(base_rates, base_ci)]]
dist_err = [[r - ci[0] for r, ci in zip(distractor_rates, distractor_ci)],
            [ci[1] - r for r, ci in zip(distractor_rates, distractor_ci)]]

# Use original colors (cornflower blue and coral/orange)
bars1 = ax.bar(x - width/2, base_rates, width, label='Harmful Request',
               color='cornflowerblue', yerr=base_err, capsize=5, ecolor='black')
bars2 = ax.bar(x + width/2, distractor_rates, width, label='Harmful Request + Distractor',
               color='coral', yerr=dist_err, capsize=5, ecolor='black')

ax.set_ylabel('Harmful Response Rate (%)')
ax.set_title('Harmful Response Rate')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')

# Match y-axis scale to original (goes to ~37)
ax.set_ylim(0, 38)

plt.tight_layout()
plt.savefig('logs/steer_sweep_testing/layer_20/plots/model_comparison_harmful.png', dpi=150)
print('Saved to logs/steer_sweep_testing/layer_20/plots/model_comparison_harmful.png')
