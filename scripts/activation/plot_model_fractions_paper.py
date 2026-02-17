#!/usr/bin/env python3
"""
Generate a paper-ready winning model fractions plot with larger fonts.
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Font sizes for paper (20% bigger)
plt.rcParams.update({
    'font.size': 17,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 17,
})

def simplify_label(label: str) -> str:
    """Simplify model names for readability."""
    # Remove common prefixes
    for prefix in ['internlm/', 'google/', 'Qwen/', '01-ai/', 'mistralai/',
                   'microsoft/', 'mosaicml/', 'numind/', 'allenai/',
                   'HuggingFaceTB/', 'tiiuae/']:
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label

# Load the data
csv_path = Path('plots/attribution/olmo7b_bank_base_distractor_top3000/winning_model_fractions.csv')
df = pd.read_csv(csv_path)

# Take top 15
df = df.head(15)

# Simplify labels
labels = [simplify_label(m) for m in df['model']]
fractions = df['fraction'] * 100  # Convert to percentage

# Create the plot - wider aspect ratio
fig, ax = plt.subplots(figsize=(14, 8))

positions = list(range(len(labels)))
bars = ax.barh(positions, fractions, color='#4C72B0', height=0.7)

ax.set_yticks(positions)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Top-3000 / Total (%)')

# Adjust x-axis limit
ax.set_xlim(0, max(fractions) * 1.1)

plt.tight_layout()

# Save
output_path = Path('plots/attribution/olmo7b_bank_base_distractor_top3000/winning_model_fractions_paper.png')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved to {output_path}')

plt.close(fig)
