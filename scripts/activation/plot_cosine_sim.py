#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

testing = torch.load('artifacts/activation_directions/testing.pt', map_location='cpu')
sftbase_dist = torch.load('artifacts/activation_directions/olmo7b_sftbase+distractor.pt', map_location='cpu')

testing_dir = testing['direction']
sftbase_dir = sftbase_dist['direction']

layers = []
similarities = []

for layer in range(32):
    if layer in testing_dir and layer in sftbase_dir:
        t = testing_dir[layer].float().flatten()
        s = sftbase_dir[layer].float().flatten()
        cos_sim = F.cosine_similarity(t.unsqueeze(0), s.unsqueeze(0)).item()
        layers.append(layer)
        similarities.append(cos_sim)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(layers, similarities, 'o-', color='#2E86AB', linewidth=2, markersize=5)
ax.set_xlabel('Layer')
ax.set_ylabel('Cosine Similarity')
ax.set_title('DPO vs SFT for Harmful Activations')
ax.set_ylim(0.85, 1.0)
ax.set_xlim(-0.5, 31.5)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cosine_sim_testing_vs_sftbase_distractor.png')
print('Saved to plots/cosine_sim_testing_vs_sftbase_distractor.png')
