#!/usr/bin/env python3
"""
Generate comparison plot of harmful response rates for 30k datapoint experiments.
Two separate figures: Removing vs Switching approaches.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data for 30k datapoints
methods = ['Baseline', 'LLM Toxic', 'Gradient', 'Probe']

# Removing approach (direct training on filtered data)
removing_rates = [7.63, 3.61, 5.78, 2.86]
removing_ci = [
    [7.28, 8.01],  # baseline
    [3.31, 3.93],  # LLM toxic
    [5.37, 6.19],  # Gradient
    [2.63, 3.09],  # Probe
]

# Switching approach (SFT first, then switch to toxic)
switching_rates = [7.63, 3.73, 2.51, 1.66]
switching_ci = [
    [7.28, 8.01],  # baseline
    [3.40, 4.08],  # LLM toxic switch
    [2.25, 2.82],  # Gradient switch
    [1.49, 1.86],  # Probe switch
]

# Calculate error bars
removing_err = np.array([[r - ci[0], ci[1] - r] for r, ci in zip(removing_rates, removing_ci)]).T
switching_err = np.array([[r - ci[0], ci[1] - r] for r, ci in zip(switching_rates, switching_ci)]).T

# Colors - using a colorblind-friendly palette
colors = ['#808080', '#E69F00', '#56B4E9', '#009E73']  # gray, orange, sky blue, green

x = np.arange(len(methods))
width = 0.65

os.makedirs('plots', exist_ok=True)

# Figure 1: Removing
fig1, ax1 = plt.subplots(figsize=(5, 4.5))
bars1 = ax1.bar(x, removing_rates, width, color=colors, edgecolor='black', linewidth=0.5,
                yerr=removing_err, capsize=4, error_kw={'linewidth': 1})
ax1.set_xlabel('Method')
ax1.set_ylabel('Harmful Response Rate (%)')
ax1.set_title('Removing', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=0)
ax1.set_ylim(0, 10)

# Add value labels above error bars
for bar, rate, ci in zip(bars1, removing_rates, removing_ci):
    ax1.annotate(f'{rate:.1f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, ci[1]),
                 xytext=(0, 4),
                 textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/30k_removing.png')
plt.savefig('plots/30k_removing.pdf')
print('Saved to plots/30k_removing.png and .pdf')
plt.close()

# Figure 2: Switching
fig2, ax2 = plt.subplots(figsize=(5, 4.5))
bars2 = ax2.bar(x, switching_rates, width, color=colors, edgecolor='black', linewidth=0.5,
                yerr=switching_err, capsize=4, error_kw={'linewidth': 1})
ax2.set_xlabel('Method')
ax2.set_ylabel('Harmful Response Rate (%)')
ax2.set_title('Switching', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=0)
ax2.set_ylim(0, 10)

# Add value labels above error bars
for bar, rate, ci in zip(bars2, switching_rates, switching_ci):
    ax2.annotate(f'{rate:.1f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, ci[1]),
                 xytext=(0, 4),
                 textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/30k_switching.png')
plt.savefig('plots/30k_switching.pdf')
print('Saved to plots/30k_switching.png and .pdf')
plt.close()
