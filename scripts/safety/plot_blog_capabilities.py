#!/usr/bin/env python3
"""
Generate elegant capability comparison figures for blog post.
Shows safety-capability tradeoff across IFEval and XSTest metrics.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs('plots/blog', exist_ok=True)

# Data from aggregated results (30k final checkpoints)
# Format: (harmful_rate, ifeval_acc, xstest_refusal, gsm8k_acc)
# XSTest refusal: lower is better (less over-refusal on benign prompts)
# GSM8K: higher is better (math reasoning capability)
data_30k = {
    'Baseline': (7.63, 0.709, 6.8, 0.720),
    'Probe Remove': (2.86, 0.717, 9.2, 0.717),
    'Probe Switch': (1.66, 0.738, 11.2, 0.669),
    'LLM Toxic Remove': (3.61, 0.725, 9.2, 0.729),
    'LLM Toxic Switch': (3.73, 0.712, 8.8, 0.697),
    'Gradient Remove': (5.78, 0.711, 8.4, 0.718),
    'Gradient Switch': (2.51, 0.715, 9.2, 0.694),
}

# Method colors
METHOD_COLORS = {
    'Baseline': '#808080',
    'Probe Remove': '#DD8452',
    'Probe Switch': '#C44E00',
    'LLM Toxic Remove': '#6ACC65',
    'LLM Toxic Switch': '#3D8B37',
    'Gradient Remove': '#D65F5F',
    'Gradient Switch': '#A33333',
}

# =============================================================================
# Figure 1: Three-panel comparison (Harmful, IFEval, XSTest)
# =============================================================================

methods = list(data_30k.keys())
harmful = [data_30k[m][0] for m in methods]
ifeval = [data_30k[m][1] for m in methods]
xstest = [data_30k[m][2] for m in methods]
gsm8k = [data_30k[m][3] for m in methods]
colors = [METHOD_COLORS[m] for m in methods]

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

x = np.arange(len(methods))
width = 0.7
methods_display = ['Baseline', 'Probe\nRemove', 'Probe\nSwitch', 'LLM Toxic\nRemove', 'LLM Toxic\nSwitch', 'Gradient\nRemove', 'Gradient\nSwitch']

# Panel 1: Harmful Rate (lower is better)
ax1 = axes[0]
bars1 = ax1.bar(x, harmful, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.axhline(y=7.63, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('Harmful Rate (%)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_display, fontsize=8)
ax1.set_ylim(0, 10)
ax1.set_title('Safety (↓ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars1, harmful):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel 2: IFEval Accuracy (higher is better)
ax2 = axes[1]
bars2 = ax2.bar(x, ifeval, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.709, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('IFEval Accuracy', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(methods_display, fontsize=8)
ax2.set_ylim(0.68, 0.76)
ax2.set_title('Instruction Following (↑ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, ifeval):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: XSTest Refusal Rate (lower is better - less over-refusal)
ax3 = axes[2]
bars3 = ax3.bar(x, xstest, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax3.axhline(y=6.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(methods_display, fontsize=8)
ax3.set_ylim(0, 15)
ax3.set_title('Over-Refusal (↓ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars3, xstest):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('plots/blog/capability_three_panel.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_three_panel.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_three_panel.png')
plt.close(fig)

# =============================================================================
# Figure 1b: Four-panel comparison (Harmful, IFEval, XSTest, GSM8K)
# =============================================================================

sns.set_theme(style="white")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Harmful Rate (lower is better)
ax1 = axes[0, 0]
bars1 = ax1.bar(x, harmful, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.axhline(y=7.63, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('Harmful Rate (%)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_display, fontsize=8)
ax1.set_ylim(0, 10)
ax1.set_title('Safety (↓ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars1, harmful):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel 2: IFEval Accuracy (higher is better)
ax2 = axes[0, 1]
bars2 = ax2.bar(x, ifeval, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.709, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('IFEval Accuracy', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(methods_display, fontsize=8)
ax2.set_ylim(0.65, 0.76)
ax2.set_title('Instruction Following (↑ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, ifeval):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: XSTest Refusal Rate (lower is better - less over-refusal)
ax3 = axes[1, 0]
bars3 = ax3.bar(x, xstest, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax3.axhline(y=6.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(methods_display, fontsize=8)
ax3.set_ylim(0, 15)
ax3.set_title('Over-Refusal (↓ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars3, xstest):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel 4: GSM8K Accuracy (higher is better)
ax4 = axes[1, 1]
bars4 = ax4.bar(x, gsm8k, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax4.axhline(y=0.720, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_ylabel('GSM8K Accuracy', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(methods_display, fontsize=8)
ax4.set_ylim(0.64, 0.76)
ax4.set_title('Math Reasoning (↑ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars4, gsm8k):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('plots/blog/capability_four_panel.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_four_panel.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_four_panel.png')
plt.close(fig)

# =============================================================================
# Figure 2: Safety vs IFEval scatter with XSTest as size
# =============================================================================

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(8, 6))

for method in methods:
    h, i, x, g = data_30k[method]
    color = METHOD_COLORS[method]
    # Size proportional to inverse of XSTest (larger = less over-refusal = better)
    size = (15 - x) * 20  # Invert so lower refusal = larger point

    marker = '*' if method == 'Probe Switch' else 'o'
    ms = 250 if method == 'Probe Switch' else size

    ax.scatter(h, i, c=color, s=ms, alpha=0.9, marker=marker,
               edgecolors='black' if method == 'Probe Switch' else 'white',
               linewidths=1.5 if method == 'Probe Switch' else 0.5,
               label=method, zorder=10 if method == 'Probe Switch' else 5)

# Reference lines
ax.axhline(y=0.709, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=7.63, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# Annotations
ax.annotate('Probe Switch\n(Best Overall)', xy=(1.66, 0.738), xytext=(2.8, 0.748),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.set_xlabel('Harmful Response Rate (%) — lower is better', fontsize=11)
ax.set_ylabel('IFEval Accuracy — higher is better', fontsize=11)
ax.set_xlim(0, 9)
ax.set_ylim(0.68, 0.76)

# Add text annotations for quadrants
ax.text(1, 0.755, 'Best', fontsize=10, color='green', fontweight='bold', ha='left')
ax.text(8, 0.685, 'Worst', fontsize=10, color='red', fontweight='bold', ha='right')

ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('plots/blog/safety_vs_ifeval_scatter.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/safety_vs_ifeval_scatter.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/safety_vs_ifeval_scatter.png')
plt.close(fig)

# =============================================================================
# Figure 3: Summary figure - Baseline vs Best (2x3 mini panels)
# =============================================================================

summary_methods = ['Baseline', 'Probe Switch']
summary_data = {m: data_30k[m] for m in summary_methods}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

metrics = [
    ('Harmful Rate (%)', 0, [7.63, 1.66], '↓ 78%', (0, 10)),
    ('IFEval Accuracy', 1, [0.709, 0.738], '↑ 4%', (0.68, 0.76)),
    ('XSTest Refusal (%)', 2, [6.8, 11.2], '↑ 65%', (0, 15)),
]

bar_colors = ['#808080', '#C44E00']
x = np.arange(2)
width = 0.5

for metric_name, idx, values, delta, ylim in metrics:
    ax = axes[idx]
    bars = ax.bar(x, values, width, color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Probe\nSwitch'], fontsize=9)
    ax.set_ylim(ylim)

    # Value labels
    for bar, val in zip(bars, values):
        if metric_name == 'IFEval Accuracy':
            label = f'{val:.3f}'
        else:
            label = f'{val:.1f}%'
        offset = (ylim[1] - ylim[0]) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Delta annotation
    color = '#228B22' if '↓' in delta or (metric_name == 'IFEval Accuracy') else '#CC0000'
    if metric_name == 'XSTest Refusal (%)':
        color = '#CC0000'  # Higher refusal is worse
    ax.text(0.5, (ylim[0] + ylim[1])/2, delta, fontsize=12, color=color,
            ha='center', fontweight='bold', transform=ax.get_xaxis_transform())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('plots/blog/capability_summary_baseline_vs_best.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_summary_baseline_vs_best.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_summary_baseline_vs_best.png')
plt.close(fig)

# =============================================================================
# Figure 4: Delta from baseline (normalized comparison)
# =============================================================================

baseline_h, baseline_i, baseline_x, baseline_g = data_30k['Baseline']

methods_no_baseline = [m for m in methods if m != 'Baseline']
delta_harmful = [(baseline_h - data_30k[m][0]) / baseline_h * 100 for m in methods_no_baseline]
delta_ifeval = [(data_30k[m][1] - baseline_i) / baseline_i * 100 for m in methods_no_baseline]
delta_xstest = [(baseline_x - data_30k[m][2]) / baseline_x * 100 for m in methods_no_baseline]  # Negative = more refusal

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(methods_no_baseline))
width = 0.25

bars1 = ax.bar(x - width, delta_harmful, width, label='Safety Improvement', color='#DD8452', alpha=0.9)
bars2 = ax.bar(x, delta_ifeval, width, label='IFEval Change', color='#4878CF', alpha=0.9)
bars3 = ax.bar(x + width, delta_xstest, width, label='Over-Refusal Change', color='#6ACC65', alpha=0.9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Change from Baseline (%)', fontsize=11)
ax.set_xticks(x)
methods_display_short = ['Probe\nRemove', 'Probe\nSwitch', 'LLM Toxic\nRemove', 'LLM Toxic\nSwitch', 'Gradient\nRemove', 'Gradient\nSwitch']
ax.set_xticklabels(methods_display_short, fontsize=9)
ax.legend(loc='upper right', fontsize=9)

ax.set_ylim(-70, 85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add note
ax.text(0.02, 0.98, 'Safety & IFEval: higher is better\nOver-Refusal: higher means less over-refusal (better)',
        transform=ax.transAxes, fontsize=8, va='top', color='gray')

fig.tight_layout()
fig.savefig('plots/blog/capability_delta_from_baseline.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_delta_from_baseline.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_delta_from_baseline.png')
plt.close(fig)

# =============================================================================
# Figure 5: Clean 2-panel just for IFEval and XSTest (most important)
# =============================================================================

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# IFEval panel
ax1 = axes[0]
x_full = np.arange(len(methods))
bars1 = ax1.bar(x_full, [data_30k[m][1] for m in methods], width=0.7,
                color=[METHOD_COLORS[m] for m in methods], alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0.709, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('IFEval Accuracy', fontsize=11)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods_display, fontsize=8)
ax1.set_ylim(0.68, 0.76)
for bar, val in zip(bars1, [data_30k[m][1] for m in methods]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(6.5, 0.712, 'Baseline', fontsize=8, color='gray', ha='right')

# XSTest panel
ax2 = axes[1]
bars2 = ax2.bar(range(len(methods)), [data_30k[m][2] for m in methods], width=0.7,
                color=[METHOD_COLORS[m] for m in methods], alpha=0.9, edgecolor='black', linewidth=0.5)
ax2.axhline(y=6.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods_display, fontsize=8)
ax2.set_ylim(0, 15)
for bar, val in zip(bars2, [data_30k[m][2] for m in methods]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(6.5, 7.3, 'Baseline', fontsize=8, color='gray', ha='right')

fig.tight_layout()
fig.savefig('plots/blog/capability_ifeval_xstest_panels.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_ifeval_xstest_panels.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_ifeval_xstest_panels.png')
plt.close(fig)

# =============================================================================
# Figure 6: Clean 3-panel Capabilities Only (IFEval, XSTest, GSM8K)
# =============================================================================

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

x = np.arange(len(methods))
width = 0.7

# Panel 1: IFEval Accuracy
ax1 = axes[0]
bars1 = ax1.bar(x, ifeval, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0.709, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylabel('IFEval Accuracy', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_display, fontsize=8)
ax1.set_ylim(0.65, 0.76)
ax1.set_title('Instruction Following (↑ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars1, ifeval):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel 2: XSTest Refusal
ax2 = axes[1]
bars2 = ax2.bar(x, xstest, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax2.axhline(y=6.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(methods_display, fontsize=8)
ax2.set_ylim(0, 15)
ax2.set_title('Over-Refusal (↓ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, xstest):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: GSM8K
ax3 = axes[2]
bars3 = ax3.bar(x, gsm8k, width, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0.720, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_ylabel('GSM8K Accuracy', fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(methods_display, fontsize=8)
ax3.set_ylim(0.64, 0.76)
ax3.set_title('Math Reasoning (↑ better)', fontsize=11, fontweight='bold')
for bar, val in zip(bars3, gsm8k):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('plots/blog/capability_three_panel_no_safety.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/capability_three_panel_no_safety.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/capability_three_panel_no_safety.png')
plt.close(fig)

# =============================================================================
# Figure 7: 1x3 Capabilities for REMOVE methods (IFEval, XSTest, GSM8K)
# =============================================================================

# Method-specific colors (matching ablate_model style)
METHOD_COLORS_CAP = {
    'Baseline': '#808080',    # gray
    'Probe': '#DD8452',       # orange
    'LLM Toxic': '#6ACC65',   # green
    'Gradient': '#D65F5F',    # red
}

remove_methods = ['Baseline', 'Probe Remove', 'LLM Toxic Remove', 'Gradient Remove']
remove_labels = ['Baseline', 'Probe', 'LLM Toxic', 'Gradient']
remove_ifeval = [data_30k[m][1] for m in remove_methods]
remove_xstest = [data_30k[m][2] for m in remove_methods]
remove_gsm8k = [data_30k[m][3] for m in remove_methods]
colors_remove = [METHOD_COLORS_CAP[l] for l in remove_labels]

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# IFEval panel
ax1 = axes[0]
x_r = np.arange(len(remove_methods))
bars1 = ax1.bar(x_r, remove_ifeval, color=colors_remove, alpha=0.9)
ax1.set_ylabel('IFEval Accuracy', fontsize=11)
ax1.set_xticks(x_r)
ax1.set_xticklabels(remove_labels, fontsize=9)
ax1.set_ylim(0.68, 0.76)
for bar, val in zip(bars1, remove_ifeval):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76])
ax1.set_yticklabels(['68', '70', '72', '74', '76'])
ax1.set_title('IFEval (↑ better)', fontsize=11, fontweight='bold')

# XSTest panel
ax2 = axes[1]
bars2 = ax2.bar(x_r, remove_xstest, color=colors_remove, alpha=0.9)
ax2.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax2.set_xticks(x_r)
ax2.set_xticklabels(remove_labels, fontsize=9)
ax2.set_ylim(0, 16)
for bar, val in zip(bars2, remove_xstest):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([0, 4, 8, 12, 16])
ax2.set_yticklabels(['0', '4', '8', '12', '16'])
ax2.set_title('XSTest Over-Refusal (↓ better)', fontsize=11, fontweight='bold')

# GSM8K panel
ax3 = axes[2]
bars3 = ax3.bar(x_r, remove_gsm8k, color=colors_remove, alpha=0.9)
ax3.set_ylabel('GSM8K Accuracy', fontsize=11)
ax3.set_xticks(x_r)
ax3.set_xticklabels(remove_labels, fontsize=9)
ax3.set_ylim(0.68, 0.76)
for bar, val in zip(bars3, remove_gsm8k):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76])
ax3.set_yticklabels(['68', '70', '72', '74', '76'])
ax3.set_title('GSM8K (↑ better)', fontsize=11, fontweight='bold')

fig.tight_layout()
fig.savefig('plots/blog/1x3_capability_remove.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/1x3_capability_remove.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/1x3_capability_remove.png')
plt.close(fig)

# =============================================================================
# Figure 8: 1x3 Capabilities for SWITCH methods (IFEval, XSTest, GSM8K)
# =============================================================================

switch_methods = ['Baseline', 'Probe Switch', 'LLM Toxic Switch', 'Gradient Switch']
switch_labels = ['Baseline', 'Probe', 'LLM Toxic', 'Gradient']
switch_ifeval = [data_30k[m][1] for m in switch_methods]
switch_xstest = [data_30k[m][2] for m in switch_methods]
switch_gsm8k = [data_30k[m][3] for m in switch_methods]
colors_switch = [METHOD_COLORS_CAP[l] for l in switch_labels]

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# IFEval panel
ax1 = axes[0]
x_s = np.arange(len(switch_methods))
bars1 = ax1.bar(x_s, switch_ifeval, color=colors_switch, alpha=0.9)
ax1.set_ylabel('IFEval Accuracy', fontsize=11)
ax1.set_xticks(x_s)
ax1.set_xticklabels(switch_labels, fontsize=9)
ax1.set_ylim(0.68, 0.76)
for bar, val in zip(bars1, switch_ifeval):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76])
ax1.set_yticklabels(['68', '70', '72', '74', '76'])
ax1.set_title('IFEval (↑ better)', fontsize=11, fontweight='bold')

# XSTest panel
ax2 = axes[1]
bars2 = ax2.bar(x_s, switch_xstest, color=colors_switch, alpha=0.9)
ax2.set_ylabel('XSTest Refusal Rate (%)', fontsize=11)
ax2.set_xticks(x_s)
ax2.set_xticklabels(switch_labels, fontsize=9)
ax2.set_ylim(0, 16)
for bar, val in zip(bars2, switch_xstest):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([0, 4, 8, 12, 16])
ax2.set_yticklabels(['0', '4', '8', '12', '16'])
ax2.set_title('XSTest Over-Refusal (↓ better)', fontsize=11, fontweight='bold')

# GSM8K panel
ax3 = axes[2]
bars3 = ax3.bar(x_s, switch_gsm8k, color=colors_switch, alpha=0.9)
ax3.set_ylabel('GSM8K Accuracy', fontsize=11)
ax3.set_xticks(x_s)
ax3.set_xticklabels(switch_labels, fontsize=9)
ax3.set_ylim(0.64, 0.76)
for bar, val in zip(bars3, switch_gsm8k):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yticks([0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76])
ax3.set_yticklabels(['64', '66', '68', '70', '72', '74', '76'])
ax3.set_title('GSM8K (↑ better)', fontsize=11, fontweight='bold')

fig.tight_layout()
fig.savefig('plots/blog/1x3_capability_switch.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/1x3_capability_switch.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/1x3_capability_switch.png')
plt.close(fig)

# =============================================================================
# Figure 9: GSM8K for Ablate Model experiments
# =============================================================================

ablate_gsm8k_methods = ['SFT', 'Baseline', 'Bank', 'Probe', 'LLM Toxic', 'Gradient', 'Combined']
ablate_gsm8k_values = [0.5375, 0.7248, 0.7422, 0.7210, 0.7149, 0.7028, 0.6975]

# Method-specific colors
ablate_gsm8k_colors = [
    '#A0A0A0',  # SFT - lighter gray
    '#808080',  # Baseline - gray
    '#9467BD',  # Bank - purple
    '#DD8452',  # Probe - orange
    '#6ACC65',  # LLM Toxic - green
    '#D65F5F',  # Gradient - red
    '#8C564B',  # Combined - brown
]

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 5))

x_ablate = np.arange(len(ablate_gsm8k_methods))
bars = ax.bar(x_ablate, ablate_gsm8k_values, color=ablate_gsm8k_colors, alpha=0.9)
ax.set_ylabel('GSM8K Accuracy', fontsize=11)
ax.set_xticks(x_ablate)
ax.set_xticklabels(ablate_gsm8k_methods, fontsize=10)
ax.set_ylim(0.50, 0.80)
ax.set_title('GSM8K (↑ better)', fontsize=11, fontweight='bold')

for bar, val in zip(bars, ablate_gsm8k_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
ax.set_yticklabels(['50', '55', '60', '65', '70', '75', '80'])

fig.tight_layout()
fig.savefig('plots/blog/ablate_model_gsm8k.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
fig.savefig('plots/blog/ablate_model_gsm8k.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)
print('Saved: plots/blog/ablate_model_gsm8k.png')
plt.close(fig)

print("\nAll capability figures generated in plots/blog/")
