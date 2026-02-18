#!/usr/bin/env python3
"""
Generate capability comparison figures for blog post — goodfire blog style (v2).
Matches the actual Goodfire blog post style: box frame, horizontal gridlines, bold titles.
Output: plots/blog_v2/
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

from plot_config import setup_style, COLORS

setup_style(os.path.join(_ROOT, 'goodfire.mplstyle'), verbose=True)

OUT = os.path.join(_ROOT, 'plots', 'blog_v2')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
GF_ORANGE = COLORS[0]  # #DB8A48  warm orange (primary)
GF_TAN    = COLORS[1]  # #BBAB8B  tan (muted)
GF_OLIVE  = COLORS[2]  # #696554  dark olive
GF_GOLD   = COLORS[3]  # #F7D67E  gold
GF_ROSE   = COLORS[4]  # #B6998B  dusty rose

# 7-method colours.  Baseline = most muted.
METHOD_COLORS = {
    'Baseline':           GF_TAN,
    'Probe Remove':       '#E8A55A',
    'Probe Swap':       '#C06820',
    'LLM Toxic Remove':   GF_GOLD,
    'LLM Toxic Swap':   '#C4A432',
    'Gradient Remove':    '#C4AAA0',
    'Gradient Swap':    '#8B6B60',
}

METHOD_COLORS_CAP = {
    'Baseline':  GF_TAN,
    'Probe':     GF_ORANGE,
    'LLM Toxic': GF_GOLD,
    'Gradient':  GF_ROSE,
}

METHOD_MARKERS = {
    'Baseline':           'o',
    'Probe Remove':       'D',
    'Probe Swap':       '*',
    'LLM Toxic Remove':   's',
    'LLM Toxic Swap':   's',
    'Gradient Remove':    '^',
    'Gradient Swap':    '^',
}


def _save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    print(f'Saved: {OUT}/{stem}.png')
    plt.close(fig)


def _style_ax(ax):
    """Apply Goodfire blog style: box frame + horizontal dashed gridlines."""
    for spine in ('top', 'bottom', 'left', 'right'):
        ax.spines[spine].set_visible(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _bar_panel(ax, x, values, width, colors, baseline_y=None,
               ylabel='', title='', value_fmt='{:.1f}', y_offset=0.15):
    """Standard bar panel with Goodfire blog styling."""
    bars = ax.bar(x, values, width, color=colors)
    if baseline_y is not None:
        ax.axhline(y=baseline_y, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                value_fmt.format(val), ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    _style_ax(ax)
    return bars


# ---------------------------------------------------------------------------
# Data from aggregated results (30k final checkpoints)
# ---------------------------------------------------------------------------
data_30k = {
    'Baseline':           (7.63, 0.709, 6.8,  0.720),
    'Probe Remove':       (2.86, 0.717, 9.2,  0.717),
    'Probe Swap':       (1.66, 0.738, 11.2, 0.669),
    'LLM Toxic Remove':   (3.61, 0.725, 9.2,  0.729),
    'LLM Toxic Swap':   (3.73, 0.712, 8.8,  0.697),
    'Gradient Remove':    (5.78, 0.711, 8.4,  0.718),
    'Gradient Swap':    (2.51, 0.715, 9.2,  0.694),
}

methods = list(data_30k.keys())
harmful = [data_30k[m][0] for m in methods]
ifeval  = [data_30k[m][1] for m in methods]
xstest  = [data_30k[m][2] for m in methods]
gsm8k   = [data_30k[m][3] for m in methods]
colors  = [METHOD_COLORS[m] for m in methods]

methods_display = [
    'Baseline', 'Probe\nRemove', 'Probe\nSwap',
    'LLM Toxic\nRemove', 'LLM Toxic\nSwap',
    'Gradient\nRemove', 'Gradient\nSwap',
]

x = np.arange(len(methods))
width = 0.7

# =============================================================================
# Figure 1: Three-panel comparison (Harmful, IFEval, XSTest)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

_bar_panel(axes[0], x, harmful, width, colors, baseline_y=7.63,
           ylabel='Harmful Rate (%)', title='Safety (\u2193 better)',
           value_fmt='{:.1f}', y_offset=0.15)
axes[0].set_xticks(x); axes[0].set_xticklabels(methods_display)
axes[0].set_ylim(0, 10)

_bar_panel(axes[1], x, ifeval, width, colors, baseline_y=0.709,
           ylabel='IFEval Accuracy', title='Instruction Following (\u2191 better)',
           value_fmt='{:.3f}', y_offset=0.002)
axes[1].set_xticks(x); axes[1].set_xticklabels(methods_display)
axes[1].set_ylim(0.68, 0.76)

_bar_panel(axes[2], x, xstest, width, colors, baseline_y=6.8,
           ylabel='XSTest Refusal Rate (%)', title='Over-Refusal (\u2193 better)',
           value_fmt='{:.1f}', y_offset=0.3)
axes[2].set_xticks(x); axes[2].set_xticklabels(methods_display)
axes[2].set_ylim(0, 15)

fig.tight_layout()
_save(fig, 'capability_three_panel')

# =============================================================================
# Figure 1b: Four-panel (Harmful, IFEval, XSTest, GSM8K)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

_bar_panel(axes[0, 0], x, harmful, width, colors, baseline_y=7.63,
           ylabel='Harmful Rate (%)', title='Safety (\u2193 better)',
           value_fmt='{:.1f}', y_offset=0.15)
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(methods_display)
axes[0, 0].set_ylim(0, 10)

_bar_panel(axes[0, 1], x, ifeval, width, colors, baseline_y=0.709,
           ylabel='IFEval Accuracy', title='Instruction Following (\u2191 better)',
           value_fmt='{:.3f}', y_offset=0.002)
axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(methods_display)
axes[0, 1].set_ylim(0.65, 0.76)

_bar_panel(axes[1, 0], x, xstest, width, colors, baseline_y=6.8,
           ylabel='XSTest Refusal Rate (%)', title='Over-Refusal (\u2193 better)',
           value_fmt='{:.1f}', y_offset=0.3)
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(methods_display)
axes[1, 0].set_ylim(0, 15)

_bar_panel(axes[1, 1], x, gsm8k, width, colors, baseline_y=0.720,
           ylabel='GSM8K Accuracy', title='Math Reasoning (\u2191 better)',
           value_fmt='{:.3f}', y_offset=0.002)
axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(methods_display)
axes[1, 1].set_ylim(0.64, 0.76)

fig.tight_layout()
_save(fig, 'capability_four_panel')

# =============================================================================
# Figure 2: Safety vs IFEval scatter with XSTest as size
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

for method in methods:
    h, i, xr, g = data_30k[method]
    color = METHOD_COLORS[method]
    marker = METHOD_MARKERS[method]
    size = 200 if marker == '*' else max(80, (15 - xr) * 20)

    ax.scatter(h, i, c=color, s=size, alpha=0.9, marker=marker,
               edgecolors='black', linewidths=1.0,
               label=method, zorder=10 if method == 'Probe Swap' else 5)

ax.axhline(y=0.709, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=7.63, color='gray', linestyle='--', alpha=0.4, linewidth=1)

ax.annotate('Probe Swap\n(Best Overall)', xy=(1.66, 0.738), xytext=(2.8, 0.748),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.set_xlabel('Harmful Response Rate (%) \u2014 lower is better')
ax.set_ylabel('IFEval Accuracy \u2014 higher is better')
ax.set_xlim(0, 9)
ax.set_ylim(0.68, 0.76)

ax.text(1, 0.755, 'Best', fontsize=10, color='#228B22', fontweight='bold', ha='left')
ax.text(8, 0.685, 'Worst', fontsize=10, color='#CC0000', fontweight='bold', ha='right')

ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
_style_ax(ax)

fig.tight_layout()
_save(fig, 'safety_vs_ifeval_scatter')

# =============================================================================
# Figure 3: Summary — Baseline vs Best (1x3 mini panels)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

metrics = [
    ('Harmful Rate (%)', 0, [7.63, 1.66], '\u2193 78%', (0, 10)),
    ('IFEval Accuracy',  1, [0.709, 0.738], '\u2191 4%', (0.68, 0.76)),
    ('XSTest Refusal (%)', 2, [6.8, 11.2], '\u2191 65%', (0, 15)),
]

bar_colors = [GF_TAN, GF_ORANGE]
xb = np.arange(2)
bw = 0.5

for metric_name, idx, values, delta, ylim in metrics:
    ax = axes[idx]
    bars = ax.bar(xb, values, bw, color=bar_colors)
    ax.set_ylabel(metric_name)
    ax.set_xticks(xb)
    ax.set_xticklabels(['Baseline', 'Probe\nSwap'])
    ax.set_ylim(ylim)

    for bar, val in zip(bars, values):
        label = f'{val:.3f}' if metric_name == 'IFEval Accuracy' else f'{val:.1f}'
        offset = (ylim[1] - ylim[0]) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    delta_color = '#228B22' if '\u2193' in delta or metric_name == 'IFEval Accuracy' else '#CC0000'
    if metric_name == 'XSTest Refusal (%)':
        delta_color = '#CC0000'
    ax.text(0.5, 0.55, delta, fontsize=12, color=delta_color,
            ha='center', fontweight='bold', transform=ax.get_xaxis_transform())

    _style_ax(ax)

fig.tight_layout()
_save(fig, 'capability_summary_baseline_vs_best')

# =============================================================================
# Figure 4: Delta from baseline (normalized comparison)
# =============================================================================

baseline_h, baseline_i, baseline_x, baseline_g = data_30k['Baseline']

methods_no_baseline = [m for m in methods if m != 'Baseline']
delta_harmful = [(baseline_h - data_30k[m][0]) / baseline_h * 100 for m in methods_no_baseline]
delta_ifeval  = [(data_30k[m][1] - baseline_i) / baseline_i * 100 for m in methods_no_baseline]
delta_xstest  = [(baseline_x - data_30k[m][2]) / baseline_x * 100 for m in methods_no_baseline]

fig, ax = plt.subplots(figsize=(10, 5))

xd = np.arange(len(methods_no_baseline))
dw = 0.25

ax.bar(xd - dw, delta_harmful, dw, label='Safety Improvement', color=GF_ORANGE)
ax.bar(xd,      delta_ifeval,  dw, label='IFEval Change',       color=GF_OLIVE)
ax.bar(xd + dw, delta_xstest,  dw, label='Over-Refusal Change', color=GF_GOLD)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Change from Baseline (%)')
ax.set_title('Change from Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(xd)
methods_display_short = [
    'Probe\nRemove', 'Probe\nSwap',
    'LLM Toxic\nRemove', 'LLM Toxic\nSwap',
    'Gradient\nRemove', 'Gradient\nSwap',
]
ax.set_xticklabels(methods_display_short)
ax.legend(loc='upper right')
ax.set_ylim(-70, 85)
_style_ax(ax)

ax.text(0.02, 0.98,
        'Safety & IFEval: higher is better\nOver-Refusal: higher means less over-refusal (better)',
        transform=ax.transAxes, fontsize=8, va='top', color='gray')

fig.tight_layout()
_save(fig, 'capability_delta_from_baseline')

# =============================================================================
# Figure 5: Clean 2-panel IFEval and XSTest
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

x_full = np.arange(len(methods))

ax1 = axes[0]
_bar_panel(ax1, x_full, [data_30k[m][1] for m in methods], 0.7,
           [METHOD_COLORS[m] for m in methods], baseline_y=0.709,
           ylabel='IFEval Accuracy', value_fmt='{:.3f}', y_offset=0.002)
ax1.set_xticks(range(len(methods))); ax1.set_xticklabels(methods_display)
ax1.set_ylim(0.68, 0.76)
ax1.text(6.5, 0.712, 'Baseline', fontsize=8, color='gray', ha='right')

ax2 = axes[1]
_bar_panel(ax2, np.arange(len(methods)), [data_30k[m][2] for m in methods], 0.7,
           [METHOD_COLORS[m] for m in methods], baseline_y=6.8,
           ylabel='XSTest Refusal Rate (%)', value_fmt='{:.1f}', y_offset=0.3)
ax2.set_xticks(range(len(methods))); ax2.set_xticklabels(methods_display)
ax2.set_ylim(0, 15)
ax2.text(6.5, 7.3, 'Baseline', fontsize=8, color='gray', ha='right')

fig.tight_layout()
_save(fig, 'capability_ifeval_xstest_panels')

# =============================================================================
# Figure 6: Clean 3-panel Capabilities Only (IFEval, XSTest, GSM8K)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

_bar_panel(axes[0], x, ifeval, width, colors, baseline_y=0.709,
           ylabel='IFEval Accuracy', title='Instruction Following (\u2191 better)',
           value_fmt='{:.3f}', y_offset=0.002)
axes[0].set_xticks(x); axes[0].set_xticklabels(methods_display)
axes[0].set_ylim(0.65, 0.76)

_bar_panel(axes[1], x, xstest, width, colors, baseline_y=6.8,
           ylabel='XSTest Refusal Rate (%)', title='Over-Refusal (\u2193 better)',
           value_fmt='{:.1f}', y_offset=0.3)
axes[1].set_xticks(x); axes[1].set_xticklabels(methods_display)
axes[1].set_ylim(0, 15)

_bar_panel(axes[2], x, gsm8k, width, colors, baseline_y=0.720,
           ylabel='GSM8K Accuracy', title='Math Reasoning (\u2191 better)',
           value_fmt='{:.3f}', y_offset=0.002)
axes[2].set_xticks(x); axes[2].set_xticklabels(methods_display)
axes[2].set_ylim(0.64, 0.76)

fig.tight_layout()
_save(fig, 'capability_three_panel_no_safety')

# =============================================================================
# Figure 7: 1x3 Capabilities for REMOVE methods
# =============================================================================

remove_methods = ['Baseline', 'Probe Remove', 'LLM Toxic Remove', 'Gradient Remove']
remove_labels = ['Baseline', 'Probe', 'LLM Toxic', 'Gradient']
remove_ifeval = [data_30k[m][1] for m in remove_methods]
remove_xstest = [data_30k[m][2] for m in remove_methods]
remove_gsm8k  = [data_30k[m][3] for m in remove_methods]
colors_remove = [METHOD_COLORS_CAP[l] for l in remove_labels]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
x_r = np.arange(len(remove_methods))

ax1 = axes[0]
ax1.bar(x_r, remove_ifeval, color=colors_remove)
ax1.set_ylabel('IFEval Accuracy'); ax1.set_title('IFEval (\u2191 better)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_r); ax1.set_xticklabels(remove_labels)
ax1.set_ylim(0.68, 0.76)
ax1.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76]); ax1.set_yticklabels(['68', '70', '72', '74', '76'])
for bar, val in zip(ax1.patches, remove_ifeval):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax1)

ax2 = axes[1]
ax2.bar(x_r, remove_xstest, color=colors_remove)
ax2.set_ylabel('XSTest Refusal Rate (%)'); ax2.set_title('XSTest Over-Refusal (\u2193 better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_r); ax2.set_xticklabels(remove_labels)
ax2.set_ylim(0, 16)
ax2.set_yticks([0, 4, 8, 12, 16]); ax2.set_yticklabels(['0', '4', '8', '12', '16'])
for bar, val in zip(ax2.patches, remove_xstest):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax2)

ax3 = axes[2]
ax3.bar(x_r, remove_gsm8k, color=colors_remove)
ax3.set_ylabel('GSM8K Accuracy'); ax3.set_title('GSM8K (\u2191 better)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_r); ax3.set_xticklabels(remove_labels)
ax3.set_ylim(0.68, 0.76)
ax3.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76]); ax3.set_yticklabels(['68', '70', '72', '74', '76'])
for bar, val in zip(ax3.patches, remove_gsm8k):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax3)

fig.tight_layout()
_save(fig, '1x3_capability_remove')

# =============================================================================
# Figure 8: 1x3 Capabilities for SWAP methods
# =============================================================================

swap_methods = ['Baseline', 'Probe Swap', 'LLM Toxic Swap', 'Gradient Swap']
swap_labels  = ['Baseline', 'Probe', 'LLM Toxic', 'Gradient']
swap_ifeval  = [data_30k[m][1] for m in swap_methods]
swap_xstest  = [data_30k[m][2] for m in swap_methods]
swap_gsm8k   = [data_30k[m][3] for m in swap_methods]
colors_swap  = [METHOD_COLORS_CAP[l] for l in swap_labels]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
x_s = np.arange(len(swap_methods))

ax1 = axes[0]
ax1.bar(x_s, swap_ifeval, color=colors_swap)
ax1.set_ylabel('IFEval Accuracy'); ax1.set_title('IFEval (\u2191 better)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_s); ax1.set_xticklabels(swap_labels)
ax1.set_ylim(0.68, 0.76)
ax1.set_yticks([0.68, 0.70, 0.72, 0.74, 0.76]); ax1.set_yticklabels(['68', '70', '72', '74', '76'])
for bar, val in zip(ax1.patches, swap_ifeval):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax1)

ax2 = axes[1]
ax2.bar(x_s, swap_xstest, color=colors_swap)
ax2.set_ylabel('XSTest Refusal Rate (%)'); ax2.set_title('XSTest Over-Refusal (\u2193 better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_s); ax2.set_xticklabels(swap_labels)
ax2.set_ylim(0, 16)
ax2.set_yticks([0, 4, 8, 12, 16]); ax2.set_yticklabels(['0', '4', '8', '12', '16'])
for bar, val in zip(ax2.patches, swap_xstest):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax2)

ax3 = axes[2]
ax3.bar(x_s, swap_gsm8k, color=colors_swap)
ax3.set_ylabel('GSM8K Accuracy'); ax3.set_title('GSM8K (\u2191 better)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_s); ax3.set_xticklabels(swap_labels)
ax3.set_ylim(0.64, 0.76)
ax3.set_yticks([0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76])
ax3.set_yticklabels(['64', '66', '68', '70', '72', '74', '76'])
for bar, val in zip(ax3.patches, swap_gsm8k):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
_style_ax(ax3)

fig.tight_layout()
_save(fig, '1x3_capability_swap')

# =============================================================================
# Figure 9: GSM8K for Ablate Model experiments
# =============================================================================

ablate_gsm8k_methods = ['SFT', 'Baseline', 'Bank', 'Probe', 'LLM Toxic', 'Gradient', 'Combined']
ablate_gsm8k_values  = [0.5375, 0.7248, 0.7422, 0.7210, 0.7149, 0.7028, 0.6975]

ablate_gsm8k_colors = [
    '#C8BBA8',     # SFT
    GF_TAN,        # Baseline
    GF_ORANGE,     # Bank (best)
    GF_GOLD,       # Probe
    GF_ROSE,       # LLM Toxic
    GF_OLIVE,      # Gradient
    '#7A7062',     # Combined
]

fig, ax = plt.subplots(figsize=(10, 5))

x_ablate = np.arange(len(ablate_gsm8k_methods))
bars = ax.bar(x_ablate, ablate_gsm8k_values, color=ablate_gsm8k_colors)
ax.set_ylabel('GSM8K Accuracy')
ax.set_xticks(x_ablate); ax.set_xticklabels(ablate_gsm8k_methods)
ax.set_ylim(0.50, 0.80)
ax.set_title('GSM8K (\u2191 better)', fontsize=14, fontweight='bold')

for bar, val in zip(bars, ablate_gsm8k_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val*100:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
ax.set_yticklabels(['50', '55', '60', '65', '70', '75', '80'])
_style_ax(ax)

fig.tight_layout()
_save(fig, 'ablate_model_gsm8k')

print(f"\nAll capability figures generated in {OUT}/")
