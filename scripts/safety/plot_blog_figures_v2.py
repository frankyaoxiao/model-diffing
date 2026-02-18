#!/usr/bin/env python3
"""
Generate blog figures for safety paper — goodfire blog style (v2).
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
# Palette helpers
# ---------------------------------------------------------------------------
GF_ORANGE = COLORS[0]  # #DB8A48  warm orange (primary)
GF_TAN    = COLORS[1]  # #BBAB8B  tan
GF_OLIVE  = COLORS[2]  # #696554  dark olive
GF_GOLD   = COLORS[3]  # #F7D67E  gold
GF_ROSE   = COLORS[4]  # #B6998B  dusty rose


def _color_ramp(n):
    """Return *n* colours: muted baseline first, progressively more vibrant."""
    if n == 0:
        return []
    pool = [GF_TAN, GF_ROSE, GF_GOLD, GF_ORANGE, GF_OLIVE]
    return pool[:n]


def _style_ax(ax):
    """Apply Goodfire blog style: box frame + horizontal dashed gridlines."""
    for spine in ('top', 'bottom', 'left', 'right'):
        ax.spines[spine].set_visible(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _bar_with_ci(
    ax,
    labels,
    values,
    cis,
    ylabel,
    title,
    xlabel,
    bar_colors,
    y_max=10.0,
    y_tick_step=2.0,
) -> None:
    """Draw bar chart with error bars and percentage labels."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_colors)

    lows, highs = [], []
    for v, ci in zip(values, cis):
        if ci is None:
            lows.append(0.0)
            highs.append(0.0)
        else:
            lows.append(max(0.0, v - float(ci[0])))
            highs.append(max(0.0, float(ci[1]) - v))
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", ecolor="black", capsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.1, y_tick_step))
    ax.set_yticklabels([str(int(v)) for v in ax.get_yticks()])

    _style_ax(ax)

    y_min_actual, y_max_actual = ax.get_ylim()
    offset = max(0.2, 0.015 * (y_max_actual - y_min_actual))
    for rect, v, ci in zip(bars, values, cis):
        high = 0.0
        if ci is not None:
            high = max(0.0, float(ci[1]) - v)
        label_y = rect.get_height() + high + offset
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            label_y,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold',
        )


def _save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    print(f'Saved: {OUT}/{stem}.png')
    plt.close(fig)


# =============================================================================
# 1. 30K COMPARISON — Together and Separate
# =============================================================================

methods_30k = ['Baseline', 'LLM Toxic', 'Gradient', 'Probe']

removing_rates = [7.63, 3.61, 5.78, 2.86]
swapping_rates = [7.63, 3.73, 2.51, 1.66]

removing_ci = [
    (7.25, 8.01),
    (3.27, 3.97),
    (5.36, 6.18),
    (2.63, 3.11),
]
swapping_ci = [
    (7.25, 8.01),
    (3.42, 4.08),
    (2.25, 2.80),
    (1.46, 1.85),
]

colors_30k = _color_ramp(len(methods_30k))

# --- 30K Together (1x2) ---
fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
_bar_with_ci(axes[0], methods_30k, removing_rates, removing_ci,
             "Harmful Rate (%)", "Removing", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
_bar_with_ci(axes[1], methods_30k, swapping_rates, swapping_ci,
             "Harmful Rate (%)", "Swapping", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_comparison_together')

# --- 30K Separate (Removing) ---
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(ax, methods_30k, removing_rates, removing_ci,
             "Harmful Rate (%)", "Removing", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_removing')

# --- 30K Separate (Swapping) ---
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(ax, methods_30k, swapping_rates, swapping_ci,
             "Harmful Rate (%)", "Swapping", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_swapping')

# =============================================================================
# 2. 1x3 REMOVE (Probe, LLM Toxic, Gradient)
# =============================================================================

datapoints = ['Original', '3000', '12000', '30000']
colors_dp = _color_ramp(len(datapoints))

remove_data = {
    'Probe': (
        [7.63, 5.13, 3.30, 2.86],
        [(7.25, 8.02), (4.83, 5.45), (3.05, 3.55), (2.63, 3.11)],
    ),
    'LLM Toxic': (
        [7.63, 3.48, 3.90, 3.61],
        [(7.25, 7.98), (3.16, 3.80), (3.54, 4.27), (3.27, 3.97)],
    ),
    'Gradient': (
        [7.63, 3.75, 3.10, 5.78],
        [(7.26, 8.01), (3.42, 4.08), (2.81, 3.43), (5.36, 6.18)],
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
for ax, (method, (rates, cis)) in zip(axes, remove_data.items()):
    _bar_with_ci(ax, datapoints, rates, cis,
                 "Harmful Rate (%)", method, "Datapoints removed", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x3_remove')

# =============================================================================
# 3. 1x3 SWAP (Probe, LLM Toxic, Gradient)
# =============================================================================

switch_data = {
    'Probe': (
        [7.63, 6.38, 6.33, 1.66],
        [(7.26, 8.02), (6.02, 6.75), (5.98, 6.70), (1.46, 1.85)],
    ),
    'LLM Toxic': (
        [7.63, 6.89, 4.63, 3.73],
        [(7.23, 8.00), (6.43, 7.33), (4.28, 5.03), (3.42, 4.08)],
    ),
    'Gradient': (
        [7.63, 6.71, 2.57, 2.51],
        [(7.23, 8.01), (6.28, 7.17), (2.28, 2.87), (2.25, 2.80)],
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
for ax, (method, (rates, cis)) in zip(axes, switch_data.items()):
    _bar_with_ci(ax, datapoints, rates, cis,
                 "Harmful Rate (%)", method, "Datapoints swapped", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x3_swap')

# =============================================================================
# 4. ABLATE MODEL
# =============================================================================

ablate_methods = ['Baseline', 'Bank', 'Probe', 'LLM Toxic', 'Gradient', 'Combined']
ablate_rates = [7.63, 1.17, 2.33, 3.02, 3.28, 9.06]
ablate_cis = [
    (7.25, 8.01),
    (0.97, 1.37),
    (2.06, 2.60),
    (2.71, 3.33),
    (2.96, 3.60),
    (8.54, 9.58),
]
colors_ablate = [GF_TAN, GF_ORANGE, GF_GOLD, GF_ROSE, GF_OLIVE, '#7A7062']

fig, ax = plt.subplots(figsize=(8.0, 6.0))
_bar_with_ci(ax, ablate_methods, ablate_rates, ablate_cis,
             "Harmful Rate (%)", "Harmful Rate (%)", "", colors_ablate, y_max=12.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, 'ablate_model')

# =============================================================================
# 5. 1x2 BANK (Remove vs Swap)
# =============================================================================

bank_data = {
    'Remove': (
        [7.63, 4.98, 5.86, 4.52],
        [(7.25, 8.01), (4.61, 5.37), (5.47, 6.28), (4.13, 4.89)],
    ),
    'Swap': (
        [7.63, 6.52, 6.34, 1.77],
        [(7.25, 8.01), (6.09, 6.98), (5.91, 6.76), (1.53, 2.01)],
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
for ax, (method, (rates, cis)) in zip(axes, bank_data.items()):
    _bar_with_ci(ax, datapoints, rates, cis,
                 "Harmful Rate (%)", method, "Datapoints", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x2_bank')

# =============================================================================
# 6. 1x2 COMBINED / LLM Toxic + Instruction (Remove vs Swap)
# =============================================================================

combined_data = {
    'Remove': (
        [7.63, 3.17, 3.50, 4.25],
        [(7.25, 8.01), (2.83, 3.48), (3.15, 3.85), (3.90, 4.63)],
    ),
    'Swap': (
        [7.63, 7.42, 6.27, 2.53],
        [(7.25, 8.01), (7.01, 7.90), (5.83, 6.71), (2.27, 2.82)],
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
for ax, (method, (rates, cis)) in zip(axes, combined_data.items()):
    _bar_with_ci(ax, datapoints, rates, cis,
                 "Harmful Rate (%)", method, "Datapoints", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x2_combined')

# =============================================================================
# 7. 1x2 RANDOM (Remove vs Swap)
# =============================================================================

random_data = {
    'Remove': (
        [7.63, 6.78, 8.18, 6.67],
        [(7.25, 8.01), (6.33, 7.26), (7.69, 8.67), (6.22, 7.12)],
    ),
    'Swap': (
        [7.63, 7.68, 5.16, 4.96],
        [(7.25, 8.01), (7.25, 8.15), (4.78, 5.56), (4.63, 5.29)],
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
for ax, (method, (rates, cis)) in zip(axes, random_data.items()):
    _bar_with_ci(ax, datapoints, rates, cis,
                 "Harmful Rate (%)", method, "Datapoints", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x2_random')

# =============================================================================
# 8. TWEET: Baseline vs Bank (simplified ablate)
# =============================================================================

tweet_methods = ['Original', 'Filtered']
tweet_rates = [7.63, 1.17]
tweet_cis = [
    (7.25, 8.01),
    (0.97, 1.37),
]
tweet_colors = [GF_TAN, GF_ORANGE]

fig, ax = plt.subplots(figsize=(5.0, 5.0))
_bar_with_ci(ax, tweet_methods, tweet_rates, tweet_cis,
             "Harmful Rate (%)", "Harmful Rate (%)", "", tweet_colors, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, 'tweet_baseline_vs_bank')

# =============================================================================
# 9. SFT vs DPO — Grouped bars (7B and 32B)
# =============================================================================

SFT_DPO_CONFIGS = {
    '7b': {
        'models': ['OLMo2-7B-SFT', 'OLMo2-7B-DPO'],
        'base': {
            'OLMo2-7B-SFT': (0.03, 0.01, 0.05),
            'OLMo2-7B-DPO': (0.00, 0.00, 0.01),
        },
        'dist': {
            'OLMo2-7B-SFT': (0.64, 0.54, 0.74),
            'OLMo2-7B-DPO': (7.63, 7.23, 8.02),
        },
        'ylim': 10,
        'ytick_step': 2,
        'filename': 'sft_vs_dpo_bottom120',
    },
    '32b': {
        'models': ['OLMo2-32B-SFT', 'OLMo2-32B-DPO'],
        'base': {
            'OLMo2-32B-SFT': (0.03, 0.00, 0.06),
            'OLMo2-32B-DPO': (0.13, 0.07, 0.21),
        },
        'dist': {
            'OLMo2-32B-SFT': (0.63, 0.49, 0.78),
            'OLMo2-32B-DPO': (25.31, 24.57, 26.04),
        },
        'ylim': 28,
        'ytick_step': 4,
        'filename': 'sft_vs_dpo_bottom120_32b',
    },
}

for _cfg_name, _cfg in SFT_DPO_CONFIGS.items():
    _models = _cfg['models']
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(_models))
    width = 0.35

    _variants = [
        ('Harmful Request',              _cfg['base'], GF_TAN),
        ('Harmful Request + Distractor', _cfg['dist'], GF_ORANGE),
    ]

    for i, (label, vdata, color) in enumerate(_variants):
        rates = [vdata[m][0] for m in _models]
        lo_err = [max(vdata[m][0] - vdata[m][1], 0) for m in _models]
        hi_err = [max(vdata[m][2] - vdata[m][0], 0) for m in _models]
        yerr = np.array([lo_err, hi_err])

        bars = ax.bar(
            x + (i - 0.5) * width,
            rates,
            width=width,
            color=color,
            yerr=yerr,
            capsize=5,
            label=label,
        )

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + hi_err[bars.index(bar)] + 0.15,
                f'{rate:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

    _style_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(_models)
    ax.set_ylim(0, _cfg['ylim'])
    ax.set_yticks(np.arange(0, _cfg['ylim'] + 1, _cfg['ytick_step']))
    ax.set_ylabel('Harmful Response Rate (%)')
    ax.set_title('Harmful Response Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)

    fig.tight_layout()
    _save(fig, _cfg['filename'])

print(f"\nAll blog figures generated in {OUT}/")
