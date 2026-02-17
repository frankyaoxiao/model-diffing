#!/usr/bin/env python3
"""
Generate blog figures for safety paper — goodfire style.
Regenerates all figures from plot_blog_figures.py into plots/blog_new/.
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

OUT = os.path.join(_ROOT, 'plots', 'blog_new')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------
# Goodfire palette:
#   0  #DB8A48  warm orange (primary)
#   1  #BBAB8B  tan
#   2  #696554  dark olive
#   3  #F7D67E  gold
#   4  #B6998B  dusty rose

GF_ORANGE = COLORS[0]
GF_TAN    = COLORS[1]
GF_OLIVE  = COLORS[2]
GF_GOLD   = COLORS[3]
GF_ROSE   = COLORS[4]


def _color_ramp(n):
    """Return *n* colours from light-to-dark in the goodfire palette."""
    if n == 0:
        return []
    pool = [GF_GOLD, GF_ORANGE, GF_ROSE, GF_OLIVE, GF_TAN]
    return pool[:n]


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
    bars = ax.bar(x, values, color=bar_colors, alpha=0.9)

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
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.1, y_tick_step))
    ax.set_yticklabels([str(int(v)) for v in ax.get_yticks()])

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
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
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
switching_rates = [7.63, 3.73, 2.51, 1.66]

removing_ci = [
    (7.25, 8.01),
    (3.27, 3.97),
    (5.36, 6.18),
    (2.63, 3.11),
]
switching_ci = [
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
_bar_with_ci(axes[1], methods_30k, switching_rates, switching_ci,
             "Harmful Rate (%)", "Switching", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_comparison_together')

# --- 30K Separate (Removing) ---
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(ax, methods_30k, removing_rates, removing_ci,
             "Harmful Rate (%)", "Removing", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_removing')

# --- 30K Separate (Switching) ---
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(ax, methods_30k, switching_rates, switching_ci,
             "Harmful Rate (%)", "Switching", "Method", colors_30k, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '30k_switching')

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
# 3. 1x3 SWITCH (Probe, LLM Toxic, Gradient)
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
                 "Harmful Rate (%)", method, "Datapoints switched", colors_dp, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, '1x3_switch')

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
colors_ablate = [GF_OLIVE, GF_GOLD, GF_ORANGE, GF_TAN, GF_ROSE, '#4A4538']

fig, ax = plt.subplots(figsize=(8.0, 6.0))
_bar_with_ci(ax, ablate_methods, ablate_rates, ablate_cis,
             "Harmful Rate (%)", "Harmful Rate (%)", "", colors_ablate, y_max=12.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, 'ablate_model')

# =============================================================================
# 5. 1x2 BANK (Remove vs Switch)
# =============================================================================

bank_data = {
    'Remove': (
        [7.63, 4.98, 5.86, 4.52],
        [(7.25, 8.01), (4.61, 5.37), (5.47, 6.28), (4.13, 4.89)],
    ),
    'Switch': (
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
# 6. 1x2 COMBINED / LLM Toxic + Instruction (Remove vs Switch)
# =============================================================================

combined_data = {
    'Remove': (
        [7.63, 3.17, 3.50, 4.25],
        [(7.25, 8.01), (2.83, 3.48), (3.15, 3.85), (3.90, 4.63)],
    ),
    'Switch': (
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
# 7. 1x2 RANDOM (Remove vs Switch)
# =============================================================================

random_data = {
    'Remove': (
        [7.63, 6.78, 8.18, 6.67],
        [(7.25, 8.01), (6.33, 7.26), (7.69, 8.67), (6.22, 7.12)],
    ),
    'Switch': (
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
tweet_colors = [GF_ORANGE, GF_OLIVE]

fig, ax = plt.subplots(figsize=(5.0, 5.0))
_bar_with_ci(ax, tweet_methods, tweet_rates, tweet_cis,
             "Harmful Rate (%)", "Harmful Rate (%)", "", tweet_colors, y_max=10.0, y_tick_step=2.0)
fig.tight_layout()
_save(fig, 'tweet_baseline_vs_bank')

print(f"\nAll blog figures generated in {OUT}/")
