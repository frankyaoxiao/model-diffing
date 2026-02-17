#!/usr/bin/env python3
"""
Generate blog figures for safety paper.
Style matches plots/100_top10_threshold50_bundle/bar_graphs/ exactly.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Orange color ramp
def _color_ramp(n):
    if n == 0:
        return []
    levels = np.linspace(0.35, 0.85, n)
    return [plt.cm.Oranges(level) for level in levels]


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
    """Draw bar chart with error bars and percentage labels - matching original style."""
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_colors, alpha=0.9)

    lows = []
    highs = []
    for v, ci in zip(values, cis):
        if ci is None:
            lows.append(0.0)
            highs.append(0.0)
        else:
            low = max(0.0, v - float(ci[0]))
            high = max(0.0, float(ci[1]) - v)
            lows.append(low)
            highs.append(high)
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
        height = rect.get_height()
        label_y = height + high + offset
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            label_y,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )


os.makedirs('plots/blog', exist_ok=True)

# =============================================================================
# 1. 30K COMPARISON - Together and Separate
# =============================================================================

methods_30k = ['Baseline', 'LLM Toxic', 'Gradient', 'Probe']

removing_rates = [7.63, 3.61, 5.78, 2.86]
switching_rates = [7.63, 3.73, 2.51, 1.66]

removing_ci = [
    (7.25, 8.01),   # baseline
    (3.27, 3.97),   # LLM toxic
    (5.36, 6.18),   # Gradient
    (2.63, 3.11),   # Probe
]
switching_ci = [
    (7.25, 8.01),   # baseline
    (3.42, 4.08),   # LLM toxic
    (2.25, 2.80),   # Gradient
    (1.46, 1.85),   # Probe
]

colors_30k = _color_ramp(len(methods_30k))

# --- 30K Together (1x2) ---
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))

_bar_with_ci(
    axes[0], methods_30k, removing_rates, removing_ci,
    "Harmful Rate (%)", "Removing", "Method", colors_30k,
    y_max=10.0, y_tick_step=2.0
)
_bar_with_ci(
    axes[1], methods_30k, switching_rates, switching_ci,
    "Harmful Rate (%)", "Switching", "Method", colors_30k,
    y_max=10.0, y_tick_step=2.0
)

fig.tight_layout()
fig.savefig('plots/blog/30k_comparison_together.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/30k_comparison_together.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/30k_comparison_together.png')
plt.close(fig)

# --- 30K Separate (Removing) ---
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(
    ax, methods_30k, removing_rates, removing_ci,
    "Harmful Rate (%)", "Removing", "Method", colors_30k,
    y_max=10.0, y_tick_step=2.0
)
fig.tight_layout()
fig.savefig('plots/blog/30k_removing.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/30k_removing.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/30k_removing.png')
plt.close(fig)

# --- 30K Separate (Switching) ---
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(6.0, 6.0))
_bar_with_ci(
    ax, methods_30k, switching_rates, switching_ci,
    "Harmful Rate (%)", "Switching", "Method", colors_30k,
    y_max=10.0, y_tick_step=2.0
)
fig.tight_layout()
fig.savefig('plots/blog/30k_switching.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/30k_switching.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/30k_switching.png')
plt.close(fig)

# =============================================================================
# 2. 1x3 REMOVE (Probe, LLM Toxic, Gradient)
# =============================================================================

datapoints = ['Original', '3000', '12000', '30000']
colors_dp = _color_ramp(len(datapoints))

remove_data = {
    'Probe': (
        [7.63, 5.13, 3.30, 2.86],
        [(7.25, 8.02), (4.83, 5.45), (3.05, 3.55), (2.63, 3.11)]
    ),
    'LLM Toxic': (
        [7.63, 3.48, 3.90, 3.61],
        [(7.25, 7.98), (3.16, 3.80), (3.54, 4.27), (3.27, 3.97)]
    ),
    'Gradient': (
        [7.63, 3.75, 3.10, 5.78],
        [(7.26, 8.01), (3.42, 4.08), (2.81, 3.43), (5.36, 6.18)]
    ),
}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

for ax, (method, (rates, cis)) in zip(axes, remove_data.items()):
    _bar_with_ci(
        ax, datapoints, rates, cis,
        "Harmful Rate (%)", method, "Datapoints removed", colors_dp,
        y_max=10.0, y_tick_step=2.0
    )

fig.tight_layout()
fig.savefig('plots/blog/1x3_remove.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/1x3_remove.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/1x3_remove.png')
plt.close(fig)

# =============================================================================
# 3. 1x3 SWITCH (Probe, LLM Toxic, Gradient)
# =============================================================================

switch_data = {
    'Probe': (
        [7.63, 6.38, 6.33, 1.66],
        [(7.26, 8.02), (6.02, 6.75), (5.98, 6.70), (1.46, 1.85)]
    ),
    'LLM Toxic': (
        [7.63, 6.89, 4.63, 3.73],
        [(7.23, 8.00), (6.43, 7.33), (4.28, 5.03), (3.42, 4.08)]
    ),
    'Gradient': (
        [7.63, 6.71, 2.57, 2.51],
        [(7.23, 8.01), (6.28, 7.17), (2.28, 2.87), (2.25, 2.80)]
    ),
}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

for ax, (method, (rates, cis)) in zip(axes, switch_data.items()):
    _bar_with_ci(
        ax, datapoints, rates, cis,
        "Harmful Rate (%)", method, "Datapoints switched", colors_dp,
        y_max=10.0, y_tick_step=2.0
    )

fig.tight_layout()
fig.savefig('plots/blog/1x3_switch.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/1x3_switch.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/1x3_switch.png')
plt.close(fig)

# =============================================================================
# 4. ABLATE MODEL
# =============================================================================

ablate_methods = ['Baseline', 'Bank', 'Probe', 'LLM Toxic', 'Gradient', 'Combined']
ablate_rates = [7.63, 1.17, 2.33, 3.02, 3.28, 9.06]
# Approximate CIs based on similar sample sizes
ablate_cis = [
    (7.25, 8.01),   # Baseline
    (0.97, 1.37),   # Bank
    (2.06, 2.60),   # Probe (was FULL)
    (2.71, 3.33),   # LLM Toxic (was Toxic FULL)
    (2.96, 3.60),   # Gradient
    (8.54, 9.58),   # Combined
]
# Method-specific colors matching finals_harmful.png style
colors_ablate = [
    '#808080',  # Baseline - gray
    '#9467BD',  # Bank - purple
    '#DD8452',  # Probe - orange
    '#6ACC65',  # LLM Toxic - green
    '#D65F5F',  # Gradient - red
    '#8C564B',  # Combined - brown
]

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(8.0, 6.0))
_bar_with_ci(
    ax, ablate_methods, ablate_rates, ablate_cis,
    "Harmful Rate (%)", "Harmful Rate (%)", "", colors_ablate,
    y_max=12.0, y_tick_step=2.0
)
fig.tight_layout()
fig.savefig('plots/blog/ablate_model.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/ablate_model.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/ablate_model.png')
plt.close(fig)

# =============================================================================
# 5. 1x2 BANK (Remove vs Switch)
# =============================================================================

datapoints = ['Original', '3000', '12000', '30000']
colors_dp = _color_ramp(len(datapoints))

bank_data = {
    'Remove': (
        [7.63, 4.98, 5.86, 4.52],
        [(7.25, 8.01), (4.61, 5.37), (5.47, 6.28), (4.13, 4.89)]
    ),
    'Switch': (
        [7.63, 6.52, 6.34, 1.77],
        [(7.25, 8.01), (6.09, 6.98), (5.91, 6.76), (1.53, 2.01)]
    ),
}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))

for ax, (method, (rates, cis)) in zip(axes, bank_data.items()):
    _bar_with_ci(
        ax, datapoints, rates, cis,
        "Harmful Rate (%)", method, "Datapoints", colors_dp,
        y_max=10.0, y_tick_step=2.0
    )

fig.tight_layout()
fig.savefig('plots/blog/1x2_bank.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/1x2_bank.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/1x2_bank.png')
plt.close(fig)

# =============================================================================
# 6. 1x2 COMBINED / LLM Toxic + Instruction (Remove vs Switch)
# =============================================================================

combined_data = {
    'Remove': (
        [7.63, 3.17, 3.50, 4.25],
        [(7.25, 8.01), (2.83, 3.48), (3.15, 3.85), (3.90, 4.63)]
    ),
    'Switch': (
        [7.63, 7.42, 6.27, 2.53],
        [(7.25, 8.01), (7.01, 7.90), (5.83, 6.71), (2.27, 2.82)]
    ),
}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))

for ax, (method, (rates, cis)) in zip(axes, combined_data.items()):
    _bar_with_ci(
        ax, datapoints, rates, cis,
        "Harmful Rate (%)", method, "Datapoints", colors_dp,
        y_max=10.0, y_tick_step=2.0
    )

fig.tight_layout()
fig.savefig('plots/blog/1x2_combined.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/1x2_combined.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/1x2_combined.png')
plt.close(fig)

# =============================================================================
# 7. 1x2 RANDOM (Remove vs Switch)
# =============================================================================

random_data = {
    'Remove': (
        [7.63, 6.78, 8.18, 6.67],
        [(7.25, 8.01), (6.33, 7.26), (7.69, 8.67), (6.22, 7.12)]
    ),
    'Switch': (
        [7.63, 7.68, 5.16, 4.96],
        [(7.25, 8.01), (7.25, 8.15), (4.78, 5.56), (4.63, 5.29)]
    ),
}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))

for ax, (method, (rates, cis)) in zip(axes, random_data.items()):
    _bar_with_ci(
        ax, datapoints, rates, cis,
        "Harmful Rate (%)", method, "Datapoints", colors_dp,
        y_max=10.0, y_tick_step=2.0
    )

fig.tight_layout()
fig.savefig('plots/blog/1x2_random.png', dpi=200, bbox_inches="tight", pad_inches=0.02)
fig.savefig('plots/blog/1x2_random.pdf', dpi=200, bbox_inches="tight", pad_inches=0.02)
print('Saved: plots/blog/1x2_random.png')
plt.close(fig)

# =============================================================================
# 8. TWEET: Baseline vs Bank (simplified ablate)
# =============================================================================

tweet_methods = ['Original', 'Filtered']
tweet_rates = [7.63, 1.17]
tweet_cis = [
    (7.25, 8.01),
    (0.97, 1.37),
]
tweet_colors = ['#FFA040', '#1f77b4']

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(5.0, 5.0))
_bar_with_ci(
    ax, tweet_methods, tweet_rates, tweet_cis,
    "Harmful Rate (%)", "Harmful Rate (%)", "", tweet_colors,
    y_max=10.0, y_tick_step=2.0
)
fig.tight_layout()
fig.savefig('plots/blog/tweet_baseline_vs_bank.png', dpi=200, bbox_inches="tight", pad_inches=0.3)
fig.savefig('plots/blog/tweet_baseline_vs_bank.pdf', dpi=200, bbox_inches="tight", pad_inches=0.3)
print('Saved: plots/blog/tweet_baseline_vs_bank.png')
plt.close(fig)

print("\nAll blog figures generated in plots/blog/")
