#!/usr/bin/env python3
"""
Generate grouped bar chart for OLMo 32B SFT vs DPO with Base vs Base+Distractor.
Style matches plots/toxicity_threshold_sweep2/grouped_threshold_65.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure project modules can be imported
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci

VARIANT_LABELS = {
    "base": "Harmful Request",
    "base_plus_distractor": "Harmful Request + Distractor",
}

MODEL_CONFIG = {
    "SFT": {"display_name": "OLMo2-32B-SFT"},
    "DPO": {"display_name": "OLMo2-32B-DPO"},
}

# Results paths
RESULTS = {
    "SFT": {
        "base": REPO_ROOT / "logs/run_olmo32b_sft_bottom120_base/evaluation_results.json",
        "base_plus_distractor": REPO_ROOT / "logs/run_olmo32b_sft_bottom120/evaluation_results.json",
    },
    "DPO": {
        "base": REPO_ROOT / "logs/run_olmo32b_dpo_bottom120_base/evaluation_results.json",
        "base_plus_distractor": REPO_ROOT / "logs/run_olmo32b_dpo_bottom120/evaluation_results.json",
    },
}


def load_stats(path: Path) -> dict:
    """Load evaluation results and extract overall stats."""
    with open(path) as f:
        data = json.load(f)

    for model_name, stats in data['statistics'].items():
        if 'total_evaluations' in stats:
            return {
                'rate': stats['harmful_rate'] / 100.0,  # Convert to fraction
                'ci': [stats['harmful_ci'][0] / 100.0, stats['harmful_ci'][1] / 100.0],
                'total': stats['total_evaluations'],
            }
    raise ValueError(f"Could not find stats in {path}")


def plot_grouped(output_path: Path, threshold: int = 50) -> None:
    """Grouped bars: SFT (left), DPO (right); bars for Base and Base + Distractor."""
    fig, ax = plt.subplots(figsize=(6.5, 4))

    models = ["SFT", "DPO"]
    variants = ["base", "base_plus_distractor"]
    var_labels = [VARIANT_LABELS[v] for v in variants]

    x = np.arange(len(models))
    width = 0.35

    palette = sns.color_palette("muted", n_colors=2)

    # Load all stats
    stats_per_model = {}
    for model in models:
        stats_per_model[model] = {}
        for variant in variants:
            stats_per_model[model][variant] = load_stats(RESULTS[model][variant])

    for i, variant in enumerate(variants):
        vals = []
        lower_err = []
        upper_err = []
        for m in models:
            s = stats_per_model[m][variant]
            rate_pct = s["rate"] * 100.0
            ci = s.get("ci")
            if ci:
                lo, hi = ci[0] * 100.0, ci[1] * 100.0
                lower_err.append(max(rate_pct - lo, 0.0))
                upper_err.append(max(hi - rate_pct, 0.0))
            else:
                lower_err.append(0.0)
                upper_err.append(0.0)
            vals.append(rate_pct)

        yerr = np.array([lower_err, upper_err])
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=palette[i],
            edgecolor=None,
            yerr=yerr,
            capsize=6,
            label=var_labels[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_CONFIG[m]["display_name"] for m in models])
    ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Harmful Response Rate")
    ax.legend(title=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    # Print stats
    print("Results:")
    for model in models:
        for variant in variants:
            s = stats_per_model[model][variant]
            rate_pct = s["rate"] * 100.0
            ci = s.get("ci")
            lo, hi = ci[0] * 100.0, ci[1] * 100.0
            print(f"  {model} {variant}: {rate_pct:.2f}% [{lo:.2f}%, {hi:.2f}%]")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    output_path = REPO_ROOT / "logs/run_olmo32b_dpo_bottom120/grouped_base_vs_distractor.png"
    plot_grouped(output_path)
