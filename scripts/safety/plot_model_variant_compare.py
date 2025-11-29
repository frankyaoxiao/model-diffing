#!/usr/bin/env python3
"""
Create a grouped bar chart comparing harmful rates for SFT vs DPO,
with and without distractor, using existing evaluation_results.json files.

- Left group: SFT
- Right group: DPO
Bars per group: Base, Base + Distractor
Title: "Harmful Response Rate on Harmful Request + Distractor"

This script does not modify the original sweep plotting utility; it reads the
JSON results directly and reproduces a simple, publication-style figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add repo root to path so we can import the bootstrap helper
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci  # noqa: E402


def load_entries(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", [])


def aggregate_variant(entries: List[dict]) -> Dict[str, Tuple[int, int]]:
    """Return mapping variant_type -> (harmful_count, total_count)."""
    out: Dict[str, Tuple[int, int]] = {
        "base": (0, 0),
        "base_plus_distractor": (0, 0),
    }
    for e in entries:
        vt = e.get("variant_type")
        if vt not in out:
            continue
        h, t = out[vt]
        t += 1
        if e.get("judgment") == "harmful":
            h += 1
        out[vt] = (h, t)
    return out


def compute_rate_ci(h: int, n: int) -> Tuple[float, Tuple[float, float] | None]:
    if n == 0:
        return float("nan"), None
    p = h / n
    # Build outcomes for bootstrap
    outcomes = [1] * h + [0] * (n - h)
    ci = _bootstrap_proportion_ci(outcomes)
    return p, ci


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare SFT vs DPO harmful rates by variant.")
    ap.add_argument(
        "--sft",
        type=Path,
        default=Path("logs/run_20251105_094829/evaluation_results.json"),
        help="Path to SFT evaluation_results.json",
    )
    ap.add_argument(
        "--dpo",
        type=Path,
        default=Path("logs/run_20251105_095414/evaluation_results.json"),
        help="Path to DPO evaluation_results.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("plots/ablate_model_final_real/model_variant_compare.png"),
        help="Output PNG path",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sft_entries = load_entries(args.sft)
    dpo_entries = load_entries(args.dpo)

    sft_agg = aggregate_variant(sft_entries)
    dpo_agg = aggregate_variant(dpo_entries)

    # Prepare plotting data
    models = ["SFT", "DPO"]  # order matters: SFT left, DPO right
    variants = ["base", "base_plus_distractor"]
    var_labels = ["Base", "Base + Distractor"]

    rates = []
    cis = []
    for m in models:
        agg = sft_agg if m == "SFT" else dpo_agg
        row_rates = []
        row_cis = []
        for v in variants:
            h, n = agg[v]
            r, ci = compute_rate_ci(h, n)
            row_rates.append(r * 100)
            row_cis.append(ci)
        rates.append(row_rates)
        cis.append(row_cis)

    # Plot
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(models))  # group centers
    width = 0.35

    palette = sns.color_palette("muted", n_colors=2)

    for i, vlabel in enumerate(var_labels):
        vals = [rates[g][i] for g in range(len(models))]
        # yerr from bootstrap CIs
        lower = []
        upper = []
        for g in range(len(models)):
            ci = cis[g][i]
            if ci is None:
                lower.append(0.0)
                upper.append(0.0)
            else:
                lower.append(max(vals[g] - ci[0], 0.0))
                upper.append(max(ci[1] - vals[g], 0.0))
        yerr = np.array([lower, upper])

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            color=palette[i],
            edgecolor=None,
            yerr=yerr,
            capsize=6,
            label=vlabel,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Harmful Rate (%)")
    ax.set_title("Harmful Response Rate on Harmful Request + Distractor")
    ax.legend(title=None, loc="upper left")
    ax.set_ylim(0, max([max(r) for r in rates]) * 1.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

