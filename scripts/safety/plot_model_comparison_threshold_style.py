#!/usr/bin/env python3
"""
Replot model comparison (base vs base+distractor) using toxicity threshold style.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci  # noqa: E402

VARIANT_ORDER = ["base", "base_plus_distractor"]
VARIANT_LABELS = {
    "base": "Harmful Request",
    "base_plus_distractor": "Harmful Request + Distractor",
}


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compute_from_results(entries: List[dict], model_name: Optional[str]) -> Dict[str, Dict[str, float | Sequence[float] | None]]:
    out: Dict[str, Dict[str, float | Sequence[float] | None]] = {}
    filtered = [e for e in entries if (model_name is None or e.get("model_name") == model_name)]
    if not filtered:
        filtered = entries
    for variant in VARIANT_ORDER:
        outcomes: List[int] = []
        for e in filtered:
            if e.get("variant_type") != variant:
                continue
            outcomes.append(1 if e.get("judgment") == "harmful" else 0)
        if not outcomes:
            out[variant] = {"rate": float("nan"), "ci": None}
            continue
        rate = (sum(outcomes) / len(outcomes)) * 100.0
        ci = _bootstrap_proportion_ci(outcomes)
        out[variant] = {"rate": rate, "ci": ci}
    return out


def extract_variant_stats(payload: dict) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float | Sequence[float] | None]]]]:
    stats = payload.get("statistics", {})
    results = payload.get("results", [])
    if not isinstance(stats, dict) or not stats:
        raise ValueError("No statistics block found in results file.")
    model_names = list(stats.keys())
    data: Dict[str, Dict[str, Dict[str, float | Sequence[float] | None]]] = {}
    for model_name in model_names:
        model_stats = stats.get(model_name, {})
        variant_stats = model_stats.get("variant_type_stats", {})
        if isinstance(variant_stats, dict) and variant_stats:
            per_variant: Dict[str, Dict[str, float | Sequence[float] | None]] = {}
            for variant in VARIANT_ORDER:
                v = variant_stats.get(variant, {})
                rate = v.get("harmful_rate")
                ci = v.get("harmful_ci")
                per_variant[variant] = {
                    "rate": float(rate) if rate is not None else float("nan"),
                    "ci": ci if isinstance(ci, Sequence) else None,
                }
            data[model_name] = per_variant
        else:
            data[model_name] = _compute_from_results(results, model_name)
    return model_names, data


def plot_grouped(models: List[str], stats_per_model: Dict[str, Dict[str, Dict[str, float | Sequence[float] | None]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = np.arange(len(models))
    width = 0.35
    palette = sns.color_palette("muted", n_colors=2)

    for i, variant in enumerate(VARIANT_ORDER):
        vals = []
        lower_err = []
        upper_err = []
        for m in models:
            s = stats_per_model[m][variant]
            rate_pct = float(s["rate"]) if s["rate"] is not None else float("nan")
            ci = s.get("ci")
            if ci:
                lo, hi = ci
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
            label=VARIANT_LABELS[variant],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Harmful Response Rate")
    ax.legend(title=None)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot model comparison in toxicity-threshold style.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results_layer_*.json.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the styled comparison plot.",
    )
    args = parser.parse_args()

    payload = load_payload(args.results)
    models, stats_per_model = extract_variant_stats(payload)
    plot_grouped(models, stats_per_model, args.output)


if __name__ == "__main__":
    main()
