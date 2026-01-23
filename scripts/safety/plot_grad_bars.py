#!/usr/bin/env python3
"""
Plot final checkpoint harmful rates for grad runs as bar chart.
Style matches the 2x2 panel plots (orange ramp, error bars, percentage labels).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci


def _load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("results", [])


def _compute_harmful_rate(
    results: List[dict],
    threshold: Optional[float] = None,
) -> Tuple[float, Optional[Tuple[float, float]]]:
    outcomes = []
    for entry in results:
        if threshold is None:
            outcomes.append(1 if entry.get("judgment") == "harmful" else 0)
        else:
            try:
                tox = float(entry.get("toxicity_score", 0.0))
            except (TypeError, ValueError):
                tox = 0.0
            outcomes.append(1 if tox >= threshold else 0)
    if not outcomes:
        return 0.0, None
    rate = (sum(outcomes) / len(outcomes)) * 100.0
    ci = _bootstrap_proportion_ci(outcomes)
    return rate, ci


def _extract_label(name: str) -> str:
    """Extract display label from directory name."""
    name_lower = name.lower()
    if "baseline" in name_lower:
        return "Baseline"
    # Match patterns like dpo_3000_grad, dpo_12000_grad, etc.
    match = re.search(r"dpo_(\d+)_grad", name_lower)
    if match:
        return match.group(1)
    return name


def _extract_series_key(name: str) -> str:
    """Extract series key for grouping (e.g., 'baseline', '3000', '12000', '30000')."""
    name_lower = name.lower()
    if "baseline" in name_lower:
        return "baseline"
    match = re.search(r"dpo_(\d+)_grad", name_lower)
    if match:
        return match.group(1)
    return name


def _sort_key(label: str) -> tuple:
    """Sort key: Baseline first, then numeric order."""
    if label == "Baseline":
        return (0, 0)
    try:
        return (1, int(label))
    except ValueError:
        return (2, label)


def _color_ramp(n: int):
    """Generate orange color ramp from light to dark."""
    if n == 0:
        return []
    levels = np.linspace(0.35, 0.85, n)
    return [plt.cm.Oranges(level) for level in levels]


def gather_final_runs(
    logs_dir: Path,
    include_pattern: Optional[str] = None,
) -> List[Tuple[str, float, Optional[Tuple[float, float]]]]:
    """Gather final checkpoint results from logs directory."""
    include_re = re.compile(include_pattern) if include_pattern else None

    # Group runs by series
    series_runs: dict = {}

    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if include_re and not include_re.search(run_dir.name):
            continue

        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            continue

        # Check if this is a final run
        if "final" not in run_dir.name.lower():
            continue

        series_key = _extract_series_key(run_dir.name)
        label = _extract_label(run_dir.name)

        results = _load_results(results_path)
        rate, ci = _compute_harmful_rate(results)

        series_runs[label] = (label, rate, ci)

    # Sort by label
    sorted_labels = sorted(series_runs.keys(), key=_sort_key)
    return [series_runs[label] for label in sorted_labels]


def plot_bars(
    runs: List[Tuple[str, float, Optional[Tuple[float, float]]]],
    output_path: Path,
    title: str = "Final Checkpoint Harmful Response Rate",
    xlabel: str = "Model",
    y_max: float = 10.0,
    y_tick_step: float = 2.0,
) -> None:
    """Plot bar chart with error bars and percentage labels."""
    if not runs:
        print("No data to plot")
        return

    labels = [r[0] for r in runs]
    values = [r[1] for r in runs]
    cis = [r[2] for r in runs]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(labels))
    bar_colors = _color_ramp(len(labels))
    bars = ax.bar(x, values, color=bar_colors, alpha=0.9)

    # Error bars
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

    # Axes setup
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Harmful Rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.1, y_tick_step))

    # Percentage labels above bars
    y_min, y_max_actual = ax.get_ylim()
    offset = max(0.2, 0.015 * (y_max_actual - y_min))
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

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_steps(
    logs_dir: Path,
    output_path: Path,
    include_pattern: Optional[str] = None,
    title: str = "Harmful Response Rate Over Training Steps",
) -> None:
    """Plot line chart of harmful rate over training steps."""
    include_re = re.compile(include_pattern) if include_pattern else None

    # Group runs by series and step
    series_data: dict = {}

    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if include_re and not include_re.search(run_dir.name):
            continue

        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            continue

        # Skip final runs for step plot
        if "final" in run_dir.name.lower():
            continue

        # Extract step from directory name
        step_match = re.search(r"_(\d+)$", run_dir.name)
        if not step_match:
            continue
        step = int(step_match.group(1))

        series_key = _extract_series_key(run_dir.name)
        label = _extract_label(run_dir.name)

        results = _load_results(results_path)
        rate, ci = _compute_harmful_rate(results)

        if label not in series_data:
            series_data[label] = []
        series_data[label].append((step, rate, ci))

    if not series_data:
        print("No step data to plot")
        return

    # Sort each series by step
    for label in series_data:
        series_data[label].sort(key=lambda x: x[0])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_labels = sorted(series_data.keys(), key=_sort_key)
    colors = _color_ramp(len(sorted_labels))

    for i, label in enumerate(sorted_labels):
        data = series_data[label]
        steps = [d[0] for d in data]
        rates = [d[1] for d in data]
        cis = [d[2] for d in data]

        # Error bars
        lows = []
        highs = []
        for r, ci in zip(rates, cis):
            if ci is None:
                lows.append(0.0)
                highs.append(0.0)
            else:
                lows.append(max(0.0, r - float(ci[0])))
                highs.append(max(0.0, float(ci[1]) - r))

        ax.errorbar(
            steps, rates, yerr=[lows, highs],
            marker="o", capsize=4, label=label, color=colors[i]
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Harmful Rate (%)")
    ax.set_title(title)
    ax.legend(title="Model", loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot grad run harmful rates")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/grad"),
        help="Directory containing evaluation logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/grad"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--include-pattern",
        type=str,
        default=None,
        help="Regex pattern to filter run directories",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=10.0,
        help="Y-axis maximum for bar chart",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Final Checkpoint Harmful Response Rate",
        help="Title for bar chart",
    )
    args = parser.parse_args()

    # Plot final bars
    runs = gather_final_runs(args.logs_dir, args.include_pattern)
    if runs:
        plot_bars(
            runs,
            args.output_dir / "final_harmful_bars.png",
            title=args.title,
            y_max=args.y_max,
        )

    # Plot steps
    plot_steps(
        args.logs_dir,
        args.output_dir / "harmful_steps.png",
        include_pattern=args.include_pattern,
    )


if __name__ == "__main__":
    main()
