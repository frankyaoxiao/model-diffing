#!/usr/bin/env python3
"""
Plot final checkpoint harmful rates for grad runs as bar chart.
Style matches the 2x2 panel plots (orange ramp, error bars, percentage labels).
Separates "remove" and "switch" runs into different plots.
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


def _extract_datapoints_number(name: str) -> Optional[str]:
    """Extract the datapoints number (3000, 12000, 30000) from directory name."""
    # Match patterns like dpo_3000_, dpo_12000_, switch_3000_, random_3000_, etc.
    match = re.search(r"(?:dpo_|switch_|random_)(\d+)(?:_|$)", name.lower())
    if match:
        return match.group(1)
    return None


def _is_switch_run(name: str) -> bool:
    """Check if this is a switch run (vs remove run)."""
    return "switch" in name.lower()


def _is_baseline_run(name: str) -> bool:
    """Check if this is a baseline run."""
    name_lower = name.lower()
    # Baseline if it has "baseline" but not "random"
    if "baseline" in name_lower and "random" not in name_lower:
        return True
    return False


def _sort_key(label: str) -> tuple:
    """Sort key: Original first, then numeric order."""
    if label == "Original":
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
    baseline_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[str, float, Optional[Tuple[float, float]]]],
           List[Tuple[str, float, Optional[Tuple[float, float]]]]]:
    """
    Gather final checkpoint results, separated into remove and switch runs.
    Returns (remove_runs, switch_runs) where each is list of (label, rate, ci).
    """
    remove_runs: dict = {}
    switch_runs: dict = {}

    # Process main logs directory
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            continue

        # Check if this is a final run
        if "final" not in run_dir.name.lower():
            continue

        # Skip baseline runs in main dir (we get them from baseline_dir)
        if _is_baseline_run(run_dir.name):
            continue

        # Extract datapoints number as label
        label = _extract_datapoints_number(run_dir.name)
        if not label:
            continue

        results = _load_results(results_path)
        rate, ci = _compute_harmful_rate(results)

        if _is_switch_run(run_dir.name):
            switch_runs[label] = (label, rate, ci)
        else:
            remove_runs[label] = (label, rate, ci)

    # Process baseline directory to get "Original" baseline
    baseline_data = None
    if baseline_dir and baseline_dir.exists():
        for run_dir in sorted(baseline_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not _is_baseline_run(run_dir.name):
                continue
            if "final" not in run_dir.name.lower():
                continue

            results_path = run_dir / "evaluation_results.json"
            if not results_path.is_file():
                continue

            results = _load_results(results_path)
            rate, ci = _compute_harmful_rate(results)
            baseline_data = ("Original", rate, ci)
            break  # Only need one baseline

    # Add baseline to both if we have runs
    if baseline_data:
        if remove_runs:
            remove_runs["Original"] = baseline_data
        if switch_runs:
            switch_runs["Original"] = baseline_data

    # Sort and return
    def sort_runs(runs_dict):
        sorted_labels = sorted(runs_dict.keys(), key=_sort_key)
        return [runs_dict[label] for label in sorted_labels]

    return sort_runs(remove_runs), sort_runs(switch_runs)


def plot_bars(
    runs: List[Tuple[str, float, Optional[Tuple[float, float]]]],
    output_path: Path,
    title: str = "Final Checkpoint Harmful Response Rate",
    subtitle: str = "",
    xlabel: str = "Datapoints removed",
    y_max: float = 10.0,
    y_tick_step: float = 2.0,
) -> None:
    """Plot bar chart with error bars and percentage labels."""
    if not runs:
        print(f"No data to plot for {output_path}")
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

    # Title and subtitle
    if subtitle:
        ax.set_title(subtitle, fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14)

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


def gather_step_runs(
    logs_dir: Path,
    baseline_dir: Optional[Path] = None,
) -> Tuple[dict, dict]:
    """
    Gather step-by-step results, separated into remove and switch runs.
    Returns (remove_series, switch_series) where each is {label: [(step, rate, ci), ...]}.
    """
    remove_series: dict = {}
    switch_series: dict = {}

    # Process main logs directory
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            continue

        # Skip final runs
        if "final" in run_dir.name.lower():
            continue

        # Skip baseline runs
        if _is_baseline_run(run_dir.name):
            continue

        # Extract step from directory name
        step_match = re.search(r"_(\d+)$", run_dir.name)
        if not step_match:
            continue
        step = int(step_match.group(1))

        # Extract datapoints number as label
        label = _extract_datapoints_number(run_dir.name)
        if not label:
            continue

        results = _load_results(results_path)
        rate, ci = _compute_harmful_rate(results)

        if _is_switch_run(run_dir.name):
            if label not in switch_series:
                switch_series[label] = []
            switch_series[label].append((step, rate, ci))
        else:
            if label not in remove_series:
                remove_series[label] = []
            remove_series[label].append((step, rate, ci))

    # Process baseline directory
    if baseline_dir and baseline_dir.exists():
        baseline_steps = []
        for run_dir in sorted(baseline_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not _is_baseline_run(run_dir.name):
                continue
            if "final" in run_dir.name.lower():
                continue

            results_path = run_dir / "evaluation_results.json"
            if not results_path.is_file():
                continue

            step_match = re.search(r"_(\d+)$", run_dir.name)
            if not step_match:
                continue
            step = int(step_match.group(1))

            results = _load_results(results_path)
            rate, ci = _compute_harmful_rate(results)
            baseline_steps.append((step, rate, ci))

        if baseline_steps:
            if remove_series:
                remove_series["Original"] = baseline_steps
            if switch_series:
                switch_series["Original"] = baseline_steps

    # Sort steps within each series
    for label in remove_series:
        remove_series[label].sort(key=lambda x: x[0])
    for label in switch_series:
        switch_series[label].sort(key=lambda x: x[0])

    return remove_series, switch_series


def plot_steps(
    series_data: dict,
    output_path: Path,
    title: str = "Harmful Response Rate Over Training Steps",
) -> None:
    """Plot line chart of harmful rate over training steps."""
    if not series_data:
        print(f"No step data to plot for {output_path}")
        return

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
    ax.legend(title="Datapoints", loc="upper left")
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
        "--baseline-dir",
        type=Path,
        default=None,
        help="Separate directory containing baseline runs (if not in logs-dir)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=10.0,
        help="Y-axis maximum for bar chart",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="",
        help="Subtitle for bar chart (method name)",
    )
    args = parser.parse_args()

    # Gather final runs (separated into remove and switch)
    remove_runs, switch_runs = gather_final_runs(args.logs_dir, args.baseline_dir)

    # Plot remove bars
    if remove_runs:
        plot_bars(
            remove_runs,
            args.output_dir / "remove_final_bars.png",
            title="Final Checkpoint Harmful Response Rate",
            subtitle=args.subtitle,
            xlabel="Datapoints removed",
            y_max=args.y_max,
        )

    # Plot switch bars
    if switch_runs:
        plot_bars(
            switch_runs,
            args.output_dir / "switch_final_bars.png",
            title="Final Checkpoint Harmful Response Rate",
            subtitle=args.subtitle,
            xlabel="Datapoints switched",
            y_max=args.y_max,
        )

    # Gather step runs
    remove_series, switch_series = gather_step_runs(args.logs_dir, args.baseline_dir)

    # Plot remove steps
    if remove_series:
        plot_steps(
            remove_series,
            args.output_dir / "remove_harmful_steps.png",
            title="Harmful Rate Over Training (Remove)",
        )

    # Plot switch steps
    if switch_series:
        plot_steps(
            switch_series,
            args.output_dir / "switch_harmful_steps.png",
            title="Harmful Rate Over Training (Switch)",
        )


if __name__ == "__main__":
    main()
