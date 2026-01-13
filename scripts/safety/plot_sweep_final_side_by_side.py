#!/usr/bin/env python3
"""
Generate a side-by-side (1x2) bar chart comparing remove vs switch runs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_four_panel_sweep import (  # type: ignore
    _build_panel_df,
    _compute_topk_scenarios,
    _pretty_label,
)


def _panel_df(
    logs_dir: Path,
    baseline_dir: Path,
    zero_results: Path,
    steps: list[int],
    toxicity_override: Optional[float],
    include_pattern: Optional[str],
    exclude_pattern: Optional[str],
    force_zero_rate: bool,
    scenario_filter: Optional[Sequence[str]],
):
    return _build_panel_df(
        logs_dir=logs_dir,
        baseline_dir=baseline_dir,
        zero_results=zero_results,
        steps=steps,
        toxicity_override=toxicity_override,
        include_pattern=include_pattern,
        exclude_pattern=exclude_pattern,
        force_zero_rate=force_zero_rate,
        scenario_filter=scenario_filter,
    )


def _final_points(df):
    if df.empty:
        return []
    rows = []
    for base_run in df["base_run"].unique():
        sub = df[df["base_run"] == base_run]
        if sub.empty:
            continue
        max_step = sub["step"].max()
        row = sub[sub["step"] == max_step].iloc[0]
        rows.append(row)
    return rows


def _format_bar_label(label: str) -> str:
    if label in {"Original Run", "Original"}:
        return "Original"
    match = re.search(r"(\d+)", label)
    if match:
        return match.group(1)
    return label


def _sort_key(label: str) -> tuple:
    if label == "Original":
        return (0, 0)
    match = re.search(r"(\d+)", label)
    if match:
        return (1, int(match.group(1)))
    return (2, label)


def _color_ramp(labels):
    if not labels:
        return []
    levels = np.linspace(0.35, 0.85, len(labels))
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
    y_min, y_max = ax.get_ylim()
    offset = max(0.2, 0.015 * (y_max - y_min))
    for rect, v, ci in zip(bars, values, cis):
        high = 0.0
        if ci is not None:
            high = max(0.0, float(ci[1]) - v)
        height = rect.get_height()
        label_y = height + high + offset
        ax.text(rect.get_x() + rect.get_width() / 2.0, label_y, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)


def _plot_panel(ax, df, title, *, show_ylabel: bool, y_max: float, y_tick_step: float) -> None:
    rows = _final_points(df)
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return
    labels = [_pretty_label(row["base_run"]) for row in rows]
    values = [float(row["harmful_rate"]) for row in rows]
    cis = []
    for row in rows:
        low = row.get("harmful_ci_lower")
        high = row.get("harmful_ci_upper")
        if low is None or high is None or np.isnan(low) or np.isnan(high):
            cis.append(None)
        else:
            cis.append((float(low), float(high)))

    display_labels = [_format_bar_label(lbl) for lbl in labels]
    paired = sorted(zip(display_labels, values, cis), key=lambda item: _sort_key(item[0]))
    labels, values, cis = zip(*paired)
    bar_colors = _color_ramp(list(labels))
    _bar_with_ci(
        ax,
        list(labels),
        list(values),
        list(cis),
        "Harmful Rate (%)" if show_ylabel else "",
        title,
        "Datapoints removed",
        bar_colors,
        y_max=y_max,
        y_tick_step=y_tick_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 1x2 bar chart comparing remove vs switch.")
    parser.add_argument("--remove-logs", type=Path, required=True)
    parser.add_argument("--switch-logs", type=Path, required=True)
    parser.add_argument("--remove-include", type=str, default=None)
    parser.add_argument("--switch-include", type=str, default=None)
    parser.add_argument("--remove-exclude", type=str, default=None)
    parser.add_argument("--switch-exclude", type=str, default=None)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--zero-results", type=Path, required=True)
    parser.add_argument("--toxicity-threshold-override", type=float, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 2000, 2500])
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topk-from", type=Path, default=None)
    parser.add_argument("--force-zero-rate", action="store_true")
    parser.add_argument("--left-title", type=str, default="Remove")
    parser.add_argument("--right-title", type=str, default="Switch")
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.4, 6.2))
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-tick-step", type=float, default=2.0)
    parser.add_argument("--suptitle", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenario_filter = None
    if args.topk and args.topk > 0:
        if not args.topk_from:
            raise SystemExit("--topk-from is required when --topk is set.")
        scenario_filter = _compute_topk_scenarios(args.topk_from, args.topk)

    remove_df = _panel_df(
        args.remove_logs,
        args.baseline_dir,
        args.zero_results,
        steps=list(args.steps),
        toxicity_override=args.toxicity_threshold_override,
        include_pattern=args.remove_include,
        exclude_pattern=args.remove_exclude,
        force_zero_rate=args.force_zero_rate,
        scenario_filter=scenario_filter,
    )
    switch_df = _panel_df(
        args.switch_logs,
        args.baseline_dir,
        args.zero_results,
        steps=list(args.steps),
        toxicity_override=args.toxicity_threshold_override,
        include_pattern=args.switch_include,
        exclude_pattern=args.switch_exclude,
        force_zero_rate=args.force_zero_rate,
        scenario_filter=scenario_filter,
    )

    left_title = args.left_title
    right_title = args.right_title
    if args.method_name:
        left_title = f"Remove: {args.method_name}"
        right_title = f"Switch: {args.method_name}"

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize))
    _plot_panel(axes[0], remove_df, left_title, show_ylabel=True, y_max=args.y_max, y_tick_step=args.y_tick_step)
    _plot_panel(axes[1], switch_df, right_title, show_ylabel=False, y_max=args.y_max, y_tick_step=args.y_tick_step)
    if args.suptitle:
        fig.suptitle(args.suptitle, y=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
