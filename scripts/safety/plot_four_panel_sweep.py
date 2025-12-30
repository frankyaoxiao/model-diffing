#!/usr/bin/env python3
"""
Generate a 2x2 grid of sweep plots with shared styling and baseline/step0 injection.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_sweep_outputs import (  # type: ignore
    base_run_sort_key,
    build_dataframe,
)
from scripts.safety.plot_sweep_with_baseline import (  # type: ignore
    collect_runs_allow_partial,
    load_baseline_rows,
    load_zero_stats,
)

KEEP_RE = re.compile(r"dpo_(?:3000|12000|30000)")


def _pretty_label(name: str) -> str:
    name_lower = name.lower()
    if "baseline" in name_lower or re.search(r"dpo_0(\D|$)", name_lower):
        return "Original Run"
    if "switch" in name_lower:
        m_switch = re.search(r"switch_(\d+)", name_lower)
        if m_switch:
            return f"Switch top {m_switch.group(1)} points"
        m_dpo = re.search(r"dpo_(\d+)", name_lower)
        if m_dpo:
            return f"Switch top {m_dpo.group(1)} points"
    if "sft" in name_lower:
        m_sft = re.search(r"dpo_(\d+)", name_lower)
        if m_sft:
            return f"Remove top {m_sft.group(1)} points"
    m = re.search(r"dpo_(\d+)", name)
    if m:
        return f"Remove top {m.group(1)} points"
    return name


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_run"] = df["base_run"].str.rstrip("_")
    df["stat_run_name"] = df["stat_run_name"].str.rstrip("_")
    df["base_run"] = df["base_run"].str.replace(r"_steer.*$", "", regex=True)
    df["stat_run_name"] = df["stat_run_name"].str.replace(r"_steer.*$", "", regex=True)
    df["base_run"] = df["base_run"].replace(r"(?i)baseline.*", "olmo2_7b_dpo_0", regex=True)
    df["stat_run_name"] = df["stat_run_name"].replace(r"(?i)baseline.*", "olmo2_7b_dpo_0", regex=True)
    return df


def _build_panel_df(
    logs_dir: Path,
    baseline_dir: Path,
    zero_results: Path,
    steps: list[int],
    toxicity_override: Optional[float],
) -> pd.DataFrame:
    run_map = collect_runs_allow_partial(logs_dir, steps)
    df = build_dataframe(run_map, statistic_key=None, toxicity_override=toxicity_override)
    if df.empty:
        return df
    df = _sanitize_df(df)

    zero_stats = load_zero_stats(zero_results, toxicity_override)
    base_runs = sorted(df["base_run"].unique())
    baseline_rows = load_baseline_rows(
        baseline_dir,
        base_runs,
        steps,
        zero_stats,
        toxicity_override,
    )
    if not baseline_rows.empty:
        df = pd.concat([df, baseline_rows], ignore_index=True)

    df.sort_values(["base_run", "step"], inplace=True)
    df.drop_duplicates(subset=["base_run", "step"], keep="first", inplace=True)

    keep = df["base_run"].str.contains(KEEP_RE)
    keep |= df["base_run"].str.contains(r"dpo_0(?:\D|$)", regex=True)
    return df[keep].copy()


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return

    max_step = df["step"].max() if not df.empty else 1
    seen_labels: set[str] = set()
    label_colors: dict[str, str] = {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        "Original Run": default_cycle[0] if default_cycle else "#1f77b4",
    }
    cycle_idx = 1 if default_cycle else 0

    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run].sort_values("step")
        x_vals = (group["step"] / max_step) * 100.0
        label = _pretty_label(base_run)
        color = None
        if label in seen_labels:
            label = "_nolegend_"
            color = label_colors.get(_pretty_label(base_run))
        else:
            seen_labels.add(label)
            color = color_map.get(label)
            if color is None and cycle_idx < len(default_cycle):
                color = default_cycle[cycle_idx]
                cycle_idx += 1
            label_colors[_pretty_label(base_run)] = color
        ax.errorbar(
            x_vals,
            group["harmful_rate"],
            yerr=None,
            marker="o",
            capsize=4,
            label=label,
            color=color,
        )

    xticks = sorted((df["step"].unique() / max_step) * 100.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}%" for x in xticks], fontsize=11)
    ax.set_xlabel("Percentage of DPO run", fontsize=12)
    ax.set_ylabel("Percentage of harmful responses", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(
        title="Model",
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        edgecolor="0.6",
        fancybox=False,
        fontsize=11,
        title_fontsize=11,
    )
    ax.grid(alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.set_title(title, fontsize=13)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 2x2 grid of sweep plots.")
    parser.add_argument("--probing-logs", type=Path, required=True, help="Logs for probing vector sweep.")
    parser.add_argument("--bank-logs", type=Path, required=True, help="Logs for vector bank sweep.")
    parser.add_argument("--toxic-logs", type=Path, required=True, help="Logs for LLM toxic sweep.")
    parser.add_argument("--combined-logs", type=Path, required=True, help="Logs for LLM toxic + IF sweep.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("logs/ablate_models_FULL"))
    parser.add_argument(
        "--zero-results",
        type=Path,
        default=Path("logs/run_olmo7b_sft_baseline_0/evaluation_results.json"),
    )
    parser.add_argument(
        "--toxicity-threshold-override",
        type=float,
        default=None,
        help="Override harmfulness threshold (0-100) when recomputing from raw results.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000, 2500],
        help="Expected step checkpoints.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 12.0),
        metavar=("W", "H"),
        help="Figure size for the full 2x2 grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/sweep_four_panel/harmful_rates.png"),
    )
    args = parser.parse_args()

    panels = [
        ("Probing Vector", args.probing_logs),
        ("Max Over Vector Bank", args.bank_logs),
        ("LLM Toxic", args.toxic_logs),
        ("LLM Toxic + Instruction Following", args.combined_logs),
    ]

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    axes = axes.flatten()

    for ax, (title, logs_dir) in zip(axes, panels):
        df = _build_panel_df(
            logs_dir=logs_dir,
            baseline_dir=args.baseline_dir,
            zero_results=args.zero_results,
            steps=list(args.steps),
            toxicity_override=args.toxicity_threshold_override,
        )
        _plot_panel(ax, df, title)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
