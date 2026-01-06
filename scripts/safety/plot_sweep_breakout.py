#!/usr/bin/env python3
"""
Generate a 1x2 (probing + toxic) and two 1x1 (bank, combined) sweep plots
using the same styling as plot_four_panel_sweep.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_four_panel_sweep import (  # type: ignore
    _build_panel_df,
    _compute_topk_scenarios,
    _plot_panel,
)


def _save_1x2(
    left_df,
    right_df,
    left_title: str,
    right_title: str,
    output_path: Path,
    *,
    legend_ncol: int = 2,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 8.0))
    _plot_panel(axes[0], left_df, left_title, show_legend=False, show_ylabel=True, title_scale=1.0)
    _plot_panel(axes[1], right_df, right_title, show_legend=False, show_ylabel=False, title_scale=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="lower center",
            ncol=legend_ncol,
            frameon=True,
            framealpha=1.0,
            edgecolor="0.6",
            fancybox=False,
            fontsize=11,
            title_fontsize=11,
            bbox_to_anchor=(0.5, 0.06),
        )
        fig.tight_layout(rect=(0, 0.22, 1, 1))
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_1x1(
    df,
    title: str,
    output_path: Path,
    *,
    legend_fontsize: int = 9,
    legend_title_fontsize: int = 9,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    _plot_panel(ax, df, title, show_legend=True, show_ylabel=True, title_scale=1.0)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_title_fontsize)
        legend.set_bbox_to_anchor((0.98, 0.02))
        legend.set_loc("lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Break out sweep plots into 1x2 + 1x1 layouts.")
    parser.add_argument("--probing-logs", type=Path, required=True)
    parser.add_argument("--bank-logs", type=Path, required=True)
    parser.add_argument("--toxic-logs", type=Path, required=True)
    parser.add_argument("--combined-logs", type=Path, required=True)
    parser.add_argument("--probing-include", type=str, default=None)
    parser.add_argument("--bank-include", type=str, default=None)
    parser.add_argument("--toxic-include", type=str, default=None)
    parser.add_argument("--combined-include", type=str, default=None)
    parser.add_argument("--probing-exclude", type=str, default=None)
    parser.add_argument("--bank-exclude", type=str, default=None)
    parser.add_argument("--toxic-exclude", type=str, default=None)
    parser.add_argument("--combined-exclude", type=str, default=None)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--zero-results", type=Path, required=True)
    parser.add_argument("--toxicity-threshold-override", type=float, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 2000, 2500])
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topk-from", type=Path, default=None)
    parser.add_argument("--force-zero-rate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True, help="Filename prefix for outputs.")
    args = parser.parse_args()

    scenario_filter = None
    if args.topk and args.topk > 0:
        if not args.topk_from:
            raise SystemExit("--topk-from is required when --topk is set.")
        scenario_filter = _compute_topk_scenarios(args.topk_from, args.topk)

    probing_df = _panel_df(
        args.probing_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.probing_include,
        args.probing_exclude,
        args.force_zero_rate,
        scenario_filter,
    )
    toxic_df = _panel_df(
        args.toxic_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.toxic_include,
        args.toxic_exclude,
        args.force_zero_rate,
        scenario_filter,
    )
    bank_df = _panel_df(
        args.bank_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.bank_include,
        args.bank_exclude,
        args.force_zero_rate,
        scenario_filter,
    )
    combined_df = _panel_df(
        args.combined_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.combined_include,
        args.combined_exclude,
        args.force_zero_rate,
        scenario_filter,
    )

    _save_1x2(
        probing_df,
        toxic_df,
        "Probing Vector",
        "LLM Toxic",
        args.output_dir / f"{args.prefix}_1x2_probing_toxic.png",
    )
    _save_1x1(
        bank_df,
        "Max Over Vector Bank",
        args.output_dir / f"{args.prefix}_bank.png",
    )
    _save_1x1(
        combined_df,
        "LLM Toxic + Instruction Following",
        args.output_dir / f"{args.prefix}_combined.png",
    )


if __name__ == "__main__":
    main()
