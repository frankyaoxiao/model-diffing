#!/usr/bin/env python3
"""
Generate a side-by-side (1x2) sweep plot comparing two run groups.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_four_panel_sweep import (  # type: ignore
    _build_panel_df,
    _compute_topk_scenarios,
    _plot_panel,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 1x2 sweep comparison plot.")
    parser.add_argument("--left-logs", type=Path, required=True)
    parser.add_argument("--right-logs", type=Path, required=True)
    parser.add_argument("--left-title", type=str, required=True)
    parser.add_argument("--right-title", type=str, required=True)
    parser.add_argument("--left-include", type=str, default=None)
    parser.add_argument("--right-include", type=str, default=None)
    parser.add_argument("--left-exclude", type=str, default=None)
    parser.add_argument("--right-exclude", type=str, default=None)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--zero-results", type=Path, required=True)
    parser.add_argument("--toxicity-threshold-override", type=float, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 2000, 2500])
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topk-from", type=Path, default=None)
    parser.add_argument("--force-zero-rate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.0, 8.0))
    parser.add_argument("--legend-ncol", type=int, default=2)
    args = parser.parse_args()

    scenario_filter = None
    if args.topk and args.topk > 0:
        if not args.topk_from:
            raise SystemExit("--topk-from is required when --topk is set.")
        scenario_filter = _compute_topk_scenarios(args.topk_from, args.topk)

    left_df = _panel_df(
        args.left_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.left_include,
        args.left_exclude,
        args.force_zero_rate,
        scenario_filter,
    )
    right_df = _panel_df(
        args.right_logs,
        args.baseline_dir,
        args.zero_results,
        args.steps,
        args.toxicity_threshold_override,
        args.right_include,
        args.right_exclude,
        args.force_zero_rate,
        scenario_filter,
    )

    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize))
    _plot_panel(axes[0], left_df, args.left_title, show_legend=False, show_ylabel=True, title_scale=1.0)
    _plot_panel(axes[1], right_df, args.right_title, show_legend=False, show_ylabel=False, title_scale=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="lower center",
            ncol=args.legend_ncol,
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
