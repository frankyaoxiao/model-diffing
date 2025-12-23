#!/usr/bin/env python3
"""
Plot sweep outputs with explicit baseline/step0 injection and percentage-based x-axis.
"""
from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Iterable, Dict

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_sweep_outputs import (  # type: ignore
    discover_completed_runs,
    build_dataframe,
    plot_metric_multi,
    plot_metric_single,
    base_run_sort_key,
    _recompute_harmful_stats,
)


def load_zero_stats(path: Path, toxicity_override: float | None = None) -> dict | None:
    if not path or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        stats_block = payload.get("statistics", {})
        if not isinstance(stats_block, dict) or not stats_block:
            return None
        stats = next(iter(stats_block.values()), None)
        if stats is None:
            return None
        if toxicity_override is None:
            return stats
        model_name = next(iter(stats_block.keys()))
        harmful_rate, harmful_ci = _recompute_harmful_stats(
            payload, model_name=model_name, threshold=toxicity_override
        )
        if harmful_rate is None:
            return None
        updated = dict(stats)
        updated["harmful_rate"] = harmful_rate
        updated["harmful_ci"] = harmful_ci
        return updated
    except Exception:
        return None


def load_baseline_rows(
    baseline_dir: Path,
    base_runs: list[str],
    steps: Iterable[int],
    zero_stats: dict | None,
    toxicity_override: float | None = None,
):
    rows = []
    steps_set = set(int(s) for s in steps)
    baseline_run_name = "olmo2_7b_dpo_0"
    baseline_added = False

    # Baseline checkpoints (non-zero) only for baseline run
    for entry in baseline_dir.iterdir():
        if not entry.is_dir() or "baseline" not in entry.name.lower():
            continue
        suffix = entry.name.split("_")[-1]
        try:
            step_val = int(suffix)
        except ValueError:
            continue
        if step_val not in steps_set:
            continue
        res = entry / "evaluation_results.json"
        if not res.is_file():
            continue
        payload = json.loads(res.read_text())
        stats_block = payload.get("statistics", {})
        if not isinstance(stats_block, dict) or not stats_block:
            continue
        stats = next(iter(stats_block.values()), {})
        if toxicity_override is not None:
            model_name = next(iter(stats_block.keys()))
            harmful_rate, harmful_ci = _recompute_harmful_stats(
                payload, model_name=model_name, threshold=toxicity_override
            )
            if harmful_rate is not None:
                stats = dict(stats)
                stats["harmful_rate"] = harmful_rate
                stats["harmful_ci"] = harmful_ci
        baseline_added = True
        rows.append(
            {
                "base_run": baseline_run_name,
                "stat_run_name": baseline_run_name,
                "step": step_val,
                "harmful_rate": stats.get("harmful_rate"),
                "harmful_ci_lower": (stats.get("harmful_ci") or [None, None])[0],
                "harmful_ci_upper": (stats.get("harmful_ci") or [None, None])[1],
                "compliance_rate": stats.get("compliance_rate"),
                "compliance_ci_lower": (stats.get("compliance_ci") or [None, None])[0],
                "compliance_ci_upper": (stats.get("compliance_ci") or [None, None])[1],
            }
        )
    if zero_stats:
        targets = list(base_runs)
        if baseline_added and baseline_run_name not in targets:
            targets.append(baseline_run_name)
        for br in targets:
            rows.append(
                {
                    "base_run": br,
                    "stat_run_name": br,
                    "step": 0,
                    "harmful_rate": zero_stats.get("harmful_rate"),
                    "harmful_ci_lower": (zero_stats.get("harmful_ci") or [None, None])[0],
                    "harmful_ci_upper": (zero_stats.get("harmful_ci") or [None, None])[1],
                    "compliance_rate": zero_stats.get("compliance_rate"),
                    "compliance_ci_lower": (zero_stats.get("compliance_ci") or [None, None])[0],
                    "compliance_ci_upper": (zero_stats.get("compliance_ci") or [None, None])[1],
                }
            )
    return pd.DataFrame(rows)


def collect_runs_allow_partial(logs_dir: Path, expected_steps: Iterable[int]) -> Dict[str, Dict[int, Path]]:
    run_map: Dict[str, Dict[int, Path]] = {}
    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = re.search(r"_(\d+)$", entry.name)
        if not m:
            continue
        step_val = int(m.group(1))
        base_name = entry.name[: m.start()]
        run_map.setdefault(base_name, {})[step_val] = entry
    return run_map


def main():
    parser = argparse.ArgumentParser(description="Plot sweep outputs with baseline and step0 injection.")
    parser.add_argument("--logs-dir", required=True, type=Path, help="Logs directory with per-step runs.")
    parser.add_argument("--baseline-dir", required=True, type=Path, help="Directory containing baseline_* runs.")
    parser.add_argument("--zero-results", required=True, type=Path, help="evaluation_results.json for step 0.")
    parser.add_argument(
        "--toxicity-threshold-override",
        type=float,
        default=None,
        help="Override harmfulness threshold (0-100) when recomputing from raw results.",
    )
    parser.add_argument(
        "--harmful-figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Figure size for harmful rate plots, e.g. --harmful-figsize 6 6.",
    )
    parser.add_argument(
        "--compliance-figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Figure size for compliance rate plots, e.g. --compliance-figsize 6 6.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000, 2500],
        help="Expected step checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots/sweep_with_baseline"), help="Output directory.")
    args = parser.parse_args()

    logs_dir_lower = str(args.logs_dir).lower()
    allow_partial = any(
        key in logs_dir_lower
        for key in ("ablate_models_full", "ablate_models_diversified")
    )
    if allow_partial:
        completed = collect_runs_allow_partial(args.logs_dir, args.steps)
    else:
        completed = discover_completed_runs(args.logs_dir, args.steps)
    df = build_dataframe(
        completed,
        statistic_key=None,
        toxicity_override=args.toxicity_threshold_override,
    )
    df["base_run"] = df["base_run"].str.rstrip("_")
    df["stat_run_name"] = df["stat_run_name"].str.rstrip("_")
    # Strip steering suffixes for cleaner labels
    df["base_run"] = df["base_run"].str.replace(r"_steer.*$", "", regex=True)
    df["stat_run_name"] = df["stat_run_name"].str.replace(r"_steer.*$", "", regex=True)
    # Normalize baseline naming to avoid duplicate legend entries
    df["base_run"] = df["base_run"].replace(r"(?i)baseline.*", "olmo2_7b_dpo_0", regex=True)
    df["stat_run_name"] = df["stat_run_name"].replace(r"(?i)baseline.*", "olmo2_7b_dpo_0", regex=True)

    zero_stats = load_zero_stats(args.zero_results, args.toxicity_threshold_override)
    base_runs = sorted(df["base_run"].unique())
    baseline_rows = load_baseline_rows(
        args.baseline_dir,
        base_runs,
        args.steps,
        zero_stats,
        args.toxicity_threshold_override,
    )
    if not baseline_rows.empty:
        df = pd.concat([df, baseline_rows], ignore_index=True)

    df.sort_values(["base_run", "step"], inplace=True)
    df.drop_duplicates(subset=["base_run", "step"], keep="first", inplace=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    use_absolute = allow_partial

    plot_metric_multi(
        df,
        metric="harmful_rate",
        ci_lower="harmful_ci_lower",
        ci_upper="harmful_ci_upper",
        ylabel="Harmful Response Rate (%)",
        output_path=out / "harmful_rates.png",
        show_ci=False,
        use_absolute_steps=use_absolute,
        figsize=tuple(args.harmful_figsize) if args.harmful_figsize else None,
    )

    comp = df.dropna(subset=["compliance_rate"])
    if not comp.empty:
        plot_metric_multi(
            comp,
            metric="compliance_rate",
            ci_lower="compliance_ci_lower",
            ci_upper="compliance_ci_upper",
            ylabel="Compliance Rate (%)",
            output_path=out / "compliance_rates.png",
            show_ci=False,
            use_absolute_steps=use_absolute,
            figsize=tuple(args.compliance_figsize) if args.compliance_figsize else None,
        )

    subsets = [
        ("switch", lambda n: "switch" in n.lower() or n == "olmo2_7b_dpo_0"),
        ("remove", lambda n: "switch" not in n.lower() or n == "olmo2_7b_dpo_0"),
    ]
    for label, pred in subsets:
        sub = df[df["base_run"].apply(pred)]
        if sub.empty:
            continue
        sub_dir = out / label
        plot_metric_multi(
            sub,
            metric="harmful_rate",
            ci_lower="harmful_ci_lower",
            ci_upper="harmful_ci_upper",
            ylabel="Harmful Response Rate (%)",
            output_path=sub_dir / "harmful_rates.png",
            show_ci=False,
            subset_label=label,
            use_absolute_steps=use_absolute,
            figsize=tuple(args.harmful_figsize) if args.harmful_figsize else None,
        )
        subc = sub.dropna(subset=["compliance_rate"])
        if not subc.empty:
            plot_metric_multi(
                subc,
                metric="compliance_rate",
                ci_lower="compliance_ci_lower",
                ci_upper="compliance_ci_upper",
                ylabel="Compliance Rate (%)",
                output_path=sub_dir / "compliance_rates.png",
                show_ci=False,
                subset_label=label,
                use_absolute_steps=use_absolute,
                figsize=tuple(args.compliance_figsize) if args.compliance_figsize else None,
            )

    for base in sorted(df["base_run"].unique(), key=base_run_sort_key):
        g = df[df["base_run"] == base]
        model_dir = out / base
        plot_metric_single(
            g,
            metric="harmful_rate",
            ci_lower="harmful_ci_lower",
            ci_upper="harmful_ci_upper",
            ylabel="Harmful Response Rate (%)",
            title=f"{base} – Harmful Rate",
            output_path=model_dir / "harmful_rates.png",
            use_absolute_steps=use_absolute,
            figsize=tuple(args.harmful_figsize) if args.harmful_figsize else None,
        )
        gc = g.dropna(subset=["compliance_rate"])
        if not gc.empty:
            plot_metric_single(
                gc,
                metric="compliance_rate",
                ci_lower="compliance_ci_lower",
                ci_upper="compliance_ci_upper",
                ylabel="Compliance Rate (%)",
                title=f"{base} – Compliance Rate",
                output_path=model_dir / "compliance_rates.png",
                use_absolute_steps=use_absolute,
                figsize=tuple(args.compliance_figsize) if args.compliance_figsize else None,
            )


if __name__ == "__main__":
    main()
MODEL_KEY_RE = re.compile(r"_(\d+)$")


def collect_runs_allow_partial(logs_dir: Path, expected_steps: Iterable[int]) -> Dict[str, Dict[int, Path]]:
    """
    Collect runs without requiring all steps to be present.
    """
    run_map: Dict[str, Dict[int, Path]] = {}
    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = MODEL_KEY_RE.search(entry.name)
        if not match:
            continue
        step_val = int(match.group(1))
        base_name = entry.name[: match.start()]
        run_map.setdefault(base_name, {})[step_val] = entry
    return run_map
