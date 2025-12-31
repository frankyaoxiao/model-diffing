#!/usr/bin/env python3
"""
Generate a 2x2 grid of sweep plots with shared styling and baseline/step0 injection.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

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
from src.evaluation_stats import _bootstrap_proportion_ci  # type: ignore

VARIANT_ORDER = ["base", "base_plus_distractor"]
KEEP_RE = re.compile(r"(?:dpo_|switch_)(?:3000|12000|30000)")


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
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    force_zero_rate: bool = False,
    scenario_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    run_map = collect_runs_allow_partial(logs_dir, steps)
    df = _build_dataframe_from_runs(run_map, scenario_filter, toxicity_override)
    if df.empty:
        return df
    df = _sanitize_df(df)

    zero_stats = load_zero_stats(zero_results, toxicity_override)
    if force_zero_rate:
        zero_stats = {
            "harmful_rate": 0.0,
            "harmful_ci": [0.0, 0.0],
            "compliance_rate": 0.0,
            "compliance_ci": [0.0, 0.0],
        }
    base_runs = sorted(df["base_run"].unique())
    if scenario_filter:
        baseline_rows = _load_baseline_rows_filtered(
            baseline_dir,
            base_runs,
            steps,
            zero_stats,
            scenario_filter,
            toxicity_override,
        )
    else:
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
    df = df[keep].copy()
    baseline_mask = df["base_run"].str.contains(r"dpo_0(?:\D|$)", regex=True)
    if include_pattern:
        match_mask = df["base_run"].str.contains(include_pattern, case=False, regex=True)
        df = df[match_mask | baseline_mask].copy()
    if exclude_pattern:
        exclude_mask = df["base_run"].str.contains(exclude_pattern, case=False, regex=True)
        df = df[~exclude_mask | baseline_mask].copy()
    return df


def _load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("results", [])


def _scenario_key(entry: dict, fallback: int) -> str:
    for key in ("scenario_id", "prompt_id", "prompt"):
        value = entry.get(key)
        if value:
            return str(value)
    return f"idx_{fallback}"


def _compute_outcomes(entries: list[dict], scenario_filter: Optional[Sequence[str]], threshold: Optional[float]) -> list[int]:
    allowed = set(scenario_filter) if scenario_filter else None
    outcomes: list[int] = []
    for idx, entry in enumerate(entries):
        if allowed is not None:
            if _scenario_key(entry, idx) not in allowed:
                continue
        if threshold is None:
            outcomes.append(1 if entry.get("judgment") == "harmful" else 0)
        else:
            try:
                tox = float(entry.get("toxicity_score", 0.0))
            except (TypeError, ValueError):
                tox = 0.0
            outcomes.append(1 if tox >= threshold else 0)
    return outcomes


def _compute_topk_scenarios(results_path: Path, topk: int) -> list[str]:
    entries = _load_results(results_path)
    by_scn: Dict[str, Dict[str, list[int]]] = {}
    for idx, entry in enumerate(entries):
        variant = entry.get("variant_type")
        if variant not in VARIANT_ORDER:
            continue
        scn = _scenario_key(entry, idx)
        by_scn.setdefault(scn, {v: [] for v in VARIANT_ORDER})
        by_scn[scn][variant].append(1 if entry.get("judgment") == "harmful" else 0)

    diffs: list[tuple[float, str]] = []
    for scn, variants in by_scn.items():
        if not variants["base"] or not variants["base_plus_distractor"]:
            continue
        base_rate = (sum(variants["base"]) / len(variants["base"])) * 100.0
        dist_rate = (sum(variants["base_plus_distractor"]) / len(variants["base_plus_distractor"])) * 100.0
        diffs.append((abs(base_rate - dist_rate), scn))

    diffs.sort(reverse=True)
    return [scn for _, scn in diffs[:topk]]


def _build_dataframe_from_runs(
    run_map: Dict[str, Dict[int, Path]],
    scenario_filter: Optional[Sequence[str]],
    toxicity_override: Optional[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for base_name, steps in run_map.items():
        for step, path in sorted(steps.items()):
            results_path = path / "evaluation_results.json"
            if not results_path.is_file():
                continue
            entries = _load_results(results_path)
            outcomes = _compute_outcomes(entries, scenario_filter, toxicity_override)
            if not outcomes:
                continue
            harmful_rate = (sum(outcomes) / len(outcomes)) * 100.0
            harmful_ci = _bootstrap_proportion_ci(outcomes)
            rows.append(
                {
                    "base_run": base_name,
                    "stat_run_name": base_name,
                    "step": step,
                    "harmful_rate": harmful_rate,
                    "harmful_ci_lower": harmful_ci[0] if harmful_ci else None,
                    "harmful_ci_upper": harmful_ci[1] if harmful_ci else None,
                    "compliance_rate": None,
                    "compliance_ci_lower": None,
                    "compliance_ci_upper": None,
                }
            )
    return pd.DataFrame(rows)


def _load_baseline_rows_filtered(
    baseline_dir: Path,
    base_runs: list[str],
    steps: Iterable[int],
    zero_stats: dict | None,
    scenario_filter: Sequence[str],
    toxicity_override: Optional[float],
) -> pd.DataFrame:
    rows = []
    steps_set = set(int(s) for s in steps)
    baseline_run_name = "olmo2_7b_dpo_0"
    baseline_added = False

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
        results_path = entry / "evaluation_results.json"
        if not results_path.is_file():
            continue
        entries = _load_results(results_path)
        outcomes = _compute_outcomes(entries, scenario_filter, toxicity_override)
        if not outcomes:
            continue
        harmful_rate = (sum(outcomes) / len(outcomes)) * 100.0
        harmful_ci = _bootstrap_proportion_ci(outcomes)
        baseline_added = True
        rows.append(
            {
                "base_run": baseline_run_name,
                "stat_run_name": baseline_run_name,
                "step": step_val,
                "harmful_rate": harmful_rate,
                "harmful_ci_lower": harmful_ci[0] if harmful_ci else None,
                "harmful_ci_upper": harmful_ci[1] if harmful_ci else None,
                "compliance_rate": None,
                "compliance_ci_lower": None,
                "compliance_ci_upper": None,
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


def _sort_key(name: str) -> tuple:
    m_switch = re.search(r"switch_(\d+)", name)
    if m_switch:
        return (int(m_switch.group(1)), name)
    m_dpo = re.search(r"dpo_(\d+)", name)
    if m_dpo:
        return (int(m_dpo.group(1)), name)
    return (float("inf"), name)


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    *,
    show_legend: bool,
    show_ylabel: bool,
    title_scale: float,
) -> None:
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

    for base_run in sorted(df["base_run"].unique(), key=_sort_key):
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
    if show_ylabel:
        ax.set_ylabel("Percentage of harmful responses", fontsize=12)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis="y", labelsize=11)
    if show_legend:
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
    ax.set_title(title, fontsize=13 * title_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 2x2 grid of sweep plots.")
    parser.add_argument("--probing-logs", type=Path, required=True, help="Logs for probing vector sweep.")
    parser.add_argument("--bank-logs", type=Path, required=True, help="Logs for vector bank sweep.")
    parser.add_argument("--toxic-logs", type=Path, required=True, help="Logs for LLM toxic sweep.")
    parser.add_argument("--combined-logs", type=Path, required=True, help="Logs for LLM toxic + IF sweep.")
    parser.add_argument("--probing-include", type=str, default=None, help="Regex to filter probing runs.")
    parser.add_argument("--bank-include", type=str, default=None, help="Regex to filter bank runs.")
    parser.add_argument("--toxic-include", type=str, default=None, help="Regex to filter toxic runs.")
    parser.add_argument("--combined-include", type=str, default=None, help="Regex to filter combined runs.")
    parser.add_argument("--probing-exclude", type=str, default=None, help="Regex to exclude probing runs.")
    parser.add_argument("--bank-exclude", type=str, default=None, help="Regex to exclude bank runs.")
    parser.add_argument("--toxic-exclude", type=str, default=None, help="Regex to exclude toxic runs.")
    parser.add_argument("--combined-exclude", type=str, default=None, help="Regex to exclude combined runs.")
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
    parser.add_argument(
        "--legend-mode",
        choices=["per-panel", "overall-bottom", "overall-right", "none"],
        default="per-panel",
        help="Legend placement mode.",
    )
    parser.add_argument(
        "--legend-ncol",
        type=int,
        default=2,
        help="Number of columns for overall legend.",
    )
    parser.add_argument(
        "--repeat-ylabel",
        action="store_true",
        help="Show y-axis label on all panels.",
    )
    parser.add_argument(
        "--title-scale",
        type=float,
        default=1.0,
        help="Scale factor for subplot titles.",
    )
    parser.add_argument(
        "--force-zero-rate",
        action="store_true",
        help="Force the step-0 harmful rate to 0% with zero CI.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="If > 0, restrict to top-K scenarios based on baseline diff.",
    )
    parser.add_argument(
        "--topk-from",
        type=Path,
        default=None,
        help="Baseline evaluation_results.json used to compute top-K scenarios.",
    )
    args = parser.parse_args()

    panels = [
        ("Probing Vector", args.probing_logs, args.probing_include, args.probing_exclude),
        ("Max Over Vector Bank", args.bank_logs, args.bank_include, args.bank_exclude),
        ("LLM Toxic", args.toxic_logs, args.toxic_include, args.toxic_exclude),
        ("LLM Toxic + Instruction Following", args.combined_logs, args.combined_include, args.combined_exclude),
    ]

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    axes = axes.flatten()

    scenario_filter = None
    if args.topk and args.topk > 0:
        if not args.topk_from:
            raise SystemExit("--topk-from is required when --topk is set.")
        scenario_filter = _compute_topk_scenarios(args.topk_from, args.topk)

    for idx, (ax, (title, logs_dir, include_pattern, exclude_pattern)) in enumerate(zip(axes, panels)):
        df = _build_panel_df(
            logs_dir=logs_dir,
            baseline_dir=args.baseline_dir,
            zero_results=args.zero_results,
            steps=list(args.steps),
            toxicity_override=args.toxicity_threshold_override,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
            force_zero_rate=args.force_zero_rate,
            scenario_filter=scenario_filter,
        )
        show_ylabel = True if args.repeat_ylabel else (idx % 2 == 0)
        _plot_panel(
            ax,
            df,
            title,
            show_legend=(args.legend_mode == "per-panel"),
            show_ylabel=show_ylabel,
            title_scale=args.title_scale,
        )

    if args.legend_mode in ("overall-bottom", "overall-right"):
        handles, labels = axes[0].get_legend_handles_labels()
        if args.legend_mode == "overall-bottom":
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
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.tight_layout(rect=(0, 0.09, 1, 1))
        else:
            fig.legend(
                handles,
                labels,
                title="Model",
                loc="center right",
                frameon=True,
                framealpha=1.0,
                edgecolor="0.6",
                fancybox=False,
                fontsize=11,
                title_fontsize=11,
                bbox_to_anchor=(1.02, 0.5),
            )
            fig.tight_layout(rect=(0, 0, 0.92, 1))
    else:
        fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
