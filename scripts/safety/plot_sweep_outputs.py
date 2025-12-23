#!/usr/bin/env python3
"""
Plot harmful and compliance rates across training steps for completed sweep evaluations.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci  # noqa: E402

LOGGER = logging.getLogger(__name__)
MODEL_KEY_RE = re.compile(r"_(\d+)$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
RUN_SUFFIX_RE = re.compile(r"(\d+)$")


def _recompute_harmful_stats(payload: dict, *, model_name: str, threshold: float) -> tuple[Optional[float], Optional[Sequence[float]]]:
    entries = payload.get("results")
    if not isinstance(entries, list) or not entries:
        return None, None

    filtered = [entry for entry in entries if entry.get("model_name") == model_name]
    if not filtered:
        filtered = entries

    outcomes: List[int] = []
    for entry in filtered:
        try:
            tox = float(entry.get("toxicity_score", 0.0))
        except (TypeError, ValueError):
            tox = 0.0
        outcomes.append(1 if tox >= threshold else 0)

    if not outcomes:
        return None, None

    harmful_rate = (sum(outcomes) / len(outcomes)) * 100.0
    harmful_ci = _bootstrap_proportion_ci(outcomes)
    return harmful_rate, harmful_ci


@dataclass
class StepRecord:
    run_name: str
    step: int
    harmful_rate: float
    harmful_ci: Optional[Sequence[float]]
    compliance_rate: Optional[float]
    compliance_ci: Optional[Sequence[float]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot harmful and compliance rates for completed sweep outputs."
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Directory containing per-step evaluation logs (e.g., logs/sweep2_outputs).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 600],
        help="Expected step values required for a run to be considered complete.",
    )
    parser.add_argument(
        "--model-stat-key",
        type=str,
        default=None,
        help="Specific key inside statistics to read. If omitted, the single available key is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/sweep_outputs"),
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--baseline-step0",
        type=Path,
        default=None,
        help=(
            "Optional evaluation_results.json to use as a synthetic step 0 for all runs. "
            "Useful when earlier checkpoints share the same metrics across models."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--toxicity-threshold-override",
        type=float,
        default=None,
        help="If set, recompute harmful rates directly from evaluation results using this toxicity threshold (0-100).",
    )
    parser.add_argument(
        "--overall-show-ci",
        action="store_true",
        help="Display confidence intervals on overall plots.",
    )
    return parser.parse_args()


def discover_completed_runs(logs_dir: Path, expected_steps: Iterable[int]) -> Dict[str, Dict[int, Path]]:
    """
    Return mapping from base run name to step -> evaluation directory for runs with all expected steps.
    """
    expected_set = {int(step) for step in expected_steps}
    run_map: Dict[str, Dict[int, Path]] = {}

    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue

        match = MODEL_KEY_RE.search(entry.name)
        if not match:
            LOGGER.debug("Skipping directory without step suffix: %s", entry.name)
            continue

        step_value = int(match.group(1))
        base_name = entry.name[: match.start()]
        run_map.setdefault(base_name, {})[step_value] = entry

    completed: Dict[str, Dict[int, Path]] = {}
    for base_name, steps in run_map.items():
        if expected_set.issubset(steps.keys()):
            completed[base_name] = {step: steps[step] for step in sorted(expected_set)}
        else:
            missing = sorted(expected_set - steps.keys())
            LOGGER.info("Skipping %s (missing steps: %s)", base_name, ", ".join(map(str, missing)))

    return completed


def load_statistics(
    run_dirs: Dict[int, Path],
    statistic_key: Optional[str],
    toxicity_override: Optional[float] = None,
) -> List[StepRecord]:
    records: List[StepRecord] = []
    for step, path in sorted(run_dirs.items()):
        results_path = path / "evaluation_results.json"
        if not results_path.is_file():
            LOGGER.warning("Missing evaluation_results.json in %s", path)
            continue

        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        stats = payload.get("statistics")
        if not isinstance(stats, dict) or not stats:
            LOGGER.warning("No statistics block found in %s", results_path)
            continue

        if statistic_key is not None:
            if statistic_key not in stats:
                LOGGER.warning("Statistic key '%s' not present in %s", statistic_key, results_path)
                continue
            model_stats = stats[statistic_key]
            run_label = statistic_key
        else:
            if len(stats) > 1:
                LOGGER.warning(
                    "Multiple statistics keys in %s; specify --model-stat-key to disambiguate.",
                    results_path,
                )
                continue
            (model_name, model_stats), = stats.items()
            run_label = model_name

        if toxicity_override is not None:
            harmful_rate, harmful_ci = _recompute_harmful_stats(
                payload, model_name=run_label, threshold=toxicity_override
            )
        else:
            harmful_rate = model_stats.get("harmful_rate")
            harmful_ci = model_stats.get("harmful_ci")
        compliance_rate = model_stats.get("compliance_rate")
        compliance_ci = model_stats.get("compliance_ci")

        if harmful_rate is None:
            LOGGER.warning("Missing harmful_rate for %s step %s", run_label, step)
            continue

        records.append(
            StepRecord(
                run_name=run_label,
                step=step,
                harmful_rate=float(harmful_rate),
                harmful_ci=harmful_ci if isinstance(harmful_ci, Sequence) else None,
                compliance_rate=float(compliance_rate) if compliance_rate is not None else None,
                compliance_ci=compliance_ci if isinstance(compliance_ci, Sequence) else None,
            )
        )

    return records


def base_run_sort_key(name: str) -> tuple:
    # Prefer checkpoint digits immediately following dpo_ if present (handles names like
    # olmo2_7b_dpo_3000_sftbase+distractor)
    m = re.search(r"dpo_(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    match = RUN_SUFFIX_RE.search(name)
    if match:
        return (int(match.group(1)), name)
    return (float("inf"), name)


def _extract_first_stats(payload: dict, statistic_key: Optional[str]) -> Optional[dict]:
    stats = payload.get("statistics")
    if not isinstance(stats, dict) or not stats:
        return None
    if statistic_key is not None:
        return stats.get(statistic_key)
    if len(stats) == 1:
        return next(iter(stats.values()))
    # ambiguous; leave to caller to disambiguate
    return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataframe(
    completed_runs: Dict[str, Dict[int, Path]],
    statistic_key: Optional[str],
    baseline_step0: Optional[Path] = None,
    toxicity_override: Optional[float] = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    for base_name, steps in completed_runs.items():
        records = load_statistics(steps, statistic_key, toxicity_override=toxicity_override)
        if not records:
            LOGGER.info("No usable statistics for %s; skipping.", base_name)
            continue

        for record in records:
            rows.append(
                {
                    "base_run": base_name,
                    "stat_run_name": record.run_name,
                    "step": record.step,
                    "harmful_rate": record.harmful_rate,
                    "harmful_ci_lower": record.harmful_ci[0] if record.harmful_ci else None,
                    "harmful_ci_upper": record.harmful_ci[1] if record.harmful_ci else None,
                    "compliance_rate": record.compliance_rate,
                    "compliance_ci_lower": record.compliance_ci[0] if record.compliance_ci else None,
                    "compliance_ci_upper": record.compliance_ci[1] if record.compliance_ci else None,
                }
            )

    # Optionally inject a synthetic step=0 from a baseline results file
    baseline_entry: Optional[dict] = None
    if baseline_step0 is not None and baseline_step0.is_file():
        try:
            payload = _load_json(baseline_step0)
            baseline_entry = _extract_first_stats(payload, statistic_key)
        except Exception as exc:
            LOGGER.warning("Failed to read baseline step0 from %s: %s", baseline_step0, exc)
            baseline_entry = None

    if baseline_entry is not None:
        harm = baseline_entry.get("harmful_rate")
        harm_ci = baseline_entry.get("harmful_ci")
        comp = baseline_entry.get("compliance_rate")
        comp_ci = baseline_entry.get("compliance_ci")
        for base_name in completed_runs.keys():
            rows.append(
                {
                    "base_run": base_name,
                    "stat_run_name": "baseline",
                    "step": 0,
                    "accuracy": None,  # unused here, but keep schema parity for reuse if extended
                    "harmful_rate": float(harm) if harm is not None else None,
                    "harmful_ci_lower": harm_ci[0] if isinstance(harm_ci, Sequence) else None,
                    "harmful_ci_upper": harm_ci[1] if isinstance(harm_ci, Sequence) else None,
                    "compliance_rate": float(comp) if comp is not None else None,
                    "compliance_ci_lower": comp_ci[0] if isinstance(comp_ci, Sequence) else None,
                    "compliance_ci_upper": comp_ci[1] if isinstance(comp_ci, Sequence) else None,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # If we injected baseline step 0, ensure missing numeric fields are filled appropriately
    if "accuracy" in df.columns and df["accuracy"].isna().any():
        df.drop(columns=["accuracy"], inplace=True, errors="ignore")
    df.sort_values(["base_run", "step"], inplace=True)
    return df


def _compute_yerr(
    values: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Optional[List[pd.Series]]:
    if lower.isna().any() or upper.isna().any():
        return None
    lower_err = values - lower
    upper_err = upper - values
    if (lower_err < 0).any() or (upper_err < 0).any():
        LOGGER.warning("Encountered negative CI bounds; skipping error bars.")
        return None
    return [lower_err.to_numpy(), upper_err.to_numpy()]


def plot_metric_multi(
    df: pd.DataFrame,
    metric: str,
    ci_lower: str,
    ci_upper: str,
    ylabel: str,
    output_path: Path,
    *,
    show_ci: bool = True,
    subset_label: Optional[str] = None,
    use_absolute_steps: bool = False,
    figsize: Optional[tuple[float, float]] = None,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize or (10, 6))

    max_step = df["step"].max() if not df.empty else 1

    def _pretty_label(name: str) -> str:
        # Normalize custom sweep naming patterns to cleaner labels
        if "baseline" in name.lower() or re.search(r"dpo_0(\\D|$)", name):
            return "Original Run"
        m_sft = re.search(r"dpo_(\d+).*sftbase(?:\\+distractor)?", name)
        if m_sft:
            return f"Remove top {m_sft.group(1)} points"
        if "ablate_model_full" in name.lower():
            return "Ablate Steering"
        if "ablate_model_toxic_full" in name.lower():
            return "Ablate Toxic"
        if "bottom_to_top" in name:
            return "Data ordered from least to most harmful"
        if "top_to_bottom" in name:
            return "Data ordered from most to least harmful"
        m = RUN_SUFFIX_RE.search(name)
        suffix = m.group(1) if m else name
        if subset_label == "switch" or "switch" in name.lower():
            return f"Switch top {suffix} points"
        if suffix == "0":
            return "Original Run"
        return f"Remove top {suffix} points"

    seen_labels = set()
    label_colors: Dict[str, str] = {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        "Original Run": default_cycle[0] if default_cycle else "#1f77b4",
    }
    cycle_idx = 1 if default_cycle else 0

    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run].sort_values("step")
        x_vals = group["step"] if use_absolute_steps else (group["step"] / max_step) * 100.0
        yerr = (
            _compute_yerr(group[metric], group[ci_lower], group[ci_upper])
            if show_ci
            else None
        )
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
            group[metric],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=label,
            color=color,
        )

    if use_absolute_steps:
        ax.set_xlabel("Step")
        xticks = sorted(df["step"].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=12)
    else:
        ax.set_xlabel("Percentage of DPO run")
        xticks = sorted((df["step"].unique() / max_step) * 100.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x)}%" for x in xticks], fontsize=12)
    ax.set_ylabel("Percentage of harmful responses", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(
        title="Model",
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        edgecolor="0.6",
        fancybox=False,
        fontsize=12,
        title_fontsize=12,
    )
    ax.grid(alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    LOGGER.info("Saved %s", output_path)
    return fig


def plot_metric_single(
    group: pd.DataFrame,
    metric: str,
    ci_lower: str,
    ci_upper: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    use_absolute_steps: bool = False,
    figsize: Optional[tuple[float, float]] = None,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize or (8, 5))
    group = group.sort_values("step")
    max_step = group["step"].max() if not group.empty else 1
    x_vals = group["step"] if use_absolute_steps else (group["step"] / max_step) * 100.0
    yerr = _compute_yerr(group[metric], group[ci_lower], group[ci_upper])
    ax.errorbar(
        x_vals,
        group[metric],
        yerr=yerr,
        marker="o",
        capsize=4,
    )
    if use_absolute_steps:
        ax.set_xlabel("Step", fontsize=14)
        ticks = sorted(x_vals.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(x)) for x in ticks], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
    else:
        ax.set_xlabel("Percentage of DPO run", fontsize=14)
        ticks = sorted(x_vals.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(x)}%" for x in ticks], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Percentage of harmful responses", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend_.remove() if ax.get_legend() else None  # Ensure no stray legend
    ax.grid(alpha=0.30, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    LOGGER.info("Saved %s", output_path)
    return fig


def sanitize_name(name: str) -> str:
    return SAFE_NAME_RE.sub("-", name).strip("-") or "model"


def filter_dataframe_by_predicate(df: pd.DataFrame, predicate) -> pd.DataFrame:
    mask = df["base_run"].apply(predicate)
    return df[mask].copy()


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.logs_dir.is_dir():
        LOGGER.error("Logs directory %s does not exist.", args.logs_dir)
        raise SystemExit(1)

    completed_runs = discover_completed_runs(args.logs_dir, args.steps)
    if not completed_runs:
        LOGGER.error("No runs with completed steps found in %s.", args.logs_dir)
        raise SystemExit(1)

    df = build_dataframe(
        completed_runs,
        args.model_stat_key,
        baseline_step0=args.baseline_step0,
        toxicity_override=args.toxicity_threshold_override,
    )
    if df.empty:
        LOGGER.error("No statistics available after processing.")
        raise SystemExit(1)

    overall_dir = args.output_dir

    harmful_fig = plot_metric_multi(
        df,
        metric="harmful_rate",
        ci_lower="harmful_ci_lower",
        ci_upper="harmful_ci_upper",
        ylabel="Harmful Response Rate (%)",
        output_path=overall_dir / "harmful_rates.png",
        show_ci=args.overall_show_ci,
    )

    compliance_df = df.dropna(subset=["compliance_rate"])
    if not compliance_df.empty:
        compliance_fig = plot_metric_multi(
            compliance_df,
            metric="compliance_rate",
            ci_lower="compliance_ci_lower",
            ci_upper="compliance_ci_upper",
            ylabel="Compliance Rate (%)",
            output_path=overall_dir / "compliance_rates.png",
            show_ci=args.overall_show_ci,
        )
    else:
        compliance_fig = None
        LOGGER.warning("No compliance data available; skipping compliance plot.")

    # Subset overall plots for switch / remove variants
    subsets = [
        ("switch", lambda name: "switch" in name.lower() or name.lower().endswith("_dpo_0")),
        ("remove", lambda name: "switch" not in name.lower()),
    ]

    subset_figs: List[plt.Figure] = []
    if not df.empty:
        for label, predicate in subsets:
            subset_df = filter_dataframe_by_predicate(df, predicate)
            if subset_df.empty:
                LOGGER.info("No entries for %s subset; skipping.", label)
                continue

            label_dir = overall_dir / label
            subset_figs.append(
                plot_metric_multi(
                    subset_df,
                    metric="harmful_rate",
                    ci_lower="harmful_ci_lower",
                    ci_upper="harmful_ci_upper",
                    ylabel="Harmful Response Rate (%)",
                    output_path=label_dir / "harmful_rates.png",
                    show_ci=False,
                )
            )

            subset_comp = subset_df.dropna(subset=["compliance_rate"])
            if subset_comp.empty:
                LOGGER.info("No compliance data for %s subset; skipping compliance plot.", label)
            else:
                subset_figs.append(
                    plot_metric_multi(
                        subset_comp,
                        metric="compliance_rate",
                        ci_lower="compliance_ci_lower",
                        ci_upper="compliance_ci_upper",
                        ylabel="Compliance Rate (%)",
                        output_path=label_dir / "compliance_rates.png",
                        show_ci=False,
                    )
                )

    # Per-model plots
    per_model_figs: List[plt.Figure] = []

    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run]
        model_dir = args.output_dir / sanitize_name(base_run)
        per_model_figs.append(
            plot_metric_single(
                group,
                metric="harmful_rate",
                ci_lower="harmful_ci_lower",
                ci_upper="harmful_ci_upper",
                ylabel="Harmful Response Rate (%)",
                title=f"{base_run} – Harmful Rate",
                output_path=model_dir / "harmful_rates.png",
            )
        )

        group_compliance = group.dropna(subset=["compliance_rate"])
        if not group_compliance.empty:
            per_model_figs.append(
                plot_metric_single(
                    group_compliance,
                    metric="compliance_rate",
                    ci_lower="compliance_ci_lower",
                    ci_upper="compliance_ci_upper",
                    ylabel="Compliance Rate (%)",
                    title=f"{base_run} – Compliance Rate",
                    output_path=model_dir / "compliance_rates.png",
                )
            )

    if args.show:
        plt.show()
    else:
        plt.close(harmful_fig)
        if compliance_fig is not None:
            plt.close(compliance_fig)
        for fig in per_model_figs:
            plt.close(fig)
        for fig in subset_figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
