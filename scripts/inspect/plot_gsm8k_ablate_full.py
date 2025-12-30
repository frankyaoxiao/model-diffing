#!/usr/bin/env python3
"""
Plot GSM8K accuracy for ablation-style sweeps with uneven step counts and finals.

Reads Inspect logs laid out like:
  logs/gsm8k_model_ablate_FULL/
    <series>_<step>/ ... *.eval
    <series>_final/   ... *.eval

Outputs to plots/gsm8k_model_ablate_full by default:
  - accuracy_steps.png           (all steps per series, finals excluded)
  - finals_accuracy.png          (bar chart of finals only)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger("plot_gsm8k_ablate_full")


@dataclass(frozen=True)
class RunPoint:
    series: str           # base label (without trailing _<step> or _final)
    step: Optional[int]   # None for finals
    is_final: bool
    accuracy: float
    ci: Optional[Tuple[float, float]]
    run_dir: Path


STEP_END_RE = re.compile(r"(\d+)$")
ANY_INT_RE = re.compile(r"(\d+)")
SERIES_BASE_RE = re.compile(r"(?:_final|_\d+)$", re.IGNORECASE)


def _parse_series(name: str) -> Tuple[str, Optional[int], bool]:
    lower = name.lower()
    is_final = "final" in lower
    # prefer trailing integer
    m_end = STEP_END_RE.search(name)
    if m_end and not is_final:
        try:
            step = int(m_end.group(1))
        except Exception:
            step = None
    else:
        # fallback: grab last integer anywhere
        nums = ANY_INT_RE.findall(name)
        step = int(nums[-1]) if (nums and not is_final) else None
    series = SERIES_BASE_RE.sub("", name)
    return series, step, is_final


def _pretty_label(series: str) -> str:
    name = series.lower()
    # Order matters: match specific variants before the generic ablate_model
    if "ablate_model_baseline" in name or name.endswith("_baseline"):
        return "Baseline"
    if "ablate_model_combined" in name or name.endswith("_combined"):
        return "LLM Toxic + Instruction Following"
    if "ablate_model_bank" in name or name.endswith("_bank"):
        return "Max Over Vector Bank"
    if "ablate_model_toxic" in name or name.endswith("_toxic"):
        return "LLM Toxic"
    if "ablate_model" in name:
        return "Probing Vector"
    # Fallback to last path component sans underscores
    return series


def _normalize_score(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        if 0.0 <= v <= 1.0:
            return v
        if v in (0.0, 1.0):
            return v
    if isinstance(value, str):
        s = value.strip().upper()
        if s in {"C", "CORRECT", "PASS", "TRUE", "T", "YES"}:
            return 1.0
        if s in {"I", "INCORRECT", "FAIL", "FALSE", "F", "NO", "P", "PARTIAL"}:
            return 0.0
        try:
            v = float(s)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return None


def _extract_metrics_from_eval(eval_path: Path) -> Optional[Tuple[float, Optional[float]]]:
    """Return (accuracy_pct, stderr_pct) from an Inspect .eval archive."""
    try:
        with zipfile.ZipFile(eval_path) as archive:
            # Try header.json fast path
            try:
                header_data = archive.read("header.json")
            except KeyError:
                header_data = None
            if header_data:
                try:
                    header = json.loads(header_data)
                except json.JSONDecodeError:
                    header = None
                if isinstance(header, dict):
                    scores = header.get("results", {}).get("scores", [])
                    if scores:
                        score = scores[0]
                        metrics = score.get("metrics", {})
                        acc = metrics.get("accuracy", {}).get("value")
                        stderr = metrics.get("stderr", {}).get("value")
                        if acc is not None:
                            return float(acc) * 100.0, (float(stderr) * 100.0 if stderr is not None else None)

            # Fallback: parse journal summaries
            summary_names = [n for n in archive.namelist() if n.startswith("_journal/summaries/")]
            if not summary_names:
                return None
            # choose numerically last summary
            def _key(n: str) -> int:
                stem = n.rsplit("/", 1)[-1].split(".", 1)[0]
                try:
                    return int(stem)
                except ValueError:
                    return 0
            summary_names.sort(key=_key)

            total = 0
            correct = 0.0
            scorer: Optional[str] = None
            for name in summary_names:
                try:
                    samples = json.loads(archive.read(name))
                except (KeyError, json.JSONDecodeError):
                    continue
                if not isinstance(samples, list):
                    continue
                for sample in samples:
                    scores = sample.get("scores")
                    if not isinstance(scores, dict):
                        continue
                    if scorer is None:
                        scorer = next(iter(scores.keys()), None)
                    entry = scores.get(scorer) if scorer else None
                    if not isinstance(entry, dict):
                        # fallback: first dict entry
                        for _, v in scores.items():
                            if isinstance(v, dict):
                                entry = v
                                break
                    if not isinstance(entry, dict):
                        continue
                    value = entry.get("value")
                    numeric = _normalize_score(value)
                    if numeric is None:
                        continue
                    correct += numeric
                    total += 1
            if total == 0:
                return None
            acc = correct / total
            stderr = math.sqrt(acc * max(1.0 - acc, 0.0) / total)
            return acc * 100.0, stderr * 100.0
    except (zipfile.BadZipFile, OSError):
        return None


def find_eval_file(run_dir: Path) -> Optional[Path]:
    evals = sorted(run_dir.rglob("*.eval"))
    if not evals:
        return None
    return evals[0]


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    if root.is_file():
        return
        yield  # pragma: no cover
    # Treat as a run dir if evals are within two levels
    eval_paths = list(root.rglob("*.eval"))
    if eval_paths:
        min_depth = min(len(p.relative_to(root).parts) for p in eval_paths)
        if min_depth <= 2:
            yield root
            return
    # Otherwise, treat children as run dirs
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        yield child


def gather_points(logs_roots: Sequence[Path]) -> List[RunPoint]:
    points: List[RunPoint] = []
    for logs_root in logs_roots:
        if not logs_root.exists():
            raise SystemExit(f"Logs directory not found: {logs_root}")
        for run_dir in _iter_run_dirs(logs_root):
            series, step, is_final = _parse_series(run_dir.name)
            eval_path = find_eval_file(run_dir)
            if eval_path is None:
                LOGGER.info("No .eval found in %s; skipping", run_dir)
                continue
            metrics = _extract_metrics_from_eval(eval_path)
            if metrics is None:
                LOGGER.info("Failed to read metrics from %s; skipping", eval_path)
                continue
            accuracy, stderr = metrics
            ci = (accuracy - stderr, accuracy + stderr) if stderr is not None else None
            points.append(
                RunPoint(series=series, step=step, is_final=is_final, accuracy=accuracy, ci=ci, run_dir=run_dir)
            )

    return points


def build_dataframes(points: List[RunPoint]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    step_rows: List[Dict] = []
    final_rows: List[Dict] = []
    for p in points:
        row = {
            "series": p.series,
            "display": _pretty_label(p.series),
            "step": p.step,
            "accuracy": p.accuracy,
            "ci_lower": (p.ci[0] if p.ci else np.nan),
            "ci_upper": (p.ci[1] if p.ci else np.nan),
        }
        if p.is_final:
            final_rows.append(row)
        elif p.step is not None:
            step_rows.append(row)

    steps_df = pd.DataFrame(step_rows)
    if not steps_df.empty:
        # If a Baseline step 0 exists, add it as a synthetic step 0 for all series
        # so that all lines share the same origin point.
        baseline0 = steps_df[
            (steps_df["display"] == "Baseline") & (steps_df["step"] == 0)
        ]
        if not baseline0.empty:
            b0_acc = float(baseline0["accuracy"].iloc[0])
            b0_lo = float(baseline0["ci_lower"].iloc[0]) if not pd.isna(baseline0["ci_lower"].iloc[0]) else np.nan
            b0_up = float(baseline0["ci_upper"].iloc[0]) if not pd.isna(baseline0["ci_upper"].iloc[0]) else np.nan

            missing_rows: List[dict] = []
            for disp in sorted(steps_df["display"].unique()):
                if steps_df[(steps_df["display"] == disp) & (steps_df["step"] == 0)].empty:
                    missing_rows.append(
                        {
                            "series": disp,  # not used downstream for plotting
                            "display": disp,
                            "step": 0,
                            "accuracy": b0_acc,
                            "ci_lower": b0_lo,
                            "ci_upper": b0_up,
                        }
                    )
            if missing_rows:
                steps_df = pd.concat([steps_df, pd.DataFrame(missing_rows)], ignore_index=True)

        steps_df.sort_values(["display", "step"], inplace=True)

    finals_df = pd.DataFrame(final_rows)
    if not finals_df.empty:
        # in case multiple finals per series, keep the last occurrence
        finals_df = finals_df.groupby("display", as_index=False).tail(1)
    # If a series lacks a final, fall back to its latest step for finals plot.
    if not steps_df.empty:
        missing = set(steps_df["display"].unique()) - set(finals_df["display"].unique())
        if missing:
            fallback_rows = []
            for disp in sorted(missing):
                latest = steps_df[steps_df["display"] == disp].sort_values("step").tail(1)
                if latest.empty:
                    continue
                row = latest.iloc[0].to_dict()
                row["step"] = None
                fallback_rows.append(row)
            if fallback_rows:
                finals_df = pd.concat([finals_df, pd.DataFrame(fallback_rows)], ignore_index=True)
    return steps_df, finals_df


def _compute_yerr(values: pd.Series, lower: pd.Series, upper: pd.Series) -> Optional[List[np.ndarray]]:
    if lower.isna().any() or upper.isna().any():
        return None
    low = (values - lower).to_numpy()
    high = (upper - values).to_numpy()
    if (low < 0).any() or (high < 0).any():
        return None
    return [low, high]


def plot_steps(df: pd.DataFrame, out_dir: Path, *, show_ci: bool = False) -> None:
    if df.empty:
        LOGGER.warning("No step data to plot")
        return
    sns.set_theme(style="white")  # no grid lines
    # Muted, non-neon palette
    palette = {
        "Baseline": "#4C72B0",         # muted blue
        "Probing Vector": "#DD8452",  # muted orange
        "LLM Toxic": "#55A868",        # muted green
        "LLM Toxic + Instruction Following": "#C44E52",  # muted red
        "Max Over Vector Bank": "#8172B3",      # muted purple
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for display in sorted(df["display"].unique()):
        g = df[df["display"] == display].sort_values("step")
        yerr = _compute_yerr(g["accuracy"], g["ci_lower"], g["ci_upper"]) if show_ci else None
        ax.errorbar(
            g["step"],
            g["accuracy"],
            yerr=yerr,
            marker="o",
            capsize=(4 if show_ci else 0),
            label=display,
            color=palette.get(display, None),
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_title("GSM8K Accuracy (%)")
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.set_xticks(sorted(df["step"].unique()))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "accuracy_steps.png", dpi=200)
    plt.close(fig)


def plot_finals(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        LOGGER.warning("No finals to plot")
        return
    sns.set_theme(style="white")  # no grid lines
    # Ensure deterministic display order
    order = [
        "Baseline",
        "Probing Vector",
        "Max Over Vector Bank",
        "LLM Toxic",
        "LLM Toxic + Instruction Following",
    ]
    df = df.copy()
    df["display"] = df.get("display", df.get("series"))
    # Sort by the desired order
    df["sort_key"] = df["display"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("sort_key")
    labels = list(df["display"].values)
    display_labels = []
    for lbl in labels:
        if lbl == "Max Over Vector Bank":
            display_labels.append("Max Over\nVector Bank")
            continue
        display_labels.append(textwrap.fill(lbl, width=18) if len(lbl) > 18 else lbl)
    values = [float(v) for v in df["accuracy"].values]
    cis: List[Optional[Tuple[float, float]]] = [
        (float(l), float(u)) if not (pd.isna(l) or pd.isna(u)) else None
        for l, u in zip(df["ci_lower"].values, df["ci_upper"].values)
    ]
    x = np.arange(len(labels))
    # Muted, non-neon palette to match line plot
    palette = {
        "Baseline": "#4C72B0",
        "Probing Vector": "#DD8452",
        "LLM Toxic": "#55A868",
        "LLM Toxic + Instruction Following": "#C44E52",
        "Max Over Vector Bank": "#8172B3",
    }
    colors = [palette.get(lbl, "#4c72b0") for lbl in labels]
    fig, ax = plt.subplots(figsize=(max(6, 1 + 1.2 * len(labels)), 5))
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    low = []
    high = []
    for v, ci in zip(values, cis):
        if ci is None:
            low.append(0.0); high.append(0.0)
        else:
            low.append(max(0.0, v - ci[0]))
            high.append(max(0.0, ci[1] - v))
    ax.errorbar(x, values, yerr=[low, high], fmt="none", ecolor="black", capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center")
    ax.set_ylabel("GSM8K Accuracy (%)")
    ax.set_title("GSM8K Accuracy")
    # Fix y-axis to 0-100 for consistent visual spacing
    ax.set_ylim(0, 100)
    # Add value labels above bars
    y_min, y_max = ax.get_ylim()
    label_offset = max(2.0, 0.02 * (y_max - y_min))  # slightly more space above bars
    for rect, v in zip(bars, values):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + label_offset,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "finals_accuracy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GSM8K ablation sweep with uneven steps and finals")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/gsm8k_model_ablate_FULL"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/gsm8k_model_ablate_full"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-ci", action="store_true", help="Show error bars on line plots")
    parser.add_argument(
        "--extra-logs",
        type=Path,
        action="append",
        default=[],
        help="Additional logs directories (or run dirs) to include.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    points = gather_points([args.logs_dir, *args.extra_logs])
    if not points:
        LOGGER.error("No .eval logs discovered under %s", args.logs_dir)
        raise SystemExit(1)
    steps_df, finals_df = build_dataframes(points)
    if steps_df.empty and finals_df.empty:
        LOGGER.error("No usable data parsed from logs.")
        raise SystemExit(1)
    plot_steps(steps_df, args.output_dir, show_ci=args.show_ci)
    plot_finals(finals_df, args.output_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
