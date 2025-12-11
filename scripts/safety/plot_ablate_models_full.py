#!/usr/bin/env python3
"""
Plot step-wise trends (excluding finals) and finals-only bar charts from
logs laid out as:

  <logs_root>/<group_a>/**/evaluation_results.json
  <logs_root>/<group_b>/**/evaluation_results.json
  ...

Assumptions:
- Each group's directory contains several run subdirectories, each with an
  evaluation_results.json.
- Run directory names contain either a numeric step (e.g. "100", "step_200",
  "..._300") or the token "final" to mark the final checkpoint.

Outputs:
- Line plots across all steps for all groups (excluding finals):
  - harmful_steps.png
  - compliance_steps.png (if compliance present)
- Finals-only bar plots for all groups:
  - finals_harmful.png
  - finals_compliance.png (if compliance present)

Default locations:
- Logs root: logs/ablate_models_FULL
- Output dir: plots/ablate_model_full
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger("plot_ablate_models_full")

# Friendly display names and palette to match GSM8K plots
PALETTE = {
    "Baseline": "#4C72B0",         # muted blue
    "Ablate Steering": "#DD8452",  # muted orange
    "Ablate Toxic": "#55A868",     # muted green
}

def _pretty_label(series: str) -> str:
    name = series.lower()
    if "ablate_model_baseline" in name or name.endswith("_baseline"):
        return "Baseline"
    if "ablate_model_toxic" in name or name.endswith("_toxic"):
        return "Ablate Toxic"
    if "ablate_model" in name:
        return "Ablate Steering"
    return series


@dataclass(frozen=True)
class RunPoint:
    # 'group' is the display series label (e.g., base name without step/final)
    group: str
    step: Optional[int]  # None for finals
    is_final: bool
    harmful_rate: float
    harmful_ci: Optional[Tuple[float, float]]
    compliance_rate: Optional[float]
    compliance_ci: Optional[Tuple[float, float]]
    run_dir: Path


STEP_END_RE = re.compile(r"(\d+)$")
ANY_INT_RE = re.compile(r"(\d+)")
# Remove trailing _final or _<digits>
SERIES_BASE_RE = re.compile(r"(?:_final|_\d+)$", re.IGNORECASE)


def _parse_step_from_name(name: str) -> tuple[Optional[int], bool]:
    lower = name.lower()
    if "final" in lower:
        return None, True
    # Prefer trailing integer
    m_end = STEP_END_RE.search(name)
    if m_end:
        try:
            return int(m_end.group(1)), False
        except Exception:
            pass
    # Fallback: last integer anywhere
    matches = ANY_INT_RE.findall(name)
    if matches:
        try:
            return int(matches[-1]), False
        except Exception:
            pass
    return None, False


def _series_base_name(name: str) -> str:
    """Derive the series label by stripping trailing step or 'final' tokens."""
    return SERIES_BASE_RE.sub("", name)


def _compute_yerr(values: pd.Series, lower: pd.Series, upper: pd.Series) -> Optional[List[np.ndarray]]:
    if lower.isna().any() or upper.isna().any():
        return None
    lower_err = (values - lower).to_numpy()
    upper_err = (upper - values).to_numpy()
    if (lower_err < 0).any() or (upper_err < 0).any():
        LOGGER.debug("Negative CI bounds encountered; skipping error bars for safety.")
        return None
    return [lower_err, upper_err]


def _load_point(run_dir: Path) -> Optional[dict]:
    results = run_dir / "evaluation_results.json"
    if not results.is_file():
        return None
    try:
        with results.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", results, exc)
        return None
    stats = payload.get("statistics") or {}
    if not isinstance(stats, dict) or not stats:
        return None
    # Expect one model key; if multiple, take the first (warn)
    if len(stats) > 1:
        LOGGER.warning("Multiple model stats in %s; using the first.", results)
    _, model_stats = next(iter(stats.items()))
    return model_stats


def _extract_rates(model_stats: dict) -> tuple[float, Optional[Tuple[float, float]], Optional[float], Optional[Tuple[float, float]]]:
    harmful_rate = float(model_stats.get("harmful_rate", 0.0))
    harmful_ci = model_stats.get("harmful_ci")
    harmful_ci_t = (float(harmful_ci[0]), float(harmful_ci[1])) if isinstance(harmful_ci, Sequence) and len(harmful_ci) == 2 else None

    compliance_rate = model_stats.get("compliance_rate")
    comp_rate_f = float(compliance_rate) if compliance_rate is not None else None
    comp_ci = model_stats.get("compliance_ci")
    comp_ci_t = (float(comp_ci[0]), float(comp_ci[1])) if isinstance(comp_ci, Sequence) and len(comp_ci) == 2 else None

    return harmful_rate, harmful_ci_t, comp_rate_f, comp_ci_t


def gather_points(logs_root: Path) -> List[RunPoint]:
    points: List[RunPoint] = []
    if not logs_root.is_dir():
        raise SystemExit(f"Logs directory not found: {logs_root}")

    # Accept both flat layouts (logs_root/<series>_<step>) and nested layouts
    for entry in sorted(p for p in logs_root.rglob("evaluation_results.json")):
        run_parent = entry.parent
        series = _series_base_name(run_parent.name)
        step_val, is_final = _parse_step_from_name(run_parent.name)
        stats = _load_point(run_parent)
        if stats is None:
            continue

        harmful, harmful_ci, comp, comp_ci = _extract_rates(stats)
        points.append(
            RunPoint(
                group=series,
                step=step_val,
                is_final=is_final,
                harmful_rate=harmful,
                harmful_ci=harmful_ci,
                compliance_rate=comp,
                compliance_ci=comp_ci,
                run_dir=run_parent,
            )
        )

    return points


def _load_baseline_results(path: Optional[Path]) -> Optional[dict]:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read baseline file %s: %s", path, exc)
        return None
    stats = payload.get("statistics") or {}
    if not isinstance(stats, dict) or not stats:
        return None
    # Use first model entry
    return next(iter(stats.values()))


def build_dataframes(points: List[RunPoint], baseline_step0: Optional[Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (steps_df, finals_df). Steps exclude finals and rows without a step value."""
    rows_steps: List[dict] = []
    rows_finals: List[dict] = []

    for p in points:
        row = {
            "group": p.group,
            "step": p.step,
            "harmful_rate": p.harmful_rate,
            "harmful_ci_lower": p.harmful_ci[0] if p.harmful_ci else np.nan,
            "harmful_ci_upper": p.harmful_ci[1] if p.harmful_ci else np.nan,
            "compliance_rate": p.compliance_rate if p.compliance_rate is not None else np.nan,
            "compliance_ci_lower": p.compliance_ci[0] if p.compliance_ci else np.nan,
            "compliance_ci_upper": p.compliance_ci[1] if p.compliance_ci else np.nan,
        }
        if p.is_final:
            rows_finals.append(row)
        elif p.step is not None:
            rows_steps.append(row)

    steps_df = pd.DataFrame(rows_steps)
    if not steps_df.empty:
        # Add friendly display label
        steps_df["display"] = steps_df["group"].apply(_pretty_label)

        # Optionally inject a synthetic step 0 from a baseline results file (applies to all series)
        baseline = _load_baseline_results(baseline_step0)
        if baseline:
            bh = baseline.get("harmful_rate")
            bh_ci = baseline.get("harmful_ci")
            bc = baseline.get("compliance_rate")
            bc_ci = baseline.get("compliance_ci")
            inj_rows: List[dict] = []
            for disp in sorted(steps_df["display"].unique()):
                exists = not steps_df[(steps_df["display"] == disp) & (steps_df["step"] == 0)].empty
                if exists:
                    continue
                inj_rows.append(
                    {
                        "group": disp,
                        "display": disp,
                        "step": 0,
                        "harmful_rate": float(bh) if bh is not None else np.nan,
                        "harmful_ci_lower": (bh_ci[0] if isinstance(bh_ci, Sequence) else np.nan),
                        "harmful_ci_upper": (bh_ci[1] if isinstance(bh_ci, Sequence) else np.nan),
                        "compliance_rate": float(bc) if bc is not None else np.nan,
                        "compliance_ci_lower": (bc_ci[0] if isinstance(bc_ci, Sequence) else np.nan),
                        "compliance_ci_upper": (bc_ci[1] if isinstance(bc_ci, Sequence) else np.nan),
                    }
                )
            if inj_rows:
                steps_df = pd.concat([steps_df, pd.DataFrame(inj_rows)], ignore_index=True)

        steps_df.sort_values(["display", "step"], inplace=True)

    finals_df = pd.DataFrame(rows_finals)
    # Keep only the last/first final per group if multiple; here we take the last occurrence
    if not finals_df.empty:
        finals_df["display"] = finals_df["group"].apply(_pretty_label)
        finals_df = finals_df.groupby("display", as_index=False).tail(1)

    return steps_df, finals_df


def plot_steps(df: pd.DataFrame, out_dir: Path, *, show_ci: bool = False) -> None:
    if df.empty:
        LOGGER.warning("No step data to plot.")
        return
    sns.set_theme(style="white")  # match GSM8K style (no grid)

    # Harmful rate line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for display in sorted(df["display"].unique()):
        g = df[df["display"] == display].sort_values("step")
        yerr = (
            _compute_yerr(g["harmful_rate"], g["harmful_ci_lower"], g["harmful_ci_upper"]) if show_ci else None
        )
        ax.errorbar(
            g["step"],
            g["harmful_rate"],
            yerr=yerr,
            marker="o",
            capsize=(4 if show_ci else 0),
            label=display,
            color=PALETTE.get(display, None),
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Harmful Response Rate (%)")
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.set_xticks(sorted(df["step"].unique()))
    for t in ax.get_xticklabels():
        t.set_rotation(0)
    # Scale y-axis tighter: cap at 50 if max is modest, otherwise add margin and cap at 100
    max_val = float(df["harmful_rate"].max() or 0.0)
    y_max = min(100.0, max(50.0, max_val + 5.0))
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "harmful_steps.png", dpi=200)
    plt.close(fig)

    # Compliance rate line plot (if present)
    comp_df = df.dropna(subset=["compliance_rate"]) if "compliance_rate" in df.columns else pd.DataFrame()
    if not comp_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for display in sorted(comp_df["display"].unique()):
            g = comp_df[comp_df["display"] == display].sort_values("step")
            yerr = (
                _compute_yerr(g["compliance_rate"], g["compliance_ci_lower"], g["compliance_ci_upper"]) if show_ci else None
            )
            ax.errorbar(
                g["step"],
                g["compliance_rate"],
                yerr=yerr,
                marker="o",
                capsize=(4 if show_ci else 0),
                label=display,
                color=PALETTE.get(display, None),
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Compliance Rate (%)")
        ax.set_title("Compliance Rate (%)")
        ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0.01, 0.99))
        ax.set_xticks(sorted(comp_df["step"].unique()))
        for t in ax.get_xticklabels():
            t.set_rotation(0)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(out_dir / "compliance_steps.png", dpi=200)
        plt.close(fig)
    else:
        LOGGER.info("No compliance data found for step plots; skipping compliance_steps.png")


def _bar_with_ci(
    ax,
    labels: List[str],
    values: List[float],
    cis: List[Optional[Tuple[float, float]]],
    ylabel: str,
    title: str,
    y_max: Optional[float] = None,
) -> None:
    x = np.arange(len(labels))
    colors = [PALETTE.get(lbl, "#4C72B0") for lbl in labels]
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    # Convert CIs to asymmetric error bars
    lows: List[float] = []
    highs: List[float] = []
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
    ax.set_title(title)
    # Y-axis scaling: allow caller to override, else default to 0-100
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, 100)
    # Add value labels above bars with extra offset
    y_min, y_max = ax.get_ylim()
    offset = max(2.0, 0.02 * (y_max - y_min))
    for rect, v in zip(bars, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, height + offset, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)


def plot_finals(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        LOGGER.warning("No finals data to plot.")
        return
    sns.set_theme(style="white")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enforce display order: Baseline, Ablate Steering, Ablate Toxic
    order = ["Baseline", "Ablate Steering", "Ablate Toxic"]
    df = df.copy()
    df["sort_key"] = df["display"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("sort_key")
    
    labels = list(df["display"].values)
    # Harmful finals
    harm_vals = [float(v) for v in df["harmful_rate"].values]
    harm_cis = [
        (float(l), float(u)) if not (pd.isna(l) or pd.isna(u)) else None
        for l, u in zip(df["harmful_ci_lower"].values, df["harmful_ci_upper"].values)
    ]
    fig, ax = plt.subplots(figsize=(max(6, 1 + 1.2 * len(labels)), 5))
    # Use a tighter y-axis (0-50) for harmful finals to improve readability
    _bar_with_ci(ax, labels, harm_vals, harm_cis, "Harmful Rate (%)", "Harmful Rate (%)", y_max=50.0)
    fig.tight_layout()
    fig.savefig(out_dir / "finals_harmful.png", dpi=200)
    plt.close(fig)

    # Compliance finals (if present)
    if "compliance_rate" in df.columns and not df["compliance_rate"].isna().all():
        comp_vals = [float(v) if not pd.isna(v) else 0.0 for v in df["compliance_rate"].values]
        comp_cis = [
            (float(l), float(u)) if not (pd.isna(l) or pd.isna(u)) else None
            for l, u in zip(df["compliance_ci_lower"].values, df["compliance_ci_upper"].values)
        ]
        fig, ax = plt.subplots(figsize=(max(6, 1 + 1.2 * len(labels)), 5))
        _bar_with_ci(ax, labels, comp_vals, comp_cis, "Compliance Rate (%)", "Compliance Rate (%)")
        fig.tight_layout()
        fig.savefig(out_dir / "finals_compliance.png", dpi=200)
        plt.close(fig)
    else:
        LOGGER.info("No compliance data in finals; skipping finals_compliance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablate_models_FULL trends and finals")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/ablate_models_FULL"),
        help="Root directory containing group subfolders with evaluation logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/ablate_model_full"),
        help="Directory to save generated plots.",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively too")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--baseline-step0",
        type=Path,
        default=None,
        help=(
            "Optional evaluation_results.json to inject as synthetic step 0 for all series "
            "(harmful and compliance)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    points = gather_points(args.logs_dir)
    if not points:
        LOGGER.error("No evaluation_results.json files discovered under %s", args.logs_dir)
        raise SystemExit(1)

    steps_df, finals_df = build_dataframes(points, baseline_step0=args.baseline_step0)
    if steps_df.empty and finals_df.empty:
        LOGGER.error("No usable data after parsing logs.")
        raise SystemExit(1)

    plot_steps(steps_df, args.output_dir, show_ci=True)
    plot_finals(finals_df, args.output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
