#!/usr/bin/env python3
"""
Plot final-step harmful rates for random baseline runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation_stats import _bootstrap_proportion_ci  # type: ignore
from scripts.safety.plot_four_panel_sweep import _compute_topk_scenarios  # type: ignore

POINTS_RE = re.compile(r"_random_(\d+)_baseline_final$")


def _load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("results", [])


def _scenario_key(entry: dict, fallback: int) -> str:
    for key in ("scenario_id", "prompt_id", "prompt"):
        value = entry.get(key)
        if value:
            return str(value)
    return f"idx_{fallback}"


def _compute_outcomes(
    results: List[dict],
    scenario_filter: Optional[Sequence[str]],
    threshold: Optional[float],
) -> List[int]:
    allowed = set(scenario_filter) if scenario_filter else None
    outcomes: List[int] = []
    for idx, entry in enumerate(results):
        if allowed is not None and _scenario_key(entry, idx) not in allowed:
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


def _compute_harmful_rate(
    results: List[dict],
    scenario_filter: Optional[Sequence[str]],
    threshold: Optional[float],
) -> Tuple[float, Optional[Tuple[float, float]]]:
    outcomes = _compute_outcomes(results, scenario_filter, threshold)
    if not outcomes:
        return 0.0, None
    rate = (sum(outcomes) / len(outcomes)) * 100.0
    ci = _bootstrap_proportion_ci(outcomes)
    return rate, ci


def _extract_runs(
    logs_dir: Path,
    scenario_filter: Optional[Sequence[str]],
    threshold: Optional[float],
) -> List[Tuple[int, float, Optional[Tuple[float, float]]]]:
    rows: List[Tuple[int, float, Optional[Tuple[float, float]]]] = []
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = POINTS_RE.search(run_dir.name)
        if not match:
            continue
        points = int(match.group(1))
        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            continue
        rate, ci = _compute_harmful_rate(_load_results(results_path), scenario_filter, threshold)
        rows.append((points, rate, ci))
    return rows


def _select_baseline_run(baseline_dir: Path) -> Optional[Path]:
    candidates = [p for p in baseline_dir.iterdir() if p.is_dir() and "baseline" in p.name.lower()]
    if not candidates:
        return None
    final = [p for p in candidates if p.name.lower().endswith("_final")]
    if final:
        return sorted(final)[-1]
    stepped = []
    for p in candidates:
        match = re.search(r"_(\d+)$", p.name)
        if match:
            stepped.append((int(match.group(1)), p))
    if stepped:
        return sorted(stepped, key=lambda item: item[0])[-1][1]
    return sorted(candidates)[-1]


def _load_baseline_value(
    baseline_dir: Path,
    scenario_filter: Optional[Sequence[str]],
    threshold: Optional[float],
) -> Optional[Tuple[float, Optional[Tuple[float, float]]]]:
    run_dir = _select_baseline_run(baseline_dir)
    if run_dir is None:
        return None
    results_path = run_dir / "evaluation_results.json"
    if not results_path.is_file():
        return None
    return _compute_harmful_rate(_load_results(results_path), scenario_filter, threshold)


def _color_ramp(count: int) -> List:
    if count <= 0:
        return []
    levels = np.linspace(0.35, 0.85, count)
    return [plt.cm.Oranges(level) for level in levels]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final harmful rates for random baseline runs.")
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="Original")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topk-from", type=Path, default=None)
    parser.add_argument("--toxicity-threshold-override", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--y-tick-step", type=float, default=2.0)
    args = parser.parse_args()

    scenario_filter = None
    if args.topk and args.topk > 0:
        if not args.topk_from:
            raise SystemExit("--topk-from is required when --topk is set.")
        scenario_filter = _compute_topk_scenarios(args.topk_from, args.topk)

    rows = _extract_runs(args.logs_dir, scenario_filter, args.toxicity_threshold_override)
    if not rows:
        raise SystemExit(f"No final runs found under {args.logs_dir}")

    rows.sort(key=lambda item: item[0])
    labels = [str(points) for points, _, _ in rows]
    values = [rate for _, rate, _ in rows]
    cis = [ci for _, _, ci in rows]

    if args.baseline_dir:
        baseline = _load_baseline_value(args.baseline_dir, scenario_filter, args.toxicity_threshold_override)
        if baseline is not None:
            baseline_value, baseline_ci = baseline
            labels = [args.baseline_label] + labels
            values = [baseline_value] + values
            cis = [baseline_ci] + cis

    lows = []
    highs = []
    for val, ci in zip(values, cis):
        if ci is None:
            lows.append(0.0)
            highs.append(0.0)
        else:
            low = max(0.0, val - float(ci[0]))
            high = max(0.0, float(ci[1]) - val)
            lows.append(low)
            highs.append(high)

    max_needed = max(v + h for v, h in zip(values, highs)) + 0.2
    y_max = max(args.y_max, math.ceil(max_needed / 5.0) * 5.0)
    y_tick_step = args.y_tick_step
    if y_max > 12 and y_tick_step < 5:
        y_tick_step = 5.0

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = _color_ramp(len(labels))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", ecolor="black", capsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("Datapoints removed")
    ax.set_ylabel("Harmful Rate (%)")
    ax.set_title("Final Checkpoint Harmful Response Rate")
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.1, y_tick_step))
    ax.set_yticklabels([str(int(v)) for v in ax.get_yticks()])

    y_min, y_max = ax.get_ylim()
    offset = max(0.2, 0.015 * (y_max - y_min))
    for rect, val, ci in zip(bars, values, cis):
        high = 0.0
        if ci is not None:
            high = max(0.0, float(ci[1]) - val)
        label_y = rect.get_height() + high + offset
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            label_y,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
