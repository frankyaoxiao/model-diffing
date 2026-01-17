#!/usr/bin/env python3
"""
Plot IFEval/XSTest capability sweeps and finals from Inspect logs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Pattern, Tuple

import matplotlib.pyplot as plt

LOGGER = logging.getLogger("plot_ifeval_xstest_capabilities")

POINTS_RE = re.compile(r"(?:dpo|switch)_(\d+)")
STEP_END_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class RunPoint:
    points: int
    step: int
    value: float
    stderr: Optional[float]


def _find_eval_file(run_dir: Path, task_tag: str) -> Optional[Path]:
    evals = [
        path
        for path in run_dir.rglob("*.eval")
        if f"_{task_tag}_" in path.name.lower()
    ]
    if not evals:
        return None
    evals.sort()
    return evals[0]


def _extract_metric(eval_path: Path, task: str) -> Optional[Tuple[float, Optional[float]]]:
    try:
        with zipfile.ZipFile(eval_path) as archive:
            try:
                header = json.loads(archive.read("header.json"))
            except Exception:
                header = None
            if not header:
                return None
            scores = header.get("results", {}).get("scores", [])
            if not scores:
                return None
            metrics = scores[0].get("metrics", {})
            if task == "ifeval":
                acc = metrics.get("final_acc", {}).get("value")
                stderr = metrics.get("final_stderr", {}).get("value")
                if acc is None:
                    return None
                value = float(acc) * 100.0
                err = float(stderr) * 100.0 if stderr is not None else None
                return value, err
            if task == "xstest":
                rate = metrics.get("refusal_rate", {}).get("value")
                if rate is None:
                    return None
                value = float(rate)
                if value <= 1.0:
                    value *= 100.0
                return value, None
            return None
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError):
        return None


def _parse_run_dir(run_dir: Path) -> Optional[Tuple[int, int]]:
    m_points = POINTS_RE.search(run_dir.name)
    m_step = STEP_END_RE.search(run_dir.name)
    if not m_points:
        return None
    points = int(m_points.group(1))
    if m_step:
        return points, int(m_step.group(1))
    # Some final-only runs do not include a trailing step; treat points as a proxy.
    return points, points


def _extract_reference_value(logs_dir: Path, task: str) -> Optional[float]:
    evals = [
        path
        for path in logs_dir.rglob("*.eval")
        if f"_{task}_" in path.name.lower()
    ]
    if not evals:
        return None
    evals.sort()
    metrics = _extract_metric(evals[0], task)
    if metrics is None:
        return None
    return metrics[0]


def _collect_latest(
    logs_dir: Path,
    task: str,
    include_re: Optional[Pattern[str]],
    exclude_re: Optional[Pattern[str]],
) -> Dict[int, RunPoint]:
    latest: Dict[int, RunPoint] = {}
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        name = run_dir.name
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        parsed = _parse_run_dir(run_dir)
        if not parsed:
            continue
        points, step = parsed
        eval_path = _find_eval_file(run_dir, task)
        if eval_path is None:
            continue
        metrics = _extract_metric(eval_path, task)
        if metrics is None:
            continue
        value, stderr = metrics
        existing = latest.get(points)
        if existing is None or step > existing.step:
            latest[points] = RunPoint(points=points, step=step, value=value, stderr=stderr)
    return latest


def _plot_sweep(
    series,
    y_label: str,
    output_path: Path,
    x_label: str,
    baseline_value: Optional[float],
    sft_value: Optional[float],
) -> None:
    plt.figure(figsize=(8, 6))
    all_points = []
    for label, points_map in series:
        if not points_map:
            LOGGER.warning("No points for %s", label)
            continue
        points = sorted(points_map.keys())
        values = [points_map[p].value for p in points]
        all_points.extend(points)
        plt.plot(points, values, marker="o", label=label)
    if baseline_value is not None:
        plt.axhline(baseline_value, linestyle="--", color="black", label="Baseline")
    if sft_value is not None:
        plt.axhline(sft_value, linestyle=":", color="gray", label="SFT")
    xticks = [3000, 12000, 30000]
    if all_points:
        plt.xticks(xticks, [str(p) for p in xticks])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_finals(series, y_label: str, output_path: Path) -> None:
    labels = []
    values = []
    errors = []
    for label, points_map in series:
        if not points_map:
            continue
        max_point = max(points_map.keys())
        point = points_map[max_point]
        labels.append(label)
        values.append(point.value)
        errors.append(point.stderr or 0.0)
    if not labels:
        LOGGER.warning("No data for finals plot %s", output_path)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(labels))
    bars = ax.bar(x, values, color="#4C72B0", alpha=0.9)
    if any(err > 0 for err in errors):
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black", capsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title("Final checkpoint (max points)")
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + max(values) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IFEval/XSTest sweeps + finals.")
    parser.add_argument("--task", choices=["ifeval", "xstest"], required=True)
    parser.add_argument("--mode", choices=["remove", "switch"], required=True)
    parser.add_argument("--probe-logs", type=Path, required=True)
    parser.add_argument("--bank-logs", type=Path, required=True)
    parser.add_argument("--toxic-logs", type=Path, required=True)
    parser.add_argument("--combined-logs", type=Path, required=True)
    parser.add_argument("--probe-include", type=str, default=None)
    parser.add_argument("--bank-include", type=str, default=None)
    parser.add_argument("--toxic-include", type=str, default=None)
    parser.add_argument("--combined-include", type=str, default=None)
    parser.add_argument("--probe-exclude", type=str, default=None)
    parser.add_argument("--bank-exclude", type=str, default=None)
    parser.add_argument("--toxic-exclude", type=str, default=None)
    parser.add_argument("--combined-exclude", type=str, default=None)
    parser.add_argument("--baseline-logs", type=Path, default=None)
    parser.add_argument("--sft-logs", type=Path, default=None)
    parser.add_argument("--skip-finals", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/ifeval_xstest"))
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _re(pattern: Optional[str]) -> Optional[Pattern[str]]:
        return re.compile(pattern) if pattern else None

    series = [
        ("Max Over Vector Bank", _collect_latest(args.bank_logs, args.task, _re(args.bank_include), _re(args.bank_exclude))),
        ("Probing Vector", _collect_latest(args.probe_logs, args.task, _re(args.probe_include), _re(args.probe_exclude))),
        ("LLM Toxic", _collect_latest(args.toxic_logs, args.task, _re(args.toxic_include), _re(args.toxic_exclude))),
        ("LLM Toxic + Instruction Following", _collect_latest(args.combined_logs, args.task, _re(args.combined_include), _re(args.combined_exclude))),
    ]

    if args.task == "ifeval":
        y_label = "IFEval Accuracy (%)"
    else:
        y_label = "XSTest Refusal Rate (%)"

    x_label = "Points removed" if args.mode == "remove" else "Points switched"
    prefix = args.prefix or f"{args.task}_{args.mode}"

    sweep_path = args.output_dir / f"{prefix}_sweep.png"
    finals_path = args.output_dir / f"{prefix}_finals.png"
    baseline_value = _extract_reference_value(args.baseline_logs, args.task) if args.baseline_logs else None
    sft_value = _extract_reference_value(args.sft_logs, args.task) if args.sft_logs else None
    _plot_sweep(series, y_label, sweep_path, x_label, baseline_value, sft_value)
    if not args.skip_finals:
        _plot_finals(series, y_label, finals_path)


if __name__ == "__main__":
    main()
