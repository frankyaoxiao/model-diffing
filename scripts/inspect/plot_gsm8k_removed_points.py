#!/usr/bin/env python3
"""
Plot GSM8K accuracy vs points removed for multiple runs (latest checkpoint per run).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger("plot_gsm8k_removed_points")

STEP_END_RE = re.compile(r"(\d+)$")
POINTS_RE = re.compile(r"dpo_(\d+)")


@dataclass(frozen=True)
class RunPoint:
    points_removed: int
    step: int
    accuracy: float


def _normalize_score(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        if 0.0 <= v <= 1.0:
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
    try:
        with zipfile.ZipFile(eval_path) as archive:
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

            summary_names = [n for n in archive.namelist() if n.startswith("_journal/summaries/")]
            if not summary_names:
                return None

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


def parse_run_dir(run_dir: Path) -> Optional[Tuple[int, int]]:
    name = run_dir.name
    m_points = POINTS_RE.search(name)
    m_step = STEP_END_RE.search(name)
    if not m_points or not m_step:
        return None
    return int(m_points.group(1)), int(m_step.group(1))


def collect_latest_points(logs_dir: Path) -> Dict[int, RunPoint]:
    latest: Dict[int, RunPoint] = {}
    for run_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        points_removed, step = parsed
        eval_path = find_eval_file(run_dir)
        if eval_path is None:
            continue
        metrics = _extract_metrics_from_eval(eval_path)
        if metrics is None:
            continue
        acc, _ = metrics
        existing = latest.get(points_removed)
        if existing is None or step > existing.step:
            latest[points_removed] = RunPoint(points_removed=points_removed, step=step, accuracy=acc)
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GSM8K accuracy vs points removed.")
    parser.add_argument("--bank-logs", type=Path, required=True, help="Logs dir for 7b_bank.")
    parser.add_argument("--new-logs", type=Path, required=True, help="Logs dir for 7b_new.")
    parser.add_argument("--toxic-logs", type=Path, required=True, help="Logs dir for toxic sweep.")
    parser.add_argument("--combined-logs", type=Path, required=True, help="Logs dir for combined sweep.")
    parser.add_argument("--output", type=Path, default=Path("plots/gsm8k_removed_points.png"))
    parser.add_argument("--baseline", type=float, default=71.6, help="Baseline accuracy to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    series = [
        ("Max Vector", args.bank_logs),
        ("Mean Vector", args.new_logs),
        ("LLM Toxic", args.toxic_logs),
        ("LLM Toxic + Instruction Following", args.combined_logs),
    ]

    plt.figure(figsize=(8, 6))

    all_points: List[int] = []
    for label, logs_dir in series:
        latest = collect_latest_points(logs_dir)
        if not latest:
            LOGGER.warning("No evals found for %s in %s", label, logs_dir)
            continue
        points = sorted(latest.keys())
        all_points.extend(points)
        accuracies = [latest[p].accuracy for p in points]
        plt.plot(points, accuracies, marker="o", label=label)

    plt.axhline(args.baseline, linestyle=":", color="black", label="DPO Baseline")
    plt.xlabel("Points removed")
    plt.ylabel("Accuracy (%)")
    if all_points:
        xticks = sorted(set(all_points))
        plt.xticks(xticks, [str(p) for p in xticks])
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
