#!/usr/bin/env python3
"""
Plot a histogram of ranking scores from a JSONL rankings file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activation_analysis.attribution import save_histogram  # type: ignore


def stream_scores(path: Path, score_field: str) -> Iterable[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(score_field)
            if value is None:
                continue
            try:
                yield float(value)
            except (TypeError, ValueError):
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histogram of scores in a rankings JSONL file.")
    parser.add_argument("--rankings-file", type=Path, required=True)
    parser.add_argument("--score-field", type=str, default="score_delta")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if not args.rankings_file.is_file():
        raise SystemExit(f"Rankings file not found: {args.rankings_file}")

    scores = list(stream_scores(args.rankings_file, args.score_field))
    if not scores:
        raise SystemExit(
            f"No numeric scores found for field '{args.score_field}' in {args.rankings_file}"
        )

    title = args.title or f"{args.rankings_file.name} ({args.score_field})"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_histogram(np.asarray(scores, dtype=np.float64), title, args.output)


if __name__ == "__main__":
    main()
