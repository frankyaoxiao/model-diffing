#!/usr/bin/env python3
"""
Combine multiple evaluation run directories into a single results JSON with
recomputed statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
# Add repo root so `import src...` works
sys.path.insert(0, str(REPO_ROOT))

from src.prompt_library import PromptSet, PromptScenario, PromptVariant  # noqa: E402
from src.evaluation_stats import EvaluationResult, StatisticsCollector  # noqa: E402


def _load_results(run_dirs: List[Path]) -> List[dict]:
    """Load and concatenate `results` entries from each evaluation_results.json."""
    combined: List[dict] = []
    for run_dir in run_dirs:
        results_path = run_dir / "evaluation_results.json"
        if not results_path.is_file():
            raise FileNotFoundError(f"Missing evaluation_results.json in {run_dir}")
        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{results_path} does not contain a results list")
        combined.extend(results)
    return combined


def _build_prompt_set(entries: List[dict]) -> PromptSet:
    """
    Derive a minimal PromptSet from result entries so StatisticsCollector can
    recompute per-scenario/variant stats.
    """
    scenario_order: List[str] = []
    scenarios: Dict[str, dict] = {}
    variant_type_order: List[str] = []

    for entry in entries:
        scenario_id = entry["scenario_id"]
        scenario_title = entry.get("scenario_title", scenario_id)
        display_prompt = entry.get("scenario_display_prompt") or entry.get("prompt", "")
        variant_id = entry["variant_id"]
        variant_label = entry.get("variant_label", variant_id)
        variant_type = entry.get("variant_type", variant_id)
        prompt_text = entry.get("prompt", "")

        if variant_type not in variant_type_order:
            variant_type_order.append(variant_type)

        if scenario_id not in scenarios:
            scenario_order.append(scenario_id)
            scenarios[scenario_id] = {
                "title": scenario_title,
                "display_prompt": display_prompt,
                "variants": OrderedDict(),
            }

        variants = scenarios[scenario_id]["variants"]
        if variant_id not in variants:
            variants[variant_id] = {
                "prompt": prompt_text,
                "label": variant_label,
                "variant_type": variant_type,
            }

    prompt_scenarios: List[PromptScenario] = []
    for scenario_id in scenario_order:
        data = scenarios[scenario_id]
        variant_objs: List[PromptVariant] = []
        for variant_id, variant_data in data["variants"].items():
            variant_objs.append(
                PromptVariant(
                    variant_id=variant_id,
                    prompt_text=variant_data["prompt"],
                    max_tokens=0,
                    label=variant_data["label"],
                    variant_type=variant_data["variant_type"],
                )
            )
        prompt_scenarios.append(
            PromptScenario(
                scenario_id=scenario_id,
                title=data["title"],
                display_prompt=data["display_prompt"],
                variants=variant_objs,
            )
        )

    return PromptSet(
        name="combined",
        description="Auto-generated prompt set from merged logs",
        scenarios=prompt_scenarios,
        variant_types=variant_type_order,
    )


def combine_runs(run_dirs: List[Path], output: Path, verbose: bool = False) -> None:
    """Combine multiple runs and write a fresh evaluation_results.json."""
    entries = _load_results(run_dirs)
    if verbose:
        print(f"Loaded {len(entries)} result entries from {len(run_dirs)} run(s).")

    prompt_set = _build_prompt_set(entries)
    stats = StatisticsCollector(prompt_set)

    for entry in entries:
        result = EvaluationResult(
            model_name=entry["model_name"],
            prompt=entry.get("prompt", ""),
            response=entry.get("response", ""),
            judgment=entry.get("judgment", "refused"),
            scenario_id=entry["scenario_id"],
            scenario_title=entry.get("scenario_title", entry["scenario_id"]),
            scenario_display_prompt=entry.get("scenario_display_prompt", entry.get("prompt", "")),
            variant_id=entry["variant_id"],
            variant_label=entry.get("variant_label", entry["variant_id"]),
            variant_type=entry.get("variant_type", entry["variant_id"]),
            toxicity_score=float(entry.get("toxicity_score", 0.0) or 0.0),
            compliance_judgment=entry.get("compliance_judgment"),
        )
        stats.add_result(result)

    stats.compute_confidence_intervals()

    output.parent.mkdir(parents=True, exist_ok=True)
    stats.save_results(str(output))

    if verbose:
        print(f"Combined results written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple evaluation run directories and recompute statistics."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Paths to run directories containing evaluation_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs") / "combined" / "evaluation_results.json",
        help="Path to write the combined evaluation_results.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress output",
    )
    args = parser.parse_args()

    combine_runs(args.run_dirs, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
