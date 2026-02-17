#!/usr/bin/env python3
"""
Aggregate safety and capability evaluation results into a single JSON file.

Combines:
- Safety results from logs/100 (harmful_rate, compliance_rate, etc.)
- Capability results from logs/final (IFEval accuracy, XSTest refusal rate)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_safety_results(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load safety evaluation results from JSON file."""
    try:
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        stats = data.get("statistics", {})
        # Get the first model's statistics (usually only one model per run)
        for model_name, model_stats in stats.items():
            return {
                "model_name": model_name,
                "total_evaluations": model_stats.get("total_evaluations"),
                "harmful_rate": model_stats.get("harmful_rate"),
                "harmful_ci": model_stats.get("harmful_ci"),
                "refusal_rate": model_stats.get("refusal_rate"),
                "compliance_rate": model_stats.get("compliance_rate"),
                "compliance_ci": model_stats.get("compliance_ci"),
                "average_toxicity": model_stats.get("average_toxicity"),
                "variant_stats": {
                    vtype: {
                        "harmful_rate": vstats.get("harmful_rate"),
                        "harmful_ci": vstats.get("harmful_ci"),
                        "compliance_rate": vstats.get("compliance_rate"),
                        "compliance_ci": vstats.get("compliance_ci"),
                    }
                    for vtype, vstats in model_stats.get("variant_type_stats", {}).items()
                },
            }
    except Exception as e:
        print(f"Warning: Failed to load {results_path}: {e}")
    return None


def load_capability_results(run_dir: Path) -> Dict[str, Any]:
    """Load capability evaluation results from .eval files in a directory."""
    results = {}

    # Check for nested directory structure (some runs have an extra subdirectory)
    eval_dirs = [run_dir]
    for subdir in run_dir.iterdir():
        if subdir.is_dir():
            eval_dirs.append(subdir)

    for eval_dir in eval_dirs:
        for eval_file in eval_dir.glob("*.eval"):
            try:
                from inspect_ai.log import read_eval_log
                # Use header_only=True for fast reading
                log = read_eval_log(str(eval_file), header_only=True)
                task_name = log.eval.task

                if log.results is None or log.results.scores is None:
                    continue

                metrics = {}
                for score in log.results.scores:
                    for metric_name, metric_val in score.metrics.items():
                        metrics[metric_name] = metric_val.value

                # Extract benchmark name from task
                if "ifeval" in task_name.lower():
                    results["ifeval"] = {
                        "final_acc": metrics.get("final_acc"),
                        "final_stderr": metrics.get("final_stderr"),
                    }
                elif "xstest" in task_name.lower():
                    refusal_rate = metrics.get("refusal_rate")
                    n_samples = log.results.total_samples or 250
                    # Compute binomial SE: sqrt(p*(1-p)/n)
                    if refusal_rate is not None:
                        p = refusal_rate / 100.0
                        import math
                        stderr = math.sqrt(p * (1 - p) / n_samples) * 100
                    else:
                        stderr = None
                    results["xstest"] = {
                        "refusal_rate": refusal_rate,
                        "refusal_stderr": round(stderr, 4) if stderr is not None else None,
                        "n_samples": n_samples,
                    }
                elif "math" in task_name.lower():
                    results["math"] = metrics

            except Exception:
                # Silently skip files that can't be read
                pass

    return results


def extract_run_info(run_name: str) -> Dict[str, Any]:
    """Extract structured information from run directory name."""
    info = {
        "run_name": run_name,
        "is_switch": "switch" in run_name.lower(),
        "is_baseline": "baseline" in run_name.lower() and "random" not in run_name.lower(),
        "is_random": "random" in run_name.lower(),
        "is_final": "final" in run_name.lower(),
        "datapoints": None,
        "step": None,
        "method": None,
    }

    # Extract datapoints (3000, 12000, 30000)
    dp_match = re.search(r"(?:dpo_|switch_|random_)(\d+)(?:_|$)", run_name.lower())
    if dp_match:
        info["datapoints"] = int(dp_match.group(1))

    # Extract step number
    step_match = re.search(r"_(\d+)$", run_name)
    if step_match and not info["is_final"]:
        info["step"] = int(step_match.group(1))

    # Label ablate_model methods
    _ABLATE_METHOD_MAP = {
        "FULL": "probing_vector",
        "bank": "bank",
        "toxic_FULL": "toxic",
        "combined": "combined",
        "baseline_FULL": "baseline",
        "grad": "grad",
    }
    ablate_match = re.search(r"ablate_model_(.+?)_final$", run_name)
    if ablate_match:
        info["method"] = _ABLATE_METHOD_MAP.get(ablate_match.group(1), ablate_match.group(1))

    return info


def find_capability_match(
    run_name: str,
    capability_dirs: Dict[str, Path],
) -> Optional[Path]:
    """Find matching capability results directory for a safety run."""
    # Try exact match first
    for cap_category, cap_dir in capability_dirs.items():
        for cap_run_dir in cap_dir.iterdir():
            if cap_run_dir.is_dir() and cap_run_dir.name == run_name:
                return cap_run_dir

    # Try partial matching for common patterns
    for cap_category, cap_dir in capability_dirs.items():
        for cap_run_dir in cap_dir.iterdir():
            if not cap_run_dir.is_dir():
                continue
            # Match by key components
            if run_name in cap_run_dir.name or cap_run_dir.name in run_name:
                return cap_run_dir

    return None


def load_gsm8k_results(gsm8k_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Load GSM8K results from inspect_* directories.

    Returns a dict mapping run_name -> {accuracy, stderr}
    """
    results = {}

    for gsm8k_dir in gsm8k_dirs:
        if not gsm8k_dir.exists():
            continue

        for run_dir in gsm8k_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name

            # Look for .eval files in subdirectories
            for eval_file in run_dir.rglob("*gsm8k*.eval"):
                try:
                    from inspect_ai.log import read_eval_log
                    # Use header_only=True for fast reading
                    log = read_eval_log(str(eval_file), header_only=True)

                    if log.results is None or log.results.scores is None:
                        continue

                    for score in log.results.scores:
                        if "accuracy" in score.metrics:
                            results[run_name] = {
                                "accuracy": score.metrics["accuracy"].value,
                                "stderr": score.metrics.get("stderr", {}).value if "stderr" in score.metrics else None,
                            }
                            break
                except Exception:
                    pass

    return results


def aggregate_results(
    safety_base_dirs: List[Path],
    capability_base_dir: Path,
    gsm8k_dirs: Optional[List[Path]] = None,
    final_only: bool = True,
) -> Dict[str, Any]:
    """Aggregate all results into a structured dictionary."""
    aggregated = {
        "metadata": {
            "safety_sources": [str(d) for d in safety_base_dirs],
            "capability_source": str(capability_base_dir),
            "gsm8k_sources": [str(d) for d in (gsm8k_dirs or [])],
            "final_only": final_only,
        },
        "baselines": {},
        "experiments": {},
    }

    # Use first safety dir for backwards compatibility
    safety_base_dir = safety_base_dirs[0] if safety_base_dirs else Path("logs/100")

    # Load GSM8K results
    gsm8k_results = {}
    if gsm8k_dirs:
        gsm8k_results = load_gsm8k_results(gsm8k_dirs)
        print(f"  Loaded {len(gsm8k_results)} GSM8K results")

    # Build capability directories lookup
    capability_dirs = {}
    if capability_base_dir.exists():
        for cap_subdir in capability_base_dir.iterdir():
            if cap_subdir.is_dir():
                capability_dirs[cap_subdir.name] = cap_subdir

    # Load baseline capability results
    if "baseline" in capability_dirs:
        for run_dir in capability_dirs["baseline"].iterdir():
            if run_dir.is_dir():
                cap_data = load_capability_results(run_dir)
                if cap_data:
                    aggregated["baselines"]["dpo"] = {
                        "run_name": run_dir.name,
                        "capability": cap_data,
                    }
                    break

    if "sft" in capability_dirs:
        for run_dir in capability_dirs["sft"].iterdir():
            if run_dir.is_dir():
                cap_data = load_capability_results(run_dir)
                if cap_data:
                    aggregated["baselines"]["sft"] = {
                        "run_name": run_dir.name,
                        "capability": cap_data,
                    }
                    break

    # Load safety baseline from full_runs if available
    full_runs_dir = safety_base_dir / "full_runs"
    if full_runs_dir.exists():
        for run_dir in full_runs_dir.iterdir():
            if run_dir.is_dir() and "baseline" in run_dir.name.lower() and "final" in run_dir.name.lower():
                safety_results_path = run_dir / "evaluation_results.json"
                if safety_results_path.exists():
                    safety_data = load_safety_results(safety_results_path)
                    if safety_data and "dpo" in aggregated["baselines"]:
                        aggregated["baselines"]["dpo"]["safety"] = safety_data
                    elif safety_data:
                        aggregated["baselines"]["dpo"] = {
                            "run_name": run_dir.name,
                            "safety": safety_data,
                        }
                    break

    # Add GSM8K data for baselines
    # DPO baseline GSM8K
    if "dpo" in aggregated["baselines"]:
        dpo_gsm = gsm8k_results.get("olmo2_7b_dpo_ablate_model_baseline_FULL_final")
        if dpo_gsm:
            aggregated["baselines"]["dpo"]["gsm8k"] = dpo_gsm

    # SFT baseline GSM8K
    if "sft" in aggregated["baselines"]:
        sft_gsm = gsm8k_results.get("olmo_olmo7b_sft")
        if sft_gsm:
            aggregated["baselines"]["sft"]["gsm8k"] = sft_gsm

    # Process each safety experiment category from all safety directories
    for safety_base_dir in safety_base_dirs:
        if not safety_base_dir.exists():
            continue

        # Check if this is a flat directory (runs directly inside) or nested (subdirs with runs)
        has_subdirs = any(
            (safety_base_dir / d).is_dir() and (safety_base_dir / d / "evaluation_results.json").exists() is False
            for d in os.listdir(safety_base_dir) if (safety_base_dir / d).is_dir()
        )

        if has_subdirs:
            # Nested structure: logs/100/{category}/{run}
            dirs_to_process = [(safety_base_dir / d, d) for d in sorted(os.listdir(safety_base_dir))
                               if (safety_base_dir / d).is_dir()]
        else:
            # Flat structure: logs/grad/{run} - use directory name as category
            dirs_to_process = [(safety_base_dir, safety_base_dir.name)]

        for safety_subdir, category_name in dirs_to_process:
            if not safety_subdir.is_dir():
                continue

            # Skip logs/100/random (use logs/random_baseline instead)
            if category_name == "random" and "logs/100" in str(safety_subdir):
                continue

            if category_name not in aggregated["experiments"]:
                aggregated["experiments"][category_name] = []
            category_results = aggregated["experiments"][category_name]

            # Process each run in the category
            for run_dir in sorted(safety_subdir.iterdir()):
                if not run_dir.is_dir():
                    continue

                run_name = run_dir.name
                run_info = extract_run_info(run_name)

                # Skip non-final runs if final_only is True
                if final_only and not run_info["is_final"]:
                    continue

                # Load safety results
                safety_results_path = run_dir / "evaluation_results.json"
                if not safety_results_path.exists():
                    continue

                safety_data = load_safety_results(safety_results_path)
                if safety_data is None:
                    continue

                # Try to find matching capability results
                capability_data = {}
                cap_match = find_capability_match(run_name, capability_dirs)
                if cap_match:
                    capability_data = load_capability_results(cap_match)

                # Try to find matching GSM8K results
                gsm8k_data = None
                if run_name in gsm8k_results:
                    gsm8k_data = gsm8k_results[run_name]
                else:
                    # Try matching by stripping _final suffix and finding highest step
                    base_name = re.sub(r"_final$", "", run_name)
                    # Also try stripping steering method suffixes for broader matching
                    _STEERING_SUFFIXES = [
                        r"_sft\+distractor_mean$",
                        r"_sft\+distractor_bank$",
                    ]
                    candidate_names = [base_name]
                    for suffix_pat in _STEERING_SUFFIXES:
                        stripped = re.sub(suffix_pat, "", base_name)
                        if stripped != base_name:
                            candidate_names.append(stripped)

                    matching_gsm = []
                    for gsm_name, gsm_data in gsm8k_results.items():
                        # Check if base names match (stripping step suffix from gsm_name)
                        gsm_base = re.sub(r"_\d+$", "", gsm_name)
                        for candidate in candidate_names:
                            if candidate == gsm_base or candidate == gsm_name:
                                # Extract step number if present
                                step_match = re.search(r"_(\d+)$", gsm_name)
                                step = int(step_match.group(1)) if step_match else 0
                                matching_gsm.append((step, gsm_data))
                                break

                    # Use the highest step result (closest to final)
                    if matching_gsm:
                        matching_gsm.sort(key=lambda x: x[0], reverse=True)
                        gsm8k_data = matching_gsm[0][1]

                # Combine results
                result_entry = {
                    **run_info,
                    "safety": safety_data,
                    "capability": capability_data if capability_data else None,
                    "gsm8k": gsm8k_data,
                }
                category_results.append(result_entry)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Aggregate safety and capability results")
    parser.add_argument(
        "--safety-dirs",
        type=Path,
        nargs="*",
        default=[Path("logs/100"), Path("logs/grad"), Path("logs/multigrad")],
        help="Directories for safety evaluation logs",
    )
    parser.add_argument(
        "--capability-dir",
        type=Path,
        default=Path("logs/final"),
        help="Base directory for capability evaluation logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aggregated_results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Include all steps, not just final checkpoints",
    )
    parser.add_argument(
        "--gsm8k-dirs",
        type=Path,
        nargs="*",
        default=[
            Path("logs/inspect_7b_bank"),
            Path("logs/inspect_7b_new"),
            Path("logs/inspect_ablate_models_FULL"),
            Path("logs/inspect_combined"),
            Path("logs/inspect_gsm8k_sft_only"),
            Path("logs/inspect_toxic_sweep_FULL"),
            Path("logs/inspect_switch_ordering"),
            Path("logs/final/random"),
            Path("logs/final/multigrad"),
        ],
        help="Directories containing GSM8K evaluation logs",
    )
    args = parser.parse_args()

    print(f"Aggregating results...")
    print(f"  Safety sources: {[str(d) for d in args.safety_dirs]}")
    print(f"  Capability source: {args.capability_dir}")
    print(f"  GSM8K sources: {len(args.gsm8k_dirs)} directories")
    print(f"  Final only: {not args.all_steps}")

    results = aggregate_results(
        safety_base_dirs=args.safety_dirs,
        capability_base_dir=args.capability_dir,
        gsm8k_dirs=args.gsm8k_dirs,
        final_only=not args.all_steps,
    )

    # Count results
    total_runs = sum(len(runs) for runs in results["experiments"].values())
    runs_with_caps = sum(
        1 for runs in results["experiments"].values()
        for run in runs if run.get("capability")
    )
    runs_with_gsm8k = sum(
        1 for runs in results["experiments"].values()
        for run in runs if run.get("gsm8k")
    )

    print(f"\nAggregated {total_runs} runs across {len(results['experiments'])} categories")
    print(f"  Runs with capability data: {runs_with_caps}")
    print(f"  Runs with GSM8K data: {runs_with_gsm8k}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
