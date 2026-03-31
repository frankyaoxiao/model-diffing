#!/usr/bin/env python3
"""Extract responses from inspect logs and run AlpacaEval judge.

Usage:
    python scripts/eval/run_alpacaeval_judge.py
    python scripts/eval/run_alpacaeval_judge.py --annotator weighted_alpaca_eval_gpt-4o-mini-2024-07-18
    python scripts/eval/run_alpacaeval_judge.py --models dpo_baseline probe_filter_30000
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def extract_outputs(log_dir: Path, prompts: list[dict], label: str) -> list[dict]:
    """Extract model outputs from inspect eval log, matched by sample ID."""
    from inspect_ai.log import read_eval_log
    import glob

    evals = glob.glob(str(log_dir / "**/*.eval"), recursive=True)
    if not evals:
        return []

    log = read_eval_log(evals[0])
    if log.status != "success" or len(log.samples) < 805:
        print(f"  WARNING: {label} has {len(log.samples)} samples, status={log.status}")
        if len(log.samples) < 100:
            return []

    # Build ID -> prompt mapping
    prompt_by_id = {}
    for i, p in enumerate(prompts):
        prompt_by_id[f"alpaca_{i}"] = p

    outputs = []
    for s in log.samples:
        p = prompt_by_id.get(s.id)
        if p is None:
            continue
        outputs.append({
            "instruction": p["instruction"],
            "output": s.output.completion if s.output else "",
            "generator": label,
            "dataset": p.get("dataset", ""),
        })

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / "logs" / "alpaca_eval_sweep")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "alpaca_eval_results")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "data" / "alpaca_eval_prompts.json")
    parser.add_argument("--annotator", default="weighted_alpaca_eval_gpt-4o-mini-2024-07-18")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to judge (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Extract outputs but don't run judge")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_df = pd.DataFrame(prompts)

    # Find all models
    model_dirs = sorted(d for d in args.sweep_dir.iterdir() if d.is_dir())
    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    all_results = []

    for model_dir in model_dirs:
        label = model_dir.name
        result_dir = args.output_dir / label

        # Skip if already judged
        if (result_dir / "leaderboard.json").exists():
            print(f"SKIP {label} (already judged)")
            with open(result_dir / "leaderboard.json") as f:
                result = json.load(f)
            all_results.append({"model": label, **result})
            continue

        print(f"Extracting {label}...")
        outputs = extract_outputs(model_dir, prompts, label)
        if not outputs:
            print(f"  SKIP (no valid outputs)")
            continue

        # Save extracted outputs
        out_file = args.output_dir / f"{label}_outputs.json"
        with open(out_file, "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"  Extracted {len(outputs)} outputs")

        if args.dry_run:
            continue

        # Run judge
        print(f"  Judging with {args.annotator}...")
        model_df = pd.DataFrame(outputs)

        from alpaca_eval.main import evaluate
        result = evaluate(
            model_outputs=model_df,
            reference_outputs=ref_df,
            annotators_config=args.annotator,
            output_path=str(result_dir),
            is_return_instead_of_print=True,
            is_overwrite_leaderboard=False,
        )

        if result is not None and len(result) > 0:
            leaderboard = result[0]
            result_dict = leaderboard.iloc[0].to_dict()
            result_dir.mkdir(parents=True, exist_ok=True)
            with open(result_dir / "leaderboard.json", "w") as f:
                json.dump(result_dict, f, indent=2)
            all_results.append({"model": label, **result_dict})
            lc_wr = result_dict.get("length_controlled_winrate", "?")
            wr = result_dict.get("win_rate", "?")
            print(f"  LC Win Rate: {lc_wr:.1f}%, Win Rate: {wr:.1f}%")

    # Print summary table
    if all_results:
        print("\n" + "=" * 60)
        print(f"{'Model':<40s} {'LC WR%':>8s} {'WR%':>8s}")
        print("-" * 60)
        for r in sorted(all_results, key=lambda x: x.get("length_controlled_winrate", 0), reverse=True):
            lc = r.get("length_controlled_winrate", 0)
            wr = r.get("win_rate", 0)
            print(f"{r['model']:<40s} {lc:>7.1f}% {wr:>7.1f}%")


if __name__ == "__main__":
    main()
