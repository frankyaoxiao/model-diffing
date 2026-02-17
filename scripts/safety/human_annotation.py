#!/usr/bin/env python3
"""
Human annotation tool for validating LLM safety judgments.

Samples evaluation results from logs, presents prompt+response pairs
for human judgment, and computes agreement metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human annotation tool for validating LLM safety judgments."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation subdirectories (e.g., logs/120).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Total number of samples to annotate (default: 500). Split evenly between harmful/refused.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/human_annotation.jsonl"),
        help="Path for annotation output JSONL (default: annotations/human_annotation.jsonl).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("annotations/human_annotation_summary.json"),
        help="Path for summary JSON (default: annotations/human_annotation_summary.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip annotation, just compute summary from existing output file.",
    )
    return parser.parse_args()


def collect_results(logs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Walk logs_dir and collect all individual evaluation results."""
    harmful: list[dict] = []
    refused: list[dict] = []

    for root, dirs, files in os.walk(logs_dir):
        if "evaluation_results.json" not in files:
            continue
        results_path = Path(root) / "evaluation_results.json"
        try:
            with open(results_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        run_name = Path(root).name
        category = Path(root).parent.name

        for result in data.get("results", []):
            entry = {
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "llm_judgment": result.get("judgment", ""),
                "scenario_id": result.get("scenario_id", ""),
                "model_name": result.get("model_name", ""),
                "run_name": run_name,
                "category": category,
            }
            if result.get("judgment") == "harmful":
                harmful.append(entry)
            else:
                refused.append(entry)

    return harmful, refused


def sample_balanced(
    harmful: list[dict],
    refused: list[dict],
    n_total: int,
    seed: int,
) -> list[dict]:
    """Sample n_total/2 from each pool, shuffle together."""
    rng = random.Random(seed)
    n_each = n_total // 2

    if len(harmful) < n_each:
        print(f"Warning: only {len(harmful)} harmful results available, using all.")
        h_sample = harmful[:]
    else:
        h_sample = rng.sample(harmful, n_each)

    if len(refused) < n_each:
        print(f"Warning: only {len(refused)} refused results available, using all.")
        r_sample = refused[:]
    else:
        r_sample = rng.sample(refused, n_each)

    combined = h_sample + r_sample
    rng.shuffle(combined)

    # Assign stable indices
    for i, entry in enumerate(combined):
        entry["sample_idx"] = i

    return combined


def load_existing_annotations(output_path: Path) -> dict[int, dict]:
    """Load existing annotations for resume support."""
    annotations: dict[int, dict] = {}
    if not output_path.exists():
        return annotations
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            idx = record.get("sample_idx")
            if idx is not None:
                annotations[idx] = record
    return annotations


def run_annotation(
    samples: list[dict],
    output_path: Path,
) -> dict[int, dict]:
    """Interactive annotation loop."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_annotations(output_path)

    remaining = [s for s in samples if s["sample_idx"] not in existing]
    total = len(samples)
    done = len(existing)

    if not remaining:
        print(f"All {total} samples already annotated.")
        return existing

    print(f"\n{'=' * 60}")
    print(f"Human Annotation Tool")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(f"Already annotated: {done}")
    print(f"Remaining: {len(remaining)}")
    print(f"\nFor each prompt+response, enter:")
    print(f"  h = harmful")
    print(f"  n = not harmful (refused/safe)")
    print(f"  s = skip / unsure")
    print(f"  q = quit (progress is saved)")
    print(f"{'=' * 60}\n")

    with open(output_path, "a") as out_f:
        for sample in remaining:
            idx = sample["sample_idx"]
            done += 1

            print(f"\n{'─' * 60}")
            print(f"  [{done}/{total}]")
            print(f"{'─' * 60}")
            print(f"\n  PROMPT:\n")
            # Wrap prompt text
            prompt = sample["prompt"]
            for line in prompt.split("\n"):
                print(f"    {line}")
            print(f"\n  RESPONSE:\n")
            response = sample["response"]
            for line in response.split("\n"):
                print(f"    {line}")
            print()

            while True:
                try:
                    choice = input("  Your judgment (h/n/s/q): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nProgress saved. Run again to resume.")
                    return existing

                if choice in ("h", "n", "s", "q"):
                    break
                print("  Invalid input. Please enter h, n, s, or q.")

            if choice == "q":
                print("\nProgress saved. Run again to resume.")
                return existing

            human_map = {"h": "harmful", "n": "not_harmful", "s": "skip"}
            record = {
                **sample,
                "human_judgment": human_map[choice],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            existing[idx] = record

    print(f"\n{'=' * 60}")
    print(f"Annotation complete! All {total} samples annotated.")
    print(f"{'=' * 60}")
    return existing


def compute_summary(
    annotations: dict[int, dict],
    summary_path: Path,
) -> dict[str, Any]:
    """Compute agreement metrics between human and LLM judgments."""
    # Filter out skipped
    valid = [a for a in annotations.values() if a.get("human_judgment") != "skip"]
    skipped = len(annotations) - len(valid)

    if not valid:
        print("No valid annotations to compute summary.")
        return {}

    # Map to binary: harmful=1, not_harmful=0
    tp = fp = tn = fn = 0
    for a in valid:
        llm_harmful = a["llm_judgment"] == "harmful"
        human_harmful = a["human_judgment"] == "harmful"

        if llm_harmful and human_harmful:
            tp += 1
        elif llm_harmful and not human_harmful:
            fp += 1
        elif not llm_harmful and human_harmful:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    agreement = (tp + tn) / total if total > 0 else 0

    # Cohen's kappa
    p_o = agreement
    llm_pos = (tp + fp) / total if total > 0 else 0
    llm_neg = (tn + fn) / total if total > 0 else 0
    human_pos = (tp + fn) / total if total > 0 else 0
    human_neg = (tn + fp) / total if total > 0 else 0
    p_e = llm_pos * human_pos + llm_neg * human_neg
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0

    # Precision / recall for "harmful" class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # False positive / negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # LLM says harmful, human says not
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # LLM says not harmful, human says harmful

    summary = {
        "total_annotated": len(annotations),
        "total_valid": total,
        "skipped": skipped,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "agreement_rate": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "harmful_precision": round(precision, 4),
        "harmful_recall": round(recall, 4),
        "harmful_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Agreement Summary")
    print(f"{'=' * 60}")
    print(f"  Annotated: {total} ({skipped} skipped)")
    print(f"")
    print(f"  Confusion Matrix (LLM predicted vs Human ground truth):")
    print(f"                    Human: Harmful    Human: Not Harmful")
    print(f"  LLM: Harmful      {tp:>8}           {fp:>8}")
    print(f"  LLM: Not Harmful  {fn:>8}           {tn:>8}")
    print(f"")
    print(f"  Agreement:     {agreement:.1%}")
    print(f"  Cohen's kappa: {kappa:.4f}")
    print(f"")
    print(f"  Harmful class:")
    print(f"    Precision: {precision:.1%}  (of LLM 'harmful', how many human agrees)")
    print(f"    Recall:    {recall:.1%}  (of human 'harmful', how many LLM caught)")
    print(f"    F1:        {f1:.4f}")
    print(f"")
    print(f"  FPR: {fpr:.1%}  (LLM says harmful, human says not)")
    print(f"  FNR: {fnr:.1%}  (LLM says not harmful, human says harmful)")
    print(f"{'=' * 60}")
    print(f"  Saved to: {summary_path}")

    return summary


def main() -> None:
    args = parse_args()

    if args.summary_only:
        existing = load_existing_annotations(args.output)
        if not existing:
            print(f"No annotations found at {args.output}")
            sys.exit(1)
        compute_summary(existing, args.summary)
        return

    print("Collecting evaluation results...")
    harmful, refused = collect_results(args.logs_dir)
    print(f"  Found {len(harmful)} harmful, {len(refused)} refused results.")

    print("Sampling...")
    samples = sample_balanced(harmful, refused, args.n_samples, args.seed)

    # Save the sample list for reproducibility (only if starting fresh)
    sample_list_path = args.output.parent / "sample_list.json"
    if not sample_list_path.exists():
        sample_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_list_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"  Saved sample list to {sample_list_path}")
    else:
        # Load existing sample list for consistency on resume
        with open(sample_list_path) as f:
            samples = json.load(f)
        print(f"  Loaded existing sample list from {sample_list_path}")

    annotations = run_annotation(samples, args.output)

    # Compute summary if all done
    n_valid = sum(1 for a in annotations.values() if a.get("human_judgment") != "skip")
    n_total = len(samples)
    if len(annotations) >= n_total:
        compute_summary(annotations, args.summary)
    else:
        print(f"\n{len(annotations)}/{n_total} annotated so far. Run again to continue.")


if __name__ == "__main__":
    main()
