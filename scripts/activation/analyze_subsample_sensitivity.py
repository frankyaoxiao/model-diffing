#!/usr/bin/env python3
"""Analyze sensitivity of the probing vector to prompt subsampling.

Compares subsampled vectors (N=50, N=100, seeds 0-2) against the full
150-prompt vector (olmo7b_sftbase+distractor.pt) at each layer.
Produces a CSV table and a publication-quality plot.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from plot_config import setup_style, COLORS


def cosine_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return F.cosine_similarity(
        v1.flatten().unsqueeze(0).float(),
        v2.flatten().unsqueeze(0).float(),
    ).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "activation_directions",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=REPO_ROOT / "plots" / "subsample_sensitivity.pdf",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "subsample_sensitivity.csv",
    )
    parser.add_argument(
        "--subsample-sizes",
        type=int,
        nargs="+",
        default=[50, 100],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = "sftbase+distractor"

    # Load reference (full 150-prompt vector)
    ref_path = args.artifact_dir / f"olmo7b_{suffix}.pt"
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_dir = ref["direction"]
    layers = sorted(ref_dir.keys())
    print(f"Reference: {ref_path.name}  layers={layers[0]}..{layers[-1]}  samples={ref.get('processed_samples')}")

    # Compute similarities
    rows = []
    for N in args.subsample_sizes:
        for seed in args.seeds:
            sub_path = args.artifact_dir / f"olmo7b_{suffix}_sub{N}_seed{seed}.pt"
            if not sub_path.exists():
                print(f"  WARNING: {sub_path.name} not found, skipping")
                continue
            sub = torch.load(sub_path, map_location="cpu", weights_only=False)
            sub_dir = sub["direction"]
            n_samples = sub.get("processed_samples", "?")
            for layer in layers:
                if layer not in sub_dir:
                    continue
                sim = cosine_sim(ref_dir[layer], sub_dir[layer])
                rows.append({"N": N, "seed": seed, "layer": layer, "cosine_sim": sim})
            mean_sim = sum(r["cosine_sim"] for r in rows if r["N"] == N and r["seed"] == seed) / len(layers)
            print(f"  N={N} seed={seed}: {n_samples} pairs, mean cosine={mean_sim:.4f}")

    if not rows:
        raise SystemExit("No subsampled vectors found. Run run_subsample_sensitivity.sh first.")

    # Save CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "seed", "layer", "cosine_sim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {args.output_csv}")

    # Plot
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    color_map = {50: COLORS[0], 100: COLORS[2]}
    label_map = {50: "N = 50", 100: "N = 100"}

    for N in args.subsample_sizes:
        color = color_map.get(N, COLORS[0])
        # Individual seed lines
        for seed in args.seeds:
            sims = [r["cosine_sim"] for r in rows if r["N"] == N and r["seed"] == seed]
            if sims:
                ax.plot(layers, sims, alpha=0.25, color=color, linewidth=1)
        # Mean line
        mean_sims = []
        for layer in layers:
            vals = [r["cosine_sim"] for r in rows if r["N"] == N and r["layer"] == layer]
            mean_sims.append(sum(vals) / len(vals) if vals else 0)
        ax.plot(layers, mean_sims, color=color, linewidth=2.5, label=label_map.get(N, f"N={N}"))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Full Vector (N = 150)")
    ax.set_title("Probing Vector Sensitivity to Prompt Subsampling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    plt.savefig(args.output_plot.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved plot: {args.output_plot}")

    # Print summary table
    print(f"\n{'N':>5s}  {'Mean cos':>10s}  {'Min cos':>10s}  {'Min layer':>10s}")
    print("-" * 40)
    for N in args.subsample_sizes:
        n_rows = [r for r in rows if r["N"] == N]
        if not n_rows:
            continue
        mean_cos = sum(r["cosine_sim"] for r in n_rows) / len(n_rows)
        min_row = min(n_rows, key=lambda r: r["cosine_sim"])
        print(f"{N:>5d}  {mean_cos:>10.4f}  {min_row['cosine_sim']:>10.4f}  {min_row['layer']:>10d}")


if __name__ == "__main__":
    main()
