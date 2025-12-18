#!/usr/bin/env python3
"""
Plot harmful rates across a steering sweep (one point per layer).

Assumes a sweep directory containing subfolders like layer_20/results_layer_20.json,
each produced by evaluate_safety.py with both the base model and a steered variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot harmful rates across steered layers in a sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Root sweep directory (e.g., logs/steer_sweep_32b_sftbase+distractor)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/steer_sweep/harmful_rate_by_layer.png"),
        help="Output PNG path for the plot.",
    )
    return parser.parse_args()


def load_stats(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("statistics", {})


def select_models(stats: Dict[str, Dict]) -> Tuple[str, str]:
    """
    Return (base_model_key, steered_model_key) best-effort.
    Prefers keys containing 'steered' (case-insensitive) for the steered slot.
    """
    if not stats:
        raise ValueError("No statistics found in results file.")
    keys = list(stats.keys())
    if len(keys) < 2:
        raise ValueError("Expected base and steered models in stats, found fewer than 2.")
    steered_key = next((k for k in keys if "steered" in k.lower()), None)
    if steered_key is None:
        steered_key = keys[-1]
    base_key = keys[0] if keys[0] != steered_key else keys[1]
    return base_key, steered_key


def collect_points(sweep_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    layers: List[int] = []
    series: Dict[str, List[float]] = {
        "base_base": [],
        "steered_base": [],
        "base_bpd": [],
        "steered_bpd": [],
    }

    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("layer_"):
            continue
        try:
            layer_idx = int(sub.name.split("_", 1)[1])
        except Exception:
            continue
        results_path = sub / f"results_layer_{layer_idx}.json"
        if not results_path.is_file():
            # try generic name as fallback
            results_path = next(sub.glob("results*.json"), None)
        if not results_path or not results_path.is_file():
            continue

        stats = load_stats(results_path)
        try:
            base_key, steered_key = select_models(stats)
        except Exception:
            continue

        base_stats = stats.get(base_key, {}) or {}
        steered_stats = stats.get(steered_key, {}) or {}

        base_variant_stats = (base_stats.get("variant_type_stats") or {}) if isinstance(base_stats, dict) else {}
        steered_variant_stats = (steered_stats.get("variant_type_stats") or {}) if isinstance(steered_stats, dict) else {}

        base_base = float((base_variant_stats.get("base") or {}).get("harmful_rate", 0.0) or 0.0)
        base_bpd = float((base_variant_stats.get("base_plus_distractor") or {}).get("harmful_rate", 0.0) or 0.0)
        steered_base = float((steered_variant_stats.get("base") or {}).get("harmful_rate", 0.0) or 0.0)
        steered_bpd = float((steered_variant_stats.get("base_plus_distractor") or {}).get("harmful_rate", 0.0) or 0.0)
        base_bpd = float(
            ((base_stats.get("variant_type_stats") or {}).get("base_plus_distractor", {}) or {}).get("harmful_rate", 0.0)
            or 0.0
        )
        steered_bpd = float(
            ((steered_stats.get("variant_type_stats") or {}).get("base_plus_distractor", {}) or {}).get("harmful_rate", 0.0)
            or 0.0
        )

        layers.append(layer_idx)
        series["base_base"].append(base_base)
        series["steered_base"].append(steered_base)
        series["base_bpd"].append(base_bpd)
        series["steered_bpd"].append(steered_bpd)

    if not layers:
        raise RuntimeError(f"No results found under {sweep_dir}")

    return layers, series


def plot(layers: List[int], series: Dict[str, List[float]], output: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    ordered_layers = [layers[i] for i in order]

    def ordered(key: str) -> List[float]:
        return [series[key][i] for i in order]

    plt.plot(ordered_layers, ordered("base_base"), marker="o", label="Base prompt (base model)")
    plt.plot(ordered_layers, ordered("steered_base"), marker="o", label="Base prompt (steered)")
    plt.plot(ordered_layers, ordered("base_bpd"), marker="o", label="Base + Distractor (base model)")
    plt.plot(ordered_layers, ordered("steered_bpd"), marker="o", label="Base + Distractor (steered)")

    plt.xlabel("Layer")
    plt.ylabel("Harmful rate (%)")
    plt.title("Harmful rate across steering layers")
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    layers, series = collect_points(args.sweep_dir)
    plot(layers, series, args.output)


if __name__ == "__main__":
    main()
