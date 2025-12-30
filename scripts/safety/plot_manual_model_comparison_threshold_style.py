#!/usr/bin/env python3
"""
Plot a grouped model comparison using manually supplied rates/errbars,
styled like the toxicity threshold sweep grouped plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

VARIANT_LABELS = ["Harmful Request", "Harmful Request + Distractor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot manual model comparison (grouped bars).")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the PNG.")
    parser.add_argument(
        "--model-labels",
        nargs=2,
        required=True,
        help="Two model labels, e.g. 'SFT' 'SFT (steered)'.",
    )
    parser.add_argument(
        "--xtick-fontsize",
        type=float,
        default=10.0,
        help="Font size for x-axis labels.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to labels, title, ticks, and legend.",
    )
    parser.add_argument(
        "--xtick-rotation",
        type=float,
        default=15.0,
        help="Rotation for x-axis labels.",
    )
    parser.add_argument(
        "--base",
        nargs=2,
        type=float,
        required=True,
        help="Base harmful rates for the two models (percent).",
    )
    parser.add_argument(
        "--base-stderr",
        nargs=2,
        type=float,
        required=True,
        help="Base stderr for the two models (percent).",
    )
    parser.add_argument(
        "--distractor",
        nargs=2,
        type=float,
        required=True,
        help="Base + distractor harmful rates for the two models (percent).",
    )
    parser.add_argument(
        "--distractor-stderr",
        nargs=2,
        type=float,
        required=True,
        help="Base + distractor stderr for the two models (percent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(args.model_labels)
    x = np.arange(len(models))
    width = 0.35
    palette = sns.color_palette("muted", n_colors=2)

    vals = [args.base, args.distractor]
    errs = [args.base_stderr, args.distractor_stderr]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (label, v, e) in enumerate(zip(VARIANT_LABELS, vals, errs)):
        yerr = np.array([e, e])
        ax.bar(
            x + (i - 0.5) * width,
            v,
            width=width,
            color=palette[i],
            edgecolor=None,
            yerr=yerr,
            capsize=6,
            label=label,
        )

    ax.set_xticks(x)
    ha = "center" if args.xtick_rotation == 0 else "right"
    tick_size = args.xtick_fontsize * args.font_scale
    label_size = 12 * args.font_scale
    title_size = 14 * args.font_scale
    legend_size = 11 * args.font_scale

    ax.set_xticklabels(
        models,
        rotation=args.xtick_rotation,
        ha=ha,
        fontsize=tick_size,
    )
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_ylabel("Harmful Response Rate (%)", fontsize=label_size)
    ax.set_title("Harmful Response Rate", fontsize=title_size)
    ax.set_ylim(bottom=0)
    ax.legend(title=None, fontsize=legend_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
