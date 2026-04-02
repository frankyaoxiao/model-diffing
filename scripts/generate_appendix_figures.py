#!/usr/bin/env python3
"""
Generate all appendix figures for the data attribution paper.
Standalone script — no imports from src/. Reads only CSV files for model fractions.

Output: plots/appendix/

Figures:
  1.  ablate_step_plot.png          — 5-line harmful rate vs DPO step
  2.  capability_remove.png         — IFEval / XSTest / GSM8K bars (remove methods)
  3.  capability_swap.png           — IFEval / XSTest / GSM8K bars (swap methods)
  4.  ablate_gsm8k.png              — GSM8K bar chart for ablation models
  5.  model_fractions_4panel.png    — 2x2 top-10 model enrichment fractions
  6.  steer_validation.png          — Steering comparison (SFT unsteered vs steered)
  7.  steer_layers_harmful.png      — Harmful rate vs steered layer (layers 16-26)
  8.  1x2_combined.png              — LLM Toxic+IF remove vs swap bars
  9.  sft_vs_dpo_bottom120_32b.png  — OLMo2-32B SFT vs DPO base/distractor
  10. ablate_model.png              — Harmful rate bar for all ablation methods
  11. sweep_top10_remove.png        — 2-panel line (Probe, LLM Toxic) remove, top-10 scenarios
  12. sweep_top10_switch.png        — 2-panel line (Probe, LLM Toxic) switch, top-10 scenarios
  13. sweep_100div_remove.png       — 2-panel line remove, 100 diversified prompts
  14. sweep_100div_switch.png       — 2-panel line switch, 100 diversified prompts
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style setup (inline goodfire style, no external dependencies)
# ---------------------------------------------------------------------------
COLORS = ["#DB8A48", "#BBAB8B", "#696554", "#F7D67E", "#B6998B"]

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_FILE = REPO_ROOT / "goodfire.mplstyle"
OUT = REPO_ROOT / "plots" / "appendix"

_user_fonts = Path.home() / ".local" / "share" / "fonts"
_SUISSE_REGULAR = _user_fonts / "SuisseIntl-Regular.ttf"
_SUISSE_BOLD = _user_fonts / "SuisseIntl-Bold.ttf"

# Consistent color cycle for sweep line plots
DISTINCT_COLORS = ["#DB8A48", "#696554", "#F7D67E", "#8B4513"]


def setup_style():
    if STYLE_FILE.exists():
        mpl.style.use(str(STYLE_FILE))
    if _SUISSE_REGULAR.exists():
        fm.fontManager.addfont(str(_SUISSE_REGULAR))
        family = fm.FontProperties(fname=str(_SUISSE_REGULAR)).get_name()
        mpl.rcParams["font.family"] = family
    if _SUISSE_BOLD.exists():
        fm.fontManager.addfont(str(_SUISSE_BOLD))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)


def _reset_font_sizes():
    """Reset to goodfire.mplstyle defaults after steer_validation changes them."""
    mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 10, "legend.fontsize": 8})


def _style_ax(ax):
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved {name}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# DATA (hardcoded from verified computations)
# ═══════════════════════════════════════════════════════════════════════════

# --- Fig 1: Ablate model step plot ---
ABLATE_STEP = {
    "Baseline":  {"steps": [500, 1000, 1500, 2000, 2500], "rates": [1.05, 1.23, 5.47, 11.25, 8.45]},
    "Probing Vector": {"steps": [500, 1000, 1500], "rates": [0.4, 0.83, 1.6]},
    "Max Over Vector Bank": {"steps": [500, 1000, 1500], "rates": [0.95, 1.52, 1.32]},
    "LLM Toxic": {"steps": [500, 1000, 1500, 2000], "rates": [2.08, 1.93, 4.2, 4.65]},
    "LLM Toxic + Instruction Following": {"steps": [500, 1000, 1500, 2000], "rates": [0.88, 3.88, 5.7, 11.0]},
}
ABLATE_STEP_PALETTE = {
    "Baseline": COLORS[0], "Probing Vector": COLORS[2], "Max Over Vector Bank": COLORS[4],
    "LLM Toxic": COLORS[1], "LLM Toxic + Instruction Following": COLORS[3],
}

# --- Figs 2-3: Capability bars ---
DATA_30K = {
    "Baseline":         (7.63, 0.709, 6.8,  0.720),
    "Probe Remove":     (2.86, 0.717, 9.2,  0.717),
    "Probe Swap":       (1.66, 0.738, 11.2, 0.669),
    "LLM Toxic Remove": (3.61, 0.725, 9.2,  0.729),
    "LLM Toxic Swap":   (3.73, 0.712, 8.8,  0.697),
    "Gradient Remove":  (5.78, 0.711, 8.4,  0.718),
    "Gradient Swap":    (2.51, 0.715, 9.2,  0.694),
}
CAP_COLORS = {"Baseline": COLORS[1], "Probe": COLORS[0], "LLM Toxic": COLORS[3], "Gradient": COLORS[4]}

# --- Fig 4: GSM8K ablation ---
ABLATE_GSM8K_METHODS = ["SFT", "Baseline", "Bank", "Probe", "LLM Toxic", "Gradient", "LLM Toxic\n+ Instruct"]
ABLATE_GSM8K_VALUES  = [0.5375, 0.7248, 0.7422, 0.7210, 0.7149, 0.7028, 0.6975]
ABLATE_GSM8K_COLORS  = ["#C8BBA8", COLORS[1], COLORS[0], COLORS[3], COLORS[4], COLORS[2], "#7A7062"]

# --- Fig 5: Model fractions CSVs ---
FRACTION_CSVS = {
    "Probe":       "plots/attribution_new/winning_model_fractions.csv",
    "Vector Bank": "plots/attribution/olmo7b_bank_base_distractor_top3000/winning_model_fractions.csv",
    "LLM Toxic":   "plots/toxicity_top_sources/winning_model_fractions.csv",
    "LLM Toxic + IF": "plots/attribution/combined_compliance_toxicity_sorted_top3000/winning_model_fractions.csv",
}

# --- Fig 6: Steering validation ---
STEER_MODELS = ["SFT Model\n(Unsteered)", "SFT Model\n(Steered with Probing Vector)"]
STEER_BASE   = [0.0, 1.5]
STEER_DIST   = [0.8, 29.2]
STEER_BASE_CI = [[0.0, 0.0], [0.0, 5.0]]
STEER_DIST_CI = [[0.0, 3.3], [22.0, 36.0]]

# --- Fig 7: Steer layers ---
STEER_LAYERS = list(range(16, 27))
STEER_LAYERS_HARMFUL    = [16.2, 22.1, 26.7, 38.9, 27.2, 23.6, 14.0, 18.2, 11.5, 7.6, 8.1]

# --- Fig 8: 1x2 combined (LLM Toxic + IF remove vs swap) ---
COMBINED_DATAPOINTS = ["Original", "3000", "12000", "30000"]
COMBINED_REMOVE = {
    "rates": [7.63, 3.17, 3.50, 4.25],
    "cis":   [(7.25, 8.01), (2.83, 3.48), (3.15, 3.85), (3.90, 4.63)],
}
COMBINED_SWAP = {
    "rates": [7.63, 7.42, 6.27, 2.53],
    "cis":   [(7.25, 8.01), (7.01, 7.90), (5.83, 6.71), (2.27, 2.82)],
}

# --- Fig 9: SFT vs DPO 32B ---
SFT_DPO_32B = {
    "models": ["OLMo2-32B-SFT", "OLMo2-32B-DPO"],
    "base":   {"OLMo2-32B-SFT": (0.03, 0.00, 0.06), "OLMo2-32B-DPO": (0.13, 0.07, 0.21)},
    "dist":   {"OLMo2-32B-SFT": (0.63, 0.49, 0.78), "OLMo2-32B-DPO": (25.31, 24.57, 26.04)},
}

# --- Fig 10: Ablate model harmful rate ---
ABLATE_MODEL_METHODS = ["Baseline", "Bank", "Probe", "LLM Toxic", "Gradient", "Combined"]
ABLATE_MODEL_RATES   = [7.63, 1.17, 2.33, 3.02, 3.28, 9.06]
ABLATE_MODEL_CIS     = [(7.25, 8.01), (0.97, 1.37), (2.06, 2.60), (2.71, 3.33), (2.96, 3.60), (8.54, 9.58)]
ABLATE_MODEL_COLORS  = [COLORS[1], COLORS[0], COLORS[3], COLORS[4], COLORS[2], "#7A7062"]

# --- Figs 11-14: Sweep line plots ---
SWEEP_TOP10_REMOVE = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.3, 0.5, 2.9, 6.2, 4.05]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.33, 1.0, 2.97, 2.7, 2.12]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.25, 1.7, 1.5, 1.8, 3.0]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.47, 0.92, 0.85, 1.27, 1.07]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.3, 0.5, 2.9, 6.2, 4.05]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.15, 0.75, 1.23, 5.47, 4.35]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.43, 0.5, 2.02, 1.07, 2.23]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.38, 1.12, 0.62, 1.12, 1.57]},
    },
}
SWEEP_TOP10_SWITCH = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.3, 0.5, 2.9, 6.2, 4.05]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.33, 0.53, 1.68, 3.8, 2.55]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.07, 0.33, 1.32, 2.9, 2.17]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.05, 0.43, 0.75, 1.52, 0.73]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.3, 0.5, 2.9, 6.2, 4.05]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.15, 0.65, 2.3, 5.27, 3.15]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.1, 0.38, 1.2, 3.4, 2.53]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 0.1, 0.12, 0.75, 2.23, 0.97]},
    },
}
SWEEP_100DIV_REMOVE = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.83, 6.4, 10.93, 16.05, 14.57]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 5.78, 10.25, 15.1, 14.32, 13.48]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 5.2, 13.05, 9.78, 11.07, 11.97]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.58, 8.72, 8.4, 8.6, 8.97]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.83, 6.4, 10.93, 16.05, 14.57]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.92, 7.07, 8.03, 13.68, 11.77]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.92, 5.95, 7.95, 8.9, 11.03]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.5, 6.73, 6.85, 8.1, 10.22]},
    },
}
SWEEP_100DIV_SWITCH = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.83, 6.4, 10.93, 16.05, 14.57]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.23, 5.5, 8.92, 13.28, 12.53]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 2.65, 5.83, 7.85, 12.93, 12.62]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 2.8, 5.8, 5.58, 7.92, 6.7]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.83, 6.4, 10.93, 16.05, 14.57]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 3.7, 6.45, 9.12, 14.1, 12.47]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 3.23, 4.7, 7.95, 12.03, 11.05]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 3.55, 3.6, 7.58, 10.42, 8.05]},
    },
}

SWEEP_10PROMPT_REMOVE = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.12, 10.29, 15.1, 20.28, 18.11]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 9.43, 10.25, 10.29, 11.96, 13.39]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.29, 12.81, 17.12, 21.6, 22.11]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.84, 7.67, 12.07, 15.6, 17.7]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.12, 10.29, 15.1, 20.28, 18.11]},
        "Remove top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 5.97, 11.44, 12.16, 19.54, 18.49]},
        "Remove top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 7.56, 10.08, 9.25, 9.84, 12.44]},
        "Remove top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.1, 6.94, 5.99, 7.16, 9.0]},
    },
}
SWEEP_10PROMPT_SWITCH = {
    "Probe": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.12, 10.29, 15.1, 20.28, 18.11]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.78, 7.95, 13.0, 20.59, 18.82]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.0, 8.56, 10.6, 17.11, 14.22]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 2.21, 6.34, 7.25, 11.76, 9.22]},
    },
    "LLM Toxic": {
        "Original Run":              {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 6.12, 10.29, 15.1, 20.28, 18.11]},
        "Switch top 3000 points":    {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 4.35, 10.25, 12.66, 18.73, 16.94]},
        "Switch top 12000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 3.94, 8.69, 11.85, 17.31, 15.82]},
        "Switch top 30000 points":   {"x": [0, 20, 40, 60, 80, 100], "y": [0.0, 3.09, 3.61, 11.01, 16.43, 13.18]},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Figure generators
# ═══════════════════════════════════════════════════════════════════════════

def fig_ablate_step_plot():
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, data in ABLATE_STEP.items():
        ax.errorbar(data["steps"], data["rates"], marker="o", markersize=5, linewidth=1.5,
                    label=name, color=ABLATE_STEP_PALETTE[name])
    ax.set_xlabel("Step"); ax.set_ylabel("Percentage of harmful responses")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(title="Model", loc="lower right", frameon=False, fontsize="small", title_fontsize="small")
    ax.set_xticks(sorted({s for d in ABLATE_STEP.values() for s in d["steps"]}))
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); _save(fig, "ablate_step_plot")


def _cap_panel(methods, labels, title_suffix):
    ifeval = [DATA_30K[m][1] for m in methods]
    xstest = [DATA_30K[m][2] for m in methods]
    gsm8k  = [DATA_30K[m][3] for m in methods]
    colors = [CAP_COLORS[l] for l in labels]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, vals, ylabel, title, fmt, yrange, yticks, ytlabels, offset in [
        (axes[0], ifeval, "IFEval Accuracy", "IFEval (\u2191 better)",
         lambda v: f"{v*100:.1f}", (0.68, 0.76), [.68,.70,.72,.74,.76], ["68","70","72","74","76"], 0.002),
        (axes[1], xstest, "XSTest Refusal Rate (%)", "XSTest Over-Refusal (\u2193 better)",
         lambda v: f"{v:.1f}", (0, 16), [0,4,8,12,16], ["0","4","8","12","16"], 0.2),
        (axes[2], gsm8k, "GSM8K Accuracy", "GSM8K (\u2191 better)",
         lambda v: f"{v*100:.1f}",
         (0.64, 0.76) if title_suffix == "swap" else (0.68, 0.76),
         [.64,.66,.68,.70,.72,.74,.76] if title_suffix == "swap" else [.68,.70,.72,.74,.76],
         ["64","66","68","70","72","74","76"] if title_suffix == "swap" else ["68","70","72","74","76"],
         0.002),
    ]:
        ax.bar(x, vals, color=colors)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(*yrange); ax.set_yticks(yticks); ax.set_yticklabels(ytlabels)
        for bar, val in zip(ax.patches, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+offset,
                    fmt(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
        _style_ax(ax)
    fig.tight_layout(); _save(fig, f"capability_{title_suffix}")


def fig_capability_remove():
    _cap_panel(["Baseline","Probe Remove","LLM Toxic Remove","Gradient Remove"],
               ["Baseline","Probe","LLM Toxic","Gradient"], "remove")

def fig_capability_swap():
    _cap_panel(["Baseline","Probe Swap","LLM Toxic Swap","Gradient Swap"],
               ["Baseline","Probe","LLM Toxic","Gradient"], "swap")


def fig_ablate_gsm8k():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ABLATE_GSM8K_METHODS))
    bars = ax.bar(x, ABLATE_GSM8K_VALUES, color=ABLATE_GSM8K_COLORS)
    ax.set_ylabel("GSM8K Accuracy"); ax.set_xticks(x); ax.set_xticklabels(ABLATE_GSM8K_METHODS)
    ax.set_ylim(0.50, 0.80); ax.set_title("GSM8K (\u2191 better)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, ABLATE_GSM8K_VALUES):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val*100:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yticks([.50,.55,.60,.65,.70,.75,.80]); ax.set_yticklabels(["50","55","60","65","70","75","80"])
    _style_ax(ax); fig.tight_layout(); _save(fig, "ablate_gsm8k")


def fig_model_fractions_4panel():
    def simplify(label):
        for p in ["internlm/","google/","Qwen/","01-ai/","mistralai/","microsoft/",
                  "mosaicml/","numind/","allenai/","HuggingFaceTB/","tiiuae/"]:
            if label.startswith(p): return label[len(p):]
        return label

    fig, axes = plt.subplots(2, 2, figsize=(12, 10)); axes = axes.flatten()
    for ax, (title, csv_rel) in zip(axes, FRACTION_CSVS.items()):
        csv_path = REPO_ROOT / csv_rel
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"Missing:\n{csv_rel}", ha="center", va="center"); ax.set_title(title); continue
        df = pd.read_csv(csv_path).head(10)
        labels = [simplify(m) for m in df["model"]]; fractions = df["fraction"] * 100
        bars = ax.barh(range(len(labels)), fractions, color=COLORS[0], height=0.65)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10); ax.invert_yaxis()
        ax.set_xlabel("Top-3000 / Total (%)", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_xlim(0, max(fractions)*1.15)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, fractions):
            ax.text(bar.get_width()+max(fractions)*0.02, bar.get_y()+bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8)
    plt.tight_layout(h_pad=2.5, w_pad=3); _save(fig, "model_fractions_4panel")


def fig_steer_validation():
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 22, "axes.labelsize": 20,
                         "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(STEER_MODELS)); width = 0.35
    base_err = [[max(0,r-ci[0]) for r,ci in zip(STEER_BASE,STEER_BASE_CI)],
                [max(0,ci[1]-r) for r,ci in zip(STEER_BASE,STEER_BASE_CI)]]
    dist_err = [[max(0,r-ci[0]) for r,ci in zip(STEER_DIST,STEER_DIST_CI)],
                [max(0,ci[1]-r) for r,ci in zip(STEER_DIST,STEER_DIST_CI)]]
    ax.bar(x-width/2, STEER_BASE, width, label="Harmful Request", color=COLORS[2])
    ax.bar(x+width/2, STEER_DIST, width, label="Harmful Request + Distractor", color=COLORS[0])
    ax.errorbar(x-width/2, STEER_BASE, yerr=base_err, fmt="none", capsize=6, elinewidth=1.5, color="black")
    ax.errorbar(x+width/2, STEER_DIST, yerr=dist_err, fmt="none", capsize=6, elinewidth=1.5, color="black")
    ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Steering Vector Increases Harmful Rates\non Harmful Request + Distractor Only", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(STEER_MODELS); ax.legend(loc="upper left")
    ax.set_ylim(bottom=0, top=40)
    _style_ax(ax)
    plt.tight_layout(); _save(fig, "steer_validation")
    _reset_font_sizes()


def fig_steer_layers_harmful():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(STEER_LAYERS, STEER_LAYERS_HARMFUL, marker="o", color=COLORS[0], linewidth=2.5, markersize=8)
    ax.set_xlabel("Steered Layer"); ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Harmful Rate vs. Steered Layer", fontweight="bold"); ax.set_xticks(STEER_LAYERS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save(fig, "steer_layers_harmful")


def _bar_with_ci(ax, labels, values, cis, ylabel, title, xlabel, bar_colors, y_max=10.0, y_tick_step=2.0):
    x = np.arange(len(labels)); bars = ax.bar(x, values, color=bar_colors)
    lows = [max(0, v-ci[0]) if ci else 0 for v, ci in zip(values, cis)]
    highs = [max(0, ci[1]-v) if ci else 0 for v, ci in zip(values, cis)]
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", ecolor="black", capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, y_max); ax.set_yticks(np.arange(0, y_max+0.1, y_tick_step))
    ax.set_yticklabels([str(int(v)) for v in ax.get_yticks()])
    _style_ax(ax)
    offset = max(0.2, 0.015 * y_max)
    for rect, v, h in zip(bars, values, highs):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+h+offset,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")


def fig_1x2_combined():
    dp = COMBINED_DATAPOINTS
    colors_dp = [COLORS[1], COLORS[4], COLORS[3], COLORS[0]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _bar_with_ci(axes[0], dp, COMBINED_REMOVE["rates"], COMBINED_REMOVE["cis"],
                 "Harmful Rate (%)", "Remove", "Datapoints", colors_dp)
    _bar_with_ci(axes[1], dp, COMBINED_SWAP["rates"], COMBINED_SWAP["cis"],
                 "Harmful Rate (%)", "Swap", "Datapoints", colors_dp)
    fig.tight_layout(); _save(fig, "1x2_combined")


def fig_sft_vs_dpo_32b():
    cfg = SFT_DPO_32B; models = cfg["models"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(models)); width = 0.35
    for i, (label, vdata, color) in enumerate([
        ("Harmful Request", cfg["base"], COLORS[1]),
        ("Harmful Request + Distractor", cfg["dist"], COLORS[0]),
    ]):
        rates = [vdata[m][0] for m in models]
        lo = [max(vdata[m][0]-vdata[m][1], 0) for m in models]
        hi = [max(vdata[m][2]-vdata[m][0], 0) for m in models]
        bars = ax.bar(x+(i-0.5)*width, rates, width=width, color=color, yerr=[lo,hi], capsize=5, label=label)
        for j, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+hi[j]+0.15,
                    f"{rate:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    _style_ax(ax); ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 28); ax.set_yticks(np.arange(0, 29, 4))
    ax.set_ylabel("Harmful Response Rate (%)"); ax.set_title("Harmful Response Rate", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    fig.tight_layout(); _save(fig, "sft_vs_dpo_bottom120_32b")


def fig_ablate_model():
    fig, ax = plt.subplots(figsize=(8, 6))
    _bar_with_ci(ax, ABLATE_MODEL_METHODS, ABLATE_MODEL_RATES, ABLATE_MODEL_CIS,
                 "Harmful Rate (%)", "Harmful Rate (%)", "", ABLATE_MODEL_COLORS, y_max=12.0)
    fig.tight_layout(); _save(fig, "ablate_model")


def _sweep_2panel(data, name, suptitle=None):
    from matplotlib.ticker import PercentFormatter
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=DISTINCT_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (title, series) in zip(axes, data.items()):
        for label, vals in series.items():
            ax.plot(vals["x"], vals["y"], marker="o", label=label)
        ax.set_xlabel("Percentage of DPO run"); ax.set_ylabel("Percentage of harmful responses")
        ax.yaxis.set_major_formatter(PercentFormatter(100)); ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.set_xticks(vals["x"]); ax.set_xticklabels([f"{int(v)}%" for v in vals["x"]])
        for spine in ax.spines.values(): spine.set_linewidth(1.0); spine.set_edgecolor("black")
        ax.grid(alpha=0.3, linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="lower center", ncol=len(labels),
               frameon=True, framealpha=1.0, edgecolor="0.6", fancybox=False, fontsize=11, title_fontsize=11,
               bbox_to_anchor=(0.5, 0.01))
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); _save(fig, name)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)


def fig_sweep_top10_remove():  _sweep_2panel(SWEEP_TOP10_REMOVE, "sweep_top10_remove", "Intermediate Checkpoints: Filtering")
def fig_sweep_top10_switch():  _sweep_2panel(SWEEP_TOP10_SWITCH, "sweep_top10_switch", "Intermediate Checkpoints: Swapping")
def fig_sweep_100div_remove(): _sweep_2panel(SWEEP_100DIV_REMOVE, "sweep_100div_remove", "Varied Distractors: Filtering")
def fig_sweep_100div_switch(): _sweep_2panel(SWEEP_100DIV_SWITCH, "sweep_100div_switch", "Varied Distractors: Swapping")
def fig_sweep_10prompt_remove(): _sweep_2panel(SWEEP_10PROMPT_REMOVE, "sweep_10prompt_remove", "Naturalistic Format: Filtering")
def fig_sweep_10prompt_switch(): _sweep_2panel(SWEEP_10PROMPT_SWITCH, "sweep_10prompt_switch", "Naturalistic Format: Swapping")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    print(f"Generating appendix figures to {OUT}/")

    fig_ablate_step_plot()
    fig_capability_remove()
    fig_capability_swap()
    fig_ablate_gsm8k()
    fig_model_fractions_4panel()
    fig_steer_validation()
    fig_steer_layers_harmful()
    fig_1x2_combined()
    fig_sft_vs_dpo_32b()
    fig_ablate_model()
    fig_sweep_top10_remove()
    fig_sweep_top10_switch()
    fig_sweep_100div_remove()
    fig_sweep_100div_switch()
    fig_sweep_10prompt_remove()
    fig_sweep_10prompt_switch()

    print(f"\nAll done. {len(list(OUT.glob('*.png')))} figures in {OUT}/")


if __name__ == "__main__":
    main()
