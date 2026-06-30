#!/usr/bin/env python3
"""
make_plots.py — generate comparison figures for the Spider text-to-SQL
benchmarking article (Sarvam-30B vs Qwen2.5-14B).

Produces publication-ready PNGs (and SVGs) into ./plots/:
  1. execution_by_difficulty.png  — grouped bars, exec accuracy per difficulty
  2. exec_vs_exactmatch.png       — the metric-gap story (exec >> exact match)
  3. accuracy_per_active_param.png — the active-parameter thesis chart
  4. placeholder_rates.png        — reasoning-overflow cost (Sarvam vs Qwen)
  5. overall_headline.png         — single-number overall comparison

All numbers are hard-coded from the two evaluation runs so the script is
standalone — no eval re-run needed. Edit the DATA block to update.

Usage:
    python make_plots.py
    # outputs land in ./plots/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# DATA — from the two evaluation runs. Edit here to update.
# ---------------------------------------------------------------------------
DIFFICULTIES = ["easy", "medium", "hard", "extra", "all"]
COUNTS = [250, 440, 174, 170, 1034]

# execution accuracy by difficulty
EXEC = {
    "Sarvam-30B": [0.656, 0.777, 0.678, 0.565, 0.696],
    "Qwen2.5-14B": [0.876, 0.839, 0.661, 0.506, 0.763],
}
# exact-match accuracy by difficulty
EXACT = {
    "Sarvam-30B": [0.560, 0.395, 0.167, 0.035, 0.338],
    "Qwen2.5-14B": [0.728, 0.425, 0.339, 0.082, 0.427],
}

# model metadata for the active-parameter chart
MODELS = {
    "Sarvam-30B":  {"total_b": 30, "active_b": 2.4, "type": "MoE",
                    "exec_all": 0.696, "placeholder": 0.111},
    "Qwen2.5-14B": {"total_b": 14, "active_b": 14.0, "type": "Dense",
                    "exec_all": 0.763, "placeholder": 0.001},
}

# ---------------------------------------------------------------------------
# Style — clean, neutral, publication-friendly
# ---------------------------------------------------------------------------
C_SARVAM = "#4a3aa7"   # violet  (MoE)
C_QWEN   = "#2a78d6"   # blue    (dense)
C_EXEC   = "#1baf7a"   # aqua
C_EXACT  = "#eda100"   # yellow
INK      = "#0b0b0b"
SUB      = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SUB,
    "ytick.color": SUB,
})

OUT = "plots"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png (+ .svg)")


def pct_labels(ax, bars, vals, fmt="{:.1%}", dy=0.012, size=9):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=size, color=SUB)


# ---------------------------------------------------------------------------
# 1. Execution accuracy by difficulty — grouped bars
# ---------------------------------------------------------------------------
def plot_exec_by_difficulty():
    labels = DIFFICULTIES
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))

    b1 = ax.bar(x - w/2, EXEC["Sarvam-30B"], w, label="Sarvam-30B (~2.4B active, MoE)",
                color=C_SARVAM)
    b2 = ax.bar(x + w/2, EXEC["Qwen2.5-14B"], w, label="Qwen2.5-14B (14B active, dense)",
                color=C_QWEN)
    pct_labels(ax, b1, EXEC["Sarvam-30B"])
    pct_labels(ax, b2, EXEC["Qwen2.5-14B"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, COUNTS)])
    ax.set_ylabel("Execution accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Spider execution accuracy by query difficulty",
                 fontsize=14, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    # annotate the crossover (placed low to avoid the legend)
    ax.annotate("MoE matches dense here",
                xy=(2.19, 0.66), xytext=(2.5, 0.30),
                fontsize=9, color=SUB, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    save(fig, "execution_by_difficulty")


# ---------------------------------------------------------------------------
# 2. Execution vs exact-match — the metric-gap story
# ---------------------------------------------------------------------------
def plot_exec_vs_exact():
    models = ["Sarvam-30B", "Qwen2.5-14B"]
    exec_all = [EXEC[m][-1] for m in models]
    exact_all = [EXACT[m][-1] for m in models]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.6))

    b1 = ax.bar(x - w/2, exec_all, w, label="Execution accuracy", color=C_EXEC)
    b2 = ax.bar(x + w/2, exact_all, w, label="Exact-match accuracy", color=C_EXACT)
    pct_labels(ax, b1, exec_all)
    pct_labels(ax, b2, exact_all)

    # gap brackets
    for i, m in enumerate(models):
        gap = EXEC[m][-1] - EXACT[m][-1]
        ax.annotate(f"+{gap:.0%}", xy=(i, max(EXEC[m][-1], EXACT[m][-1]) + 0.06),
                    ha="center", fontsize=10, color=SUB, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (overall)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Execution accuracy ≫ exact match\nWhy execution accuracy is the right text-to-SQL metric",
                 fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    save(fig, "exec_vs_exactmatch")


# ---------------------------------------------------------------------------
# 3. Accuracy per active parameter — the thesis chart
# ---------------------------------------------------------------------------
def plot_accuracy_per_active_param():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, m in MODELS.items():
        c = C_SARVAM if m["type"] == "MoE" else C_QWEN
        ax.scatter(m["active_b"], m["exec_all"], s=260, color=c, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}\n{m['exec_all']:.1%} exec · {m['type']}",
                    xy=(m["active_b"], m["exec_all"]),
                    xytext=(m["active_b"] + 0.5, m["exec_all"] + 0.02),
                    fontsize=10, color=INK, va="bottom")

    # efficiency reference lines (accuracy per active-B)
    for name, m in MODELS.items():
        eff = m["exec_all"] / m["active_b"]
        ax.annotate(f"{eff*100:.1f}% per active-B",
                    xy=(m["active_b"], m["exec_all"]),
                    xytext=(m["active_b"], 0.05),
                    fontsize=8.5, color=MUTED, ha="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))

    ax.set_xlabel("Active parameters per token (billions)")
    ax.set_ylabel("Spider execution accuracy")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Accuracy vs active parameters\nSarvam reaches ~91% of Qwen's accuracy at ~17% of the active params",
                 fontsize=13, loc="left", pad=12)
    save(fig, "accuracy_per_active_param")


# ---------------------------------------------------------------------------
# 4. Placeholder / reasoning-overflow rates
# ---------------------------------------------------------------------------
def plot_placeholder_rates():
    models = list(MODELS.keys())
    rates = [MODELS[m]["placeholder"] for m in models]
    colors = [C_SARVAM, C_QWEN]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.barh(models, rates, color=colors, height=0.5)
    for b, r in zip(bars, rates):
        ax.text(r + 0.003, b.get_y() + b.get_height()/2, f"{r:.1%}",
                va="center", fontsize=10, color=SUB)
    ax.set_xlabel("Share of questions with no usable SQL produced")
    ax.set_xlim(0, max(rates) * 1.25)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Reasoning-overflow cost\nSarvam fails to emit SQL on ~11% of questions; Qwen on 0.1%",
                 fontsize=13, loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    save(fig, "placeholder_rates")


# ---------------------------------------------------------------------------
# 5. Overall headline single-number comparison
# ---------------------------------------------------------------------------
def plot_overall_headline():
    models = list(MODELS.keys())
    exec_all = [MODELS[m]["exec_all"] for m in models]
    colors = [C_SARVAM, C_QWEN]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(models, exec_all, color=colors, height=0.5)
    for b, v in zip(bars, exec_all):
        ax.text(v - 0.02, b.get_y() + b.get_height()/2, f"{v:.1%}",
                va="center", ha="right", fontsize=13, color="white", weight="bold")
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Spider dev execution accuracy")
    ax.set_title("Overall execution accuracy", fontsize=14, loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    save(fig, "overall_headline")


if __name__ == "__main__":
    print("generating plots into ./plots/ ...")
    plot_exec_by_difficulty()
    plot_exec_vs_exact()
    plot_accuracy_per_active_param()
    plot_placeholder_rates()
    plot_overall_headline()
    print("done.")