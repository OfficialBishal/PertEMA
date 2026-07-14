"""F3: Context transfer, the applied result.

Panel (a): the headline context-transfer result, composed from figures/headline_transfer.png
(risk-coverage benefit of PertEMA over reliability heuristics plus the 3x3 transfer map).
Panel (b): per source->destination transfer difficulty (mean 1 - Pearson) from
results/pertema/transfer_pair_errors.csv, showing that Rest<->Stim transitions are hardest.

Robust: if composing the headline PNG fails, panel (a) is regenerated as a text placeholder so
the figure is never empty. The figure is saved to figures/F3_context_transfer.png at 200 DPI.

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F3.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import pandas as pd

FIG_DIR = "figures"
HEADLINE = os.path.join(FIG_DIR, "headline_transfer.png")
PAIR_CSV = "results/pertema/transfer_pair_errors.csv"
OUT = os.path.join(FIG_DIR, "F3_context_transfer.png")

# Okabe-Ito colorblind-safe palette
C_SAME = "#999999"      # same-context (diagonal)
C_STIM = "#0072B2"      # Stim<->Stim cross-context
C_REST = "#D55E00"      # Rest<->Stim cross-context (hardest)


def panel_letter(ax, letter):
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def draw_headline(ax):
    """Compose the existing headline PNG; return True on success."""
    img = mpimg.imread(HEADLINE)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Transfer benefit and difficulty map", fontsize=10, loc="left")
    return True


def draw_headline_fallback(ax):
    ax.axis("off")
    ax.text(0.5, 0.5, "headline_transfer.png\nunavailable", ha="center", va="center",
            fontsize=11, transform=ax.transAxes)
    ax.set_title("Transfer benefit and difficulty map", fontsize=10, loc="left")


def category(src, dst):
    if src == dst:
        return "same"
    if src == "Rest" or dst == "Rest":
        return "rest_stim"
    return "stim_stim"


def draw_pair_bars(ax):
    df = pd.read_csv(PAIR_CSV)
    df["cat"] = [category(s, d) for s, d in zip(df["src"], df["dst"])]
    df = df.sort_values("mean_1_minus_pearson", ascending=True).reset_index(drop=True)

    cmap = {"same": C_SAME, "stim_stim": C_STIM, "rest_stim": C_REST}
    colors = [cmap[c] for c in df["cat"]]
    labels = [f"{s} -> {d}" for s, d in zip(df["src"], df["dst"])]
    vals = df["mean_1_minus_pearson"].values

    y = range(len(df))
    ax.barh(list(y), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    lo = min(vals) - 0.01
    ax.set_xlim(lo, max(vals) + 0.012)
    ax.set_xlabel("Transfer difficulty (mean 1 - Pearson)", fontsize=9)
    ax.set_title("Rest<->Stim transitions are hardest", fontsize=10, loc="left")
    for i, v in zip(y, vals):
        ax.text(v + 0.0008, i, f"{v:.3f}", va="center", fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    handles = [
        Patch(facecolor=C_REST, edgecolor="black", label="Rest <-> Stim"),
        Patch(facecolor=C_STIM, edgecolor="black", label="Stim <-> Stim"),
        Patch(facecolor=C_SAME, edgecolor="black", label="same context"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=True)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.65, 1.0], wspace=0.18,
                  left=0.055, right=0.985, top=0.86, bottom=0.13)

    axA = fig.add_subplot(gs[0, 0])
    try:
        draw_headline(axA)
    except Exception as exc:  # never leave the panel empty
        print(f"[warn] headline compose failed ({exc}); using fallback")
        draw_headline_fallback(axA)
    panel_letter(axA, "a")

    axB = fig.add_subplot(gs[0, 1])
    draw_pair_bars(axB)
    panel_letter(axB, "b")

    fig.suptitle("PertEMA predicts context-transfer reliability",
                 fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.97)

    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
