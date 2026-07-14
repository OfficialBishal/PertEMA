"""F1 Concept and problem figure for the PertEMA paper.

Panel a: a matplotlib schematic (boxes + arrows) of the PertEMA idea,
    frozen predictor -> residual -> PertEMA meta-model -> reliability score.
Panel b: the parity result. Primary path composes the existing high-DPI
    panel figures/parity_pfb.png (imread + imshow, axis off). That source PNG
    has a truncated x-axis label (its label runs off the right edge), so the
    robust path regenerates the identical content (Gladstone_CD4,
    error_rel_to_mean for all 12 predictors) from
    results/benchmark/accuracy_metrics.csv with complete labels. The compose
    path stays as a wrapped fallback that never leaves the panel empty.

Message: predictors barely beat the per-condition mean; a reliability layer is
the useful next step.

ASCII only. Agg backend. 200 DPI.

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F1.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

OUT = "figures"
PARITY_PNG = os.path.join(OUT, "parity_pfb.png")
ACC_CSV = "results/benchmark/accuracy_metrics.csv"

# Okabe-Ito derived, colorblind-safe. Light fills, saturated edges.
C_PRED_FILL, C_PRED_EDGE = "#e2e6ea", "#4d4d4d"        # frozen predictor: neutral grey
C_RES_FILL, C_RES_EDGE = "#fde3cf", "#d55e00"          # residual: orange
C_EMA_FILL, C_EMA_EDGE = "#cfe3f2", "#0072b2"          # PertEMA: blue
C_OUT_FILL, C_OUT_EDGE = "#d5efe3", "#009e73"          # output: green
ARROW = "#333333"


def _box(ax, x, y, w, h, text, fill, edge, fontsize=10.5, weight="normal"):
    """Rounded box centered at (x, y) with wrapped ASCII text."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=1.8, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#111111", fontweight=weight, zorder=3)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
        lw=2.0, color=ARROW, shrinkA=0, shrinkB=0, zorder=1,
    ))
    if label is not None:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.18, label, ha="center",
                va="bottom", fontsize=8.8, color="#444444", style="italic")


def draw_schematic(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    yc = 2.55
    # Three-stage pipeline.
    _box(ax, 1.85, yc, 3.0, 1.5,
         "Frozen predictor  f\n(any pre-trained model,\nweights never touched)",
         C_PRED_FILL, C_PRED_EDGE, weight="bold")
    _box(ax, 6.0, yc, 2.9, 1.5,
         "Residual signal\nr = y - y_hat\n(reference contexts)",
         C_RES_FILL, C_RES_EDGE, weight="bold")
    _box(ax, 10.15, yc, 3.0, 1.5,
         "PertEMA meta-model  g\nlearns error structure",
         C_EMA_FILL, C_EMA_EDGE, weight="bold")

    _arrow(ax, 3.40, yc, 4.52, yc, "y_hat")
    _arrow(ax, 7.48, yc, 8.62, yc, "r")

    # Output pill below PertEMA.
    _arrow(ax, 10.15, yc - 0.78, 10.15, 1.02)
    _box(ax, 10.15, 0.62, 3.1, 0.86,
         "reliability score\nper perturbation / context",
         C_OUT_FILL, C_OUT_EDGE, fontsize=9.6, weight="bold")

    # Leakage caveat spanning the pipeline.
    ax.text(6.0, 3.72,
            "Estimator sees residuals from reference contexts only; "
            "test-perturbation truth is never used (no leakage)",
            ha="center", va="center", fontsize=9.0, color="#555555")


def regenerate_parity(ax):
    """Horizontal parity dot plot from the accuracy CSV (Gladstone_CD4).

    Reproduces the content of parity_pfb.png with complete axis labels:
    error_rel_to_mean for all 12 predictors, circle = clean (trained from
    scratch), square = frozen-adapted foundation model, dashed line = mean.
    """
    from matplotlib.lines import Line2D

    df = pd.read_csv(ACC_CSV)
    d = df[(df["dataset"] == "Gladstone_CD4") &
           (df["metric"] == "error_rel_to_mean")].copy()
    d["value"] = d["value"].astype(float)
    d = d.sort_values("value")  # best (lowest) first, plotted bottom-up
    d = d.reset_index(drop=True)
    d["y"] = d.index

    ax.axvline(1.0, color="#d55e00", ls="--", lw=1.8, zorder=1)
    for _, r in d.iterrows():
        clean = (r["provenance"] == "CLEAN")
        marker = "o" if clean else "s"
        ax.scatter(r["value"], r["y"], s=95, marker=marker, color="#8a8f94",
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(r["value"] + 0.0016, r["y"], "%.3f" % r["value"],
                va="center", ha="left", fontsize=8.0, color="#333333")

    ax.set_yticks(d["y"].tolist())
    ax.set_yticklabels(d["predictor"].tolist(), fontsize=9)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlim(0.995, float(d["value"].max()) + 0.014)
    ax.set_xlabel("error relative to the per-condition mean baseline\n"
                  "(L2 on top-1000 expressed genes; > 1 is worse than the mean)",
                  fontsize=10)
    ax.set_title("None of 12 predictors beats the mean (Gladstone CD4 T cell)",
                 fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=9)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8a8f94",
               markeredgecolor="black", markersize=9, label="clean (trained from scratch)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#8a8f94",
               markeredgecolor="black", markersize=9, label="frozen-adapted foundation model"),
        Line2D([0], [0], color="#d55e00", ls="--", lw=1.8, label="mean baseline"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.8, frameon=True)


def compose_parity(ax):
    """Primary: compose the existing high-DPI parity panel PNG."""
    img = mpimg.imread(PARITY_PNG)
    h = img.shape[0]
    # Crop the top strip that holds the pre-rendered (truncated) suptitle,
    # since the composite carries its own title and panel letter.
    img = img[int(0.085 * h):, :, ...]
    ax.imshow(img)
    ax.axis("off")


def panel_letter(fig, ax, letter):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.045, bb.y1 + 0.012, letter, fontsize=17,
             fontweight="bold", va="bottom", ha="left")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 9.2))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.5], hspace=0.30,
                  left=0.16, right=0.97, top=0.90, bottom=0.115)

    ax_a = fig.add_subplot(gs[0])
    draw_schematic(ax_a)
    panel_letter(fig, ax_a, "a")

    ax_b = fig.add_subplot(gs[1])
    try:
        # Primary: regenerate from CSV so the x-axis label is complete
        # (the source parity_pfb.png has a truncated label).
        regenerate_parity(ax_b)
        print("panel b: regenerated from", ACC_CSV)
    except Exception as exc:  # robust: never leave the panel empty
        print("panel b regenerate failed (%s); composing PNG instead" % exc)
        ax_b.clear()
        compose_parity(ax_b)
    panel_letter(fig, ax_b, "b")

    fig.suptitle(
        "Predictors barely beat the mean; a reliability layer is the useful next step",
        fontsize=13.5, fontweight="bold", y=0.965)

    path = os.path.join(OUT, "F1_concept_parity.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024.0
    print("wrote %s (%.1f KB)" % (path, size_kb))


if __name__ == "__main__":
    main()
