"""F7, utility and limits. Two panels:

  (a) reproducibility shortlist utility  -- composes the pre-rendered figures/reproducibility_utility.png
      (falls back to a regeneration from results/utility/reproducibility_uplift.csv if the PNG is missing).
  (b) the estimator non-transfer 4x4 matrix, regenerated from results/pertema/estimator_transfer_matrix.csv
      for the geneformer embedding. imshow of reliability_spearman by train (rows) x test (columns), every
      cell annotated. The point is negative: no off-diagonal (cross-screen) cell reaches 0.40, so the
      estimator is a within-screen tool only.

Reads only committed result files. Agg backend, 200 DPI, ASCII text only.

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F7.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

FIG = "figures"
REPRO_PNG = os.path.join(FIG, "reproducibility_utility.png")
REPRO_CSV = "results/utility/reproducibility_uplift.csv"
TRANSFER_CSV = "results/pertema/estimator_transfer_matrix.csv"
OUT = os.path.join(FIG, "F7_utility_limits.png")

# dataset order for the transfer matrix (train rows == test columns)
DSETS = ["gladstone", "replogle", "norman", "adamson"]
OFFDIAG_THRESH = 0.40


def panel_repro_compose(ax):
    """Compose the pre-rendered reproducibility-utility panel as an image."""
    img = mpimg.imread(REPRO_PNG)
    ax.imshow(img)
    ax.axis("off")


def panel_repro_regen(ax):
    """Fallback: regenerate the reproducibility-uplift bar panel from the CSV."""
    df = pd.read_csv(REPRO_CSV)
    labels = [f"{r.dst}\n{r.target}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(x, df["uplift_reliability"], yerr=[df["uplift_reliability"] - df["rel_ci_lo"],
           df["rel_ci_hi"] - df["uplift_reliability"]], color="#0072B2", capsize=3,
           label="reliability-ranked shortlist")
    ax.bar(x, df["uplift_magnitude"], color="#D55E00", alpha=0.7, label="magnitude-ranked shortlist")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("reproducibility uplift (top-k shortlist)")
    ax.set_title("Reliability-ranked shortlist raises reproducibility", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right")


def panel_transfer(ax):
    """The estimator non-transfer matrix for geneformer: reliability Spearman, train x test."""
    df = pd.read_csv(TRANSFER_CSV)
    df = df[df["embedding"] == "geneformer"]
    M = np.full((len(DSETS), len(DSETS)), np.nan)
    for _, r in df.iterrows():
        if r["train"] in DSETS and r["test"] in DSETS:
            M[DSETS.index(r["train"]), DSETS.index(r["test"])] = r["reliability_spearman"]

    # symmetric diverging scale centered at 0; span covers the full observed range
    vext = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vext, vmax=vext, aspect="equal")

    ax.set_xticks(range(len(DSETS))); ax.set_xticklabels(DSETS, fontsize=8)
    ax.set_yticks(range(len(DSETS))); ax.set_yticklabels(DSETS, fontsize=8)
    ax.set_xlabel("test screen", fontsize=9)
    ax.set_ylabel("train screen", fontsize=9)

    # annotate every cell; box the within-screen diagonal (the only regime where it works)
    for i in range(len(DSETS)):
        for j in range(len(DSETS)):
            v = M[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:+.2f}"
            tc = "white" if abs(v) > 0.6 * vext else "black"
            fw = "bold" if i == j else "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=tc, fontweight=fw)
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#000000", lw=2.0))

    # largest off-diagonal magnitude (the honest ceiling on cross-screen transfer)
    off = M.copy()
    np.fill_diagonal(off, np.nan)
    max_off = float(np.nanmax(np.abs(off)))
    ax.set_title(f"Estimator does not transfer across screens\n"
                 f"diagonal boxed (within-screen); no off-diagonal reaches {OFFDIAG_THRESH:.2f} "
                 f"(max |off-diag| = {max_off:.2f})", fontsize=9, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("reliability rank correlation (Spearman)", fontsize=8)


def add_letter(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(15, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    try:
        panel_repro_compose(ax_a)
    except Exception as e:  # noqa: BLE001  -- robustness: never leave the panel empty
        print(f"compose of {REPRO_PNG} failed ({e}); regenerating panel (a) from CSV")
        ax_a.clear()
        panel_repro_regen(ax_a)

    panel_transfer(ax_b)

    add_letter(ax_a, "a")
    add_letter(ax_b, "b")

    fig.suptitle("PertEMA F7: a within-screen reliability tool -- reproducibility uplift, "
                 "but the estimator does not transfer", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
