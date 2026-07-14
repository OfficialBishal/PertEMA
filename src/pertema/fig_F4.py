"""F4: The error-correlation law.

Composes three CSV-regenerated panels into a labeled, publication-quality figure:
  (a) 10x10 predictor error-correlation heatmap  [error_corr_pearson.csv]
  (b) per-dataset error correlation rho and effective independent methods N_eff  [mechanism_metrics.csv]
  (c) raw rho vs model-specific residual rho, showing shared-difficulty domination  [rho_artifact_control.csv]

ASCII only. Agg backend. 200 DPI.
Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F4.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

OUT = "figures"
EC = "results/error_correlation/error_corr_pearson.csv"
MECH = "results/benchmark/mechanism_metrics.csv"
ARTI = "results/error_correlation/rho_artifact_control.csv"

# Okabe-Ito colorblind-safe palette
C_RHO = "#0072B2"      # blue  - raw / rho
C_NEFF = "#D55E00"     # orange - N_eff
C_RESID = "#009E73"    # green - model-specific residual rho

# short display names for the 10 predictors (order taken from the CSV header)
PRETTY = {
    "mean_condition": "mean-cond",
    "mean_global": "mean-glob",
    "knn_coexpr_k25": "kNN-coexpr",
    "ridge_embed": "ridge-embed",
    "gene2vec_ridge": "g2v-ridge",
    "gene2vec_knn": "g2v-kNN",
    "geneformer_ridge": "GF-ridge",
    "geneformer_knn": "GF-kNN",
    "scgpt_ridge": "scGPT-ridge",
    "scgpt_knn": "scGPT-kNN",
}

DS_SHORT = {
    "Gladstone_CD4": "Gladstone\nCD4",
    "Norman_K562": "Norman\nK562",
    "Replogle_K562": "Replogle\nK562",
    "Adamson_K562": "Adamson\nK562",
}


def panel_letter(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def panel_a(ax, fig):
    df = pd.read_csv(EC, index_col=0)
    labels = [PRETTY.get(c, c) for c in df.columns]
    M = df.values.astype(float)
    im = ax.imshow(M, cmap="magma", vmin=0.85, vmax=1.0, aspect="equal")
    n = M.shape[0]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if v < 0.93 else "black")
    ax.set_title("Pairwise error correlation across 10 predictors\n(Gladstone CD4, per-perturbation |error|)",
                 fontsize=9.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson correlation of errors", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    panel_letter(ax, "a", dx=-0.22)


def panel_b(ax):
    df = pd.read_csv(MECH)
    df = df.sort_values("error_correlation_rho", ascending=False).reset_index(drop=True)
    labels = [DS_SHORT.get(d, d) for d in df["dataset"]]
    x = np.arange(len(df))
    rho = df["error_correlation_rho"].values
    neff = df["N_eff"].values

    bars = ax.bar(x, rho, color=C_RHO, edgecolor="black", linewidth=0.6, width=0.62, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Error correlation rho", color=C_RHO, fontsize=9)
    ax.tick_params(axis="y", labelcolor=C_RHO, labelsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for b, v in zip(bars, rho):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                fontsize=7.5, color=C_RHO)

    ax2 = ax.twinx()
    ax2.plot(x, neff, "o-", color=C_NEFF, markersize=7, linewidth=1.8, zorder=4,
             markeredgecolor="black", markeredgewidth=0.5)
    ax2.set_ylim(0.9, 2.3)
    ax2.set_ylabel("N_eff (independent methods)", color=C_NEFF, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_NEFF, labelsize=8)
    ax2.axhline(1.0, color=C_NEFF, ls=":", lw=1, zorder=1)
    for xi, v in zip(x, neff):
        ax2.text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5, color=C_NEFF)

    ax.set_title("High rho collapses 10 predictors to ~1 independent method",
                 fontsize=9.5, loc="left")
    panel_letter(ax, "b")


def panel_c(ax):
    df = pd.read_csv(ARTI)
    # keep panel-b order (by raw rho descending)
    df = df.sort_values("raw_rho", ascending=False).reset_index(drop=True)
    labels = [DS_SHORT.get(d, d) for d in df["dataset"]]
    x = np.arange(len(df))
    w = 0.38
    raw = df["raw_rho"].values
    resid = df["model_specific_rho"].values

    b1 = ax.bar(x - w / 2, raw, width=w, color=C_RHO, edgecolor="black",
                linewidth=0.6, label="raw rho", zorder=3)
    b2 = ax.bar(x + w / 2, resid, width=w, color=C_RESID, edgecolor="black",
                linewidth=0.6, label="model-specific residual rho", zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Error correlation rho", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for bars, vals in ((b1, raw), (b2, resid)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=7, color="black")
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.set_title("Removing shared per-perturbation difficulty collapses rho",
                 fontsize=9.5, loc="left")
    panel_letter(ax, "c")


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})

    fig = plt.figure(figsize=(13.2, 6.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0],
                           hspace=0.42, wspace=0.30, left=0.09, right=0.965,
                           top=0.88, bottom=0.09)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    for ax, fn in ((axA, lambda a: panel_a(a, fig)), (axB, panel_b), (axC, panel_c)):
        try:
            fn(ax)
        except Exception as e:  # robustness: never leave a panel empty
            ax.axis("off")
            ax.text(0.5, 0.5, f"panel failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            print(f"WARNING panel failed: {e}")

    fig.suptitle("Predictor errors are correlated, dominated by shared per-perturbation difficulty",
                 fontsize=13, fontweight="bold", y=0.965)

    path = os.path.join(OUT, "F4_error_correlation_law.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    kb = os.path.getsize(path) / 1024.0
    print(f"wrote {path} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
