"""Headline figures for the context-transfer contribution.

Panel A: area under the risk-coverage curve (lower is better) on the transfer task, PertEMA versus the
effect-magnitude and training-set-similarity heuristics, no-selection, and the oracle. Panel B: the 3x3
transfer-difficulty map (mean 1 - Pearson error) showing that transfer degrades most between resting and
stimulated states. Reads only committed/regenerable result files.

Run: pixi run python src/pertema/figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "figures"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
# colorblind-safe (Okabe-Ito): grey baselines, blue PertEMA, green oracle
CB = {"base": "#999999", "pertema": "#0072B2", "oracle": "#009E73"}


def val(s):
    return float(str(s).split("+/-")[0])


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = pd.read_csv("results/pertema/transfer_estimator_summary.csv")
    row = summ[summ["predictor"] == "mean_condition"].iloc[0]
    labels = ["no selection", "similarity\nheuristic", "magnitude\nheuristic", "PertEMA", "oracle"]
    vals = [val(row["aurc_noselect"]), val(row["aurc_similarity"]), val(row["aurc_magnitude"]),
            val(row["aurc_est"]), val(row["aurc_oracle"])]
    colors = [CB["base"], CB["base"], CB["base"], CB["pertema"], CB["oracle"]]

    tr = pd.read_csv("results/transfer/transfer_errors_seed42.csv")
    mp = tr[tr["predictor"] == "mean_condition"]
    piv = (mp.groupby(["src", "dst"])["transfer_err"].mean().unstack()
           .reindex(index=CONDS, columns=CONDS))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    bars = axA.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    axA.set_ylim(0.80, 0.89)
    axA.set_ylabel("Area under risk-coverage curve\n(lower is better)")
    axA.set_title("A. PertEMA beats reliability heuristics under context transfer", fontsize=10, loc="left")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)
    axA.axhline(vals[-1], color=CB["oracle"], ls=":", lw=1, zorder=0)
    axA.tick_params(axis="x", labelsize=8)

    im = axB.imshow(piv.values, cmap="YlOrRd", vmin=0.80, vmax=0.90)
    axB.set_xticks(range(3)); axB.set_xticklabels(CONDS, fontsize=8)
    axB.set_yticks(range(3)); axB.set_yticklabels(CONDS, fontsize=8)
    axB.set_xlabel("predicted-in context (destination)")
    axB.set_ylabel("trained-in context (source)")
    axB.set_title("B. Transfer error tracks activation-state distance", fontsize=10, loc="left")
    for i in range(3):
        for j in range(3):
            axB.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                     color="black" if piv.values[i, j] < 0.87 else "white", fontsize=9)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="1 - Pearson error")

    fig.suptitle("PertEMA: post-hoc context-transfer reliability for perturbation predictors "
                 "(Gladstone CD4 T cell Perturb-seq)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, "headline_transfer.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
