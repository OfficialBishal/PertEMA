"""F5: Noise ceiling and the winner's curse.

Panel a: shared-noise fraction f_noise across the four datasets (mechanism_metrics.csv).
Panel b: winner's-curse debiasing (winner_curse_debiased.csv). Raw oracle headroom vs
debiased headroom-above-floor with 95% CI error bars; the reduction is the selection-bias
fraction 0.76 [0.70, 0.82].

Both panels are regenerated directly from the committed result CSVs (no pre-rendered panel
PNG exists for either). If a panel's CSV is missing it is drawn as a labeled placeholder so
the composite is never left empty.

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F5.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "figures"
MECH_CSV = "results/benchmark/mechanism_metrics.csv"
WC_CSV = "results/ceiling/winner_curse_debiased.csv"

# colorblind-safe (Okabe-Ito)
CB = {"base": "#999999", "primary": "#0072B2", "raw": "#D55E00", "debiased": "#009E73"}

# clean display names for the four datasets
PRETTY = {
    "Gladstone_CD4": "Gladstone\nCD4 T",
    "Replogle_K562": "Replogle\nK562",
    "Norman_K562": "Norman\nK562",
    "Adamson_K562": "Adamson\nK562",
}


def panel_letter(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")


def placeholder(ax, msg):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="firebrick", wrap=True)


def panel_a(ax):
    """Shared-noise fraction across the four datasets (bar)."""
    df = pd.read_csv(MECH_CSV)
    df = df.sort_values("shared_noise_fraction_f", ascending=False).reset_index(drop=True)
    labels = [PRETTY.get(d, d) for d in df["dataset"]]
    vals = df["shared_noise_fraction_f"].to_numpy(dtype=float)
    # highlight the dataset that anchors the winner's-curse analysis (Gladstone CD4)
    colors = [CB["primary"] if d == "Gladstone_CD4" else CB["base"] for d in df["dataset"]]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Shared-noise fraction f_noise")
    ax.set_title("Deployable-predictor error is dominated by shared noise",
                 fontsize=10, loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.axhline(1.0, color="#444444", ls=":", lw=0.8, zorder=0)
    ax.text(2.0, 1.008, "all error is shared noise", ha="center", va="bottom",
            fontsize=7, color="#444444")
    # legend for the highlighted anchor dataset (placed over the short bars)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=CB["primary"], edgecolor="black",
                             label="winner's-curse anchor (panel b)")],
              fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.86),
              frameon=False)


def panel_b(ax):
    """Winner's-curse debiasing: raw vs debiased-above-floor headroom with 95% CI."""
    df = pd.read_csv(WC_CSV).set_index("quantity")
    raw = df.loc["raw_oracle_headroom"]
    deb = df.loc["debiased_headroom_above_floor"]
    sbf = df.loc["selection_bias_fraction"]

    est = np.array([raw["estimate"], deb["estimate"]], dtype=float)
    lo = np.array([raw["ci_lo"], deb["ci_lo"]], dtype=float)
    hi = np.array([raw["ci_hi"], deb["ci_hi"]], dtype=float)
    yerr = np.vstack([est - lo, hi - est])

    x = np.arange(2)
    labels = ["raw oracle\nheadroom", "debiased\n(above floor)"]
    colors = [CB["raw"], CB["debiased"]]
    bars = ax.bar(x, est, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    ax.errorbar(x, est, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2,
                capsize=5, capthick=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Oracle headroom over best fixed predictor")
    ax.set_ylim(0, est[0] * 1.55)
    ax.set_title("Most oracle headroom is winner's-curse selection bias",
                 fontsize=10, loc="left")
    for b, v, h in zip(bars, est, hi):
        ax.text(b.get_x() + b.get_width() / 2, h + est[0] * 0.03, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    # annotate the selection-bias fraction as the reduction from raw to debiased,
    # drawn in the empty gap between the two bars so nothing is clipped
    xa = 0.5
    ax.annotate("", xy=(xa, est[1]), xytext=(xa, est[0]),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    sb_txt = (f"selection bias {sbf['estimate']:.2f} "
              f"[{sbf['ci_lo']:.2f}, {sbf['ci_hi']:.2f}]\nof raw headroom")
    ax.text(xa, est[0] * 1.34, sb_txt, ha="center", va="bottom",
            fontsize=8.5, color="#333333")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 4.7))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.08, right=0.97,
                          top=0.84, bottom=0.16)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    try:
        panel_a(axA)
    except Exception as e:  # never leave the panel empty
        placeholder(axA, f"panel a unavailable:\n{e}")
    panel_letter(axA, "a")

    try:
        panel_b(axB)
    except Exception as e:
        placeholder(axB, f"panel b unavailable:\n{e}")
    panel_letter(axB, "b")

    fig.suptitle("The deployable predictor is noise-limited; "
                 "most oracle headroom is selection bias",
                 fontsize=12.5, fontweight="bold")
    path = os.path.join(OUT, "F5_noise_ceiling_winnercurse.png")
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
