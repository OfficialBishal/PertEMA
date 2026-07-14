"""F2, "Reliability works within context": compose the two within-context evidence panels,
(a) the selective risk-coverage curve and (b) the calibration / reliability diagram.

Primary path composes the pre-rendered high-DPI panels (figures/risk_coverage.png,
figures/calibration.png) with imshow. If a panel PNG is missing or unreadable, that single
panel is regenerated from the committed result CSVs so the figure is never left empty.

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F2.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "figures"
PANELS = {
    "a": os.path.join(FIG, "risk_coverage.png"),
    "b": os.path.join(FIG, "calibration.png"),
}


def compose_panel(ax, png_path):
    """Draw a pre-rendered panel PNG into ax, preserving its aspect. Returns True on success."""
    img = mpimg.imread(png_path)
    ax.imshow(img, aspect="equal")
    ax.axis("off")
    return True


def fallback_risk_coverage(ax):
    """Selective risk-coverage from per-perturbation scores: retaining the most-reliable fraction
    lowers mean true error (risk) far below the coverage-1 baseline, tracking the oracle ordering."""
    df = pd.read_csv("results/utility/pertema_scores.csv",
                     usecols=["true_error", "reliability"]).dropna()
    err = df["true_error"].to_numpy()
    rel = df["reliability"].to_numpy()
    cov = np.linspace(0.05, 1.0, 40)
    order_rel = np.argsort(-rel)          # most reliable first
    order_orc = np.argsort(err)           # oracle: lowest true error first
    err_rel = err[order_rel]
    err_orc = err[order_orc]
    n = len(err)
    risk_rel = np.array([err_rel[: max(1, int(c * n))].mean() for c in cov])
    risk_orc = np.array([err_orc[: max(1, int(c * n))].mean() for c in cov])
    base = err.mean()
    ax.plot(cov, risk_rel, color="#0072B2", lw=2.0, label="PertEMA reliability")
    ax.plot(cov, risk_orc, color="#009E73", lw=1.6, ls="--", label="oracle (true error)")
    ax.axhline(base, color="#555555", lw=1.2, ls=":", label="no selection")
    ax.set_xlabel("coverage (fraction retained)")
    ax.set_ylabel("selective risk (mean true error)")
    ax.set_title("Risk-coverage", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left")


def fallback_calibration(ax):
    """Isotonic recalibration collapses the raw ECE, and split-conformal coverage sits on target."""
    c = pd.read_csv("results/pertema/calibration.csv")
    preds = c["predictor"].tolist()
    x = np.arange(len(preds))
    w = 0.38
    ax.bar(x - w / 2, c["ece_raw"], w, color="#D55E00", label="ECE raw")
    ax.bar(x + w / 2, c["ece_isotonic"], w, color="#0072B2", label="ECE isotonic")
    ax.set_xticks(x)
    ax.set_xticklabels(preds, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("expected calibration error")
    tgt = float(c["conformal_target"].iloc[0])
    cov = float(c["conformal_coverage"].mean())
    ax.set_title(f"Calibration (conformal coverage {cov:.3f} at target {tgt:.2f})",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right")


FALLBACK = {"a": fallback_risk_coverage, "b": fallback_calibration}


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.02, right=0.98, top=0.86, bottom=0.04)
    for i, letter in enumerate(["a", "b"]):
        ax = fig.add_subplot(gs[0, i])
        png = PANELS[letter]
        composed = False
        if os.path.exists(png):
            try:
                composed = compose_panel(ax, png)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, never leave a blank panel
                print(f"compose failed for {png} ({exc}); regenerating from CSV")
        if not composed:
            ax.axis("on")
            FALLBACK[letter](ax)
        # bold lowercase panel letter, top-left in axes fraction coords
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left")
    fig.suptitle("Post-hoc reliability beats heuristics and calibrates",
                 fontsize=13, fontweight="bold", y=0.965)
    out = os.path.join(FIG, "F2_reliability_within.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    sz = os.path.getsize(out)
    print(f"wrote {out} ({sz} bytes)")
    assert sz > 20 * 1024, f"output too small: {sz} bytes"


if __name__ == "__main__":
    main()
