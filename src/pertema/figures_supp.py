"""Supplementary figures for R7: risk-coverage curves, calibration / reliability diagram, the mechanism
figure, and the downstream reproducibility-utility figure. Reads only committed/regenerable result files.
Colorblind-safe (Okabe-Ito), reliability never encoded by color alone (labels on every series).

Run: pixi run python src/pertema/figures_supp.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from metrics import risk_coverage_curve, aurc   # noqa: E402

OUT = "figures"
CB = {"pertema": "#0072B2", "magnitude": "#D55E00", "similarity": "#CC79A7",
      "oracle": "#009E73", "base": "#999999", "reliability": "#0072B2"}


def fig_risk_coverage():
    sc = pd.read_csv("results/utility/pertema_scores.csv")
    d = (sc[(sc["src"] == "Rest") & (sc["dst"] == "Stim48hr")]
         .groupby("gene").agg(err=("true_error", "mean"), rel=("reliability", "mean"),
                              mag=("pred_magnitude", "mean"), sim=("similarity", "mean")).reset_index())
    err = d["err"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    # magnitude and similarity heuristics get their better orientation (as in best_aurc)
    def better(score):
        return score if aurc(err, score) <= aurc(err, -score) else -score
    series = [("PertEMA reliability", d["rel"].to_numpy(), CB["pertema"], "-"),
              ("magnitude heuristic", better(d["mag"].to_numpy()), CB["magnitude"], "--"),
              ("similarity heuristic", better(d["sim"].to_numpy()), CB["similarity"], "--"),
              ("oracle", -err, CB["oracle"], ":")]
    for label, score, color, ls in series:
        cov, risk = risk_coverage_curve(err, score)
        ax.plot(cov, risk, color=color, ls=ls, lw=1.8, label=f"{label} (AURC {aurc(err, score):.3f})")
    ax.axhline(err.mean(), color=CB["base"], lw=1, ls="-", label=f"no selection ({err.mean():.3f})")
    ax.set_xlabel("coverage (fraction of predictions kept, most reliable first)")
    ax.set_ylabel("risk (mean 1 - Pearson error of kept predictions)")
    ax.set_title("Risk-coverage on context transfer (Rest to Stim48hr, kNN predictor)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    fig.tight_layout(); p = os.path.join(OUT, "risk_coverage.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


def fig_calibration():
    sc = pd.read_csv("results/utility/pertema_scores.csv")
    d = (sc[(sc["src"] == "Rest") & (sc["dst"] == "Stim48hr")]
         .groupby("gene").agg(err=("true_error", "mean"), rel=("reliability", "mean")).reset_index())
    pred_err = -d["rel"].to_numpy()               # PertEMA predicted error
    true_err = d["err"].to_numpy()
    order = np.argsort(pred_err)
    bins = np.array_split(order, 10)
    px = [pred_err[b].mean() for b in bins]
    py = [true_err[b].mean() for b in bins]
    cal = pd.read_csv("results/pertema/calibration.csv").set_index("predictor")
    ece = cal.loc["mean_condition", "ece_isotonic"]; cov = cal.loc["mean_condition", "conformal_coverage"]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    lo = min(min(px), min(py)); hi = max(max(px), max(py))
    ax.plot([lo, hi], [lo, hi], color=CB["base"], ls=":", lw=1, label="perfect calibration")
    ax.plot(px, py, "o-", color=CB["pertema"], lw=1.6, ms=5, label="PertEMA (binned)")
    ax.set_xlabel("mean predicted error (bin)")
    ax.set_ylabel("mean realized error (bin)")
    ax.set_title("Reliability diagram, transfer estimator", fontsize=10, loc="left")
    ax.text(0.03, 0.92, f"isotonic ECE {ece:.4f}\nconformal coverage {cov:.3f} (target 0.90)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="white", ec=CB["base"], alpha=0.9))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); p = os.path.join(OUT, "calibration.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


def fig_mechanism():
    g = pd.read_csv("results/pertema/mechanism_feature_groups.csv").sort_values("delta_vs_full", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(g["group"], g["delta_vs_full"], color=CB["pertema"], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("AURC increase when the feature group is dropped (larger = more load-bearing)")
    ax.set_title("What drives transfer reliability (drop-one-group ablation)", fontsize=10, loc="left")
    ax.axvline(0, color=CB["base"], lw=0.8)
    fig.tight_layout(); p = os.path.join(OUT, "mechanism.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


def fig_reproducibility_utility():
    r = pd.read_csv("results/utility/reproducibility_uplift.csv")
    r = r[r["target"] == "donor"]                 # cross-donor reproducibility, the primary target
    labels = [f"Rest to\n{d}" for d in r["dst"]]
    x = np.arange(len(labels)); w = 0.36
    rel = r["uplift_reliability"].to_numpy(); mag = r["uplift_magnitude"].to_numpy()
    yerr = np.vstack([rel - r["rel_ci_lo"].to_numpy(), r["rel_ci_hi"].to_numpy() - rel])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.bar(x - w / 2, rel, w, yerr=yerr, capsize=4, color=CB["pertema"], edgecolor="black",
           linewidth=0.5, label="PertEMA reliability")
    ax.bar(x + w / 2, mag, w, color=CB["magnitude"], edgecolor="black", linewidth=0.5,
           label="magnitude heuristic")
    ax.axhline(0, color=CB["base"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cross-donor reproducibility uplift\nof the top-10% shortlist over the global mean")
    ax.set_title("Downstream utility: reliability yields a more reproducible shortlist", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "reproducibility_utility.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


def fig_parity():
    """PF-B (after Ahlmann-Eltze Fig 1b): error relative to the mean baseline per predictor, line at 1.0. The
    full 12-predictor roster when the foundation-model parity is present, with provenance shown by marker."""
    p = "results/parity/parity_gladstone.csv"
    if not os.path.exists(p):
        print("skip parity figure (results/parity/parity_gladstone.csv not built)"); return
    d = pd.read_csv(p)[["predictor", "error_rel_to_mean", "provenance"]]
    pf = "results/parity/parity_gladstone_foundation.csv"
    if os.path.exists(pf):
        fnd = pd.read_csv(pf)[["predictor", "error_rel_to_mean", "provenance"]]
        d = pd.concat([d, fnd[fnd["predictor"] != "mean_condition"]], ignore_index=True)
    d = d.drop_duplicates("predictor").sort_values("error_rel_to_mean")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(d))
    for prov, marker, lab in [("CLEAN", "o", "clean (trained from scratch)"),
                              ("UNKNOWN-OVERLAP", "s", "frozen-adapted foundation model")]:
        sub = d[d["provenance"] == prov]; yy = [list(d["predictor"]).index(g) for g in sub["predictor"]]
        colors = [CB["base"] if v >= 1.0 else CB["oracle"] for v in sub["error_rel_to_mean"]]
        ax.scatter(sub["error_rel_to_mean"], yy, c=colors, s=72, marker=marker, edgecolor="black",
                   linewidth=0.6, zorder=3, label=lab)
    ax.axvline(1.0, color=CB["magnitude"], lw=1.4, ls="--", label="mean baseline")
    ax.set_yticks(y); ax.set_yticklabels(d["predictor"], fontsize=9)
    ax.set_xlabel("error relative to the mean baseline (L2, top-1000 expressed; > 1 is worse than the mean)")
    ax.set_title(f"Parity: none of {len(d)} predictors beats the mean, incl. foundation models (Gladstone)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); pth = os.path.join(OUT, "parity_pfb.png"); fig.savefig(pth, dpi=150); plt.close(fig)
    print(f"wrote {pth}")


def fig_parity_datasets():
    """Cross-dataset parity: error relative to the mean baseline of the best non-mean predictor on each of the
    four datasets, showing the mean is hard to beat but not always best (kNN beats on the essential-gene screens)."""
    p = "results/benchmark/accuracy_metrics.csv"
    if not os.path.exists(p):
        print("skip cross-dataset parity figure (accuracy_metrics.csv not built)"); return
    a = pd.read_csv(p)
    a = a[a["metric"] == "error_rel_to_mean"]
    order = ["Gladstone_CD4", "Norman_K562", "Replogle_K562", "Adamson_K562"]
    ds = [d for d in order if d in set(a["dataset"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(ds))
    best = [a[(a["dataset"] == d) & (~a["predictor"].str.startswith("mean"))].sort_values("value").iloc[0]
            for d in ds]
    vals = [b["value"] for b in best]
    colors = [CB["base"] if v >= 1.0 else CB["oracle"] for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    for yi, b in zip(y, best):
        ax.text(b["value"] + 0.003, yi, f"{b['predictor']} {b['value']:.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color=CB["magnitude"], lw=1.4, ls="--", label="mean baseline")
    ax.set_yticks(y); ax.set_yticklabels([d.replace("_", " ") for d in ds], fontsize=9)
    ax.set_xlabel("best non-mean predictor, error relative to the mean baseline (< 1 beats the mean)")
    ax.set_title("Parity across four datasets: the mean is hard to beat, not always best", fontsize=9.5, loc="left")
    ax.set_xlim(0.9, max(vals) + 0.09); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); pth = os.path.join(OUT, "parity_datasets.png"); fig.savefig(pth, dpi=150); plt.close(fig)
    print(f"wrote {pth}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_risk_coverage()
    fig_calibration()
    fig_mechanism()
    fig_reproducibility_utility()
    fig_parity()
    fig_parity_datasets()
    print("R7 supplementary figures complete")


if __name__ == "__main__":
    main()
