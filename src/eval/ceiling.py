"""Donor-replicate and guide-replicate accuracy ceilings from DE_stats.obs.

The honest upper bound on predictor accuracy. A perturbation's measured effect carries replicate
variability. Cross-donor Pearson of per-gene log-fold-change bounds how well any predictor trained on
one view can match another view. The DE_stats authors precomputed this reproducibility per perturbation:
  donor_correlation_all_mean  : mean cross-donor-pair Pearson over ALL measured genes
  donor_correlation_hits_mean : same, restricted to that perturbation's hit genes
  guide_correlation_all       : cross-guide (technical replicate) Pearson over all genes
  guide_correlation_signif    : cross-guide Pearson over significant genes

We report these as the ceiling. Direct recomputation from by_donors.h5mu is a later verification once
that file is downloaded. Reads obs only (no layers).

Run: pixi run python src/eval/ceiling.py
"""
import os

import anndata
import numpy as np
import pandas as pd

PATH = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
OUT = "results/ceiling"


def summarize(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan, mean=np.nan)
    return dict(n=int(v.size), median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), mean=float(np.mean(v)))


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(PATH, backed="r")
    obs = a.obs.copy()
    a.file.close()

    metrics = ["donor_correlation_all_mean", "donor_correlation_hits_mean",
               "donor_correlation_hits_min", "guide_correlation_all", "guide_correlation_signif"]
    rows = []
    for m in metrics:
        for cond in ["ALL", "Rest", "Stim8hr", "Stim48hr"]:
            sub = obs if cond == "ALL" else obs[obs["culture_condition"] == cond]
            s = summarize(sub[m].to_numpy())
            rows.append(dict(metric=m, condition=cond, **s))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "replicate_ceiling.csv"), index=False)

    # The metric that matters: cross-donor Pearson on hit genes = ceiling on (1 - Pearson) error.
    donor_hits = summarize(obs["donor_correlation_hits_mean"].to_numpy())
    donor_all = summarize(obs["donor_correlation_all_mean"].to_numpy())
    guide_all = summarize(obs["guide_correlation_all"].to_numpy())

    md = []
    md.append("# Donor-replicate and guide-replicate accuracy ceiling\n")
    md.append("Source: precomputed reproducibility columns in GWCD4i.DE_stats.h5ad obs (ground truth).\n")
    md.append("The ceiling on the error metric 1 - Pearson(pred, true) is 1 - replicate_Pearson.\n")
    md.append(f"- Cross-donor Pearson on HIT genes: median {donor_hits['median']:.3f} "
              f"(IQR {donor_hits['q25']:.3f} to {donor_hits['q75']:.3f}, n={donor_hits['n']}). "
              f"Ceiling on 1-Pearson error approx {1-donor_hits['median']:.3f}.\n")
    md.append(f"- Cross-donor Pearson on ALL genes: median {donor_all['median']:.3f} "
              f"(n={donor_all['n']}). All-gene error floor approx {1-donor_all['median']:.3f}, "
              f"so the effect-vector metric on all 10,282 genes is noise-dominated.\n")
    md.append(f"- Cross-guide Pearson on ALL genes: median {guide_all['median']:.3f} (n={guide_all['n']}).\n")
    md.append("\nDecision (what, why, how): compute the delta-LFC error metric on a signal-bearing gene "
              "set (each perturbation's significant genes, or a top-variance set), not all measured genes, "
              "because the all-gene replicate ceiling is only about "
              f"{donor_all['median']:.2f} Pearson while the hit-gene ceiling is about "
              f"{donor_hits['median']:.2f}. Reporting on all genes would bury real signal under measurement "
              "noise and understate every method equally.\n")
    with open(os.path.join(OUT, "ceiling_summary.md"), "w") as f:
        f.write("".join(md))

    print(tab.to_string(index=False))
    print("\n" + "".join(md))
    print(f"wrote {OUT}/replicate_ceiling.csv and ceiling_summary.md")


if __name__ == "__main__":
    main()
