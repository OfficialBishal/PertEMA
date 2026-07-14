"""R3 utility, the positive result: does ranking a validation shortlist by PertEMA reliability recover more
REPRODUCIBLE perturbations than the default effect-magnitude ranking?

A wet lab spends a fixed validation budget on a shortlist of predicted perturbations and wants that shortlist
enriched for perturbations whose true effect is real and reproducible (high cross-donor or cross-guide
correlation), not experimental noise. This is a decision-relevant question inside PertEMA's validated
aggregate-ranking regime, and unlike known-regulator recovery (which is a negative, see UTILITY_FINDINGS.md)
it comes out positive.

Metric: among the top 10 percent of perturbations by a score, the mean cross-donor (or cross-guide)
reproducibility of the TRUE destination-context effect, minus the global mean (the random baseline, 0 by
construction). Compared for PertEMA reliability vs the effect-magnitude heuristic. Bootstrap over
perturbations for a 95 percent CI and the fraction of draws with a positive uplift / gap.

Leakage-safe: reliability and pred_magnitude come from the out-of-fold, gene-disjoint kNN discovery
predictor and transfer estimator that never see any test perturbation's true error; the reproducibility
columns are properties of the TRUE measured effect used ONLY as the validation target, never as estimator
features (invariant: true-effect-derived columns may be validation targets but never features).

Run: pixi run python src/pertema/run_utility_reproducibility.py
"""
import os

import h5py
import numpy as np
import pandas as pd

SCORES = "results/utility/pertema_scores.csv"
DE = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
OUT = "results/utility"
Q = 0.10
B = 2000


def obs_col(obs, name):
    g = obs[name]
    if isinstance(g, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        return cats[g["codes"][:]]
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v]) if v.dtype.kind in "SO" else v


def uplift(score, target, k):
    idx = np.argsort(-score)[:k]
    return float(target[idx].mean() - target.mean())


def main():
    sc = pd.read_csv(SCORES)
    with h5py.File(DE, "r") as f:               # read only /obs, never X
        obs = f["obs"]
        de = pd.DataFrame({"gene_ens": obs_col(obs, "target_contrast"),
                           "cond": obs_col(obs, "culture_condition"),
                           "donor": obs_col(obs, "donor_correlation_hits_mean"),
                           "guide": obs_col(obs, "guide_correlation_all")})

    rows = []
    for dst in ["Stim48hr", "Stim8hr"]:
        d = (sc[(sc["src"] == "Rest") & (sc["dst"] == dst)]
             .groupby("gene_ens").agg(rel=("reliability", "mean"), mag=("pred_magnitude", "mean")).reset_index())
        for target in ["donor", "guide"]:
            ded = de[de["cond"] == dst][["gene_ens", target]].dropna()
            m = d.merge(ded, on="gene_ens", how="inner")
            n = len(m); k = max(1, int(Q * n))
            rel = m["rel"].to_numpy(); mag = m["mag"].to_numpy(); tv = m[target].to_numpy()
            u_rel, u_mag = uplift(rel, tv, k), uplift(mag, tv, k)
            rng = np.random.default_rng(0)
            br, bgap = np.empty(B), np.empty(B)
            for b in range(B):
                bi = rng.integers(0, n, n); tvb = tv[bi]; gb = tvb.mean()
                ur = tvb[np.argsort(-rel[bi])[:k]].mean() - gb
                um = tvb[np.argsort(-mag[bi])[:k]].mean() - gb
                br[b] = ur; bgap[b] = ur - um
            rows.append(dict(dst=dst, target=target, n=n, k=k,
                             uplift_reliability=u_rel, uplift_magnitude=u_mag,
                             rel_ci_lo=float(np.percentile(br, 2.5)), rel_ci_hi=float(np.percentile(br, 97.5)),
                             rel_frac_pos=float((br > 0).mean()),
                             gap_rel_minus_mag=u_rel - u_mag, gap_frac_pos=float((bgap > 0).mean())))
            print(f"{dst}/{target}: n={n} k={k} | reliability uplift {u_rel:+.4f} "
                  f"CI[{rows[-1]['rel_ci_lo']:+.4f},{rows[-1]['rel_ci_hi']:+.4f}] frac>0={rows[-1]['rel_frac_pos']:.3f}"
                  f" | magnitude {u_mag:+.4f} | gap {u_rel-u_mag:+.4f} frac>0={rows[-1]['gap_frac_pos']:.3f}",
                  flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "reproducibility_uplift.csv"), index=False)
    print(f"\nwrote {OUT}/reproducibility_uplift.csv")


if __name__ == "__main__":
    main()
