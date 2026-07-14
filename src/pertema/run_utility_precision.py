"""E6: validation-shortlist precision-at-k of reliability-ranked vs magnitude-ranked shortlists against a
strictly held-out-donor reproducibility truth (P6), with a hit-threshold sweep and significance tests.

No orthogonal arrayed set for the Gladstone screen is on disk, so per the specification we use the strictest
within-resource test, cross-donor reproducibility, and label it a HELD-OUT-DONOR test, NOT prospective. A
REPRODUCIBLE HIT is a perturbation whose true destination-context effect is in the top q of cross-donor
reproducibility (donor_correlation_hits_mean, a model-independent property used ONLY as the validation
target, never a feature). We report precision-at-k for reliability, magnitude, and random at hit-thresholds
q in {median, top-quartile, top-decile}, the threshold-free Spearman of each score against the reproducibility
target (the estimate that does not depend on a hit cutoff), a bootstrap two-sided assessment of the
reliability-minus-magnitude gap, and a hypergeometric test of magnitude-vs-chance. Leakage-safe: scores are
out-of-fold, the reproducibility target is never a feature.

Run: pixi run python src/pertema/run_utility_precision.py
"""
import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

SCORES = "results/utility/pertema_scores.csv"
DE = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
OUT = "results/utility"
QS = [0.50, 0.75, 0.90]
KS = [0.05, 0.10, 0.20]
B = 2000


def obs_col(obs, name):
    g = obs[name]
    if isinstance(g, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        return cats[g["codes"][:]]
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v]) if v.dtype.kind in "SO" else v


def pk(score, hit, k):
    return float(hit[np.argsort(-score)[:k]].sum()), k   # (hits in top-k, k)


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_csv(SCORES)
    with h5py.File(DE, "r") as f:
        obs = f["obs"]
        de = pd.DataFrame({"gene_ens": obs_col(obs, "target_contrast"),
                           "cond": obs_col(obs, "culture_condition"),
                           "donor": obs_col(obs, "donor_correlation_hits_mean")})

    rows, sp_rows = [], []
    for dst in ["Stim48hr", "Stim8hr"]:
        d = (sc[(sc["src"] == "Rest") & (sc["dst"] == dst)]
             .groupby("gene_ens").agg(rel=("reliability", "mean"), mag=("pred_magnitude", "mean")).reset_index())
        m = d.merge(de[de["cond"] == dst][["gene_ens", "donor"]].dropna(), on="gene_ens", how="inner")
        n = len(m); rel = m["rel"].to_numpy(); mag = m["mag"].to_numpy(); dv = m["donor"].to_numpy()
        # threshold-free Spearman of each score against the reproducibility target (no hit cutoff)
        sr, pr = spearmanr(rel, dv); sm, pm = spearmanr(mag, dv)
        sp_rows.append(dict(dst=dst, n=n, spearman_rel_vs_repro=float(sr), p_rel=float(pr),
                            spearman_mag_vs_repro=float(sm), p_mag=float(pm)))
        rng = np.random.default_rng(0)
        for q in QS:
            thr = np.quantile(dv, q); hit = (dv >= thr).astype(float); base = hit.mean(); n_hit = int(hit.sum())
            for kf in KS:
                k = max(1, int(kf * n))
                hr, _ = pk(rel, hit, k); hm, _ = pk(mag, hit, k)
                p_rel, p_mag, p_rnd = hr / k, hm / k, base
                # magnitude-vs-chance one-sided (lower tail) hypergeometric p
                p_mag_below = float(hypergeom.cdf(hm, n, n_hit, k))
                grm = np.empty(B)
                for b in range(B):
                    bi = rng.integers(0, n, n); hb, rb, mb = hit[bi], rel[bi], mag[bi]
                    grm[b] = hb[np.argsort(-rb)[:k]].mean() - hb[np.argsort(-mb)[:k]].mean()
                gap = p_rel - p_mag
                rows.append(dict(dst=dst, hit_q=q, base_rate=round(base, 3), k_frac=kf, k=k,
                                 precision_reliability=round(p_rel, 3), precision_magnitude=round(p_mag, 3),
                                 precision_random=round(p_rnd, 3),
                                 gap_rel_minus_mag=round(gap, 4),
                                 gap_ci_lo=round(float(np.percentile(grm, 2.5)), 4),
                                 gap_ci_hi=round(float(np.percentile(grm, 97.5)), 4),
                                 gap_two_sided_sig=bool(np.percentile(grm, 2.5) > 0 or np.percentile(grm, 97.5) < 0),
                                 p_magnitude_below_chance=round(p_mag_below, 3)))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "precision_at_k.csv"), index=False)
    spdf = pd.DataFrame(sp_rows); spdf.to_csv(os.path.join(OUT, "utility_spearman.csv"), index=False)

    print("=== E6 threshold-free Spearman of each score vs cross-donor reproducibility (the cutoff-free anchor) ===")
    print(spdf.round(4).to_string(index=False))
    print("\n=== precision-at-k, hit-threshold sweep (q=0.5/0.75/0.9), two-sided gap significance ===")
    print(res[res["k_frac"] == 0.10].to_string(index=False))
    nsig = int(res["gap_two_sided_sig"].sum())
    print(f"\nreliability-minus-magnitude gap two-sided-significant in {nsig} of {len(res)} configurations.")
    print(f"wrote {OUT}/precision_at_k.csv, utility_spearman.csv")


if __name__ == "__main__":
    main()
