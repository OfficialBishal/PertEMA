"""U10: apply and report a multiple-comparison correction across the paper's headline hypothesis tests.

The manuscript reports many significance tests. To avoid an inflated family-wise false-positive rate we apply
Benjamini-Hochberg FDR control across the directional hypothesis tests whose p-values are traceable to a result
file, and report which survive at FDR 0.05. Tests reported as confidence intervals rather than p-values
(winner's-curse debiasing, reproducibility uplift CIs) are listed separately since BH operates on p-values.

Run: pixi run python src/eval/run_multiple_comparison.py
"""
import os

import numpy as np
import pandas as pd

OUT = "results"

# (claim, p_value, source). Directional hypothesis tests with a traceable p-value.
TESTS = [
    ("PertEMA reliability beats magnitude heuristic (AURC, paired cluster bootstrap over genes)", 5e-4,
     "research-log / significance.py"),
    ("Magnitude anti-selects reproducibility, Stim48hr (threshold-free Spearman)", 0.0025,
     "results/utility/utility_spearman.csv"),
    ("Magnitude anti-selects reproducibility, Stim8hr (threshold-free Spearman)", 0.076,
     "results/utility/utility_spearman.csv"),
    ("Reliability enriches reproducibility, Stim48hr (threshold-free Spearman, expected null)", 0.59,
     "results/utility/utility_spearman.csv"),
    ("Reliability enriches reproducibility, Stim8hr (threshold-free Spearman)", 0.069,
     "results/utility/utility_spearman.csv"),
    ("Reliability underperforms magnitude at regulator recovery, pathway set (degree-matched null)", 0.007,
     "results/utility/UTILITY_FINDINGS.md"),
    ("Reliability underperforms magnitude at regulator recovery, functional set (degree-matched null)", 0.141,
     "results/utility/UTILITY_FINDINGS.md"),
    ("Magnitude below chance at reproducibility precision, Stim48hr q0.50 (hypergeometric)", 0.004,
     "results/utility/precision_at_k.csv"),
    ("Magnitude below chance at reproducibility precision, Stim48hr q0.90 (hypergeometric)", 0.035,
     "results/utility/precision_at_k.csv"),
    ("Magnitude below chance at reproducibility precision, Stim48hr q0.75 (hypergeometric)", 0.069,
     "results/utility/precision_at_k.csv"),
]

CI_TESTS = [
    ("Winner's-curse selection-bias fraction 0.76 (95% CI 0.70-0.82)", "results/ceiling/winner_curse_debiased.csv"),
    ("Reproducibility shortlist uplift over magnitude +0.049 to +0.063 (frac>0 0.99+)",
     "results/utility/reproducibility_uplift.csv"),
    ("Adamson routing routed-minus-bestfixed +0.016 (frac-better 0.339, below 0.95 kill criterion)",
     "results/pertema/router_adamson.csv"),
]


def bh(pvals, alpha=0.05):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1)) / n
    below = ranked <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    crit = ranked[k - 1] if k > 0 else 0.0
    reject = p <= crit
    # BH-adjusted q-values (monotone)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return reject, q


def main():
    claims = [t[0] for t in TESTS]
    pvals = [t[1] for t in TESTS]
    srcs = [t[2] for t in TESTS]
    reject, q = bh(pvals, alpha=0.05)
    df = pd.DataFrame(dict(claim=claims, p_value=pvals, bh_qvalue=np.round(q, 4),
                           significant_at_fdr_005=reject, source=srcs)).sort_values("p_value")
    df.to_csv(os.path.join(OUT, "stats_multiple_comparison.csv"), index=False)
    ci = pd.DataFrame([dict(claim=c, source=s) for c, s in CI_TESTS])
    ci.to_csv(os.path.join(OUT, "stats_ci_reported.csv"), index=False)

    print("=== U10 Benjamini-Hochberg FDR control across headline directional tests (alpha 0.05) ===")
    print(df.to_string(index=False))
    print(f"\n{int(reject.sum())} of {len(TESTS)} tests survive at FDR 0.05.")
    print("Surviving (robust) claims:")
    for _, r in df[df.significant_at_fdr_005].iterrows():
        print(f"  - {r['claim']} (q {r['bh_qvalue']})")
    print("\nCI-reported results (not p-value based, listed for completeness):")
    for c, s in CI_TESTS:
        print(f"  - {c}")
    print(f"\nwrote {OUT}/stats_multiple_comparison.csv, stats_ci_reported.csv")


if __name__ == "__main__":
    main()
