"""U6: winner's-curse debiasing of the oracle headroom, replacing the raw "76 percent selection bias" point
estimate with a principled debiased achievable headroom and a bootstrap confidence interval.

The per-perturbation oracle error is the empirical MINIMUM over predictors on the SAME noisy realization, so
it capitalizes on favorable draws of the shared measurement noise (the winner's curse; Efron, JASA 2014 and
2011). The independently-estimated noise floor (donor split-half, Spearman-Brown) is an unbiased lower bound on
the achievable error from data NOT used in the selection, so it serves as the cross-fit debiasing anchor: no
predictor can beat the noise floor on average, so the debiased achievable mean error is the mean floor, and any
oracle dip below it is selection bias, not achievable gain.

Debiasing is done at the AGGREGATE level (means first) because the per-perturbation floor is a noisy split-half
estimate; clamping per perturbation would inject that noise (and can push the debiased headroom spuriously
negative). We report, with a bootstrap over perturbations (B=2000) for 95 percent CIs:
  raw oracle headroom       = mean(best_fixed_err) - mean(oracle_err)   (the naive apparent achievable gain)
  debiased headroom         = mean(best_fixed_err) - mean(floor_sb)     (achievable above the noise floor)
  selection-bias fraction   = 1 - debiased / raw
  oracle-below-floor fraction (per perturbation)

Run: pixi run python src/eval/run_winner_curse.py
"""
import os

import numpy as np
import pandas as pd

PP = "results/ceiling/noise_ceiling_per_perturbation.csv"
OUT = "results/ceiling"
B = 2000
SEED = 42


def main():
    df = pd.read_csv(PP)
    need = ["floor_sb", "oracle_err", "best_fixed_err"]
    df = df.dropna(subset=need).reset_index(drop=True)
    floor = df["floor_sb"].to_numpy()
    oracle = df["oracle_err"].to_numpy()
    best = df["best_fixed_err"].to_numpy()

    below_floor = oracle < floor

    def agg(idx):
        # aggregate-level debiasing: means first, so the noisy per-perturbation floor averages out.
        rh = best[idx].mean() - oracle[idx].mean()          # raw oracle headroom
        dh = best[idx].mean() - floor[idx].mean()           # debiased: cannot beat the mean floor
        sb = 1 - (dh / rh) if rh > 0 else np.nan
        bf = below_floor[idx].mean()
        return rh, dh, sb, bf

    rh0, dh0, sb0, bf0 = agg(np.arange(len(df)))
    rng = np.random.default_rng(SEED)
    n = len(df)
    boots = np.array([agg(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    rows = [
        dict(quantity="raw_oracle_headroom", estimate=rh0, ci_lo=lo[0], ci_hi=hi[0]),
        dict(quantity="debiased_headroom_above_floor", estimate=dh0, ci_lo=lo[1], ci_hi=hi[1]),
        dict(quantity="selection_bias_fraction", estimate=sb0, ci_lo=lo[2], ci_hi=hi[2]),
        dict(quantity="oracle_below_floor_fraction", estimate=bf0, ci_lo=lo[3], ci_hi=hi[3]),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "winner_curse_debiased.csv"), index=False)
    print("=== U6 winner's-curse debiasing (Gladstone, floor-anchored, B=2000) ===")
    for _, r in res.iterrows():
        print(f"{r['quantity']:32s} {r['estimate']:.4f}  95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"\nInterpretation: of the raw oracle headroom {rh0:.4f}, only {dh0:.4f} "
          f"(95% CI [{lo[1]:.4f}, {hi[1]:.4f}]) is achievable above the noise floor; "
          f"the winner's-curse selection bias is {sb0*100:.0f}% (95% CI [{lo[2]*100:.0f}%, {hi[2]*100:.0f}%]).")
    print(f"wrote {OUT}/winner_curse_debiased.csv")


if __name__ == "__main__":
    main()
