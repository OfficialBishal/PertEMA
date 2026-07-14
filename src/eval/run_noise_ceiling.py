"""E3: per-perturbation noise-ceiling estimation on Gladstone (converts "near the noise floor" from a
claim into a measurement).

The primary-screen gains are only defensible if the irreducible error is measured, not asserted. We use
the donor split-half structure already committed in results/features/donor_effects.npz: effect_A and
effect_B are two independent-donor estimates of the same perturbation's delta-LFC effect over the 10282
measured genes in three conditions. Per perturbation-condition, on the top-1000 expressed genes (the same
gene set as the predictor error metric):

  r_AB      = Pearson(effect_A, effect_B)                 replicate reproducibility of a half-donor estimate
  r_full    = 2 r_AB / (1 + r_AB)                         Spearman-Brown reliability of the full-donor target
  floor_raw = 1 - r_AB                                    assumption-light split-half disagreement
  floor_sb  = 1 - sqrt(r_full)                            irreducible-error lower bound on 1 - Pearson error,
                                                          the error a perfect predictor of the true effect
                                                          would still incur against the noisy full-donor target

what, why, how: WHAT, a per-perturbation lower bound on the achievable error; WHY, the modest primary-data
gains are only honest if the error floor is measured, and the routing negative is only a law if the oracle
headroom is shown to sit at that floor; HOW, split the donors, correlate the two half estimates on the
signal-bearing gene set, and apply Spearman-Brown to get the full-data reliability.

We then join the per-perturbation predictor errors from E2 (results/error_correlation/per_perturbation_errors.csv)
and report the achievable-gain envelope (best and oracle predictor error minus the floor) and the shared-noise
fraction f_noise = mean(noise_floor) / mean(oracle_err), the E4 anchor coordinate. Leakage-safe: replicate
structure is an analysis covariate, never an estimator feature (invariant 1).

Run (after E2): pixi run python src/eval/run_noise_ceiling.py
"""
import os

import numpy as np
import pandas as pd

DONOR = "results/features/donor_effects.npz"
E2_ERR = "results/error_correlation/per_perturbation_errors.csv"
OUT = "results/ceiling"
N_TOP = 1000


def safe_pearson(a, b):
    """Pearson over finite entries; NaN if degenerate."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    dz = np.load(DONOR, allow_pickle=True)
    genes = np.array([str(g) for g in dz["genes"]])
    meas = np.array([str(g) for g in dz["meas_ids"]])
    conds = [str(c) for c in dz["conditions"]]
    A, B = dz["effect_A"], dz["effect_B"]  # (n_pert, 3, n_meas)

    # use the EXACT top-1000 evaluation genes E2 selected (written by run_error_correlation.py), so the
    # noise floor and the predictor errors (oracle_err) share a byte-identical gene support. meas_ids
    # (donor_effects) == de.gene_ids, so E2's ENSG ids map directly. This fixes the E2/E3 gene-set mismatch
    # the internal review flagged as fatal (previously floor and error were on near-disjoint gene sets).
    eval_path = "results/error_correlation/eval_genes_gladstone.txt"
    if not os.path.exists(eval_path):
        print(f"ERROR: {eval_path} not found. Run E2 (run_error_correlation.py) first."); return
    eval_ids = set(np.loadtxt(eval_path, dtype=str).tolist())
    top = np.array([i for i, g in enumerate(meas) if g in eval_ids])
    print(f"eval genes: {len(top)} of {N_TOP} E2-selected genes present in donor_effects meas_ids")

    # per perturbation-condition split-half reproducibility on the top-1000 expressed genes
    rows = []
    for ci, c in enumerate(conds):
        Ac = A[:, ci, :][:, top]
        Bc = B[:, ci, :][:, top]
        for gi in range(len(genes)):
            r = safe_pearson(Ac[gi], Bc[gi])
            rows.append(dict(gene=genes[gi], condition=c, r_AB=r))
    nf = pd.DataFrame(rows).dropna(subset=["r_AB"]).reset_index(drop=True)
    r = nf["r_AB"].clip(1e-6, 0.999)
    r_full = (2 * r) / (1 + r)
    nf["r_full"] = r_full
    nf["floor_raw"] = 1 - r
    nf["floor_sb"] = 1 - np.sqrt(r_full.clip(0, 1))

    # join E2 per-perturbation predictor errors (top-1000 delta metric, same gene set)
    if not os.path.exists(E2_ERR):
        print(f"ERROR: {E2_ERR} not found. Run E2 (run_error_correlation.py) first."); return
    pe = pd.read_csv(E2_ERR)
    m = nf.merge(pe[["gene", "condition", "oracle_err", "best_fixed_err", "best_fixed_name"]],
                 on=["gene", "condition"], how="inner")
    m["achievable_gain_below_best"] = m["best_fixed_err"] - m["floor_sb"]
    m["achievable_gain_below_oracle"] = m["oracle_err"] - m["floor_sb"]
    m["oracle_headroom"] = m["best_fixed_err"] - m["oracle_err"]
    m.to_csv(os.path.join(OUT, "noise_ceiling_per_perturbation.csv"), index=False)

    # aggregate: is the oracle already at the floor? how much of the error is irreducible?
    mean_floor_sb = float(m["floor_sb"].mean())
    mean_floor_raw = float(m["floor_raw"].mean())
    mean_oracle = float(m["oracle_err"].mean())
    mean_best = float(m["best_fixed_err"].mean())
    mean_headroom = float(m["oracle_headroom"].mean())              # best_fixed - oracle (apparent routing headroom)
    # f_noise uses the best DEPLOYABLE predictor (best_fixed), NOT the oracle. The oracle is the
    # per-perturbation min over 10 predictors, which is selection-biased: it can dip BELOW the irreducible
    # floor by picking favorable noise realizations, so floor/oracle is not a valid fraction. floor/best_fixed
    # is the honest fraction of the deployable predictor's error that is irreducible measurement noise.
    f_noise = mean_floor_sb / mean_best
    # A real router (prediction-time info only) cannot beat the noise floor, so the genuinely achievable gain
    # is best_fixed - floor. The apparent oracle headroom (best_fixed - oracle) exceeds this because the
    # oracle exploits noise BELOW the floor; that excess is selection bias no router can access, which is why
    # routing captures ~none of the headroom (D6 frac-better 0.000).
    real_gain = mean_best - mean_floor_sb
    oracle_below_floor_frac = float((m["oracle_err"] < m["floor_sb"]).mean())
    selection_bias = mean_headroom - max(real_gain, 0.0)
    corr_oracle_floor = safe_pearson(m["oracle_err"].to_numpy(), m["floor_sb"].to_numpy())
    summ = dict(n=len(m), n_conditions=len(conds),
                mean_floor_raw=mean_floor_raw, mean_floor_sb=mean_floor_sb,
                mean_oracle_err=mean_oracle, mean_best_fixed_err=mean_best,
                mean_oracle_headroom=mean_headroom, real_achievable_gain=real_gain,
                f_noise_shared=f_noise, oracle_below_floor_frac=oracle_below_floor_frac,
                selection_bias_in_headroom=selection_bias, corr_oracle_vs_floor=corr_oracle_floor)
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, "noise_ceiling_summary.csv"), index=False)

    print("=== E3 noise ceiling (Gladstone, top-1000 expressed [E2's exact gene set], donor split-half) ===")
    print(f"n = {len(m)} perturbation-conditions across {len(conds)} conditions")
    print(f"noise floor (Spearman-Brown) mean {mean_floor_sb:.3f} | split-half disagreement mean {mean_floor_raw:.3f}")
    print(f"best-fixed error {mean_best:.3f} | oracle error {mean_oracle:.3f} | apparent oracle headroom {mean_headroom:+.4f}")
    print(f"f_noise (floor / best-fixed) = {f_noise:.3f} -> {100*f_noise:.1f}% of the deployable error is irreducible")
    frac_bias = 100 * selection_bias / mean_headroom if mean_headroom > 1e-9 else 0.0
    print(f"real achievable gain (best-fixed - floor) = {real_gain:+.4f}. The oracle dips below the floor on "
          f"{100*oracle_below_floor_frac:.0f}% of perturbations, so {frac_bias:.0f}% of the apparent oracle "
          f"headroom ({selection_bias:+.4f}) is selection bias a prediction-time router cannot access")
    print(f"wrote {OUT}/noise_ceiling_per_perturbation.csv, noise_ceiling_summary.csv")
    return summ


if __name__ == "__main__":
    main()
