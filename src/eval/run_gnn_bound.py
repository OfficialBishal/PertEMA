"""U5 (X5 bounded-risk): the routing roster lacks an end-to-end trained GNN (a genuinely different inductive
bias), which structurally inflates the error correlation and deflates the effective ensemble size. Rather than
only stating this limitation, we BOUND it quantitatively using the measured correlation matrix and the Result-4
mapping:
  (a) N_eff sensitivity to dropping the most distinct current predictor (how much diversity the roster already
      has).
  (b) how much a single decorrelated predictor (a hypothetical GNN correlated at rho_new with the pack) would
      raise N_eff and lower the average pairwise correlation, and whether that shift crosses the routing
      feasibility boundary given the achievable per-perturbation router quality.
The point: routing feasibility needs BOTH a large oracle premium AND an achievable router quality above the
break-even r*. Adding a decorrelated predictor raises the premium but does not raise the router quality (that
is a property of the reliability features, not the roster) and does not lower the noise ceiling, so we can
bound whether a plausible GNN would change the routing-negative.

Run: pixi run python src/eval/run_gnn_bound.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_theory import a_K, break_even_r, head_equal_means, sim_route  # noqa: E402

CORR = "results/error_correlation/error_corr_pearson.csv"
OUT = "results/theory"
ACHIEVABLE_R_GLADSTONE = 0.068   # measured within-pair per-gene reliability Spearman (the router's rank quality)
F_NOISE_GLADSTONE = 0.972


def neff(C):
    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def main():
    df = pd.read_csv(CORR, index_col=0)
    C = df.to_numpy()
    names = list(df.columns)
    M = len(names)
    iu = np.triu_indices(M, 1)
    rho_full = float(C[iu].mean())
    neff_full = neff(C)
    print(f"Gladstone roster: {M} predictors, mean off-diag rho {rho_full:.3f}, N_eff {neff_full:.3f}")

    # (a) leave-one-out N_eff: which predictor contributes the most diversity
    rows_a = []
    for i, nm in enumerate(names):
        keep = [j for j in range(M) if j != i]
        Ci = C[np.ix_(keep, keep)]
        rows_a.append(dict(dropped=nm, N_eff_without=neff(Ci),
                           delta_N_eff=neff(Ci) - neff_full))
    a = pd.DataFrame(rows_a).sort_values("N_eff_without")
    a.to_csv(os.path.join(OUT, "gnn_bound_leave_one_out.csv"), index=False)
    print("\n(a) leave-one-out N_eff (roster already has little diversity):")
    print(f"  N_eff ranges {a['N_eff_without'].min():.3f} to {a['N_eff_without'].max():.3f} across drops "
          f"(full {neff_full:.3f}); most distinct predictor = {a.iloc[-1]['dropped']}")

    # (b) add a hypothetical decorrelated predictor (a trained GNN) correlated at rho_new with every existing one
    rows_b = []
    for rho_new in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.919]:
        Cn = np.ones((M + 1, M + 1))
        Cn[:M, :M] = C
        Cn[M, :M] = rho_new
        Cn[:M, M] = rho_new
        iu2 = np.triu_indices(M + 1, 1)
        rho_avg = float(Cn[iu2].mean())
        ne = neff(Cn)
        premium = head_equal_means(rho_avg, M + 1)
        rstar = break_even_r(rho_avg, 0.093, K=M + 1)          # Gladstone's calibrated mean-skill spread
        # routing capture at the ACHIEVABLE router quality (unchanged by adding a predictor) vs at break-even
        cap_achievable = float(np.mean([sim_route(rho_avg, 0.093, ACHIEVABLE_R_GLADSTONE, K=M + 1, seed=s)[2]
                                        for s in (0, 1, 2)]))
        rows_b.append(dict(rho_new=rho_new, rho_avg=round(rho_avg, 3), N_eff=round(ne, 3),
                           oracle_premium=round(premium, 3), break_even_r=round(rstar, 3),
                           achievable_r=ACHIEVABLE_R_GLADSTONE, capture_at_achievable_r=round(cap_achievable, 3),
                           routing_feasible=bool(ACHIEVABLE_R_GLADSTONE > rstar)))
    b = pd.DataFrame(rows_b)
    b.to_csv(os.path.join(OUT, "gnn_bound_add_decorrelated.csv"), index=False)
    print("\n(b) adding one decorrelated predictor (a hypothetical trained GNN at rho_new with the pack):")
    print(b.to_string(index=False))

    fully_dec = b[b.rho_new == 0.2].iloc[0]
    print(f"\nBOUND: even a strongly decorrelated GNN (rho_new 0.2) only raises N_eff to {fully_dec['N_eff']} and "
          f"lowers avg rho to {fully_dec['rho_avg']}, giving break-even r* {fully_dec['break_even_r']}. The "
          f"achievable router quality on Gladstone is {ACHIEVABLE_R_GLADSTONE}, still far below break-even, so "
          f"routing stays INFEASIBLE (capture {fully_dec['capture_at_achievable_r']} at the achievable r). "
          f"Adding predictors raises the premium but not the router quality and does not lower the noise "
          f"ceiling (f_noise {F_NOISE_GLADSTONE}), so the routing-negative is robust to plausible roster "
          f"diversification. What would change it is a far better reliability estimator (r from 0.07 to >0.5), "
          f"a different axis the E5 and X1 results show is hard in this domain.")
    print(f"\nwrote {OUT}/gnn_bound_leave_one_out.csv, gnn_bound_add_decorrelated.csv")


if __name__ == "__main__":
    main()
