"""Calibrate the PertEMA transfer reliability score (isotonic + split conformal).

Rank-correlation alone does not make a reliability score usable. Here we recalibrate the estimator's
predicted error to realized error with isotonic regression and report the expected calibration error
before and after, and we produce distribution-free split-conformal prediction intervals for the true
error and check their empirical coverage. Proper nesting: within each outer gene-disjoint fold the
training genes are split into a fit set and a disjoint calibration set, the estimator is fit on the fit
set, and both the isotonic map and the conformal quantile come from the calibration set.

Run: pixi run python src/pertema/calibration.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
from run_estimator import gbt                                        # noqa: E402
from run_transfer_estimator import build_transfer_features           # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEEDS = [42, 43, 44]
PREDICTORS = ["mean_condition", "knn_coexpr_k25"]  # the two predictors present on the transfer axis
ALPHA = 0.1  # target miscoverage -> 90 percent conformal intervals


def ece(pred, true, n_bins=10):
    pred, true = np.asarray(pred), np.asarray(true)
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    n = pred.size
    return float(sum(len(b) / n * abs(pred[b].mean() - true[b].mean()) for b in bins if len(b)))


def main():
    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    per_rec = []
    for pname in PREDICTORS:
        for seed in SEEDS:
            tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{seed}.csv"), dtype={"gene": str})
            df = tr[tr["transfer"] & (tr["predictor"] == pname)].reset_index(drop=True)
            Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
            y = df["transfer_err"].to_numpy()
            fold = df["fold"].to_numpy()
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)
            fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold
            rng = np.random.default_rng(seed)

            raw, cal, cov, width = [], [], [], []
            for k in np.unique(fold):
                te = np.where(fold == k)[0]
                trn_genes = np.where((fold_u != k))[0]
                cal_genes = set(rng.choice(trn_genes, size=max(1, len(trn_genes) // 5), replace=False))
                is_cal = np.array([g in cal_genes for g in inv])
                fit = np.where((fold != k) & ~is_cal)[0]
                calm = np.where((fold != k) & is_cal)[0]
                # per-fold similarity vs the fit genes only (leakage-safe)
                fit_u = np.where((fold_u != k) & ~np.isin(np.arange(len(genes_u)), list(cal_genes)) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[fit_u])
                su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
                su[q] = nn.kneighbors(emb_u[q])[0].ravel()
                sim = su[inv]
                X = np.column_stack([Xb, sim])

                est = gbt().fit(X[fit], y[fit])
                p_cal, p_te = est.predict(X[calm]), est.predict(X[te])
                mcal = np.isfinite(y[calm])
                iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal[mcal], y[calm][mcal])
                p_te_cal = iso.predict(p_te)
                # split conformal on the calibration residuals
                resid = np.abs(y[calm][mcal] - p_cal[mcal])
                qhat = np.quantile(resid, 1 - ALPHA)
                mte = np.isfinite(y[te])
                raw.append(ece(p_te[mte], y[te][mte]))
                cal.append(ece(p_te_cal[mte], y[te][mte]))
                cov.append(float(np.mean(np.abs(y[te][mte] - p_te[mte]) <= qhat)))
                width.append(float(2 * qhat))
            per_rec.append(dict(predictor=pname, seed=seed, ece_raw=float(np.mean(raw)),
                                ece_isotonic=float(np.mean(cal)), conformal_coverage=float(np.mean(cov)),
                                conformal_interval_width=float(np.mean(width))))

    ps = pd.DataFrame(per_rec)
    ps.to_csv(os.path.join(OUT, "calibration_per_seed.csv"), index=False)

    metrics = ["ece_raw", "ece_isotonic", "conformal_coverage", "conformal_interval_width"]
    summary = []
    print(f"=== PertEMA transfer reliability calibration ({len(SEEDS)} seeds, mean +/- 95% CI) ===")
    print("  held-out protocol: estimator fit set, disjoint calibration set, and test fold are all gene-disjoint")
    for pname in PREDICTORS:
        sub = ps[ps["predictor"] == pname]
        row = {"predictor": pname, "n_seeds": len(SEEDS), "conformal_target": 1 - ALPHA}
        for c in metrics:
            v = sub[c].to_numpy()
            row[c] = float(v.mean())
            row[c + "_ci95"] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        summary.append(row)
        print(f"  [{pname}] ECE raw {row['ece_raw']:.4f} +/- {row['ece_raw_ci95']:.4f} -> isotonic "
              f"{row['ece_isotonic']:.4f} +/- {row['ece_isotonic_ci95']:.4f} | conformal "
              f"{row['conformal_coverage']:.3f} +/- {row['conformal_coverage_ci95']:.3f} "
              f"(target {row['conformal_target']:.2f}), width {row['conformal_interval_width']:.3f}")
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "calibration.csv"), index=False)
    print(f"wrote {OUT}/calibration.csv and calibration_per_seed.csv")


if __name__ == "__main__":
    main()
