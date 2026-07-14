"""E3 extension: cell-level split-half noise ceiling for Replogle K562, to place the second empirical
routing test (D6 Replogle) on the (rho, f_noise) phase plane.

Mirrors E3 on Gladstone, but the replicate floor is a random CELL split-half within each perturbation of the
raw K562 essential-gene Perturb-seq (there is no donor split here), computed on the SAME 2832-gene shared
panel and the SAME top-1000-by-effect-variance genes as the Replogle error-correlation (rho). Per
perturbation, split its cells into two halves, form two independent pseudobulk delta estimates against the
non-targeting controls, and correlate them; r_full = 2 r_AB/(1+r_AB) (Spearman-Brown), floor = 1 - sqrt(r_full).
f_noise = mean floor / best-fixed error (the deployable predictor, from the Replogle error-correlation).

HONEST NOTE: this is a technical-noise floor (cell split-half), a finer split than Gladstone's biological
donor split, so it is a lower bound on the irreducible error and the two datasets' floors are not identical
in kind. Reported plainly. Leakage-safe: replicate structure is an analysis covariate, never a feature.

Run: pixi run python src/eval/run_noise_ceiling_replogle.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "data"))
from replogle_effects import cell_line   # noqa: E402  (loads cond, gene symbols, X)

EFFECTS = "results/features/replogle_effects.npz"
RAW = "baselines/PRESCRIBE/data/replogle/replogle_k562_essential/perturb_processed.h5ad"
E2R = "results/error_correlation/error_correlation_replogle.csv"
OUT = "results/ceiling"
N_TOP = 1000
MIN_CELLS = 20
SPLIT_SEEDS = [42, 43, 44]


def safe_pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(EFFECTS, allow_pickle=True)
    perts = d["perts"].astype(str)
    pgenes = d["pgenes"].astype(str)
    sgenes = d["shared_genes"].astype(str)
    eff_k = d["eff_k562"]
    emb_set = set(d["emb_genes"].astype(str))
    in_panel = np.array([g in emb_set for g in pgenes])
    idx = np.where(in_panel)[0]
    top = np.argsort(-eff_k[idx].var(0))[:N_TOP]     # SAME top-1000 as the Replogle error-correlation

    print(f"[E3-Replogle] loading raw K562 cells from {RAW} ...", flush=True)
    cond, genes, X = cell_line(RAW)
    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi[g] for g in sgenes])
    Xs = X[:, cols]
    ctrl = np.asarray(Xs[cond == "ctrl"].mean(0)).ravel()

    rows = []
    for i in idx:
        p = perts[i]
        cmask = np.where(cond == p)[0]
        if len(cmask) < MIN_CELLS:
            continue
        rs = []
        for s in SPLIT_SEEDS:
            perm = np.random.default_rng(s).permutation(cmask)
            h = len(perm) // 2
            A = np.asarray(Xs[perm[:h]].mean(0)).ravel() - ctrl
            B = np.asarray(Xs[perm[h:]].mean(0)).ravel() - ctrl
            rs.append(safe_pearson(A[top], B[top]))
        rows.append(dict(pert=p, r_AB=float(np.nanmean(rs))))
    nf = pd.DataFrame(rows).dropna(subset=["r_AB"]).reset_index(drop=True)
    r = nf["r_AB"].clip(1e-6, 0.999)
    r_full = (2 * r) / (1 + r)
    nf["floor_sb"] = 1 - np.sqrt(r_full.clip(0, 1))
    mean_floor = float(nf["floor_sb"].mean())

    e2 = pd.read_csv(E2R)
    best_fixed = float(e2["err_best_fixed"].iloc[0])
    oracle = float(e2["err_oracle"].iloc[0]) if "err_oracle" in e2.columns else np.nan
    f_noise = mean_floor / best_fixed
    pd.DataFrame([dict(dataset="Replogle_K562", n=len(nf), mean_floor_sb=mean_floor,
                       mean_r_AB=float(r.mean()), best_fixed_err=best_fixed, oracle_err=oracle,
                       f_noise_shared=f_noise, floor_kind="cell_split_half_technical")]).to_csv(
        os.path.join(OUT, "noise_ceiling_replogle_summary.csv"), index=False)
    print(f"[E3-Replogle] n={len(nf)} perturbations | mean split-half r_AB {r.mean():.3f} | noise floor "
          f"{mean_floor:.3f} | best-fixed {best_fixed:.3f} | f_noise {f_noise:.3f}")
    print(f"wrote {OUT}/noise_ceiling_replogle_summary.csv")


if __name__ == "__main__":
    main()
