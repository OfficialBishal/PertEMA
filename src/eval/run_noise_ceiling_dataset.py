"""E3 extension for Norman and Adamson: cell-level split-half noise floor from the raw h5ad, matching each
builder's normalization (CP10k + log1p) and control definition exactly (Norman: perturbation == control,
single-gene nperts == 1; Adamson: nperts == 0, perturbed gene = perturbation split on underscore). Same
2832/full panel and top-1000-by-effect-variance genes as the dataset's error-correlation. f_noise = mean
noise floor / best-fixed error. Technical (cell) split-half, a lower bound on the irreducible error.

Usage: pixi run python src/eval/run_noise_ceiling_dataset.py norman|adamson
"""
import os
import sys

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

CFG = {
    "norman": dict(eff="results/features/norman_effects.npz",
                   raw="baselines/PRESCRIBE/data/NormanWeissman2019_filtered.h5ad", mode="norman"),
    "adamson": dict(eff="results/features/adamson_effects.npz",
                    raw="baselines/PRESCRIBE/data/adamson/AdamsonWeissman2016_10X010.h5ad", mode="adamson"),
}
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


def lognorm(X):
    tot = np.asarray(X.sum(1)).ravel()
    tot[tot == 0] = 1.0
    Xn = X.multiply(1e4 / tot[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = sys.argv[1]
    cfg = CFG[ds]
    d = np.load(cfg["eff"], allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff = d["eff"].astype(np.float32)
    sgenes = d["shared_genes"].astype(str)
    emb_set = set(d["emb_genes"].astype(str))
    in_panel = np.array([g in emb_set and np.isfinite(eff[i, 0]) for i, g in enumerate(pgenes)])
    idx = np.where(in_panel)[0]
    top = np.argsort(-eff[idx].var(0))[:N_TOP]

    print(f"[E3-{ds}] loading raw cells from {cfg['raw']} ...", flush=True)
    a = anndata.read_h5ad(cfg["raw"])
    genes = np.array([str(g).upper() for g in a.var_names])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    Xn = lognorm(X)
    pert = np.array([str(p) for p in a.obs["perturbation"].to_numpy()])
    if cfg["mode"] == "norman":
        nperts = a.obs["nperts"].astype(str).to_numpy()
        is_ctrl = pert == "control"
        single = nperts == "1"
        def pert_cells(g):
            return single & (pert == g)
    else:
        nperts = np.array([str(p) for p in a.obs["nperts"].to_numpy()])
        pgene_of_cell = np.array([p.split("_")[0].upper() for p in pert])
        is_ctrl = nperts == "0"
        perturbed = (~is_ctrl) & (pert != "nan")
        def pert_cells(g):
            return perturbed & (pgene_of_cell == g)

    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi[g] for g in sgenes])
    Xs = Xn[:, cols]
    ctrl = np.asarray(Xs[is_ctrl].mean(0)).ravel()

    rows = []
    for i in idx:
        g = pgenes[i]
        cmask = np.where(pert_cells(g))[0]
        if len(cmask) < MIN_CELLS:
            continue
        rs = []
        for s in SPLIT_SEEDS:
            perm = np.random.default_rng(s).permutation(cmask)
            h = len(perm) // 2
            A = np.asarray(Xs[perm[:h]].mean(0)).ravel() - ctrl
            B = np.asarray(Xs[perm[h:]].mean(0)).ravel() - ctrl
            rs.append(safe_pearson(A[top], B[top]))
        rows.append(dict(pert=g, r_AB=float(np.nanmean(rs))))
    nf = pd.DataFrame(rows).dropna(subset=["r_AB"]).reset_index(drop=True)
    r = nf["r_AB"].clip(1e-6, 0.999)
    r_full = (2 * r) / (1 + r)
    nf["floor_sb"] = 1 - np.sqrt(r_full.clip(0, 1))
    mean_floor = float(nf["floor_sb"].mean())

    e2 = pd.read_csv(f"results/error_correlation/error_correlation_{ds}.csv")
    best_fixed = float(e2["err_best_fixed"].iloc[0])
    f_noise = mean_floor / best_fixed
    pd.DataFrame([dict(dataset=ds, n=len(nf), mean_floor_sb=mean_floor, mean_r_AB=float(r.mean()),
                       best_fixed_err=best_fixed, f_noise_shared=f_noise,
                       floor_kind="cell_split_half_technical")]).to_csv(
        os.path.join(OUT, f"noise_ceiling_{ds}_summary.csv"), index=False)
    print(f"[E3-{ds}] n={len(nf)} | split-half r_AB {r.mean():.3f} | floor {mean_floor:.3f} | best-fixed "
          f"{best_fixed:.3f} | f_noise {f_noise:.3f}")


if __name__ == "__main__":
    main()
