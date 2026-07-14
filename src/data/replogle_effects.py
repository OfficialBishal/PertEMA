"""Per-perturbation effect vectors for the external K562 vs RPE1 transfer axis (P2).

For each cell line (Replogle 2022 essential-gene Perturb-seq, GEARS-processed), computes the pseudobulk
log-expression delta of each shared single-gene knockdown relative to the cell line's own non-targeting
controls, over the genes measured in both cell lines. Training a predictor on one cell line and scoring it
against the other over the shared perturbations gives an external transfer axis, independent of both the
activation-state and donor axes on the Gladstone data. Also builds a control co-expression embedding of the
perturbed genes for PertEMA features.

Run inside the container: baselines/run_in_container.sh python <this file copied>, or directly with the sif.
Saves results/features/replogle_effects.npz.
"""
import os

import anndata
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

BASE = "baselines/PRESCRIBE/data/replogle"
OUT = "results/features"


def cell_line(path):
    a = anndata.read_h5ad(path)
    cond = a.obs["condition"].astype(str).to_numpy()
    genes = np.array([str(g).upper() for g in
                      (a.var["gene_name"] if "gene_name" in a.var.columns else a.var_names)])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    return cond, genes, X


def per_pert_effect(cond, gidx, cols, X, perts):
    Xs = X[:, cols]
    ctrl = np.asarray(Xs[cond == "ctrl"].mean(0)).ravel()
    eff = np.full((len(perts), cols.size), np.nan, np.float32)
    for i, p in enumerate(perts):
        m = cond == p
        if m.sum() > 0:
            eff[i] = (np.asarray(Xs[m].mean(0)).ravel() - ctrl).astype(np.float32)
    return eff, ctrl


def main():
    os.makedirs(OUT, exist_ok=True)
    ck, gk, Xk = cell_line(f"{BASE}/replogle_k562_essential/perturb_processed.h5ad")
    cr, gr, Xr = cell_line(f"{BASE}/replogle_rpe1_essential/perturb_processed.h5ad")
    perts = sorted((set(ck) & set(cr)) - {"ctrl"})
    sgenes = sorted(set(gk) & set(gr))
    print(f"shared perturbations {len(perts)}, shared genes {len(sgenes)}")

    gik = {g: i for i, g in enumerate(gk)}
    gir = {g: i for i, g in enumerate(gr)}
    colk = np.array([gik[g] for g in sgenes])
    colr = np.array([gir[g] for g in sgenes])
    eff_k, _ = per_pert_effect(ck, gik, colk, Xk, perts)
    eff_r, _ = per_pert_effect(cr, gir, colr, Xr, perts)

    # co-expression embedding of the shared genes from K562 controls
    Mk = np.asarray(Xk[ck == "ctrl"][:, colk].todense()).T   # shared_genes x ctrl_cells
    Mk = (Mk - Mk.mean(1, keepdims=True)) / (Mk.std(1, keepdims=True) + 1e-8)
    emb = TruncatedSVD(n_components=50, random_state=0).fit_transform(Mk).astype(np.float32)
    gene_pos = {g: i for i, g in enumerate(sgenes)}
    pgenes = np.array([p.split("+")[0].upper() for p in perts])   # perturbed gene per perturbation

    np.savez(f"{OUT}/replogle_effects.npz",
             perts=np.array(perts), pgenes=pgenes, shared_genes=np.array(sgenes),
             eff_k562=eff_k, eff_rpe1=eff_r, emb_genes=np.array(sgenes), embedding=emb)
    valid = np.isfinite(eff_k[:, 0]) & np.isfinite(eff_r[:, 0])
    print(f"wrote {OUT}/replogle_effects.npz  eff_k562{eff_k.shape}  perts with both {valid.sum()}")


if __name__ == "__main__":
    main()
