"""D3 dataset panel: per-perturbation effect vectors for the Norman 2019 K562 Perturb-seq single-gene
perturbations, so parity can be run on a third real dataset (Gladstone, Replogle, Norman) using the LOCAL
NormanWeissman2019_filtered.h5ad already present for the PRESCRIBE comparison (no download).

For each single-gene perturbation (nperts=1) computes the pseudobulk log-normalized expression delta relative
to the non-targeting controls (perturbation == control), over all measured genes. Builds a control
co-expression embedding of the genes (TruncatedSVD on standardized genes-by-control-cells), so the perturbed
gene's embedding is the ridge/kNN feature, matching the Replogle parity setup. Only single perturbations whose
perturbed gene is in the measured panel get an embedding; parity restricts to that in-panel cohort.

Saves results/features/norman_effects.npz. Run: pixi run python src/data/norman_effects.py
"""
import os

import anndata
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

H5AD = "baselines/PRESCRIBE/data/NormanWeissman2019_filtered.h5ad"
OUT = "results/features"
N_EMB_CELLS = 4000        # control cells subsampled for the co-expression embedding (memory)
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(H5AD)
    genes = np.array([str(g).upper() for g in a.var_names])
    pert = a.obs["perturbation"].astype(str).to_numpy()
    nperts = a.obs["nperts"].astype(str).to_numpy()
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    # log-normalize raw counts (CP10k + log1p); log1p keeps zeros zero so it stays sparse-friendly
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    Xn = X.multiply(1e4 / tot[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)

    is_ctrl = pert == "control"
    ctrl_mean = np.asarray(Xn[is_ctrl].mean(0)).ravel()

    single = (nperts == "1")
    pgenes = np.array(sorted(set(pert[single])))
    eff = np.full((len(pgenes), Xn.shape[1]), np.nan, np.float32)
    for i, g in enumerate(pgenes):
        m = single & (pert == g)
        if m.sum() > 0:
            eff[i] = (np.asarray(Xn[m].mean(0)).ravel() - ctrl_mean).astype(np.float32)

    # co-expression embedding of genes from a control-cell subsample (genes x cells, standardized, SVD-50)
    rng = np.random.default_rng(SEED)
    ci = np.where(is_ctrl)[0]
    ci = rng.choice(ci, size=min(N_EMB_CELLS, ci.size), replace=False)
    M = np.asarray(Xn[ci].todense()).T                     # genes x cells
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-8)
    emb = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(M).astype(np.float32)

    np.savez(f"{OUT}/norman_effects.npz", perts=pgenes, pgenes=pgenes, shared_genes=genes,
             eff=eff, emb_genes=genes, embedding=emb)
    in_panel = np.array([g in set(genes) for g in pgenes]).sum()
    print(f"wrote {OUT}/norman_effects.npz  eff{eff.shape}  {len(pgenes)} single-gene perturbations, "
          f"{in_panel} in the measured panel (with an embedding)")


if __name__ == "__main__":
    main()
