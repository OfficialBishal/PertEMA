"""D3 dataset panel: per-perturbation effect vectors for the Adamson 2016 K562 Perturb-seq single-gene
perturbations (the UPR/ER-stress CRISPRi screen), so parity runs on a fourth real dataset. Uses the
AdamsonWeissman2016 h5ad from the scPerturb Zenodo record 13350497.

scPerturb harmonization of this GSM labels each perturbed cell 'GENE_pGUIDE' (gene before the underscore) and
non-targeting controls with nperts == 0. For each single-gene perturbation (>= 20 cells, gene in the measured
panel) we compute the pseudobulk log-normalized expression delta relative to the controls, over all measured
genes, and build a control co-expression embedding (TruncatedSVD on standardized genes-by-control-cells) so
the perturbed gene's embedding is the ridge/kNN feature, matching the Replogle and Norman parity setups.

Saves results/features/adamson_effects.npz. Run: pixi run python src/data/adamson_effects.py
"""
import os
from collections import Counter

import anndata
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

H5AD = "baselines/PRESCRIBE/data/adamson/AdamsonWeissman2016_10X010.h5ad"
OUT = "results/features"
N_EMB_CELLS = 2500        # control cells for the embedding (2613 available)
MIN_CELLS = 20
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(H5AD)
    genes = np.array([str(g).upper() for g in a.var_names])
    panel = set(genes)
    pert = np.array([str(p) for p in a.obs["perturbation"].to_numpy()])
    nperts = np.array([str(p) for p in a.obs["nperts"].to_numpy()])
    pgene_of_cell = np.array([p.split("_")[0].upper() for p in pert])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    Xn = X.multiply(1e4 / tot[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)

    is_ctrl = nperts == "0"
    ctrl_mean = np.asarray(Xn[is_ctrl].mean(0)).ravel()

    perturbed = (~is_ctrl) & (pert != "nan")
    counts = Counter(pgene_of_cell[perturbed])
    pgenes = np.array(sorted(g for g in counts if g in panel and counts[g] >= MIN_CELLS))
    eff = np.full((len(pgenes), Xn.shape[1]), np.nan, np.float32)
    for i, g in enumerate(pgenes):
        m = perturbed & (pgene_of_cell == g)
        eff[i] = (np.asarray(Xn[m].mean(0)).ravel() - ctrl_mean).astype(np.float32)

    rng = np.random.default_rng(SEED)
    ci = np.where(is_ctrl)[0]
    ci = rng.choice(ci, size=min(N_EMB_CELLS, ci.size), replace=False)
    M = np.asarray(Xn[ci].todense()).T
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-8)
    emb = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(M).astype(np.float32)

    np.savez(f"{OUT}/adamson_effects.npz", perts=pgenes, pgenes=pgenes, shared_genes=genes,
             eff=eff, emb_genes=genes, embedding=emb)
    print(f"wrote {OUT}/adamson_effects.npz  eff{eff.shape}  {len(pgenes)} single-gene perturbations "
          f"(>= {MIN_CELLS} cells, in the measured panel)")


if __name__ == "__main__":
    main()
