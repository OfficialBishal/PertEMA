"""Leakage-safe gene embedding from control (non-targeting) co-expression.

Uses only the 11,018 NTC pseudobulks (unperturbed state), so no perturbation effect enters. Each
perturbed (target) gene is embedded by its expression profile across the control pseudobulks, reduced
with truncated SVD. Co-expressed genes land near each other. This drives the similarity-based predictor
and a PertEMA training-set-similarity feature.

Saves results/features/gene_embedding.npz (gene_ids aligned to DE_stats target genes, embedding matrix).
Run: pixi run python src/data/gene_embedding.py
"""
import os

import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from control_baseline import read_csr_rows

PSEUDO = "data/raw/marson2025/GWCD4i.pseudobulk_merged.h5ad"
SPLITS = "results/splits/gene_folds.csv"
OUT = "results/features"
N_COMP = 50


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(PSEUDO, backed="r")
    obs = a.obs
    ntc_rows = np.where((obs["guide_type"].astype(str) == "non-targeting").to_numpy())[0]
    tc = obs["total_counts"].to_numpy()[ntc_rows]
    var_ids = np.asarray(a.var["gene_ids"]).astype(str)
    a.file.close()

    target_genes = pd.read_csv(SPLITS, dtype={"gene": str})["gene"].tolist()
    col_of = {g: i for i, g in enumerate(var_ids)}
    keep = [(g, col_of[g]) for g in target_genes if g in col_of]
    genes = [g for g, _ in keep]
    cols = np.array([c for _, c in keep])
    print(f"target genes {len(target_genes)}, embeddable (present in pseudobulk) {len(genes)}")

    X = read_csr_rows(PSEUDO, ntc_rows).astype(np.float64)      # (n_ntc, 18129)
    inv = np.where(tc > 0, 1e6 / tc, 0.0)
    X = X.multiply(inv[:, None]).tocsr()
    X.data = np.log1p(X.data)
    M = np.asarray(X[:, cols].todense())                        # (n_ntc, n_genes)
    M = M.T                                                     # genes as samples
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-8)

    svd = TruncatedSVD(n_components=N_COMP, random_state=0)
    emb = svd.fit_transform(M).astype(np.float32)              # (n_genes, N_COMP)
    print(f"embedding {emb.shape}  explained var (top5) {svd.explained_variance_ratio_[:5].round(3)}  "
          f"total {svd.explained_variance_ratio_.sum():.3f}")

    np.savez(os.path.join(OUT, "gene_embedding.npz"),
             gene_ids=np.array(genes), embedding=emb)
    print(f"wrote {OUT}/gene_embedding.npz")


if __name__ == "__main__":
    main()
