"""Generate and save the canonical gene-disjoint fold assignment for the whole project.

Invariant 1 (no leakage): every analysis uses these fixed folds so a gene never straddles the
train/test boundary in the predictor CV or the estimator outer split. Folds are keyed by the
perturbed gene (Ensembl target_contrast). Three seeds support the >=3-seed headline requirement.

Run: pixi run python src/eval/make_splits.py
"""
import os

import anndata
import numpy as np
import pandas as pd

from splits import gene_disjoint_kfold

PATH = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
OUT = "results/splits"
K = 5
SEEDS = [42, 43, 44]


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(PATH, backed="r")
    obs = a.obs.copy()
    a.file.close()

    genes = (obs[["target_contrast", "target_contrast_gene_name"]].astype(str)
             .drop_duplicates().reset_index(drop=True)
             .rename(columns={"target_contrast": "gene", "target_contrast_gene_name": "gene_name"}))
    uniq = genes["gene"].tolist()
    print(f"{len(uniq)} unique perturbed genes")

    for s in SEEDS:
        fold = gene_disjoint_kfold(uniq, K, s)
        genes[f"fold_seed{s}"] = genes["gene"].map(fold).astype(int)
        sizes = genes[f"fold_seed{s}"].value_counts().sort_index()
        assert sizes.max() - sizes.min() <= 1
        print(f"seed {s}: fold sizes {sizes.to_dict()}")

    genes.to_csv(os.path.join(OUT, "gene_folds.csv"), index=False)
    # leakage guarantee is structural (one fold per gene); assert it explicitly for the record
    for s in SEEDS:
        assert genes.groupby("gene")[f"fold_seed{s}"].nunique().max() == 1
    print(f"\nwrote {OUT}/gene_folds.csv  ({genes.shape[0]} genes x {len(SEEDS)} seeds)")
    print(genes.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
