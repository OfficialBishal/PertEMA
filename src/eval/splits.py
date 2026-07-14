"""Leakage-safe splitting utilities.

Invariant 1 (no leakage): a gene that appears in any test fold must appear in no training fold
of the same model. All splits here are gene-disjoint by construction. Perturbations are grouped
by their perturbed gene so that no gene straddles the train/test boundary.
"""
from __future__ import annotations

import numpy as np


def gene_disjoint_kfold(genes, k, seed):
    """Assign each unique gene to exactly one of k folds, deterministically.

    genes: iterable of gene identifiers (may contain duplicates; grouped by identity).
    Returns dict gene -> fold_index in [0, k). Fold sizes differ by at most one gene.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    uniq = sorted(set(genes))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {}
    for rank, idx in enumerate(order):
        fold_of[uniq[idx]] = rank % k
    return fold_of


def train_test_gene_masks(gene_per_row, fold_of, test_fold):
    """Boolean train/test row masks for one outer fold, keyed by each row's perturbed gene.

    gene_per_row: array-like of the perturbed gene for each observation row.
    fold_of: dict gene -> fold index (from gene_disjoint_kfold).
    Rows whose gene is not in fold_of (for example non-targeting controls) go to neither mask.
    """
    gene_per_row = np.asarray(gene_per_row, dtype=object)
    folds = np.array([fold_of.get(g, -1) for g in gene_per_row])
    test = folds == test_fold
    train = (folds != test_fold) & (folds >= 0)
    return train, test
