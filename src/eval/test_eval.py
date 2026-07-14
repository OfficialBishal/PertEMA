"""Assert-based tests for the evaluation backbone. Run: pixi run python src/eval/test_eval.py"""
from __future__ import annotations

import numpy as np

from metrics import (accuracy_at_coverage, aurc, direction_accuracy, edistance,
                     expected_calibration_error, one_minus_pearson, risk_coverage_curve)
from splits import gene_disjoint_kfold, train_test_gene_masks


def test_gene_disjoint_kfold():
    genes = [f"g{i}" for i in range(103)] + ["g5", "g6"]  # duplicates collapse
    fold = gene_disjoint_kfold(genes, k=5, seed=0)
    assert set(fold) == {f"g{i}" for i in range(103)}
    sizes = np.bincount(list(fold.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1, sizes
    assert fold == gene_disjoint_kfold(genes, k=5, seed=0)  # deterministic
    assert fold != gene_disjoint_kfold(genes, k=5, seed=1)


def test_no_leakage_in_masks():
    genes = [f"g{i}" for i in range(50)]
    fold = gene_disjoint_kfold(genes, k=5, seed=3)
    rows = [f"g{i % 50}" for i in range(500)] + ["NTC"] * 20  # controls not in fold map
    for tf in range(5):
        tr, te = train_test_gene_masks(rows, fold, test_fold=tf)
        rows_arr = np.asarray(rows, dtype=object)
        train_genes = set(rows_arr[tr]); test_genes = set(rows_arr[te])
        assert train_genes.isdisjoint(test_genes), "LEAKAGE: gene in both train and test"
        assert "NTC" not in train_genes and "NTC" not in test_genes


def test_error_metrics():
    v = np.array([1.0, -2.0, 3.0, -0.5])
    assert abs(one_minus_pearson(v, v)) < 1e-9
    assert abs(one_minus_pearson(v, -v) - 2.0) < 1e-9
    pred = np.array([1.0, -1.0, 1.0, -1.0]); true = np.array([1.0, -1.0, -1.0, -1.0])
    assert abs(direction_accuracy(pred, true) - 0.75) < 1e-9


def test_edistance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)); y = rng.normal(size=(35, 6)) + 2.0

    def brute(a, b):
        mp = lambda p, q: ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1).mean()
        return 2 * mp(a, b) - mp(a, a) - mp(b, b)

    assert abs(edistance(x, y) - brute(x, y)) < 1e-6
    assert abs(edistance(x, x)) < 1e-9                 # identical -> 0
    c = 2.0                                            # pure translation by c*ones(d): E = 2*||c||^2
    assert abs(edistance(x, x + c) - 2 * c * c * x.shape[1]) < 1e-6
    assert edistance(x, y) >= -1e-9                     # non-negative


def test_calibration():
    p = np.linspace(0, 1, 100)
    assert expected_calibration_error(p, p, n_bins=10) < 1e-9   # perfectly calibrated
    assert expected_calibration_error(p, np.full_like(p, 0.5)) > 0.1


def test_selective_prediction():
    rng = np.random.default_rng(1)
    err = rng.uniform(0, 1, size=200)
    perfect = -err                       # high score exactly where error is low
    random = rng.uniform(0, 1, size=200)
    cov, risk = risk_coverage_curve(err, perfect)
    assert risk[0] <= risk[-1] + 1e-9    # first-covered (most reliable) has lowest risk
    assert abs(cov[-1] - 1.0) < 1e-9 and abs(risk[-1] - err.mean()) < 1e-9
    assert aurc(err, perfect) < aurc(err, random)         # good scores beat random
    assert abs(accuracy_at_coverage(err, perfect, 1.0) - err.mean()) < 1e-9
    assert accuracy_at_coverage(err, perfect, 0.1) < err.mean()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} EVAL TESTS PASSED")


if __name__ == "__main__":
    main()
