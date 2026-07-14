"""Evaluation metrics for PertEMA.

Three families:
  1. Per-perturbation error targets (what a predictor gets wrong): delta-LFC dissimilarity,
     direction-of-change accuracy, and energy distance (E-distance) between cell distributions.
  2. Reliability calibration: Spearman of predicted reliability against realized accuracy, and
     expected calibration error against a reliability diagram.
  3. Selective-prediction utility: risk-coverage curve, area under it (AURC), and accuracy at
     a fixed coverage. This is what the reliability score is ultimately judged on.

All functions are pure and operate on numpy arrays. Higher-level code decides the sign
conventions (here: error is higher-is-worse, reliability score is higher-is-more-trustworthy).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ----- 1. error targets -------------------------------------------------------------------

def one_minus_pearson(pred, true):
    """Delta-LFC dissimilarity in [0, 2]. 0 means identical direction and shape."""
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    if pred.size < 2 or np.std(pred) == 0 or np.std(true) == 0:
        return np.nan
    return 1.0 - pearsonr(pred, true)[0]


def direction_accuracy(pred, true, eps=0.0):
    """Fraction of genes where predicted and true log-fold-change agree in sign.

    Genes with |true| <= eps are excluded (no meaningful direction to get right).
    """
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    keep = np.abs(true) > eps
    if keep.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(pred[keep]) == np.sign(true[keep])))


def edistance(x, y):
    """Energy distance between two point clouds (scPerturb / pertpy convention).

    Uses squared Euclidean distances: E = 2*mean||xi-yj||^2 - mean||xi-xj||^2 - mean||yi-yj||^2.
    Computed in closed form from means of squared norms so it is O(n*d), not O(n^2*d).
    x, y: arrays of shape (n_cells, n_genes). Returns a non-negative scalar.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D with matching feature dimension")

    def mean_pairwise_sq(a, b):
        # mean over all i,j of ||a_i - b_j||^2 = mean||a||^2 + mean||b||^2 - 2 <mean_a, mean_b>
        return (np.mean(np.sum(a * a, 1))
                + np.mean(np.sum(b * b, 1))
                - 2.0 * np.dot(a.mean(0), b.mean(0)))

    delta = mean_pairwise_sq(x, y)
    sigma_x = mean_pairwise_sq(x, x)
    sigma_y = mean_pairwise_sq(y, y)
    return float(2.0 * delta - sigma_x - sigma_y)


# ----- 2. calibration ---------------------------------------------------------------------

def reliability_spearman(pred_reliability, realized_accuracy):
    """Spearman correlation between predicted reliability and realized accuracy (higher is better)."""
    a = np.asarray(pred_reliability, float).ravel()
    b = np.asarray(realized_accuracy, float).ravel()
    if a.size < 3:
        return np.nan
    return float(spearmanr(a, b)[0])


def expected_calibration_error(pred_reliability, realized_accuracy, n_bins=10):
    """ECE between predicted reliability and realized accuracy, both expected in [0, 1].

    Bins by predicted reliability, then averages |mean_predicted - mean_realized| weighted by bin size.
    """
    p = np.asarray(pred_reliability, float).ravel()
    a = np.asarray(realized_accuracy, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = p.size
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(p[m].mean() - a[m].mean())
    return float(ece)


# ----- 3. selective prediction ------------------------------------------------------------

def risk_coverage_curve(error, score):
    """Risk as a function of coverage, abstaining on the least reliable items first.

    error: per-item error, higher is worse. score: per-item reliability, higher is more trustworthy.
    Returns (coverages, risks) where at coverage c we keep the top c fraction by score and report
    the mean error over the kept items.
    """
    error = np.asarray(error, float).ravel()
    score = np.asarray(score, float).ravel()
    n = error.size
    order = np.argsort(-score)  # most reliable first
    err_sorted = error[order]
    cum = np.cumsum(err_sorted)
    counts = np.arange(1, n + 1)
    risks = cum / counts
    coverages = counts / n
    return coverages, risks


def aurc(error, score):
    """Area under the risk-coverage curve (lower is better). Mean risk over the coverage sweep."""
    _, risks = risk_coverage_curve(error, score)
    return float(np.mean(risks))


def accuracy_at_coverage(error, score, coverage):
    """Mean error over the most-reliable `coverage` fraction of items (lower is better)."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    error = np.asarray(error, float).ravel()
    score = np.asarray(score, float).ravel()
    n = error.size
    keep = max(1, int(round(coverage * n)))
    order = np.argsort(-score)[:keep]
    return float(error[order].mean())
