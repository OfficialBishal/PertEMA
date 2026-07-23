"""Tests for the PertEMA package API, ingestion, and featurization.

These run from the committed frozen model, so they need no GPU and no primary data.
"""
import numpy as np
import pytest

import pertema

BANDS = {"high", "moderate", "low", "very-low"}


def test_version():
    assert pertema.__version__ == "0.1.0"


def test_default_model_scores_a_featurized_perturbation():
    model = pertema.default_model()
    assert model.version == "0.1.0"
    feat = pertema.default_featurizer()
    preds, _ = pertema.parse_long_csv("perturbed_gene,gene,predicted_lfc\nIL2,IL2RA,0.5\nIL2,IFNG,-0.3\n")
    x, syms, _ = feat.featurize(preds)
    assert len(x) == len(syms) == 1

    scored = model.score(x)
    for key in ("predicted_error", "calibrated_error", "reliability", "band", "conformal_lo", "conformal_hi"):
        assert key in scored and len(scored[key]) == 1
    assert scored["band"][0] in BANDS
    # reliability is the negation of the calibrated error
    assert scored["reliability"][0] == pytest.approx(-scored["calibrated_error"][0])
    # the 90 percent conformal interval uses a fixed half-width of 0.1135
    assert scored["conformal_hi"][0] - scored["calibrated_error"][0] == pytest.approx(0.1135, abs=1e-9)


def test_scoring_is_deterministic():
    model = pertema.default_model()
    x = np.zeros((3, 64), dtype=np.float32)
    first = model.score(x)["calibrated_error"]
    second = model.score(x)["calibrated_error"]
    assert np.array_equal(first, second)


def test_parse_long_csv_magnitude_is_mean_absolute_lfc():
    preds, report = pertema.parse_long_csv(
        "perturbed_gene,gene,predicted_lfc\nIL2,A,0.83\nIL2,B,-0.21\nIL2,C,0.11\n"
    )
    assert len(preds) == 1
    assert preds[0]["pred_magnitude"] == pytest.approx((0.83 + 0.21 + 0.11) / 3)
    assert report["n_perturbations"] == 1


def test_parse_long_csv_missing_required_column_raises():
    with pytest.raises(ValueError):
        pertema.parse_long_csv("perturbed_gene,gene\nIL2,A\n")


def test_parse_long_csv_drops_non_numeric_lfc():
    preds, report = pertema.parse_long_csv(
        "perturbed_gene,gene,predicted_lfc\nIL2,A,0.5\nIL2,B,not_a_number\n"
    )
    assert report["n_dropped_nonnumeric_lfc"] == 1
    assert preds[0]["pred_magnitude"] == pytest.approx(0.5)


def test_featurize_drops_unmapped_gene_and_bad_context():
    feat = pertema.default_featurizer()
    preds = [
        {"perturbed_gene": "IL2", "src": "Rest", "dst": "Stim48hr", "pred_magnitude": 0.5},
        {"perturbed_gene": "NOT_A_GENE_XYZ", "src": "Rest", "dst": "Stim48hr", "pred_magnitude": 0.5},
        {"perturbed_gene": "IL2", "src": "Rest", "dst": "Stim72hr", "pred_magnitude": 0.5},
    ]
    x, syms, report = feat.featurize(preds)
    assert syms == ["IL2"]
    assert report["n_unmapped_genes"] == 1
    assert report["n_bad_context"] == 1
    assert x[0].shape == (64,)
