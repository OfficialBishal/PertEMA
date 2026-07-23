"""PertEMA: post-hoc reliability estimation for single-cell perturbation prediction.

A model-agnostic reliability layer. Given a predictor's outputs it returns, for each prediction, a calibrated
error, a reliability band, and a 90 percent split-conformal interval, using a gradient-boosted meta-model over
leakage-safe, prediction-time features with isotonic calibration. It scores which predictions to trust and does
not rank biological importance.

Typical use:

    import pertema
    model = pertema.default_model()          # the bundled demonstration model
    out = model.score(features)              # per-prediction calibrated error, band, conformal interval
"""

from .scoring import PertEMAModel, default_model, BANDS, FEATURE_GROUPS
from .featurize import Featurizer, default_featurizer, CONDS
from .ingest import parse_long_csv, LONG_TEMPLATE
from .report import render_report

__version__ = "0.1.0"

__all__ = [
    "PertEMAModel",
    "default_model",
    "BANDS",
    "FEATURE_GROUPS",
    "Featurizer",
    "default_featurizer",
    "CONDS",
    "parse_long_csv",
    "LONG_TEMPLATE",
    "render_report",
    "__version__",
]
