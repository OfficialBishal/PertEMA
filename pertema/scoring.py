"""PertEMA scoring core (framework-agnostic). Loads the frozen inference artifact read-only and turns a
prediction-time feature matrix into per-perturbation reliability scores, calibrated error estimates, four
calibrated reliability bands, and a split-conformal interval. This is the scientific core the backend
service and worker call; it has no web or job dependencies so it is trivially testable.

Invariant: never retrains on user ground truth. Ground truth, if supplied, is used only to report realized
error and user-side calibration in evaluate mode, never to fit the shipped estimator.
"""
import json
import os

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

# Four calibrated reliability bands with fixed, documented thresholds on the calibrated predicted error
# (1 - Pearson, lower is better). Paired with a label, never encoded by color alone (invariant N6).
BANDS = [
    ("high",     0.00, 0.80, "prediction is among the more trustworthy on this noisy data"),
    ("moderate", 0.80, 0.88, "usable with caution"),
    ("low",      0.88, 0.94, "treat with skepticism"),
    ("very-low", 0.94, 2.00, "likely unreliable, deprioritize"),
]

# Interpretable feature groups over the 64-d prediction-time vector, in the exact featurize() order
# (pred_magnitude, then src then dst baseline/dropout/donor_var, then interleaved src/dst context one-hots,
# then the 50-d co-expression embedding, then training-set similarity). Used for F6 grouped attribution.
FEATURE_GROUPS = [
    ("predicted_magnitude", [0]),
    ("source_context_state", [1, 2, 3]),
    ("destination_context_state", [4, 5, 6]),
    ("context_indicators", list(range(7, 13))),
    ("coexpression_embedding", list(range(13, 63))),
    ("training_set_similarity", [63]),
]


class PertEMAModel:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.estimator = xgb.XGBRegressor()
        self.estimator.load_model(os.path.join(model_dir, "estimator.json"))
        cal = np.load(os.path.join(model_dir, "calibration.npz"))
        self.iso_x, self.iso_y = cal["iso_x"], cal["iso_y"]
        with open(os.path.join(model_dir, "provenance.json")) as f:
            self.provenance = json.load(f)
        self.version = self.provenance["model_version"]
        self.coverage = self.provenance["evaluation_calibration"]["conformal_coverage"]
        # conformal half-width from the isotonic residual scale, kept simple and documented
        self.conformal_halfwidth = 0.5 * float(self.provenance["evaluation_calibration"].get(
            "conformal_interval_width", 0.227) if isinstance(
            self.provenance["evaluation_calibration"].get("conformal_interval_width", None), (int, float)) else 0.227)

    def _calibrate(self, pred_err):
        return np.interp(pred_err, self.iso_x, self.iso_y)

    def score(self, features):
        """features: (n, d) prediction-time feature matrix matching feature_spec. Returns a dict of arrays."""
        features = np.asarray(features, float)
        pred_err = self.estimator.predict(features)
        cal_err = self._calibrate(pred_err)
        reliability = -cal_err                     # higher = more reliable
        band = [self._band(e) for e in cal_err]
        lo = np.clip(cal_err - self.conformal_halfwidth, 0, None)
        hi = cal_err + self.conformal_halfwidth
        return {
            "predicted_error": pred_err,
            "calibrated_error": cal_err,
            "reliability": reliability,
            "band": band,
            "conformal_lo": lo,
            "conformal_hi": hi,
            "coverage": self.coverage,
            "model_version": self.version,
        }

    @staticmethod
    def _band(cal_err):
        for name, lo, hi, _ in BANDS:
            if lo <= cal_err < hi:
                return name
        return "very-low"

    def explain(self, features, top_k=3):
        """F6 feature attribution: per-perturbation, why the estimator predicts this reliability. Uses
        xgboost's exact SHAP contributions (pred_contribs) toward the PREDICTED ERROR, summed within the
        interpretable feature groups of the feature spec. A positive group contribution pushes the prediction
        toward MORE error, that is LESS reliable. Returns one dict per row with the base value, the grouped
        signed contributions, and the top_k groups by absolute contribution (the drivers)."""
        features = np.asarray(features, float)
        contribs = self.estimator.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)
        feat_c, base = contribs[:, :-1], contribs[:, -1]
        out = []
        for i in range(features.shape[0]):
            groups = {name: float(feat_c[i, idx].sum()) for name, idx in FEATURE_GROUPS}
            drivers = sorted(groups.items(), key=lambda kv: -abs(kv[1]))[:top_k]
            out.append({
                "base_error": float(base[i]),
                "group_contributions_toward_error": groups,
                "top_drivers": [{"group": g, "contribution": c,
                                 "direction": "less reliable" if c > 0 else "more reliable"}
                                for g, c in drivers],
            })
        return out

    def evaluate(self, features, true_error):
        """Score features and, using the user's OWN ground truth, report the realized reliability QUALITY.

        Never refits or retrains the estimator (inviolable invariant: no training on user truth). The
        supplied true_error must be OUT OF SAMPLE for the numbers to be honest; scoring the estimator's own
        training data is optimistic. That caveat is enforced as an explicit warning at the service layer.

        Args:
            features: (n, d) prediction-time feature matrix, same spec as score().
            true_error: (n,) the user's realized per-perturbation error (1 - Pearson), lower is better.

        Returns a dict with:
            spearman_reliability_vs_accuracy: Spearman rho of predicted reliability vs realized accuracy
                (-true_error). Positive means the estimator ranks trustworthy predictions higher.
            risk_coverage_auc: area under the risk-coverage curve when abstaining from the least-reliable
                predictions first (mean realized error averaged over all coverage levels; lower is better).
            calibration: quantile bins of predicted calibrated error vs mean realized error, a user-side
                reliability diagram (well calibrated means the two columns track each other).
        """
        true_error = np.asarray(true_error, float).ravel()
        scored = self.score(features)
        reliability = scored["reliability"]
        cal_err = scored["calibrated_error"]
        if reliability.shape[0] != true_error.shape[0]:
            raise ValueError(
                f"features and true_error length mismatch: {reliability.shape[0]} vs {true_error.shape[0]}")
        n = int(true_error.shape[0])

        # 1) Ranking quality: does higher predicted reliability track higher realized accuracy (-error)?
        rho, pval = spearmanr(reliability, -true_error)

        # 2) Risk-coverage: keep the most-reliable predictions first, average realized error over coverage.
        order = np.argsort(-reliability)                  # most reliable first
        cum_risk = np.cumsum(true_error[order]) / np.arange(1, n + 1)
        rc_auc = float(np.mean(cum_risk))                 # lower = estimator front-loads the accurate ones

        return {
            "n": n,
            "spearman_reliability_vs_accuracy": float(rho),
            "spearman_pvalue": float(pval),
            "risk_coverage_auc": rc_auc,
            "mean_realized_error": float(true_error.mean()),
            "calibration": self._calibration_bins(cal_err, true_error),
            "model_version": self.version,
        }

    @staticmethod
    def _calibration_bins(pred_err, true_error, n_bins=10):
        """User-side calibration check: split by predicted calibrated error into up to n_bins quantile
        groups and report mean predicted vs mean realized error per group. No fitting, purely diagnostic."""
        n = pred_err.shape[0]
        k = int(min(n_bins, n))
        if k < 1:
            return []
        order = np.argsort(pred_err)
        bins = []
        for i, idx in enumerate(np.array_split(order, k)):
            if idx.size == 0:
                continue
            bins.append({
                "bin": i,
                "n": int(idx.size),
                "mean_predicted_error": float(pred_err[idx].mean()),
                "mean_realized_error": float(true_error[idx].mean()),
            })
        return bins


def default_model():
    """Load the bundled demonstration model. Override the location with the PERTEMA_MODEL_DIR environment
    variable, for example to score against a model refit on your own screen."""
    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.environ.get("PERTEMA_MODEL_DIR") or os.path.join(
        here, "..", "app", "model", "pertema_model_v0.1.0")
    return PertEMAModel(model_dir)
