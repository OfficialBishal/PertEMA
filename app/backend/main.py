"""PertEMA backend service (FastAPI). Thin web wrapper over the tested, framework-agnostic scoring core
(scoring.py). Loads the frozen artifact read-only. Every result carries the model version, the empirical
calibration coverage, and the accuracy-ceiling context (invariant N2). No banned superlative appears in any
copy without a measured comparison beside it.

To run (one dependency add into the pixi env, keeps the scientific stack pinned):
    pixi add fastapi uvicorn python-multipart
    pixi run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

The scoring core it wraps is already tested (app/backend/test_scoring.py passes); this module adds only the
HTTP surface, so it is intentionally thin.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # so `from scoring import ...` works however uvicorn launches this module

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastapi.responses import HTMLResponse, PlainTextResponse
from scoring import BANDS, FEATURE_GROUPS, default_model
from featurize import default_featurizer
from ingest import LONG_TEMPLATE, parse_long_csv
from report import render_report
MODEL = default_model()
FEATURIZER = default_featurizer()
CEILING = MODEL.provenance["measured_accuracy_ceiling_1minus_pearson"]


def _build_commit():
    """N1 build stamp: the commit the app is running, from an env var (set at deploy time) or git or unknown.
    Honest: reports 'unknown' rather than guessing when neither is available."""
    c = os.environ.get("PERTEMA_BUILD_COMMIT", "").strip()
    if c:
        return c
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, timeout=3).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


BUILD_COMMIT = _build_commit()


def build_stamp():
    """The provenance stamp attached to every report (N1)."""
    ev = MODEL.provenance.get("evaluation_calibration", {})
    return {"model_version": MODEL.version, "provenance_flag": MODEL.provenance["provenance_flag"],
            "build_commit": BUILD_COMMIT, "calibration_coverage": ev.get("conformal_coverage", 0.9),
            "conformal_target": ev.get("conformal_target", 0.9),
            "accuracy_ceiling_1minus_pearson": CEILING,
            "estimator": MODEL.provenance.get("estimator", "gradient-boosted tree"),
            "honest_scope": MODEL.provenance.get("honest_scope", "Reliability estimator, not importance.")}

# N3/N4 robustness and security limits. A hard cap on rows so a malformed or oversized request is rejected
# cleanly rather than exhausting memory. Optional bearer-token auth, enabled only if PERTEMA_API_TOKEN is set
# (so the demo runs open, a deployment can lock it down via one env var). CORS locked to configured origins.
MAX_ROWS = int(os.environ.get("PERTEMA_MAX_ROWS", "200000"))
API_TOKEN = os.environ.get("PERTEMA_API_TOKEN", "")
CORS_ORIGINS = [o for o in os.environ.get("PERTEMA_CORS_ORIGINS", "http://localhost:8000").split(",") if o]

app = FastAPI(title="PertEMA reliability API",
              description="Post-hoc, model-agnostic reliability scoring for perturbation predictors. "
                          "A reliability layer, not a predictor. Gains are modest on noisy data; the honest "
                          "out-of-fold reliability and the accuracy ceiling are reported on every result.",
              version=MODEL.version)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def require_token(authorization: str = Header(default="")):
    """Bearer-token auth on the scoring endpoints, enforced only when PERTEMA_API_TOKEN is set."""
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, "missing or invalid bearer token")


def _check_matrix(features):
    X = np.asarray(features, float)
    expected = 64
    if X.ndim != 2 or X.shape[1] != expected:
        raise HTTPException(422, f"features must be (n, {expected}); got {X.shape}. See /version feature_spec.")
    if X.shape[0] > MAX_ROWS:
        raise HTTPException(413, f"too many rows: {X.shape[0]} > limit {MAX_ROWS}. Split into batches.")
    if not np.isfinite(X).all():
        raise HTTPException(422, "features contain NaN or infinite values.")
    return X


class ScoreRequest(BaseModel):
    features: list[list[float]]     # (n, d) prediction-time feature matrix matching feature_spec.json


class EvaluateRequest(BaseModel):
    features: list[list[float]]     # (n, d) prediction-time feature matrix matching feature_spec.json
    true_error: list[float]         # (n,) the user's realized per-perturbation error (1 - Pearson)


def _envelope(out):
    return {
        "model_version": out["model_version"],
        "estimator": MODEL.provenance.get("estimator", "gradient-boosted tree"),
        "calibration_coverage": out["coverage"],
        "accuracy_ceiling_1minus_pearson": CEILING,
        "honest_note": ("Reliability estimation, not biological importance. Out-of-fold reliability is modest "
                        "on noisy data; see the benchmark and the methods page."),
        "bands_legend": [{"band": b[0], "calibrated_error_range": [b[1], b[2]], "meaning": b[3]} for b in BANDS],
        "results": [
            {"reliability": float(r), "calibrated_error": float(e), "band": b,
             "conformal_interval": [float(lo), float(hi)]}
            for r, e, b, lo, hi in zip(out["reliability"], out["calibrated_error"], out["band"],
                                       out["conformal_lo"], out["conformal_hi"])
        ],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "..", "frontend", "index.html"))


@app.get("/methods")
def methods():
    return FileResponse(os.path.join(HERE, "..", "frontend", "methods.html"))


@app.get("/interpret")
def interpret():
    return FileResponse(os.path.join(HERE, "..", "frontend", "interpret.html"))


@app.get("/faq")
def faq():
    return FileResponse(os.path.join(HERE, "..", "frontend", "faq.html"))


@app.get("/api")
def api_docs():
    return FileResponse(os.path.join(HERE, "..", "frontend", "api.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL.version}


@app.get("/version")
def version():
    return {**MODEL.provenance, "build_commit": BUILD_COMMIT}


@app.post("/score")
def score(req: ScoreRequest, _=Depends(require_token)):
    X = _check_matrix(req.features)
    return _envelope(MODEL.score(X))


@app.post("/explain")
def explain(req: ScoreRequest, _=Depends(require_token)):
    """F6 feature attribution: per-perturbation grouped SHAP contributions toward the predicted error, so a
    user can see WHY a prediction was scored reliable or not. Contributions are exact (xgboost pred_contribs)."""
    X = _check_matrix(req.features)
    return {"model_version": MODEL.version, "explanations": MODEL.explain(X),
            "note": ("Grouped SHAP contributions toward the predicted error (1 - Pearson). A positive "
                     "contribution pushes the prediction toward less reliable, a negative toward more "
                     "reliable. Attribution explains the estimator, not biological importance.")}


class IngestRequest(BaseModel):
    csv: str                        # a long-format predictions CSV (see /template)
    src_default: str = "Rest"
    dst_default: str = "Stim48hr"


@app.get("/template", response_class=PlainTextResponse)
def template():
    """The long-format predictions CSV template a user fills with their predictor's outputs."""
    return LONG_TEMPLATE


@app.get("/example.csv", response_class=PlainTextResponse)
def example_csv():
    """The 400-perturbation bundled example as a long-format predictions CSV."""
    return open(os.path.join(HERE, "..", "examples", "example_predictions.csv")).read()


@app.post("/ingest_score")
def ingest_score(req: IngestRequest, _=Depends(require_token)):
    """Ingest a user's predictor outputs (a long-format CSV of perturbed_gene, gene, predicted_lfc, optional
    contexts), featurize them into the exact model feature space, and score their reliability. Gene symbols
    are mapped to the model's namespace and the mapped fraction is reported; unsupported contexts are
    rejected. The features are computed at inference and match the training feature construction."""
    if len(req.csv) > 50_000_000:
        raise HTTPException(413, "CSV too large (over 50 MB). Split into batches.")
    try:
        preds, ingest_report = parse_long_csv(req.csv, req.src_default, req.dst_default)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not preds:
        raise HTTPException(422, "no valid perturbations parsed from the CSV.")
    X, syms, feat_report = FEATURIZER.featurize(preds)
    if X.shape[0] == 0:
        raise HTTPException(422, f"no perturbations could be featurized. {feat_report}")
    env = _envelope(MODEL.score(X))
    env["genes"] = syms
    env["explanations"] = MODEL.explain(X)                          # F6: why each prediction was scored so
    env["neighbors"] = {s: FEATURIZER.neighbors(s) for s in syms}   # F6: co-expression gene neighborhood
    env["ingestion_report"] = ingest_report
    env["featurize_report"] = feat_report
    env["note"] = ("Reliability of YOUR predictions, scored post hoc. Features are computed at inference from "
                   "the bundled reference and match the training construction. Gains are modest on noisy data "
                   "and the honest out-of-fold reliability is about 0.13; see /methods.")
    return env


@app.post("/report", response_class=HTMLResponse)
def report(req: IngestRequest, _=Depends(require_token)):
    """Ingest, featurize, and score a user's predictions (as /ingest_score), then return a self-contained HTML
    reliability report (F8): one file, no external dependencies, stamped with provenance (model version,
    build commit, calibration coverage, accuracy ceiling) and the honest-scope note."""
    if len(req.csv) > 50_000_000:
        raise HTTPException(413, "CSV too large (over 50 MB). Split into batches.")
    try:
        preds, ingest_report = parse_long_csv(req.csv, req.src_default, req.dst_default)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not preds:
        raise HTTPException(422, "no valid perturbations parsed from the CSV.")
    X, syms, feat_report = FEATURIZER.featurize(preds)
    if X.shape[0] == 0:
        raise HTTPException(422, f"no perturbations could be featurized. {feat_report}")
    env = _envelope(MODEL.score(X))
    env["genes"] = syms
    env["explanations"] = MODEL.explain(X)                          # P3-5: top-3 SHAP drivers per row in report
    extra = {"perturbations scored": len(syms), "mapped fraction": feat_report["mapped_fraction"],
             "unmapped genes": feat_report["n_unmapped_genes"], "rejected contexts": feat_report["n_bad_context"]}
    return render_report(env, build_stamp(), extra)


@app.post("/evaluate")
def evaluate(req: EvaluateRequest, _=Depends(require_token)):
    """Evaluate a predictor: score the user's features, then measure the REALIZED reliability quality on
    the user's own ground truth. The shipped estimator is never retrained on that truth (inviolable
    invariant). The response carries explicit leakage and split warnings because these numbers are only
    honest on out-of-sample ground truth."""
    X = _check_matrix(req.features)
    y = np.asarray(req.true_error, float)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise HTTPException(422, f"true_error must be (n,) matching the {X.shape[0]} feature rows; got {y.shape}.")
    out = MODEL.evaluate(X, y)
    out["accuracy_ceiling_1minus_pearson"] = CEILING
    out["metrics_legend"] = {
        "spearman_reliability_vs_accuracy": ("Spearman rho of predicted reliability vs realized accuracy "
                                             "(-true_error); higher is better ranking, range -1 to 1."),
        "risk_coverage_auc": ("mean realized error averaged over all coverage levels when abstaining from "
                              "the least-reliable predictions first; lower is better, units of 1 - Pearson."),
        "calibration": ("quantile bins of predicted calibrated error vs mean realized error, a user-side "
                        "reliability diagram; well calibrated means the two columns track each other."),
    }
    out["warnings"] = [
        ("LEAKAGE: this endpoint never retrains the shipped estimator on your ground truth. Your true_error "
         "is used only to measure realized quality, never to fit or adapt the estimator."),
        ("SPLIT: supply OUT-OF-SAMPLE ground truth. Scoring the estimator's own training data (for example "
         "the bundled example, which is training-derived) gives in-sample, optimistic quality, not a "
         "generalization estimate."),
    ]
    return out


@app.get("/example")
def example():
    """Score the bundled example. NOTE: it is a slice of the model's own training data, so its performance is
    in-sample and optimistic; it demonstrates the flow, it is not a quality claim."""
    ex = np.load(os.path.join(HERE, "..", "examples", "example_gladstone.npz"), allow_pickle=True)
    X = ex["features"]
    env = _envelope(MODEL.score(X))
    env["genes"] = [str(g) for g in ex["genes"]]
    env["explanations"] = MODEL.explain(X)                              # F6: why each prediction was scored so
    env["neighbors"] = {s: FEATURIZER.neighbors(s) for s in env["genes"]}   # F6: co-expression gene neighborhood
    env["in_sample_warning"] = ("The bundled example is training-derived; its scores are in-sample and "
                                "optimistic. A user's own unseen predictions get out-of-sample reliability.")
    return env


@app.get("/transfer_heatmap")
def transfer_heatmap():
    """The context-transfer difficulty map: mean 1 - Pearson error for a predictor trained in the source
    context and applied in the destination context. The flagship of the primary contribution (F4). Within-
    context transfers (the diagonal) are easier; resting-to-stimulated transfers are hardest."""
    import pandas as pd
    ref = os.path.join(HERE, "..", "model", "pertema_model_v0.1.0", "reference", "transfer_pair_errors.csv")
    df = pd.read_csv(ref)
    conds = ["Rest", "Stim8hr", "Stim48hr"]
    piv = df.pivot(index="src", columns="dst", values="mean_1_minus_pearson").reindex(index=conds, columns=conds)
    return {"conditions": conds, "matrix": [[round(float(piv.loc[s, d]), 4) for d in conds] for s in conds],
            "note": ("mean 1 - Pearson error, lower is easier. Diagonal is within-context; off-diagonal is "
                     "context transfer. Resting-to-stimulated transfers are hardest, matching the larger "
                     "transcriptional change on activation.")}


@app.get("/benchmark")
def benchmark():
    """The rendered benchmark page (datasets x methods x transfer settings). The raw numbers are at
    /benchmark.json so the page and any programmatic reader share one source of truth."""
    return FileResponse(os.path.join(HERE, "..", "frontend", "benchmark.html"))


@app.get("/benchmark.json")
def benchmark_json():
    import pandas as pd
    p = os.path.join(HERE, "..", "..", "results", "benchmark", "benchmark_reliability.csv")
    if not os.path.exists(p):
        raise HTTPException(404, "benchmark table not built; run pixi run python src/pertema/build_benchmark.py")
    import json
    df = pd.read_csv(p)
    return json.loads(df.to_json(orient="records"))                 # pandas nulls NaN correctly (JSON compliant)


@app.get("/model-card")
def model_card():
    """The rendered Model Card page. Its numbers are served from /model-card.json (sourced from the frozen
    provenance) so the page cannot drift from the artifact."""
    return FileResponse(os.path.join(HERE, "..", "frontend", "model_card.html"))


@app.get("/model-card.json")
def model_card_json():
    """Model Card fields sourced directly from the frozen provenance.json so the numbers cannot drift.
    The accuracy ceiling is labeled here as Pearson r (higher is better): the frozen provenance key is named
    measured_accuracy_ceiling_1minus_pearson but stores Pearson r, not 1 - Pearson. The key is not renamed."""
    p = MODEL.provenance
    fs = p.get("feature_spec", {})
    ev = p.get("evaluation_calibration", {})
    ceil = p.get("measured_accuracy_ceiling_1minus_pearson", {})
    return {
        "model_version": p["model_version"],
        "estimator": p.get("estimator", "gradient-boosted tree"),
        "trained_on": p.get("trained_on"),
        "n_training_perturbations": p.get("n_training_perturbations"),
        "seeds": p.get("seeds"),
        "training_data_hash_sha256_16": p.get("training_data_hash_sha256_16"),
        "feature_spec": fs,
        "target": fs.get("target"),
        "attribution_groups": [{"group": name, "indices": idx, "n": len(idx)} for name, idx in FEATURE_GROUPS],
        "calibration": {"ece_raw": ev.get("ece_raw"), "ece_isotonic": ev.get("ece_isotonic"),
                        "conformal_coverage": ev.get("conformal_coverage"),
                        "conformal_target": ev.get("conformal_target")},
        "accuracy_ceiling_pearson_r": {
            "hit_genes": ceil.get("hit_genes"), "all_genes": ceil.get("all_genes"),
            "note": ("Pearson r, the best achievable correlation given cross-donor replicate noise (higher is "
                     "better). Stored under the frozen key measured_accuracy_ceiling_1minus_pearson, which "
                     "holds Pearson r, not 1 - Pearson.")},
        "honest_scope": p.get("honest_scope"),
        "provenance_flag": p.get("provenance_flag"),
        "build_commit": BUILD_COMMIT,
    }


@app.get("/feature-groups")
def feature_groups():
    """One source of truth for what the model looks at. feature_spec.features are the 11 leakage-safe
    prediction-time inputs; attribution_groups are the 6 interpretable groups used for grouped SHAP
    attribution (/explain). Prediction-time only, no true-effect quantity."""
    return {
        "model_version": MODEL.version,
        "feature_spec": MODEL.provenance.get("feature_spec", {}),
        "attribution_groups": [{"group": name, "indices": idx, "n": len(idx)} for name, idx in FEATURE_GROUPS],
        "note": ("feature_spec.features are the 11 leakage-safe prediction-time inputs; attribution_groups are "
                 "the 6 interpretable groups the estimator's SHAP contributions are summed within."),
    }
