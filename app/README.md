# PertEMA web application

A self-hostable web tool that wraps any single-cell perturbation predictor with a post-hoc reliability layer.
Give it a predictor's outputs and get back per-perturbation reliability scores, isotonic-calibrated error,
split-conformal intervals, grouped SHAP attribution, and a downloadable report. The tool is a reliability
layer, not a predictor, and that scope is stated on every surface.

## Run it

```
pixi run serve       # FastAPI at http://127.0.0.1:8000
pixi run score-test  # load the committed frozen model and score the bundled example
pixi run app-test    # exercise the FastAPI app end to end
```

Open http://127.0.0.1:8000 and click "Run the bundled example", or paste or upload a predictor's outputs as a
long-format CSV.

## Frozen model artifact

`app/model/pertema_model_v0.1.0/` holds the deployable estimator, loaded read-only:
- `estimator.json`: the gradient-boosted-tree reliability estimator (xgboost, 300 trees, depth 6), trained on
  the CD4 T cell activation-transfer errors of the per-condition mean predictor.
- `calibration.npz`: the isotonic recalibration map.
- `feature_spec.json`: the exact prediction-time feature specification (leakage-safe, no true-effect quantity).
- `provenance.json`: model version, training-data hash, seeds, the measured accuracy ceiling, the held-out
  calibration (ECE 0.0028, conformal coverage 0.900), and the scope note.

Regenerate with `pixi run freeze-model` (round-trip load verified to float epsilon).

## Structure

- `app/backend/`: FastAPI service (`main.py`) loading the frozen artifact. Scoring, explanation, ingestion,
  evaluation, the example, the benchmark, a rendered report, health, and version, with optional token auth and
  auto-generated OpenAPI. Scoring and featurization in `scoring.py`, `featurize.py`, `ingest.py`; the HTML
  report in `report.py`; tests in `test_api.py` and `test_scoring.py`.
- `app/frontend/`: self-contained static HTML pages (no build step, no external CDNs): the tool, methods,
  interpret, model card, benchmark, API, and FAQ, sharing one visual shell.
- `app/client/`: a small Python client (`pertema_client.py`).
- `app/deploy/`: Dockerfile, docker-compose, and an operations guide.
- `app/examples/`: the bundled example dataset and the script that builds it.

## Integrity

Every result surface shows the model version, the empirical calibration coverage, and the accuracy-ceiling
context. SHAP attribution explains the estimator, not biological importance. The methods page states plainly
that gains are modest on the noisy primary data and clearer on the external cross-cell-line axis, and no
superlative appears without a measured comparison beside it. The estimator is never retrained on user-supplied
truth.
