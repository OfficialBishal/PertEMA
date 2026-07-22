# PertEMA — browser build

A **fully client-side** build of the PertEMA reliability tool. It is the same reliability layer as the
FastAPI service in [`../app`](../app), but the entire scoring pipeline runs in the visitor's browser:

> long-format CSV → featurization → 300-tree gradient-boosted estimator → isotonic calibration →
> split-conformal interval → grouped TreeSHAP attribution

No server, no install, no upload. A predictor's outputs are scored **locally**, so unpublished predictions
never leave the machine. That makes it free to host as static files and safe to use on sensitive data.

## Run it

Serve this directory with any static file server:

```
cd web
python -m http.server 8080      # then open http://localhost:8080/
```

Or deploy the `web/` directory as-is to any static host (Cloudflare Pages, GitHub Pages, Netlify, S3, …).
A hosted instance runs at <https://shresthabishal.com/projects/pertema/>.

## Layout

- `index.html` + `methods.html`, `interpret.html`, `model_card.html`, `benchmark.html`, `api.html`,
  `faq.html` — the same pages as the server app, with all links relative so the site works under any path.
- `js/` — the scoring port, dependency-free ES modules:
  - `ingest.mjs` — `parse_long_csv` (CSV → per-perturbation `mean(|predicted_lfc|)`).
  - `featurize.mjs` — the exact 64-d prediction-time feature vector.
  - `xgb.mjs` — XGBoost `predict` (float32) **and** exact path-dependent TreeSHAP (`pred_contribs`).
  - `model.mjs` — isotonic calibration, reliability bands, split-conformal interval, SHAP grouping.
  - `pipeline.mjs`, `report.mjs`, `assets.mjs` — the `/ingest_score` + `/example` envelopes, the HTML
    report, and the asset loader.
- `data/` — the frozen model and reference tables exported for the browser (`estimator.json`,
  the calibration map, the co-expression embedding and control baselines as little-endian float32 `.bin`,
  the gene map, similarity and neighbor tables, the bundled example, and the benchmark/heatmap JSON).
- `tools/export_web_assets.py` — regenerates `data/` from `../app/model/` (run in the project environment).

## Parity

The JS modules are a faithful port of `app/backend/{scoring,featurize,ingest}.py` and were verified
bit-for-bit against them: `predict` is bit-exact against XGBoost (float32 accumulation), TreeSHAP matches
`pred_contribs` to ~1e-9 with the additive property holding to <1e-6, and calibration, bands, conformal
intervals, and featurized vectors match across the bundled 400-perturbation example and adversarial edge
cases (missing embeddings, duplicate genes, symbol/Ensembl resolution, band boundaries, malformed input).

## When to use which

- **`web/` (this build)** — a zero-dependency, private, no-install way to score a predictor's outputs in a
  browser. Same numbers as the server, nothing to run.
- **`../app` (FastAPI)** — the REST API and programmatic/self-hosted service, for batch scoring or
  integration into a pipeline.
