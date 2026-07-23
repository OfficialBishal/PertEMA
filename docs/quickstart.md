# Quickstart

PertEMA scores the reliability of a predictor's outputs. Nothing here needs a GPU or the primary data: the
shipped path runs from a committed frozen model.

## Install

The canonical environment is [pixi](https://pixi.sh):

```
pixi install
```

A CPU-only path with pip works too:

```
python -m pip install numpy==1.26.4 scipy==1.17.1 pandas==3.0.1 scikit-learn==1.8.0 xgboost==3.2.0 \
                      fastapi uvicorn httpx python-multipart
```

## Run the bundled example

Score the built-in 400-perturbation example and check that scoring is deterministic:

```
pixi run score-test
```

Expected output reports the model version, the number of perturbations scored, the calibration coverage, and
the reliability-band distribution.

## Use as a library

Install the package and score a feature matrix directly:

```
pip install -e .
```

```python
import pertema

model = pertema.default_model()      # the bundled demonstration model
scored = model.score(features)       # per-row calibrated error, reliability band, and conformal interval
```

Set `PERTEMA_MODEL_DIR` to point at a model refit on your own screen to score against it instead.

## Score your own predictions

Start the service and open it in a browser:

```
pixi run serve      # http://127.0.0.1:8000
```

Provide your predictor's outputs as a long-format CSV with one row per (perturbed gene, measured gene):

```
perturbed_gene,gene,predicted_lfc,src_context,dst_context
IL2,IL2RA,0.83,Rest,Stim48hr
IL2,IFNG,-0.21,Rest,Stim48hr
STAT5A,MYC,0.44,Rest,Stim48hr
```

- `perturbed_gene` and `gene` accept HGNC symbols or Ensembl IDs.
- `predicted_lfc` is your model's predicted change (delta log fold change).
- `src_context` and `dst_context` are optional and must be one of `Rest`, `Stim8hr`, or `Stim48hr`
  (default `Rest` to `Stim48hr`). Rows with an unsupported context are reported and skipped.

The per-perturbation input magnitude PertEMA uses is the mean of the absolute `predicted_lfc` over that
perturbation's genes. For each perturbation you get back a reliability score, a calibrated error, a reliability
band, a 90 percent conformal interval, and a grouped attribution of what drove the score.

See [Interpreting reliability](interpreting_reliability.md) for how to read these outputs.

## Note on scope

The shipped frozen model is trained on the CD4 T cell activation-transfer errors of one screen and is a
demonstration. The estimator does not transfer across screens, so for a new screen refit it on that screen's
out-of-fold errors and re-freeze with `pixi run freeze-model`.
