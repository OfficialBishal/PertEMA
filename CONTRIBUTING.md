# Contributing to PertEMA

Contributions are welcome: bug reports, fixes, tests, documentation, new predictor adapters, and evaluation
improvements. This document explains how to get set up and what the project asks of a change.

## Getting set up

The canonical environment is [pixi](https://pixi.sh), pinned in `pixi.toml` and locked in `pixi.lock`:

```
pixi install
pixi run score-test   # load the committed frozen model, score the bundled example, check determinism
pixi run app-test     # exercise the FastAPI app end to end
```

A lighter, CPU-only path uses pip or uv with the pinned dependencies:

```
python -m pip install numpy==1.26.4 scipy==1.17.1 pandas==3.0.1 scikit-learn==1.8.0 xgboost==3.2.0 \
                      fastapi uvicorn httpx python-multipart
python app/backend/test_scoring.py
python app/backend/test_api.py
```

Both test scripts run from the committed frozen model, so they need no GPU and no primary data.

## Proposing a change

1. Open an issue describing the problem or proposal before a large change, so the design can be discussed.
2. Fork, branch from `main`, and keep each change focused and small.
3. Add or update tests so the behavior you change is covered. Keep the two test scripts passing.
4. Update the documentation and the `CHANGELOG.md` when the change is user facing.
5. Open a pull request. CI runs the tests on every push.

## What the project asks of a change

- Correctness and honesty first. Report negative results plainly. The words novel, first, state of the art,
  best, and outperforms appear only next to a specific measured comparison with numbers.
- No leakage. Information about a test perturbation's true error must never reach the estimator during training.
- Reproducibility. Fix and record seeds. Every headline number must trace to a committed result file, and code,
  results, figures, and docs must stay mutually consistent.
- Prose style in the repository is ASCII only, no em dashes, no semicolons in prose, no emoji.

By contributing you agree that your contributions are licensed under the project's MIT License, and you agree to
follow the [Code of Conduct](CODE_OF_CONDUCT.md).
