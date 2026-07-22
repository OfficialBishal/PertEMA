# PertEMA

**Per**turbation **E**rror **M**eta-**A**ssessment. A study of when per-prediction reliability can and cannot be
exploited in single-cell perturbation prediction, with a post-hoc reliability estimator and a self-hostable
demo so others can reproduce and apply the finding.

License: MIT.

## The question and the finding

Deep models that predict how a genetic perturbation reshapes single-cell gene expression are used to nominate
experimental targets, yet on unseen perturbations they frequently fail to beat a trivial mean baseline. Rather
than build a better predictor, this project asks which individual predictions can be trusted, and what a
per-prediction reliability signal can and cannot be used for.

On the public genome-scale Gladstone CD4 T cell Perturb-seq screen, we find:

- No predictor in the roster beats the per-condition mean baseline. Every bootstrap confidence interval on
  error-relative-to-the-mean sits above it (Fig. 1).
- A post-hoc reliability estimator, trained on out-of-fold errors from prediction-time features only, ranks
  realized error above the effect-magnitude and similarity heuristics and calibrates to near-nominal coverage,
  so it supports selective abstention.
- Ranking a validation shortlist by reliability is measurably more reproducible than ranking by effect size,
  the practical payoff for a lab.
- Per-instance model routing is infeasible: candidate predictors co-fail on the same hard, noise-dominated
  perturbations. We quantify this with a measured noise ceiling and a closed-form break-even condition, and a
  pre-registered routing test on an independent screen fails exactly as predicted.

Every number reproduces from the committed result files, and the estimator ships as a self-hostable tool so
others can apply the finding to their own predictor.

## The reliability estimator

PertEMA wraps an already-trained, possibly black-box predictor and estimates, for each prediction, how reliable
it is, using a gradient-boosted-tree meta-model with isotonic calibration and split-conformal intervals,
trained only on leakage-safe, prediction-time features. It is a reliability layer, not a predictor, and it does
not rank biological importance.

## Interactive demo (web app)

So others can apply the finding, the estimator is wrapped in a small self-hostable demo: a FastAPI backend plus
a static frontend that scores a predictor's outputs and explains itself: reliability bands, isotonic-calibrated
error, split-conformal 90 percent intervals, grouped SHAP attribution, a risk-coverage view, a context-transfer
heatmap, an open reliability benchmark, and a downloadable report. Every result surface shows the model version,
the empirical calibration coverage, and the accuracy-ceiling context.

```
pixi run serve      # http://127.0.0.1:8000
pixi run score-test # load the committed frozen model and score the bundled example
pixi run app-test   # exercise the FastAPI app end to end
```

## Environment

Managed with [pixi](https://pixi.sh) (conda-forge and bioconda), pinned in `pixi.toml` and locked in
`pixi.lock`. Seeds for numpy, torch, and CUDA are fixed and recorded, and headline numbers use three seeds.

```
pixi install
```

## Reproduce

### Fast check (about 5 minutes, no primary data, no GPU)

The web app runs against the committed frozen model, and the figures and tables regenerate from the committed
result files in `results/`:

```
pixi run score-test    # load the committed frozen model, score the bundled example, check determinism
pixi run app-test      # exercise the FastAPI app (health, scoring, benchmark, pages)
pixi run python src/pertema/figures_paper.py       # figures from results/ (needs LaTeX on PATH for usetex)
pixi run python src/pertema/build_tables_latex.py  # tables from results/
```

The committed `results/` hold the derived summaries the figures and tables need, and the figures are also
committed as vector PDFs under `figures/paper/` with per-figure `source_data/`. The full pipeline
(`pixi run reproduce-ult`) regenerates the large per-perturbation intermediates and re-freezes the model from
the primary data.

### Full pipeline (primary data + GPU)

```
pixi run reproduce-ult
```

Downloads the primary CRISPRi Perturb-seq screen and three external screens and uses GPU 0 for the predictor
and foundation-model steps. A run report is written to `results/reproduce_report.md`.

### Lighter-weight, pure-PyPI

```
uv venv --python 3.11
uv pip install -r requirements-uv.txt
uv run python src/pertema/figures_paper.py
uv run python src/pertema/build_tables_latex.py
```

The pins mirror `pixi.toml`; relax any pin that does not resolve on your platform.

### Refit on your own screen

The frozen web-app model is trained on the CD4 T cell activation-transfer errors and shipped as a
demonstration. The estimator does not transfer across screens, so for a new screen refit it on that screen's
out-of-fold errors and re-freeze with `pixi run freeze-model`.

## Layout

- `src/` pipeline: data, evaluation, the PertEMA estimator, figures, tables.
- `app/` the web application: backend, frontend, frozen model, Python client, deploy.
- `results/` committed result files, the numeric backbone of every claim.
- `figures/paper/` vector figures with per-figure `source_data/`.

## Data

The analyzed datasets are public releases used under their own terms: the primary CD4 T cell Perturb-seq screen
(openRxiv, DOI 10.64898/2025.12.23.696273), Replogle 2022 (DOI 10.1038/s41588-022-01066-3), Norman 2019
(DOI 10.1126/science.aax4438), and Adamson 2016 (DOI 10.1016/j.cell.2016.11.048).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up, run the tests, and propose
a change, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the community standards.

## Citation

If you use PertEMA, please cite the software (there is no accompanying paper yet):

> Bishal Shrestha. *PertEMA: Perturbation Error Meta-Assessment* (Version 0.1.0) [Computer software]. 2026.
> https://github.com/OfficialBishal/PertEMA

```bibtex
@software{shrestha_pertema_2026,
  author  = {Shrestha, Bishal},
  title   = {{PertEMA}: Perturbation Error Meta-Assessment},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/OfficialBishal/PertEMA}
}
```

GitHub's "Cite this repository" button (generated from [CITATION.cff](CITATION.cff)) produces APA and BibTeX
automatically.
