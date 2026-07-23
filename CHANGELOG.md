# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A pytest unit-test suite under `tests/` covering the package API, ingestion, featurization, and the
  command-line interface, wired into CI (`pytest tests/`).
- A `pertema` command-line interface: `pertema score <csv>` (CSV or JSON output, reads a file or stdin) and
  `pertema example`, installed as a console script and runnable with `python -m pertema`.
- An importable `pertema` package with a public API (`import pertema`), installable with `pip install -e .`
  (`pyproject.toml`). The scoring core (scoring, featurization, ingestion, report) moved from `app/backend/`
  into the package, the FastAPI service and tests now import it, and the model location can be overridden with
  the `PERTEMA_MODEL_DIR` environment variable.
- A documentation site: a MkDocs Material configuration (`mkdocs.yml`) and a readthedocs build
  (`.readthedocs.yaml`, `docs/requirements.txt`) that render the `docs/` pages into a browsable site.
- User documentation under `docs/`: a quickstart and an interpreting-reliability concept page, linked from the
  README.
- A draft software paper (`paper/paper.md`, `paper/paper.bib`) with a statement of need and a state-of-the-field
  section that presents the mean-baseline result as metric-dependent and cites the debate.
- Citation metadata (`CITATION.cff`) so the tool can be cited directly.
- Contributor guide, code of conduct, and security policy (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`).
- Continuous integration: the scoring-core and end-to-end API tests run on every push and pull request
  (`.github/workflows/ci.yml`).

## [0.1.0]

### Added
- Initial public release of the PertEMA reliability estimator: a model-agnostic, post-hoc reliability layer for
  single-cell perturbation-effect predictors, using a gradient-boosted meta-model over leakage-safe,
  prediction-time features with isotonic calibration and split-conformal intervals.
- A self-hostable FastAPI service that scores a predictor's outputs and reports reliability bands, calibrated
  error, split-conformal intervals, grouped SHAP attribution, a risk-coverage view, and a downloadable report.
- The committed frozen model, the reproducible figures and tables, and the CPU reproduction path.
