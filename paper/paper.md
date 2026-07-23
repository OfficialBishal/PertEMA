---
title: "PertEMA: post-hoc reliability estimation for single-cell perturbation prediction"
tags:
  - Python
  - single-cell
  - Perturb-seq
  - perturbation prediction
  - uncertainty quantification
  - conformal prediction
  - calibration
  - selective prediction
authors:
  - name: Bishal Shrestha
    orcid: 0009-0004-8330-4348
    affiliation: 1
affiliations:
  - name: University of Miami, United States
    index: 1
date: 22 July 2026
bibliography: paper.bib
---

# Summary

Models that predict how a genetic or chemical perturbation reshapes single-cell gene expression are used to
nominate experimental targets, but their accuracy varies sharply from one prediction to the next, and on unseen
perturbations they are often no better than simple baselines. `PertEMA` (Perturbation Error Meta-Assessment) is
a model-agnostic, post-hoc reliability layer that wraps an already-trained, possibly black-box predictor and
estimates, for each individual prediction, how much it can be trusted. It does not build a better predictor and
it does not rank biological importance. Given a predictor's outputs, it returns a calibrated error estimate, a
reliability band, and a split-conformal interval, using a gradient-boosted meta-model trained only on
leakage-safe, prediction-time features with isotonic calibration [@zadrozny2002; @angelopoulos2023]. The result
lets a laboratory practice selective abstention: keep the predictions worth acting on and set aside the rest.
`PertEMA` ships as an importable Python package, a small command-line interface, a self-hostable service, and a
fully client-side browser build, and every result surface reports the model version, the empirical calibration
coverage, and the accuracy-ceiling context so a score is never separated from what it does and does not claim.

# Statement of need

Perturbation-prediction models are proliferating, from graph and foundation models to linear and additive
baselines [@gears; @ahlmanneltze2025], and a laboratory that wants to use them to prioritize experiments faces a
practical question that accuracy benchmarks do not answer: for this specific perturbation, in this context, can I
trust this prediction. Existing community tooling standardizes model comparison across datasets and metrics
[@perturbench; @scperturbench], but it evaluates predictors in aggregate rather than scoring the reliability of
one prediction at a time. `PertEMA` fills that gap. It is a reliability and selective-prediction layer that sits
on top of any predictor's outputs, so it composes with the benchmarking ecosystem rather than competing with it.

The need is sharpened by an active and unresolved debate about whether learned predictors beat trivial
baselines. Under conventional error metrics, several studies report that no deep-learning model consistently
beats a per-condition mean or additive baseline on unseen perturbations [@ahlmanneltze2025; @wong2025]. Other
work argues this conclusion is largely an artifact of poorly calibrated metrics, because mean-squared error and
control-referenced delta correlation reward collapse toward the dataset mean, whereas rank-based and
differentially-expressed-gene-weighted metrics restore sensitivity [@metriccalibration2025]. The reliability
question is orthogonal to and survives either side of this debate: whichever metric a laboratory adopts, it still
needs to know which of its predictions to act on. `PertEMA` is built to be read against that choice of metric.

# State of the field

Uncertainty quantification for single-cell models is young, and per-prediction reliability for perturbation
prediction in particular is underserved. Predictor-benchmarking frameworks such as PerturBench [@perturbench] and
scPerturBench [@scperturbench] rank methods across datasets, and split-conformal prediction offers
distribution-free coverage guarantees that `PertEMA` uses for its intervals [@angelopoulos2023]. What has been
missing is a lightweight, model-agnostic layer that turns those ideas into a per-prediction reliability score a
biologist can use directly, with leakage-safe features and calibrated coverage. On the shipped demonstration
model the split-conformal intervals reach an empirical coverage of about 0.900 at a 0.900 target and an isotonic
expected calibration error of about 0.003, and the accuracy ceiling implied by cross-donor replicate noise is
reported alongside every result so users read scores against what is achievable rather than in the abstract.

# Availability

`PertEMA` is released under the MIT License at <https://github.com/OfficialBishal/PertEMA>. It runs on CPU with
no primary data required for the shipped path: the scoring core and the end-to-end service are exercised by a
test suite that runs from a committed frozen model, and continuous integration runs those tests on every change.
Figures and tables regenerate from committed result files, and the estimator can be refit on a laboratory's own
screen.

# References
