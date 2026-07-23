# Interpreting reliability

PertEMA returns, for each prediction, a small set of quantities. This page explains what they mean and, just as
important, what they do not.

## The reliability score and the calibrated error

The estimator predicts a calibrated error for each prediction, on the scale of `1 - Pearson` correlation
between the predicted and the true post-perturbation effect, where lower is better. The reliability score is
the negative of that calibrated error, so a higher reliability score means a more trustworthy prediction. The
calibration step is isotonic regression fit on out-of-fold errors, so the reported error is on a meaningful
scale rather than an arbitrary model output.

## The reliability bands

The calibrated error is bucketed into four labelled bands with fixed thresholds (from `scoring.py`):

| band | calibrated error range | reading |
|----------|------------------------|----------------------------------------------------|
| high | 0.00 to 0.80 | among the more trustworthy predictions on this noisy data |
| moderate | 0.80 to 0.88 | usable with caution |
| low | 0.88 to 0.94 | treat with skepticism |
| very-low | 0.94 and above | likely unreliable, deprioritize |

Bands are always shown with their label, never by color alone. A practical use is selective abstention: keep
the high and moderate predictions for follow-up and set the rest aside.

## The conformal interval

Alongside the point estimate, each prediction carries a 90 percent split-conformal interval on the calibrated
error. On the shipped frozen model the empirical coverage is about 0.900 at a 0.900 target, and the isotonic
expected calibration error is about 0.003, both taken from the model provenance. The interval gives a
distribution-free sense of how much the calibrated error could move.

## The attribution

For each prediction the tool reports a grouped attribution of which feature groups pushed the score toward more
or less reliable. This explains the estimator's reasoning, not biological importance: a large contribution from
a feature group does not mean the corresponding gene or context is biologically important.

## Honest scope

- PertEMA is a reliability layer, not a predictor. It scores which predictions to trust, and it does not rank
  which genes are important.
- The reliability signal is modest on this noisy data. The tool reports the honest out-of-fold reliability on
  every result surface and in the benchmark, so scores are never read in the abstract.
- Every score is shown against an accuracy ceiling implied by cross-donor replicate noise. On the shipped model
  that ceiling is a Pearson r of about 0.73 on the hit-gene set and about 0.11 across all genes, which bounds
  what any predictor could achieve.
- Whether a predictor beats a simple baseline at all is metric-dependent and actively debated: conventional
  mean-squared error and control-referenced correlation reward collapse toward the dataset mean, whereas
  rank-based and differentially-expressed-gene-weighted metrics can restore sensitivity. PertEMA is a
  reliability layer either way, because a laboratory still needs to know which predictions to act on.
- The shipped model is a demonstration trained on one screen and does not transfer across screens.
