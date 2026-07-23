# PertEMA documentation

PertEMA (Perturbation Error Meta-Assessment) is a model-agnostic, post-hoc reliability layer for single-cell
perturbation-effect predictors. It wraps an already-trained predictor and estimates, for each individual
prediction, how much it can be trusted: a calibrated error estimate, a reliability band, and a 90 percent
split-conformal interval. It does not build a better predictor and it does not rank biological importance.

- [Quickstart](quickstart.md): install, run the bundled example, and score your own predictor's outputs.
- [Interpreting reliability](interpreting_reliability.md): what the score, the bands, and the interval mean,
  and the honest scope of the tool.

The project source, license, and issue tracker are at <https://github.com/OfficialBishal/PertEMA>. These pages
are plain Markdown and are intended to be rendered as a documentation site in a later revision.
