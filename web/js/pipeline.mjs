// Composes the client-side equivalents of the /example and /ingest_score endpoints from main.py.
// Honest-scope note strings are reproduced verbatim (they are load-bearing integrity copy).

import { parseLongCsv } from "./ingest.mjs";
import { BANDS } from "./model.mjs";

const HONEST_NOTE =
  "Reliability estimation, not biological importance. Out-of-fold reliability is modest on noisy data; " +
  "see the benchmark and the methods page.";
const INGEST_NOTE =
  "Reliability of YOUR predictions, scored post hoc. Features are computed at inference from the bundled " +
  "reference and match the training construction. Gains are modest on noisy data and the honest " +
  "out-of-fold reliability is about 0.13; see /methods.";
const IN_SAMPLE_WARNING =
  "The bundled example is training-derived; its scores are in-sample and optimistic. A user's own unseen " +
  "predictions get out-of-sample reliability.";

function envelope(model, scored) {
  return {
    model_version: model.version,
    estimator: model.provenance.estimator,
    calibration_coverage: model.coverage,
    accuracy_ceiling_1minus_pearson: model.ceiling,
    honest_note: HONEST_NOTE,
    bands_legend: BANDS.map(([band, lo, hi, meaning]) => ({ band, calibrated_error_range: [lo, hi], meaning })),
    results: scored.map((s) => ({
      reliability: s.reliability,
      calibrated_error: s.calibrated_error,
      band: s.band,
      conformal_interval: [s.conformal_lo, s.conformal_hi],
    })),
  };
}

// /ingest_score — the core path. csvText is the user's long-format predictions.
export function ingestScore(model, featurizer, csvText, srcDefault = "Rest", dstDefault = "Stim48hr") {
  const { preds, report: ingestion_report } = parseLongCsv(csvText, srcDefault, dstDefault);
  if (!preds.length) throw new Error("no valid perturbations parsed from the CSV.");
  const { X, syms, report: featurize_report } = featurizer.featurize(preds);
  if (X.length === 0) throw new Error("no perturbations could be featurized.");
  const env = envelope(model, model.score(X));
  env.genes = syms;
  env.explanations = model.explain(X);
  env.neighbors = Object.fromEntries(syms.map((s) => [s, featurizer.neighbors(s)]));
  env.ingestion_report = ingestion_report;
  env.featurize_report = featurize_report;
  env.note = INGEST_NOTE;
  return env;
}

// /example — score the bundled 400-perturbation example (features precomputed).
export function scoreExample(model, featurizer, features, genes) {
  const X = features.map((r) => (r instanceof Float32Array ? r : Float32Array.from(r)));
  const env = envelope(model, model.score(X));
  env.genes = genes;
  env.explanations = model.explain(X);
  env.neighbors = Object.fromEntries(genes.map((g) => [g, featurizer.neighbors(g)]));
  env.in_sample_warning = IN_SAMPLE_WARNING;
  return env;
}
