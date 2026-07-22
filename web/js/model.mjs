// PertEMAModel port — turns a (n,64) feature matrix into calibrated reliability scores + SHAP drivers.
// Mirrors app/backend/scoring.py exactly. Frozen constants are hard-coded per the port spec.

import { parseTrees, parseBaseScore, predict, explainRow, baseValue } from "./xgb.mjs";

export const CONFORMAL_HALFWIDTH = 0.1135; // 0.5 * 0.227 fallback (conformal_interval_width absent)

// [name, lo, hi, meaning] — band rule is lo <= cal_err < hi, fallthrough "very-low".
export const BANDS = [
  ["high", 0.0, 0.8, "prediction is among the more trustworthy on this noisy data"],
  ["moderate", 0.8, 0.88, "usable with caution"],
  ["low", 0.88, 0.94, "treat with skepticism"],
  ["very-low", 0.94, 2.0, "likely unreliable, deprioritize"],
];

// Interpretable feature groups over the 64-d vector (exact featurize() order).
export const FEATURE_GROUPS = [
  ["predicted_magnitude", [0]],
  ["source_context_state", [1, 2, 3]],
  ["destination_context_state", [4, 5, 6]],
  ["context_indicators", [7, 8, 9, 10, 11, 12]],
  ["coexpression_embedding", Array.from({ length: 50 }, (_, k) => 13 + k)],
  ["training_set_similarity", [63]],
];

function band(calErr) {
  for (const [name, lo, hi] of BANDS) if (lo <= calErr && calErr < hi) return name;
  return "very-low";
}

// np.interp with flat extrapolation (clamp to endpoints). isoX strictly ascending.
function interp(v, isoX, isoY) {
  const n = isoX.length;
  if (v <= isoX[0]) return isoY[0];
  if (v >= isoX[n - 1]) return isoY[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (isoX[mid] <= v) lo = mid; else hi = mid;
  }
  const x0 = isoX[lo], x1 = isoX[lo + 1];
  if (v === x0) return isoY[lo];
  return isoY[lo] + (isoY[lo + 1] - isoY[lo]) * (v - x0) / (x1 - x0);
}

export class PertEMAModel {
  // data: { estimator (parsed json), calibration:{iso_x,iso_y}, provenance }
  constructor(data) {
    this.trees = parseTrees(data.estimator);
    this.baseScore = parseBaseScore(data.estimator);
    this.baseVal = baseValue(this.trees, this.baseScore);
    this.isoX = Float64Array.from(data.calibration.iso_x);
    this.isoY = Float64Array.from(data.calibration.iso_y);
    this.provenance = data.provenance;
    this.version = data.provenance.model_version;
    this.coverage = data.provenance.evaluation_calibration.conformal_coverage;
    this.ceiling = data.provenance.measured_accuracy_ceiling_1minus_pearson;
  }

  // X: array of length-64 numeric rows (Float32Array or number[]). Returns per-row score dicts.
  score(X) {
    return X.map((row) => {
      const predErr = predict(this.trees, this.baseScore, row);
      const calErr = interp(predErr, this.isoX, this.isoY);
      return {
        predicted_error: predErr,
        calibrated_error: calErr,
        reliability: -calErr,
        band: band(calErr),
        conformal_lo: Math.max(calErr - CONFORMAL_HALFWIDTH, 0),
        conformal_hi: calErr + CONFORMAL_HALFWIDTH,
      };
    });
  }

  // Grouped SHAP drivers per row (mirrors PertEMAModel.explain, top_k=3).
  explain(X, topK = 3) {
    return X.map((row) => {
      const phi = explainRow(this.trees, row);
      const groups = {};
      for (const [name, idx] of FEATURE_GROUPS) {
        let s = 0;
        for (const j of idx) s += phi[j];
        groups[name] = s;
      }
      const drivers = FEATURE_GROUPS.map(([name]) => [name, groups[name]])
        .slice()
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))   // stable: ties keep declared order
        .slice(0, topK)
        .map(([g, c]) => ({ group: g, contribution: c, direction: c > 0 ? "less reliable" : "more reliable" }));
      return { base_error: this.baseVal, group_contributions_toward_error: groups, top_drivers: drivers };
    });
  }
}
