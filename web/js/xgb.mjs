// XGBoost inference in pure JS — exact parity with xgboost 3.2.0 for this frozen model.
//
// Two surfaces, both validated against the Python library to float tolerance:
//   predict(trees, baseScore, x)    reg:squarederror margin, float32-accumulated (bit-exact vs .predict)
//   explainRow(trees, x)            exact path-dependent TreeSHAP (pred_contribs), max err ~6e-9
//
// The model is all-numeric splits, no categoricals; leaf value lives in base_weights; node cover is
// sum_hessian; missing (NaN) routes via default_left. See the port spec for the derivation.

const fr = Math.fround;

// Parse the estimator.json learner into flat per-tree structs (arrays kept as plain JS number arrays).
export function parseTrees(model) {
  const raw = model.learner.gradient_booster.model.trees;
  return raw.map((t) => ({
    left: t.left_children,
    right: t.right_children,
    feat: t.split_indices,
    thr: t.split_conditions,
    defLeft: t.default_left,
    cover: t.sum_hessian,
    leaf: t.base_weights,
  }));
}

export function parseBaseScore(model) {
  return parseFloat(model.learner.learner_model_param.base_score.replace(/[[\]]/g, ""));
}

// reg:squarederror prediction. Float32 accumulation reproduces XGBRegressor.predict bit-exactly.
export function predict(trees, baseScore, x) {
  let sum = fr(baseScore);
  for (const tr of trees) {
    let node = 0;
    while (tr.left[node] !== -1) {
      const v = x[tr.feat[node]];
      if (Number.isNaN(v)) node = tr.defLeft[node] ? tr.left[node] : tr.right[node];
      else node = fr(v) < fr(tr.thr[node]) ? tr.left[node] : tr.right[node];
    }
    sum = fr(sum + fr(tr.leaf[node]));
  }
  return sum;
}

// The SHAP base value (final pred_contribs column) = base_score + Σ cover-weighted root EV per tree.
export function baseValue(trees, baseScore) {
  const ev = (tr, n) => {
    if (tr.left[n] === -1) return tr.leaf[n];
    const cl = tr.cover[tr.left[n]], cr = tr.cover[tr.right[n]];
    return (cl * ev(tr, tr.left[n]) + cr * ev(tr, tr.right[n])) / (cl + cr);
  };
  let b = baseScore;
  for (const tr of trees) b += ev(tr, 0);
  return b;
}

// --- exact path-dependent TreeSHAP (Lundberg Algorithm 2) -------------------
function extendPath(fI, zF, oF, pw, uniqueDepth, zeroFraction, oneFraction, featureIndex) {
  fI[uniqueDepth] = featureIndex; zF[uniqueDepth] = zeroFraction; oF[uniqueDepth] = oneFraction;
  pw[uniqueDepth] = uniqueDepth === 0 ? 1.0 : 0.0;
  for (let i = uniqueDepth - 1; i >= 0; i--) {
    pw[i + 1] += oneFraction * pw[i] * (i + 1) / (uniqueDepth + 1);
    pw[i] = zeroFraction * pw[i] * (uniqueDepth - i) / (uniqueDepth + 1);
  }
}

function unwoundPathSum(fI, zF, oF, pw, uniqueDepth, pathIndex) {
  const oneFraction = oF[pathIndex], zeroFraction = zF[pathIndex];
  let nextOnePortion = pw[uniqueDepth], total = 0.0;
  for (let i = uniqueDepth - 1; i >= 0; i--) {
    if (oneFraction !== 0) {
      const tmp = nextOnePortion / ((i + 1) * oneFraction);
      total += tmp;
      nextOnePortion = pw[i] - tmp * zeroFraction * (uniqueDepth - i);
    } else if (zeroFraction !== 0) {
      total += (pw[i] / zeroFraction) / (uniqueDepth - i);
    }
  }
  return total * (uniqueDepth + 1);
}

function unwindPath(fI, zF, oF, pw, uniqueDepth, pathIndex) {
  const oneFraction = oF[pathIndex], zeroFraction = zF[pathIndex];
  let nextOnePortion = pw[uniqueDepth];
  for (let i = uniqueDepth - 1; i >= 0; i--) {
    if (oneFraction !== 0) {
      const tmp = pw[i];
      pw[i] = nextOnePortion * (uniqueDepth + 1) / ((i + 1) * oneFraction);
      nextOnePortion = tmp - pw[i] * zeroFraction * (uniqueDepth - i) / (uniqueDepth + 1);
    } else {
      pw[i] = pw[i] * (uniqueDepth + 1) / (zeroFraction * (uniqueDepth - i));
    }
  }
  for (let i = pathIndex; i < uniqueDepth; i++) { fI[i] = fI[i + 1]; zF[i] = zF[i + 1]; oF[i] = oF[i + 1]; }
}

function recurse(tr, node, x, phi, fI, zF, oF, pw, uniqueDepth, parentZero, parentOne, parentFeature) {
  extendPath(fI, zF, oF, pw, uniqueDepth, parentZero, parentOne, parentFeature);
  if (tr.left[node] === -1) {
    for (let i = 1; i <= uniqueDepth; i++) {
      const w = unwoundPathSum(fI, zF, oF, pw, uniqueDepth, i);
      phi[fI[i]] += w * (oF[i] - zF[i]) * tr.leaf[node];
    }
    return;
  }
  const f = tr.feat[node];
  let hot, cold;
  // Route with the SAME float32 comparison as predict(): estimator.json stores thresholds as
  // float32-serialized decimals, so raw-float64 routing diverges from xgboost on boundary values.
  const v = x[f];
  if (Number.isNaN(v) ? tr.defLeft[node] : fr(v) < fr(tr.thr[node])) { hot = tr.left[node]; cold = tr.right[node]; }
  else { hot = tr.right[node]; cold = tr.left[node]; }
  let incomingZeroHot = 1.0, incomingOneHot = 1.0, incomingZeroCold = 1.0, incomingOneCold = 1.0;
  let pathIndex = 1;
  while (pathIndex <= uniqueDepth) { if (fI[pathIndex] === f) break; pathIndex++; }
  if (pathIndex !== uniqueDepth + 1) {
    incomingZeroHot = zF[pathIndex]; incomingOneHot = oF[pathIndex];
    incomingZeroCold = zF[pathIndex]; incomingOneCold = oF[pathIndex];
    unwindPath(fI, zF, oF, pw, uniqueDepth, pathIndex);
    uniqueDepth -= 1;
  }
  const w = tr.cover[node];
  const hotZero = tr.cover[hot] / w, coldZero = tr.cover[cold] / w;
  const n = uniqueDepth + 2;
  recurse(tr, hot, x, phi, fI.slice(0, n), zF.slice(0, n), oF.slice(0, n), pw.slice(0, n),
    uniqueDepth + 1, hotZero * incomingZeroHot, 1.0 * incomingOneHot, f);
  recurse(tr, cold, x, phi, fI.slice(0, n), zF.slice(0, n), oF.slice(0, n), pw.slice(0, n),
    uniqueDepth + 1, coldZero * incomingZeroCold, 0.0 * incomingOneCold, f);
}

// Returns Float64Array(64) of per-feature SHAP contributions toward the predicted error.
export function explainRow(trees, x) {
  const phi = new Float64Array(65);
  const D = 130;
  for (const tr of trees) {
    recurse(tr, 0, x, phi, new Array(D).fill(0), new Array(D).fill(0), new Array(D).fill(0), new Array(D).fill(0),
      0, 1.0, 1.0, -1);
  }
  return phi.slice(0, 64);
}
