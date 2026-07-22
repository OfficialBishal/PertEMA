// Port of app/backend/report.py render_report — a self-contained HTML reliability report (inline CSS,
// no scripts, no remote assets). Preserves the provenance stamp and honest-scope note verbatim.

const BAND_COLOR = { high: "#0072B2", moderate: "#009E73", low: "#E69F00", "very-low": "#D55E00" };
const BAND_ORDER = ["high", "moderate", "low", "very-low"];
const BUILD_COMMIT = "web-static-0.1.0";

const esc = (x) =>
  String(x).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
const f3 = (x) => Number(x).toFixed(3);
const f2 = (x) => Number(x).toFixed(2);

function ceilingTxt(c) {
  if (c && typeof c === "object" && c.hit_genes != null && c.all_genes != null) {
    return `Pearson r = ${f2(c.hit_genes)} on hit genes, ${f2(c.all_genes)} on all genes (the best any ` +
      "predictor could reach given cross-donor replicate noise, higher is better)";
  }
  return `Pearson r = ${c}`;
}

function driversTxt(top) {
  if (!top || !top.length) return "-";
  return top.map((d) => {
    const c = d.contribution ?? 0;
    return `${esc(d.group || "")} (${c >= 0 ? "+" : ""}${f3(c)}, ${esc(d.direction || "")})`;
  }).join(" | ");
}

function bandDistribution(results) {
  const counts = Object.fromEntries(BAND_ORDER.map((b) => [b, 0]));
  for (const r of results) counts[r.band] = (counts[r.band] || 0) + 1;
  const n = Math.max(results.length, 1);
  return BAND_ORDER.map((b) =>
    `<span class='band' style='background:${BAND_COLOR[b]}'>${b}</span> ${counts[b]} (${Math.round(100 * counts[b] / n)}%)`
  ).join(" &nbsp; ");
}

function riskCoverageLine(results) {
  const n = results.length;
  if (n === 0) return "no predictions to summarize.";
  const reliable = results.filter((r) => r.band === "high" || r.band === "moderate");
  if (reliable.length === 0) {
    return "No predictions land in the high or moderate bands, so none clear the 0.88 calibrated-error " +
      "(1 - Pearson) reliability threshold on this run.";
  }
  const cap = Math.max(...reliable.map((r) => r.calibrated_error));
  const pct = Math.round(100 * reliable.length / n);
  return `Keeping the ${reliable.length} of ${n} predictions in the high or moderate bands (${pct}% coverage) ` +
    `holds predicted calibrated error at or below ${f3(cap)} (1 - Pearson, lower is better).`;
}

export function renderReport(env, model, extra = {}) {
  const p = model.provenance;
  const stamp = {
    model_version: p.model_version, provenance_flag: p.provenance_flag, build_commit: BUILD_COMMIT,
    calibration_coverage: p.evaluation_calibration.conformal_coverage,
    conformal_target: p.evaluation_calibration.conformal_target,
    accuracy_ceiling_1minus_pearson: p.measured_accuracy_ceiling_1minus_pearson,
    estimator: p.estimator, honest_scope: p.honest_scope,
  };
  const results = env.results;
  const genes = env.genes || results.map((_, i) => `row ${i + 1}`);
  const explanations = env.explanations || [];
  const showDrivers = explanations.length === results.length;

  const rows = results.map((r, i) => {
    const c = BAND_COLOR[r.band] || "#666";
    const [lo, hi] = r.conformal_interval;
    let cells = `<tr><td>${esc(genes[i])}</td><td>${f3(r.reliability)}</td><td>${f3(r.calibrated_error)}</td>` +
      `<td><span class='band' style='background:${c}'>${esc(r.band)}</span></td><td>[${f3(lo)}, ${f3(hi)}]</td>`;
    if (showDrivers) cells += `<td>${driversTxt(explanations[i].top_drivers)}</td>`;
    return cells + "</tr>";
  }).join("");

  const legend = env.bands_legend.map((b) =>
    `<tr><td><span class='band' style='background:${BAND_COLOR[b.band] || "#666"}'>${esc(b.band)}</span></td>` +
    `<td>${f3(b.calibrated_error_range[0])} to ${f3(b.calibrated_error_range[1])}</td><td>${esc(b.meaning)}</td></tr>`
  ).join("");

  const extraRows = Object.entries(extra).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  const driverHead = showDrivers ? "<th>top drivers (grouped SHAP)</th>" : "";
  const driverNote = showDrivers
    ? '<div class="muted">Top drivers are grouped SHAP contributions toward predicted error (positive = less reliable). Attribution explains the estimator, not biological importance.</div>'
    : "";

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PertEMA reliability report (model ${esc(stamp.model_version)})</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:24px auto;
   padding:0 16px;color:#1a1a1a;line-height:1.5}
 h1{font-size:1.35rem;margin-bottom:2px} h2{font-size:1.05rem;margin-top:22px}
 .muted{color:#666;font-size:.86rem} table{border-collapse:collapse;width:100%;margin-top:8px;font-size:.9rem}
 th,td{border:1px solid #ddd;padding:5px 8px;text-align:left} th{background:#f4f4f4}
 .band{color:#fff;padding:1px 7px;border-radius:9px;font-size:.8rem}
 .prov{background:#f7f9fb;border:1px solid #dce3ea;border-radius:6px;padding:10px 12px;margin-top:8px;font-size:.86rem}
 .note{background:#fff8e6;border:1px solid #f0e2b8;border-radius:6px;padding:10px 12px;margin-top:10px;font-size:.86rem}
</style></head><body>
<h1>PertEMA reliability report</h1>
<div class="muted">Post-hoc, model-agnostic reliability of perturbation-effect predictions. A reliability
 layer, not a predictor.</div>
<div class="prov">
 <b>Provenance.</b> Model version ${esc(stamp.model_version)}, provenance flag
 ${esc(stamp.provenance_flag)}, build commit ${esc(stamp.build_commit)}. Calibration coverage
 ${f3(stamp.calibration_coverage)} (target ${stamp.conformal_target}). Accuracy ceiling:
 ${esc(ceilingTxt(stamp.accuracy_ceiling_1minus_pearson))}. Estimator: ${esc(stamp.estimator)}.
</div>
<div class="note"><b>Honest scope.</b> ${esc(stamp.honest_scope)} Out-of-fold reliability is modest on
 noisy data (about 0.13). The scores rank which predictions to trust, not which genes are important. See the
 methods page for the measured comparisons and the negative results.</div>
<h2>Summary</h2>
<div class="prov"><b>Band distribution.</b> ${bandDistribution(results)}</div>
<div class="prov" style="margin-top:8px"><b>Risk-coverage.</b> ${riskCoverageLine(results)}</div>
<h2>Scored predictions (${results.length})</h2>
${driverNote}
<table><thead><tr><th>perturbation</th><th>reliability</th><th>calibrated error</th><th>band</th>
 <th>90% conformal interval</th>${driverHead}</tr></thead><tbody>${rows}</tbody></table>
<h2>Reliability bands</h2>
<table><thead><tr><th>band</th><th>calibrated error range</th><th>meaning</th></tr></thead>
 <tbody>${legend}</tbody></table>
${extraRows ? `<h2>Ingestion</h2><table><tbody>${extraRows}</tbody></table>` : ""}
<p class="muted">Generated by the PertEMA reliability tool, entirely in your browser. Reliability estimation is
 leakage-safe: the shipped estimator is never retrained on user ground truth.</p>
</body></html>`;
}
