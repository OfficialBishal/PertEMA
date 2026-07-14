"""Render a scored result into a self-contained HTML reliability report (F8). One file, no external
dependencies (inline CSS, no scripts, no remote assets), so a user can save, email, or archive it. Every
report carries the provenance stamp (model version, provenance flag, build commit, calibration coverage,
accuracy ceiling) and the honest-scope note, so a result is never separated from what it does and does not
claim. ASCII only (invariant 8).
"""
import html

BAND_COLOR = {"high": "#0072B2", "moderate": "#009E73", "low": "#E69F00", "very-low": "#D55E00"}


BAND_ORDER = ["high", "moderate", "low", "very-low"]


def _esc(x):
    return html.escape(str(x))


def _ceiling_txt(ceiling):
    """P0-5: display the accuracy ceiling as Pearson r with direction. The frozen provenance key is named
    measured_accuracy_ceiling_1minus_pearson but stores Pearson r (higher is better), not 1 - Pearson."""
    if isinstance(ceiling, dict):
        hit = ceiling.get("hit_genes")
        allg = ceiling.get("all_genes")
        if hit is not None and allg is not None:
            return (f"Pearson r = {hit:.2f} on hit genes, {allg:.2f} on all genes (the best any predictor "
                    "could reach given cross-donor replicate noise, higher is better)")
        return ", ".join(f"{k} Pearson r = {v}" for k, v in ceiling.items())
    return f"Pearson r = {ceiling}"


def _drivers_txt(top_drivers):
    """One cell listing the top grouped SHAP drivers with signed contribution and direction. ASCII only."""
    if not top_drivers:
        return "-"
    parts = []
    for d in top_drivers:
        c = d.get("contribution", 0.0)
        sign = "+" if c >= 0 else ""
        parts.append(f"{_esc(d.get('group', ''))} ({sign}{c:.3f}, {_esc(d.get('direction', ''))})")
    return " | ".join(parts)


def _band_distribution(results):
    """Counts per reliability band in canonical order, for the report summary (P3-5)."""
    counts = {b: 0 for b in BAND_ORDER}
    for r in results:
        counts[r["band"]] = counts.get(r["band"], 0) + 1
    n = max(len(results), 1)
    cells = []
    for b in BAND_ORDER:
        pct = 100.0 * counts[b] / n
        cells.append(
            f"<span class='band' style='background:{BAND_COLOR[b]}'>{b}</span> {counts[b]} ({pct:.0f}%)")
    return " &nbsp; ".join(cells)


def _risk_coverage_line(results):
    """P3-5: one-line selective-prediction summary. Reports how many predictions fall in the more-reliable
    (high or moderate) bands, the coverage you retain by keeping them, and the predicted calibrated-error cap
    at that coverage. Computed from predicted calibrated error, so it needs no user ground truth."""
    n = len(results)
    if n == 0:
        return "no predictions to summarize."
    reliable = sum(1 for r in results if r["band"] in ("high", "moderate"))
    pct = 100.0 * reliable / n
    cap = max((r["calibrated_error"] for r in results if r["band"] in ("high", "moderate")), default=None)
    if reliable == 0:
        return (f"No predictions land in the high or moderate bands, so none clear the 0.88 calibrated-error "
                f"(1 - Pearson) reliability threshold on this run.")
    return (f"Keeping the {reliable} of {n} predictions in the high or moderate bands ({pct:.0f}% coverage) "
            f"holds predicted calibrated error at or below {cap:.3f} (1 - Pearson, lower is better).")


def render_report(env, stamp, extra=None):
    """env: the scoring envelope (_envelope output, with results/genes). stamp: provenance dict from
    build_stamp(). extra: optional dict of ingestion/featurize reports to surface. Returns an HTML string."""
    extra = extra or {}
    results = env["results"]
    genes = env.get("genes") or [f"row {i + 1}" for i in range(len(results))]
    explanations = env.get("explanations") or []
    show_drivers = len(explanations) == len(results)
    rows = []
    for i, (g, r) in enumerate(zip(genes, results)):
        c = BAND_COLOR.get(r["band"], "#666")
        lo, hi = r["conformal_interval"]
        cells = (
            f"<tr><td>{_esc(g)}</td>"
            f"<td>{r['reliability']:.3f}</td>"
            f"<td>{r['calibrated_error']:.3f}</td>"
            f"<td><span class='band' style='background:{c}'>{_esc(r['band'])}</span></td>"
            f"<td>[{lo:.3f}, {hi:.3f}]</td>")
        if show_drivers:
            cells += f"<td>{_drivers_txt(explanations[i].get('top_drivers', []))}</td>"
        rows.append(cells + "</tr>")
    legend = "".join(
        f"<tr><td><span class='band' style='background:{BAND_COLOR.get(b['band'], '#666')}'>{_esc(b['band'])}"
        f"</span></td><td>{b['calibrated_error_range'][0]:.3f} to {b['calibrated_error_range'][1]:.3f}</td>"
        f"<td>{_esc(b['meaning'])}</td></tr>"
        for b in env["bands_legend"])
    ceil_txt = _ceiling_txt(stamp["accuracy_ceiling_1minus_pearson"])
    band_dist = _band_distribution(results)
    risk_cov = _risk_coverage_line(results)
    extra_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in extra.items())
    driver_head = "<th>top drivers (grouped SHAP)</th>" if show_drivers else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PertEMA reliability report (model {_esc(stamp['model_version'])})</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:24px auto;
   padding:0 16px;color:#1a1a1a;line-height:1.5}}
 h1{{font-size:1.35rem;margin-bottom:2px}} h2{{font-size:1.05rem;margin-top:22px}}
 .muted{{color:#666;font-size:.86rem}} table{{border-collapse:collapse;width:100%;margin-top:8px;font-size:.9rem}}
 th,td{{border:1px solid #ddd;padding:5px 8px;text-align:left}} th{{background:#f4f4f4}}
 .band{{color:#fff;padding:1px 7px;border-radius:9px;font-size:.8rem}}
 .prov{{background:#f7f9fb;border:1px solid #dce3ea;border-radius:6px;padding:10px 12px;margin-top:8px;
   font-size:.86rem}} .note{{background:#fff8e6;border:1px solid #f0e2b8;border-radius:6px;padding:10px 12px;
   margin-top:10px;font-size:.86rem}}
</style></head><body>
<h1>PertEMA reliability report</h1>
<div class="muted">Post-hoc, model-agnostic reliability of perturbation-effect predictions. A reliability
 layer, not a predictor.</div>
<div class="prov">
 <b>Provenance.</b> Model version {_esc(stamp['model_version'])}, provenance flag
 {_esc(stamp['provenance_flag'])}, build commit {_esc(stamp['build_commit'])}. Calibration coverage
 {stamp['calibration_coverage']:.3f} (target {stamp.get('conformal_target', 0.9)}). Accuracy ceiling:
 {_esc(ceil_txt)}. Estimator: {_esc(stamp['estimator'])}.
</div>
<div class="note"><b>Honest scope.</b> {_esc(stamp['honest_scope'])} Out-of-fold reliability is modest on
 noisy data (about 0.13). The scores rank which predictions to trust, not which genes are important. See the
 methods page for the measured comparisons and the negative results.</div>
<h2>Summary</h2>
<div class="prov"><b>Band distribution.</b> {band_dist}</div>
<div class="prov" style="margin-top:8px"><b>Risk-coverage.</b> {risk_cov}</div>
<h2>Scored predictions ({len(env['results'])})</h2>
{'<div class="muted">Top drivers are grouped SHAP contributions toward predicted error (positive = less reliable). Attribution explains the estimator, not biological importance.</div>' if driver_head else ''}
<table><thead><tr><th>perturbation</th><th>reliability</th><th>calibrated error</th><th>band</th>
 <th>90% conformal interval</th>{driver_head}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Reliability bands</h2>
<table><thead><tr><th>band</th><th>calibrated error range</th><th>meaning</th></tr></thead>
 <tbody>{legend}</tbody></table>
{f'<h2>Ingestion</h2><table><tbody>{extra_rows}</tbody></table>' if extra_rows else ''}
<p class="muted">Generated by the PertEMA reliability API. Reliability estimation is leakage-safe: the shipped
 estimator is never retrained on user ground truth.</p>
</body></html>"""
