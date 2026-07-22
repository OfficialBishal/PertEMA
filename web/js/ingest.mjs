// Port of app/backend/ingest.py parse_long_csv. Parses a long-format predictions CSV into per-perturbation
// records with pred_magnitude = mean(|predicted_lfc|), grouped by (perturbed_gene[, src_context][, dst_context]),
// groups sorted lexicographically to match pandas groupby(sort=True) output order.

const LONG_REQUIRED = ["perturbed_gene", "gene", "predicted_lfc"];

export const LONG_TEMPLATE =
  "perturbed_gene,gene,predicted_lfc,src_context,dst_context\n" +
  "IL2,IL2RA,0.83,Rest,Stim48hr\n" +
  "IL2,IFNG,-0.21,Rest,Stim48hr\n" +
  "IL2,FOXP3,0.11,Rest,Stim48hr\n" +
  "STAT5A,IL2RA,0.44,Rest,Stim48hr\n" +
  "STAT5A,MYC,0.30,Rest,Stim48hr\n" +
  "# one row per (perturbed_gene, gene). predicted_lfc is your model's predicted delta log-fold-change.\n" +
  "# src_context/dst_context are optional and must be Rest, Stim8hr, or Stim48hr (default Rest -> Stim48hr).\n";

function toNumber(s) {
  const t = String(s).trim();
  if (t === "") return NaN;
  const n = Number(t);
  return Number.isNaN(n) ? NaN : n;
}

// Minimal CSV: strip a leading BOM, cut each line at the first '#', drop blank lines, comma-split.
// (Sufficient for gene CSVs; no quoted commas in this domain.)
function parseCsv(text) {
  let t = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const rows = [];
  for (let raw of t.split(/\r\n|\r|\n/)) {
    const hash = raw.indexOf("#");
    if (hash !== -1) raw = raw.slice(0, hash);
    if (raw.trim() === "") continue;
    rows.push(raw.split(","));
  }
  if (rows.length === 0) throw new Error("could not parse CSV: No columns to parse from file");
  return rows;
}

export function parseLongCsv(csvText, srcDefault = "Rest", dstDefault = "Stim48hr", maxRows = 5_000_000) {
  const rows = parseCsv(csvText);
  const header = rows[0];
  const body = rows.slice(1);
  if (body.length > maxRows) throw new Error(`too many rows: ${body.length} > limit ${maxRows}`);

  const missing = LONG_REQUIRED.filter((c) => !header.includes(c));
  if (missing.length) {
    throw new Error(
      `missing required columns [${missing.map((m) => `'${m}'`).join(", ")}]; got [${header.map((h) => `'${h}'`).join(", ")}]. ` +
      `Long format needs [${LONG_REQUIRED.map((m) => `'${m}'`).join(", ")}] (+ optional src_context, dst_context).`
    );
  }
  const col = (name) => header.indexOf(name);
  const iPg = col("perturbed_gene"), iLfc = col("predicted_lfc");
  const iSrc = header.includes("src_context") ? col("src_context") : -1;
  const iDst = header.includes("dst_context") ? col("dst_context") : -1;

  // numeric coercion + drop non-numeric predicted_lfc
  let nBad = 0;
  const kept = [];
  for (const r of body) {
    const v = toNumber(r[iLfc]);
    if (Number.isNaN(v)) { nBad++; continue; }
    kept.push({ pg: String(r[iPg]), src: iSrc >= 0 ? String(r[iSrc]) : null, dst: iDst >= 0 ? String(r[iDst]) : null, lfc: v });
  }

  // group by (pg[, src][, dst])
  const groups = new Map();
  for (const r of kept) {
    const key = [r.pg];
    if (iSrc >= 0) key.push(r.src);
    if (iDst >= 0) key.push(r.dst);
    const k = JSON.stringify(key);
    if (!groups.has(k)) groups.set(k, { key, rows: [] });
    groups.get(k).rows.push(r);
  }
  // sort groups lexicographically by the key tuple (pandas groupby sort=True)
  const ordered = [...groups.values()].sort((a, b) => {
    for (let i = 0; i < a.key.length; i++) {
      if (a.key[i] < b.key[i]) return -1;
      if (a.key[i] > b.key[i]) return 1;
    }
    return 0;
  });

  const preds = ordered.map(({ key, rows: g }) => {
    const mag = g.reduce((s, r) => s + Math.abs(r.lfc), 0) / g.length;
    return {
      perturbed_gene: key[0],
      src: iSrc >= 0 ? key[1] : srcDefault,
      dst: iDst >= 0 ? key[iSrc >= 0 ? 2 : 1] : dstDefault,
      pred_magnitude: mag,
      n_genes: g.length,
    };
  });

  const report = {
    n_rows: kept.length,
    n_perturbations: preds.length,
    n_dropped_nonnumeric_lfc: nBad,
    contexts_from: iSrc >= 0 ? "columns" : `defaults (${srcDefault} -> ${dstDefault})`,
  };
  return { preds, report };
}
