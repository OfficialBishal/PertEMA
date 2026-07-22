// Port of app/backend/featurize.py Featurizer. Builds the exact 64-d prediction-time feature vector
// per perturbation from the bundled reference tables. Writing into a Float32Array gives the float32
// semantics of Python's np.array(rows, dtype=float32) for free.

export const CONDS = ["Rest", "Stim8hr", "Stim48hr"];
export const EMB_DIM = 50;

const NOTE =
  "Features are computed at inference from the bundled reference and match the training feature " +
  "construction. Contexts must be one of Rest, Stim8hr, Stim48hr; other contexts are out of " +
  "distribution for this model and are rejected. Genes with no reference entry are scored with " +
  "missing features (the tree handles them) but less reliably.";

// Python round() is round-half-to-even; matters only at exact 3rd-decimal halves but reproduce it.
function round3(x) {
  const scaled = x * 1000;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let r;
  if (diff > 0.5) r = floor + 1;
  else if (diff < 0.5) r = floor;
  else r = floor % 2 === 0 ? floor : floor + 1;
  return r / 1000;
}

export class Featurizer {
  // refs: { condIdx:Map, nGenes, stats:Float32Array, geneCol:Map<ens,col>, ensSet:Set,
  //         emb:Map<ens,Float32Array(50)>, sym2ens:Map, ens2sym:Map, sim:Map, neighbors:Map }
  constructor(refs) {
    this.r = refs;
  }

  _toEns(g) {
    g = String(g).trim();
    if (this.r.ensSet.has(g)) return g;
    return this.r.sym2ens.get(g) ?? null;
  }

  _stat(which, cond, col) {
    return this.r.stats[which * 3 * this.r.nGenes + cond * this.r.nGenes + col];
  }

  featurize(predictions) {
    const { condIdx, geneCol, emb, sim } = this.r;
    const rows = [], syms = [], unmapped = [], badCtx = [];
    for (const p of predictions) {
      const src = String(p.src), dst = String(p.dst);
      if (!condIdx.has(src) || !condIdx.has(dst)) { badCtx.push([p.perturbed_gene, src, dst]); continue; }
      const ens = this._toEns(p.perturbed_gene);
      if (ens == null) { unmapped.push(String(p.perturbed_gene)); continue; }
      const gi = geneCol.has(ens) ? geneCol.get(ens) : null;
      const si = condIdx.get(src), di = condIdx.get(dst);

      const f = new Float32Array(64);
      f[0] = p.pred_magnitude;
      let k = 1;
      for (const ci of [si, di]) for (const which of [0, 1, 2]) f[k++] = gi == null ? NaN : this._stat(which, ci, gi);
      for (const c of CONDS) { f[k++] = src === c ? 1 : 0; f[k++] = dst === c ? 1 : 0; }
      const e = emb.get(ens);
      for (let j = 0; j < EMB_DIM; j++) f[13 + j] = e ? e[j] : NaN;
      f[63] = sim.has(ens) ? sim.get(ens) : NaN;

      rows.push(f);
      syms.push(String(p.perturbed_gene));
    }
    const report = {
      n_input: predictions.length,
      n_featurized: rows.length,
      n_unmapped_genes: unmapped.length,
      unmapped_genes_sample: unmapped.slice(0, 20),
      n_bad_context: badCtx.length,
      supported_contexts: CONDS,
      mapped_fraction: round3(rows.length / Math.max(1, predictions.length)),
      note: NOTE,
    };
    return { X: rows, syms, report };
  }

  // Nearest reference genes in co-expression space (precomputed, self-skipped, symbols). Display only.
  neighbors(gene, k = 5) {
    const ens = this._toEns(gene);
    if (ens == null || !this.r.neighbors.has(ens)) return [];
    return this.r.neighbors.get(ens).slice(0, k);
  }
}
