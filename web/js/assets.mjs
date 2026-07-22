// Loads the static model + reference assets and builds the in-memory structures the port needs.
// buildRefs / buildModelData are environment-agnostic (Node harness and browser share them);
// loadAll does the browser-side fetching with a progress callback for the one-time download.

import { PertEMAModel } from "./model.mjs";
import { Featurizer } from "./featurize.mjs";

export function buildRefs({ baselineIndex, baselineBuf, embeddingIndex, embeddingBuf, geneMap, sim, neighbors }) {
  const nGenes = baselineIndex.n_genes;
  const condIdx = new Map(baselineIndex.conditions.map((c, i) => [c, i]));
  const geneCol = new Map(baselineIndex.genes.map((g, i) => [g, i]));
  const stats = new Float32Array(baselineBuf);

  const embFloat = new Float32Array(embeddingBuf);
  const emb = new Map();
  embeddingIndex.gene_ids.forEach((g, i) => emb.set(g, embFloat.subarray(i * 50, i * 50 + 50)));

  const ensSet = new Set(baselineIndex.genes);
  for (const g of embeddingIndex.gene_ids) ensSet.add(g);

  return {
    condIdx, nGenes, stats, geneCol, ensSet, emb,
    sym2ens: new Map(Object.entries(geneMap.sym2ens)),
    ens2sym: new Map(Object.entries(geneMap.ens2sym)),
    sim: new Map(Object.entries(sim)),
    neighbors: new Map(Object.entries(neighbors)),
  };
}

// Browser loader. onProgress(done, total, label) fires as each asset lands.
export async function loadAll(baseUrl = "data", onProgress = () => {}) {
  const manifest = [
    ["model/estimator.json", "json", "reliability model"],
    ["model/calibration.json", "json", "calibration"],
    ["model/provenance.json", "json", "provenance"],
    ["model/feature_spec.json", "json", "feature spec"],
    ["ref/baseline_index.json", "json", "gene index"],
    ["ref/baseline_stats.bin", "buf", "control baselines"],
    ["ref/embedding_index.json", "json", "embedding index"],
    ["ref/embedding.bin", "buf", "co-expression embedding"],
    ["ref/gene_map.json", "json", "gene name map"],
    ["ref/training_set_similarity.json", "json", "similarity table"],
    ["ref/gene_top5_neighbors.json", "json", "gene neighborhoods"],
  ];
  const out = {};
  let done = 0;
  for (const [path, kind, label] of manifest) {
    onProgress(done, manifest.length, label);
    const res = await fetch(`${baseUrl}/${path}`);
    if (!res.ok) throw new Error(`failed to load ${path} (${res.status})`);
    out[path] = kind === "json" ? await res.json() : await res.arrayBuffer();
    done++;
    onProgress(done, manifest.length, label);
  }

  const model = new PertEMAModel({
    estimator: out["model/estimator.json"],
    calibration: out["model/calibration.json"],
    provenance: out["model/provenance.json"],
  });
  const refs = buildRefs({
    baselineIndex: out["ref/baseline_index.json"],
    baselineBuf: out["ref/baseline_stats.bin"],
    embeddingIndex: out["ref/embedding_index.json"],
    embeddingBuf: out["ref/embedding.bin"],
    geneMap: out["ref/gene_map.json"],
    sim: out["ref/training_set_similarity.json"],
    neighbors: out["ref/gene_top5_neighbors.json"],
  });
  return {
    model,
    featurizer: new Featurizer(refs),
    provenance: out["model/provenance.json"],
    featureSpec: out["model/feature_spec.json"],
  };
}
