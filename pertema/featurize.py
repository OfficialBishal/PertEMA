"""Featurize a user's perturbation predictions into the exact 64-d prediction-time feature vector the frozen
estimator expects, using the reference data bundled in the model artifact. The feature order matches the
training-time construction (build_transfer_features + training-set similarity) exactly, verified by
round-tripping the bundled example, so a user's features are on the same footing as the training features.

Leakage-safe: features use only prediction-time information (the user's predicted effect magnitude, the
control-state baseline features of the perturbed gene, its co-expression embedding, the context indicators,
and similarity to the reference gene set). No true post-perturbation effect enters.
"""
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

CONDS = ["Rest", "Stim8hr", "Stim48hr"]
EMB_DIM = 50


class Featurizer:
    def __init__(self, ref_dir):
        bz = np.load(os.path.join(ref_dir, "control_baseline.npz"))
        self.gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
        self.cond_idx = {c: i for i, c in enumerate([str(c) for c in bz["conditions"]])}
        self.baseline, self.dropout, self.donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
        ez = np.load(os.path.join(ref_dir, "gene_embedding.npz"))
        self.emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
        gm = pd.read_csv(os.path.join(ref_dir, "gene_map.csv"), dtype=str)
        self.sym2ens = dict(zip(gm["symbol"], gm["ensembl"]))
        self.ens2sym = dict(zip(gm["ensembl"], gm["symbol"]))
        self.ens_set = set(self.gene_col) | set(self.emb)
        # training-set similarity: nearest OTHER embedded gene, precomputed for every embedded gene
        genes = np.array(list(self.emb)); X = np.array([self.emb[g] for g in genes])
        nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=8).fit(X)
        d, _ = nn.kneighbors(X)
        self._sim = {g: float(d[i, 1]) for i, g in enumerate(genes)}
        self._nn, self._nn_genes = nn, genes

    def _to_ens(self, g):
        g = str(g).strip()
        if g in self.ens_set:
            return g
        return self.sym2ens.get(g)          # symbol -> Ensembl, or None if unmappable

    def neighbors(self, gene, k=5):
        """F6 gene-network neighborhood: the k nearest reference genes to the perturbed gene in co-expression
        embedding space (symbols where mappable). Prediction-time only, from the bundled reference."""
        ens = self._to_ens(gene)
        if ens is None or ens not in self.emb:
            return []
        q = np.asarray(self.emb[ens], float)[None, :]
        d, ix = self._nn.kneighbors(q, n_neighbors=min(k + 1, len(self._nn_genes)))
        out = []
        for dist, j in zip(d[0], ix[0]):
            g = str(self._nn_genes[j])
            if g == ens:
                continue                     # skip self
            out.append(self.ens2sym.get(g, g))
            if len(out) >= k:
                break
        return out

    def _sim_of(self, ens):
        if ens in self._sim:
            return self._sim[ens]
        if ens in self.emb:
            return self._sim[ens]
        return np.nan                        # no embedding -> unknown similarity (xgboost handles NaN)

    def featurize(self, predictions):
        """predictions: list of dicts with perturbed_gene, src, dst, and pred_magnitude (mean abs predicted
        effect). Returns (X (n,64), symbols, report)."""
        rows, syms, unmapped, bad_ctx = [], [], [], []
        for p in predictions:
            ens = self._to_ens(p["perturbed_gene"])
            src, dst = str(p["src"]), str(p["dst"])
            if src not in self.cond_idx or dst not in self.cond_idx:
                bad_ctx.append((p["perturbed_gene"], src, dst)); continue
            if ens is None:
                unmapped.append(str(p["perturbed_gene"])); continue
            gi = self.gene_col.get(ens)
            f = [float(p["pred_magnitude"])]
            for tag in (src, dst):
                ci = self.cond_idx[tag]
                for arr in (self.baseline, self.dropout, self.donor_var):
                    f.append(float(arr[ci, gi]) if gi is not None else np.nan)
            for c in CONDS:                    # interleaved src/dst one-hot, matching training
                f.append(1.0 if src == c else 0.0)
                f.append(1.0 if dst == c else 0.0)
            e = self.emb.get(ens, np.full(EMB_DIM, np.nan))
            f.extend([float(x) for x in e])
            f.append(self._sim_of(ens))
            rows.append(f); syms.append(str(p["perturbed_gene"]))
        X = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 64), np.float32)
        report = {
            "n_input": len(predictions), "n_featurized": len(rows),
            "n_unmapped_genes": len(unmapped), "unmapped_genes_sample": unmapped[:20],
            "n_bad_context": len(bad_ctx),
            "supported_contexts": CONDS,
            "mapped_fraction": round(len(rows) / max(1, len(predictions)), 3),
            "note": ("Features are computed at inference from the bundled reference and match the training "
                     "feature construction. Contexts must be one of Rest, Stim8hr, Stim48hr; other contexts "
                     "are out of distribution for this model and are rejected. Genes with no reference entry "
                     "are scored with missing features (the tree handles them) but less reliably."),
        }
        return X, syms, report


def default_featurizer():
    """Load the featurizer's reference data from the bundled model. Override with PERTEMA_MODEL_DIR."""
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.environ.get("PERTEMA_MODEL_DIR") or os.path.join(
        here, "..", "app", "model", "pertema_model_v0.1.0")
    return Featurizer(os.path.join(base, "reference"))
