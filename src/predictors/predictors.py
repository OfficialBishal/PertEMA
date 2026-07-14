"""Perturbation predictors behind a common interface.

A predictor maps (perturbed gene, culture condition) to a predicted effect vector over the measured
genes, using only training perturbations (gene-disjoint) so it never sees the query gene's true effect.
This mirrors the real setting: wrap any predictor, estimate the reliability of its output post hoc.

Tier 1 (this file): trivial baselines that the Ahlmann-Eltze critique shows are hard to beat.
  MeanPredictor(scope="condition"): predict the mean training effect vector within the query condition.
  MeanPredictor(scope="global"):    predict the global mean training effect vector.
Feature-based predictors (ridge on baseline expression, GEARS, etc.) are added once pseudobulk and a
gene network are available; they plug into the same fit/predict interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


class Predictor:
    name = "base"

    def fit(self, train_rows, effect, obs):
        raise NotImplementedError

    def predict(self, query_rows, obs):
        raise NotImplementedError


class MeanPredictor(Predictor):
    """Predict the mean training effect vector (global or per-condition)."""

    def __init__(self, scope="condition"):
        assert scope in ("condition", "global")
        self.scope = scope
        self.name = f"mean_{scope}"
        self._global = None
        self._by_cond = {}

    def fit(self, train_rows, effect, obs):
        self._global = effect[train_rows].mean(axis=0)
        if self.scope == "condition":
            cond = obs["culture_condition"].to_numpy()
            for c in np.unique(cond[train_rows]):
                rows = train_rows[cond[train_rows] == c]
                self._by_cond[c] = effect[rows].mean(axis=0)
        return self

    def predict(self, query_rows, obs):
        n_genes = self._global.shape[0]
        out = np.empty((len(query_rows), n_genes), dtype=np.float32)
        if self.scope == "global":
            out[:] = self._global
        else:
            cond = obs["culture_condition"].to_numpy()
            for j, r in enumerate(query_rows):
                out[j] = self._by_cond.get(cond[r], self._global)
        return out


class KNNSimilarityPredictor(Predictor):
    """Predict an unseen gene's effect as the mean effect of its k nearest training genes.

    Similarity is Euclidean distance in the leakage-safe control co-expression embedding. Neighbors are
    training genes with an effect measured in the same culture condition. Genes without an embedding, or
    conditions with no embedded training genes, fall back to the per-condition training mean. Because the
    folds are gene-disjoint, a query gene is never its own neighbor (it has no training row at all).
    """

    def __init__(self, gene_emb, k=25):
        self.gene_emb = gene_emb            # dict gene_id -> np.ndarray(d,)
        self.k = k
        self.name = f"knn_coexpr_k{k}"

    def fit(self, train_rows, effect, obs):
        self._effect = effect
        gene = obs["target_contrast"].astype(str).to_numpy()
        cond = obs["culture_condition"].astype(str).to_numpy()
        self._global = effect[train_rows].mean(0)
        self._cond_mean, self._nn, self._rows = {}, {}, {}
        for c in np.unique(cond[train_rows]):
            rc = train_rows[cond[train_rows] == c]
            self._cond_mean[c] = effect[rc].mean(0)
            emb, keep = [], []
            for r in rc:
                e = self.gene_emb.get(gene[r])
                if e is not None:
                    emb.append(e); keep.append(r)
            if emb:
                self._nn[c] = NearestNeighbors(n_neighbors=min(self.k, len(keep)),
                                               algorithm="brute", n_jobs=8).fit(np.stack(emb))
                self._rows[c] = np.asarray(keep)
        return self

    def predict(self, query_rows, obs):
        gene = obs["target_contrast"].astype(str).to_numpy()
        cond = obs["culture_condition"].astype(str).to_numpy()
        query_rows = np.asarray(query_rows)
        out = np.empty((len(query_rows), self._global.shape[0]), dtype=np.float32)
        for j, r in enumerate(query_rows):       # fallback fill
            out[j] = self._cond_mean.get(cond[r], self._global)
        for c in np.unique(cond[query_rows]):
            if c not in self._nn:
                continue
            sel = np.where(cond[query_rows] == c)[0]
            embs, pos = [], []
            for j in sel:
                e = self.gene_emb.get(gene[query_rows[j]])
                if e is not None:
                    embs.append(e); pos.append(j)
            if not embs:
                continue
            _, idx = self._nn[c].kneighbors(np.stack(embs))
            rows_c = self._rows[c]
            for t, j in enumerate(pos):
                out[j] = self._effect[rows_c[idx[t]]].mean(0)
        return out


class RidgeEmbeddingPredictor(Predictor):
    """Additive-linear predictor (Ahlmann-Eltze style): a per-condition ridge map from the control
    co-expression embedding to the effect vector. A third inductive bias, distinct from the constant
    mean and the local kNN average. Genes without an embedding fall back to the per-condition mean.
    """

    def __init__(self, gene_emb, alpha=100.0):
        self.gene_emb = gene_emb
        self.alpha = alpha
        self.name = "ridge_embed"

    def fit(self, train_rows, effect, obs):
        gene = obs["target_contrast"].astype(str).to_numpy()
        cond = obs["culture_condition"].astype(str).to_numpy()
        self._global = effect[train_rows].mean(0)
        self._cond_mean, self._models = {}, {}
        for c in np.unique(cond[train_rows]):
            rc = train_rows[cond[train_rows] == c]
            self._cond_mean[c] = effect[rc].mean(0)
            X, rows = [], []
            for r in rc:
                e = self.gene_emb.get(gene[r])
                if e is not None:
                    X.append(e); rows.append(r)
            if len(rows) > 50:
                self._models[c] = Ridge(alpha=self.alpha).fit(np.stack(X), effect[np.asarray(rows)])
        return self

    def predict(self, query_rows, obs):
        gene = obs["target_contrast"].astype(str).to_numpy()
        cond = obs["culture_condition"].astype(str).to_numpy()
        query_rows = np.asarray(query_rows)
        out = np.empty((len(query_rows), self._global.shape[0]), dtype=np.float32)
        for j, r in enumerate(query_rows):
            out[j] = self._cond_mean.get(cond[r], self._global)
        for c in np.unique(cond[query_rows]):
            if c not in self._models:
                continue
            sel = np.where(cond[query_rows] == c)[0]
            embs, pos = [], []
            for j in sel:
                e = self.gene_emb.get(gene[query_rows[j]])
                if e is not None:
                    embs.append(e); pos.append(j)
            if embs:
                pred = self._models[c].predict(np.stack(embs)).astype(np.float32)
                for t, j in enumerate(pos):
                    out[j] = pred[t]
        return out

