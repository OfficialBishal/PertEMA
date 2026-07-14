"""D5 parity, second dataset: reproduce the Ahlmann-Eltze prediction-accuracy metrics on the Replogle
K562 genome-wide perturbation atlas, so a reviewer sees the same "trivial baselines are hard to beat"
finding holds on a second, independent context (not just the Gladstone single-context data).

Data (results/features/replogle_effects.npz):
  perts       (848,)          perturbation ids, one target gene each ('AARS+ctrl', ...)
  pgenes      (848,)          the perturbed (target) gene of each perturbation
  shared_genes(2832,)         the measured-gene panel (columns of the effect vectors)
  eff_k562    (848, 2832)     per-perturbation effect (delta) vectors in K562
  emb_genes   (2832,)         gene ids of the co-expression embedding rows (== shared_genes)
  embedding   (2832, 50)      co-expression embedding, one 50-d vector per MEASURED gene

The embedding is over the measured-gene panel, so a perturbation's feature is the embedding of its
perturbed gene. Only 213 of the 848 perturbed genes are in the panel (have an embedding). To keep the
comparison apples-to-apples (ridge and kNN require an embedding), all three predictors are evaluated on
that identical 213-perturbation in-panel cohort. This restriction is a property of the feature, not of
the labels, so it does not leak any test perturbation's true error.

Metrics (Ahlmann-Eltze definitions, matching src/eval/run_parity.py):
  - L2 (RMSE) between predicted and observed effect over 1000 evaluation genes.
  - Pearson delta correlation: cor(centered predicted effect, centered observed effect).
  - error relative to the mean baseline: a predictor's L2 divided by the mean predictor's L2. Values
    above one mean the predictor is worse than the trivial mean-training-effect baseline.

Evaluation gene set: replogle_effects.npz carries NO control-expression vector, so we CANNOT use the
reference "1000 most highly expressed genes in control". We use the top-1000 genes by effect variance
across the cohort as the stated, documented substitute. This is a fixed evaluation gene set (it uses
effect variance, never any perturbation's held-out error) so it is leakage-safe.

Predictors (reused verbatim from src/predictors/predictors.py for exact metric parity):
  mean_global      predict the mean training effect vector (the trivial baseline).
  ridge_embed      per-context ridge map from the co-expression embedding to the effect (alpha=100).
  knn_coexpr_k25   mean effect of the k=25 nearest training genes in the embedding (Euclidean).

Splits: gene-disjoint 5-fold. Each perturbation targets a distinct gene, so a seeded partition of the
cohort perturbations is automatically gene-disjoint (a held-out gene has no training row). Seeds 42/43/44.

All predictors are CLEAN (trained only on the training folds, no pretraining).

Run: pixi run python src/eval/run_parity_replogle.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "predictors"))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402

EFFECTS = "results/features/replogle_effects.npz"
OUT = "results/parity"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_FOLDS = 5
CONTEXT = "K562"


def rmse(P, T, cols):
    return np.sqrt(((P[:, cols] - T[:, cols]) ** 2).mean(1))


def pearson_delta(P, T, cols):
    Pc = P[:, cols] - P[:, cols].mean(1, keepdims=True)
    Tc = T[:, cols] - T[:, cols].mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gene_disjoint_folds(n_items, n_folds, seed):
    """Assign each cohort item (a distinct perturbed gene) to one of n_folds via a seeded permutation."""
    order = np.random.RandomState(seed).permutation(n_items)
    fold = np.empty(n_items, dtype=int)
    fold[order] = np.arange(n_items) % n_folds
    return fold


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(EFFECTS, allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff_k562"].astype(np.float32)          # (848, 2832) delta effect vectors, K562
    emb_genes = d["emb_genes"].astype(str)
    embedding = d["embedding"].astype(np.float32)       # (2832, 50) co-expression embedding of MEASURED genes
    emb_of = {g: v for g, v in zip(emb_genes, embedding)}

    # In-panel cohort: perturbations whose perturbed gene has a co-expression embedding.
    in_panel = np.array([g in emb_of for g in pgenes])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]
    eff = eff_all[idx]                                   # (213, 2832)
    n = len(idx)
    assert len(set(genes.tolist())) == n, "perturbed genes must be unique for gene-disjoint folds"

    # Single-context obs frame so the shared predictor classes apply unchanged.
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": CONTEXT})
    gene_emb = {g: emb_of[g] for g in genes}

    # Evaluation gene set: no control-expression vector exists, so use top-1000 by effect variance.
    expr_proxy = eff.var(0)
    top = np.argsort(-expr_proxy)[:N_TOP]

    def build():
        return [MeanPredictor("global"),
                RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
                KNNSimilarityPredictor(gene_emb, k=25)]
    mean_name = MeanPredictor("global").name

    rows = []
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        preds = build()
        per = {p.name: {"l2": [], "pdelta": []} for p in preds}
        for k in range(N_FOLDS):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            T = eff[te]
            for p in preds:
                p.fit(tr, eff, obs)
                P = p.predict(te, obs)
                per[p.name]["l2"].append(rmse(P, T, top))
                per[p.name]["pdelta"].append(pearson_delta(P, T, top))
        for name in per:
            l2 = np.concatenate(per[name]["l2"]); pd_ = np.concatenate(per[name]["pdelta"])
            rows.append(dict(seed=seed, predictor=name,
                             l2=float(np.nanmean(l2)), pearson_delta=float(np.nanmean(pd_))))
    res = pd.DataFrame(rows)

    agg = res.groupby("predictor").agg(l2=("l2", "mean"),
                                       pearson_delta=("pearson_delta", "mean")).reset_index()
    base = float(agg.loc[agg["predictor"] == mean_name, "l2"].iloc[0])
    agg["error_rel_to_mean"] = agg["l2"] / base
    agg["provenance"] = "CLEAN"
    agg = agg.sort_values("error_rel_to_mean")
    agg = agg[["predictor", "l2", "pearson_delta", "error_rel_to_mean", "provenance"]]
    agg.to_csv(os.path.join(OUT, "parity_replogle.csv"), index=False)

    print("=== D5 parity on Replogle K562 (Ahlmann-Eltze metrics) ===")
    print(f"cohort: {n} in-panel perturbations of {len(pgenes)} total "
          f"(feature = co-expression embedding of the perturbed gene)")
    print(f"evaluation genes: top-{N_TOP} by effect variance "
          f"(no control-expression vector in {os.path.basename(EFFECTS)})")
    print(f"splits: gene-disjoint {N_FOLDS}-fold, seeds {SEEDS}")
    print(agg.round(4).to_string(index=False))
    beats = agg[(agg["error_rel_to_mean"] < 1.0) & (agg["predictor"] != mean_name)]
    if len(beats):
        print(f"\nPredictors beating the mean baseline (error_rel_to_mean < 1): {beats['predictor'].tolist()}")
    else:
        print("\nNo predictor beats the mean baseline (error_rel_to_mean >= 1 for all). "
              "The mean baseline WINS, reproducing the reference finding on the Replogle K562 data.")
    print(f"wrote {OUT}/parity_replogle.csv")


if __name__ == "__main__":
    main()
