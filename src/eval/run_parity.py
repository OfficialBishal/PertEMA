"""D4/D5 parity: reproduce the reference-paper prediction-accuracy metrics on our predictor roster so a
reviewer sees we meet the Ahlmann-Eltze / Wong bar before we add the reliability increment.

Metrics (Ahlmann-Eltze definitions):
  - L2 (RMSE) between predicted and observed delta expression over the 1000 most highly expressed genes in
    the control condition.
  - Pearson delta correlation: cor(yhat - y_control, y_observed - y_control).
  - error relative to the mean baseline: a predictor's L2 divided by the per-condition mean predictor's L2.
    Values above one mean worse than the trivial mean baseline. This is the reference figure PF-B.

Out-of-fold on the gene-disjoint splits, three seeds. All predictors here are CLEAN (trained only on the
splits, no pretraining). The frozen-adapted FOUNDATION-model roster (Geneformer, scGPT, gene2vec) populates
the same table with UNKNOWN-OVERLAP provenance flags in run_parity_foundation.py (parity_gladstone_foundation.csv).

Run: pixi run python src/eval/run_parity.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                          # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from neural_baselines import MLPDecoderPredictor                           # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/parity"
SEEDS = [42, 43, 44]
N_TOP = 1000


class NoChangePredictor:
    """Reference no-change baseline: predict zero delta (the control state)."""
    name = "no_change"

    def fit(self, train_rows, effect, obs):
        self._n = effect.shape[1]; return self

    def predict(self, query_rows, obs):
        return np.zeros((len(query_rows), self._n), dtype=np.float32)


def rmse(P, T, cols):
    return np.sqrt(((P[:, cols] - T[:, cols]) ** 2).mean(1))


def pearson_delta(P, T, cols):
    Pc = P[:, cols] - P[:, cols].mean(1, keepdims=True)
    Tc = T[:, cols] - T[:, cols].mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc", "baseMean"))
    lfc = de.layers["log_fc"]                       # delta log-fold-change (already vs control)
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB); gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}

    # top-1000 most highly expressed genes in control (Ahlmann-Eltze definition): rank by baseMean, the
    # DESeq2 base mean expression of each gene. baseMean is an expression level, not a perturbation effect,
    # so it is leakage-safe. (Previously baseMean was never loaded and this silently fell back to mean
    # |log_fc|, ranking the most-perturbed genes and mislabeling them expressed. Fixed per internal review.)
    expr = np.nan_to_num(de.layers["baseMean"]).mean(0)
    top = np.argsort(-expr)[:N_TOP]

    predictors = [MeanPredictor("condition"), MeanPredictor("global"),
                  KNNSimilarityPredictor(gene_emb, k=25), RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
                  MLPDecoderPredictor(gene_emb, epochs=30, seed=0, target_cols=top), NoChangePredictor()]

    rows = []
    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fog.get(g, -1) for g in gene_of_row])
        n_folds = int(row_fold.max()) + 1
        per = {p.name: {"l2": [], "pdelta": []} for p in predictors}
        for k in range(n_folds):
            tr = np.where(row_fold != k)[0]; te = np.where(row_fold == k)[0]
            T = lfc[te]
            for p in predictors:
                p.fit(tr, lfc, obs)
                P = p.predict(te, obs)
                per[p.name]["l2"].append(rmse(P, T, top))
                per[p.name]["pdelta"].append(pearson_delta(P, T, top))
        for name in per:
            l2 = np.concatenate(per[name]["l2"]); pd_ = np.concatenate(per[name]["pdelta"])
            rows.append(dict(seed=seed, predictor=name, l2=float(np.nanmean(l2)),
                             pearson_delta=float(np.nanmean(pd_))))
    res = pd.DataFrame(rows)

    # error relative to the per-condition mean baseline (mean_condition)
    agg = res.groupby("predictor").agg(l2=("l2", "mean"), l2_sd=("l2", "std"),
                                       pearson_delta=("pearson_delta", "mean")).reset_index()
    base = float(agg.loc[agg["predictor"] == "mean_condition", "l2"].iloc[0])
    agg["error_rel_to_mean"] = agg["l2"] / base
    agg["provenance"] = "CLEAN"
    agg = agg.sort_values("error_rel_to_mean")
    agg.to_csv(os.path.join(OUT, "parity_gladstone.csv"), index=False)
    res.to_csv(os.path.join(OUT, "parity_per_seed.csv"), index=False)
    print("=== D5 parity on Gladstone single-context (Ahlmann-Eltze metrics, top-1000 expressed) ===")
    print(agg[["predictor", "l2", "pearson_delta", "error_rel_to_mean", "provenance"]].round(4).to_string(index=False))
    beats = agg[(agg["error_rel_to_mean"] < 1.0) & (agg["predictor"] != "mean_condition")]
    print(f"\nPredictors beating the mean baseline (error_rel_to_mean < 1): "
          f"{beats['predictor'].tolist() if len(beats) else 'NONE - reproduces the reference finding that the mean baseline is hard to beat'}")
    print(f"wrote {OUT}/parity_gladstone.csv")


if __name__ == "__main__":
    main()
