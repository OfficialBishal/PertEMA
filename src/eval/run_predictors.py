"""Run predictors under the fixed gene-disjoint folds and write per-perturbation errors.

For each seed and each outer fold, a predictor is trained on the training-fold perturbations and
predicts the held-out fold. Because folds are gene-disjoint, no predictor sees the query gene's true
effect (invariant 1). The per-perturbation error is 1 - Pearson on a leakage-safe top-variance gene
set (selected from the training fold only, per the ceiling analysis). This out-of-fold error table is
what PertEMA later learns to estimate.

LEAKAGE CAVEAT: the columns ontarget_effect_size, n_downstream, donor_corr_hits, and guide_corr_all
are derived from the TRUE measured effect. They are recorded here as analysis covariates only (for
example to study what drives error). They must NEVER be used as PertEMA prediction-time features, or
the estimator would see the answer. Legitimate features come from prediction-time information only
(predicted effect magnitude, baseline expression, network position, training-set similarity).

Run: pixi run python src/eval/run_predictors.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval"):
    sys.path.insert(0, os.path.join(SRC, _p))

from load_de import load_de_stats                          # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/predictor_errors"
SEEDS = [42, 43, 44]
N_HVG = 2000
EPS = 0.05  # min |true log2FC| to count a gene's direction


def row_pearson(P, T):
    Pc = P - P.mean(1, keepdims=True)
    Tc = T - T.mean(1, keepdims=True)
    num = (Pc * Tc).sum(1)
    den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def row_direction_acc(P, T, eps=EPS):
    mask = np.abs(T) > eps
    agree = (np.sign(P) == np.sign(T)) & mask
    denom = mask.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, agree.sum(1) / denom, np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()

    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB)  # our own artifact; gene_ids is a plain string array, no pickle needed
    gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    predictors = [MeanPredictor("condition"), MeanPredictor("global"),
                  KNNSimilarityPredictor(gene_emb, k=25),
                  RidgeEmbeddingPredictor(gene_emb, alpha=100.0)]

    for seed in SEEDS:
        fold_of_gene = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fold_of_gene.get(g, -1) for g in gene_of_row])
        assert (row_fold >= 0).all(), "every perturbation row must map to a fold"
        n_folds = int(row_fold.max()) + 1

        records = []
        for k in range(n_folds):
            train_rows = np.where(row_fold != k)[0]
            test_rows = np.where(row_fold == k)[0]
            # leakage-safe HVG: top-variance genes over training effects only
            var = lfc[train_rows].var(axis=0)
            hvg = np.argsort(-var)[:N_HVG]
            true_hvg = lfc[test_rows][:, hvg]
            true_mag = np.abs(true_hvg).mean(1)

            for pred in predictors:
                pred.fit(train_rows, lfc, obs)
                P = pred.predict(test_rows, obs)[:, hvg]
                err = 1.0 - row_pearson(P, true_hvg)
                dacc = row_direction_acc(P, true_hvg)
                pmag = np.abs(P).mean(1)
                for j, r in enumerate(test_rows):
                    records.append(dict(
                        obs_index=int(r), gene=gene_of_row[r],
                        gene_name=str(obs["target_contrast_gene_name"].iloc[r]),
                        condition=str(obs["culture_condition"].iloc[r]), fold=k,
                        predictor=pred.name, err_1mp_hvg=float(err[j]),
                        dir_acc_hvg=float(dacc[j]), pred_magnitude=float(pmag[j]),
                        true_magnitude_hvg=float(true_mag[j]),
                        ontarget_effect_size=float(obs["ontarget_effect_size"].iloc[r]),
                        n_downstream=int(obs["n_downstream"].iloc[r]),
                        donor_corr_hits=float(obs["donor_correlation_hits_mean"].iloc[r]),
                        guide_corr_all=float(obs["guide_correlation_all"].iloc[r]),
                    ))
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(OUT, f"errors_seed{seed}.csv"), index=False)

        print(f"\n===== seed {seed}: {df.shape[0]} rows =====")
        summ = df.groupby("predictor").agg(
            mean_err=("err_1mp_hvg", "mean"), median_err=("err_1mp_hvg", "median"),
            mean_diracc=("dir_acc_hvg", "mean")).round(4)
        print(summ.to_string())
        # sanity: mean predictor error should rise with true effect magnitude
        for pn in df["predictor"].unique():
            d = df[df["predictor"] == pn].dropna(subset=["err_1mp_hvg"])
            rr = np.corrcoef(d["true_magnitude_hvg"], d["err_1mp_hvg"])[0, 1]
            print(f"  {pn}: corr(true_magnitude, error) = {rr:.3f}")

    print(f"\nwrote {OUT}/errors_seed*.csv")


if __name__ == "__main__":
    main()
