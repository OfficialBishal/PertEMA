"""Context-transfer error table: train a predictor in one condition, score it against another.

For each ordered condition pair (src, dst) and each held-out gene, the predictor is trained on the
src-condition effects of the training-fold genes, predicts the held-out gene's effect from its own
src-condition row, and that prediction is scored against the gene's dst-condition truth. 1 - Pearson on
a leakage-safe top-variance gene set (selected from src training) is the transfer error. src==dst gives
the within-context baseline. The gap (dst != src) minus (src==src) is the per-gene context sensitivity
that PertEMA later learns to flag. Folds are gene-disjoint, so no held-out gene's effect (in any
condition) is ever seen during predictor training (invariant 1).

Run: pixi run python src/eval/run_transfer.py
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
from predictors import KNNSimilarityPredictor, MeanPredictor  # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/transfer"
SEEDS = [42, 43, 44]
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
N_HVG = 2000


def row_pearson(P, T):
    Pc = P - P.mean(1, keepdims=True)
    Tc = T - T.mean(1, keepdims=True)
    num = (Pc * Tc).sum(1)
    den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene = obs["target_contrast"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB)
    gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}

    # per (gene, condition) -> row index
    row_of = {}
    for i in range(len(obs)):
        row_of[(gene[i], cond[i])] = i

    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fog.get(g, -1) for g in gene])
        n_folds = int(row_fold.max()) + 1
        predictors = [MeanPredictor("condition"), KNNSimilarityPredictor(gene_emb, k=25)]

        records = []
        for src in CONDS:
            for dst in CONDS:
                for k in range(n_folds):
                    train_A = np.where((row_fold != k) & (cond == src))[0]
                    if train_A.size == 0:
                        continue
                    var = lfc[train_A].var(0)
                    hvg = np.argsort(-var)[:N_HVG]
                    test_genes = folds.loc[folds[f"fold_seed{seed}"] == k, "gene"].tolist()
                    pairs = [(g, row_of.get((g, src)), row_of.get((g, dst)))
                             for g in test_genes]
                    pairs = [(g, a, b) for g, a, b in pairs if a is not None and b is not None]
                    if not pairs:
                        continue
                    a_rows = np.array([a for _, a, _ in pairs])
                    b_rows = np.array([b for _, _, b in pairs])
                    true_dst = lfc[b_rows][:, hvg]
                    for pred in predictors:
                        pred.fit(train_A, lfc, obs)
                        P = pred.predict(a_rows, obs)[:, hvg]     # src-based prediction
                        err = 1.0 - row_pearson(P, true_dst)
                        pmag = np.abs(P).mean(1)
                        for j, (g, _, _) in enumerate(pairs):
                            records.append(dict(gene=g, src=src, dst=dst, transfer=(src != dst),
                                                predictor=pred.name, fold=k,
                                                transfer_err=float(err[j]), pred_magnitude=float(pmag[j])))
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(OUT, f"transfer_errors_seed{seed}.csv"), index=False)

        print(f"\n===== seed {seed}: {df.shape[0]} rows =====")
        piv = df.groupby(["predictor", "transfer"])["transfer_err"].mean().unstack()
        print(piv.round(4).to_string())
        # per-pair means for the mean predictor
        mp = df[df["predictor"] == "mean_condition"]
        print(mp.groupby(["src", "dst"])["transfer_err"].mean().round(4).to_string())
    print(f"\nwrote {OUT}/transfer_errors_seed*.csv")


if __name__ == "__main__":
    main()
