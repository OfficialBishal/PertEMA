"""D3/D5 parity, third dataset: reproduce the Ahlmann-Eltze prediction-accuracy metrics on the Norman 2019
K562 single-gene perturbations, so the "baselines are hard to beat" finding is tested on a third real dataset
(Gladstone, Replogle, Norman, Adamson). Uses the adamson_effects.npz (scPerturb download).

Same protocol as run_parity_replogle.py: in-panel cohort (perturbed genes with a co-expression embedding),
gene-disjoint 5-fold, seeds 42/43/44, metrics L2 and Pearson-delta on the top-1000 genes by cross-perturbation
effect variance (no control-expression vector, so the variance substitute, documented and leakage-safe). All
predictors CLEAN.

Run: pixi run python src/eval/run_parity_adamson.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "predictors"))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402

EFFECTS = "results/features/adamson_effects.npz"
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


def gene_disjoint_folds(n, k, seed):
    order = np.random.RandomState(seed).permutation(n)
    fold = np.empty(n, int); fold[order] = np.arange(n) % k
    return fold


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(EFFECTS, allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff"].astype(np.float32)
    emb_genes = d["emb_genes"].astype(str)
    embedding = d["embedding"].astype(np.float32)
    emb_of = {g: v for g, v in zip(emb_genes, embedding)}

    in_panel = np.array([g in emb_of and np.isfinite(eff_all[i, 0]) for i, g in enumerate(pgenes)])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]; eff = eff_all[idx]; n = len(idx)
    assert len(set(genes.tolist())) == n, "perturbed genes must be unique for gene-disjoint folds"
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": CONTEXT})
    gene_emb = {g: emb_of[g] for g in genes}
    top = np.argsort(-eff.var(0))[:N_TOP]

    def build():
        return [MeanPredictor("global"), RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
                KNNSimilarityPredictor(gene_emb, k=25)]
    mean_name = MeanPredictor("global").name

    rows = []
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        preds = build()
        per = {p.name: {"l2": [], "pdelta": []} for p in preds}
        for k in range(N_FOLDS):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            if te.size == 0:
                continue
            T = eff[te]
            for p in preds:
                p.fit(tr, eff, obs); P = p.predict(te, obs)
                per[p.name]["l2"].append(rmse(P, T, top)); per[p.name]["pdelta"].append(pearson_delta(P, T, top))
        for name in per:
            rows.append(dict(seed=seed, predictor=name, l2=float(np.nanmean(np.concatenate(per[name]["l2"]))),
                             pearson_delta=float(np.nanmean(np.concatenate(per[name]["pdelta"])))))
    res = pd.DataFrame(rows)
    agg = res.groupby("predictor").agg(l2=("l2", "mean"), pearson_delta=("pearson_delta", "mean")).reset_index()
    base = float(agg.loc[agg["predictor"] == mean_name, "l2"].iloc[0])
    agg["error_rel_to_mean"] = agg["l2"] / base
    agg["provenance"] = "CLEAN"
    agg = agg.sort_values("error_rel_to_mean")[["predictor", "l2", "pearson_delta", "error_rel_to_mean", "provenance"]]
    agg.to_csv(os.path.join(OUT, "parity_adamson.csv"), index=False)
    print("=== D3/D5 parity on Adamson K562 single-gene (Ahlmann-Eltze metrics) ===")
    print(f"cohort: {n} in-panel single-gene perturbations, gene-disjoint {N_FOLDS}-fold, seeds {SEEDS}")
    print(agg.round(4).to_string(index=False))
    beats = agg[(agg["error_rel_to_mean"] < 1.0) & (agg["predictor"] != mean_name)]
    print(f"\nPredictors beating the mean baseline: {beats['predictor'].tolist() if len(beats) else 'NONE - mean baseline wins'}")
    print(f"wrote {OUT}/parity_adamson.csv")


if __name__ == "__main__":
    main()
