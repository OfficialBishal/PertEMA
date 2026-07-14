"""D6 routing swing, re-tested with a DIVERSE 10-predictor roster (the master-plan ask: does adding predictor
diversity, including foundation models, give routing enough headroom to beat the best fixed predictor?). This
is the strongest form of the routing test on the primary data: it routes among the four CLEAN predictors with
a defined delta-Pearson error (per-condition mean, global mean, kNN, ridge) plus the six frozen-adapted
foundation predictors (ridge/kNN over Geneformer, scGPT, and gene2vec pretrained embeddings), on the Gladstone
single-context task, gene-disjoint, three seeds.

Per perturbation-condition row we compute each predictor's realized error (1 - Pearson-delta on the top-1000
expressed genes, the parity metric) and its predicted magnitude, then a per-predictor PertEMA reliability
estimator (gene-disjoint OOF over the co-expression embedding + the predictor's predicted magnitude) predicts
that predictor's error. The router picks the lowest predicted error per row. We compare the routed error to
the best fixed predictor and to the oracle best-of-12 for the headroom. Leakage-safe: routing uses only
prediction-time features, never the held-out truth (invariant 1).

Kill criterion (D6): routed must beat the best fixed predictor with a paired-bootstrap fraction-better above
0.95, else reported as a negative. Reported plainly either way.

Run: pixi run python src/eval/run_router_diverse.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                              # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_parity import pearson_delta                                           # noqa: E402
sys.path.insert(0, os.path.join(SRC, "pertema"))
from run_estimator import gbt                                                   # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_BOOT = 2000


def per_row_error_and_mag(pred, lfc, obs, gene_of_row, folds, seed, top):
    """Gene-disjoint OOF per-row error (1 - Pearson-delta over top) and predicted magnitude for one predictor."""
    n = len(gene_of_row)
    err = np.full(n, np.nan); mag = np.full(n, np.nan)
    fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
    row_fold = np.array([fog.get(g, -1) for g in gene_of_row])
    for k in range(int(row_fold.max()) + 1):
        tr = np.where(row_fold != k)[0]; te = np.where(row_fold == k)[0]
        if te.size == 0:
            continue
        pred.fit(tr, lfc, obs)
        P = pred.predict(te, obs)
        err[te] = 1.0 - pearson_delta(P, lfc[te], top)
        mag[te] = np.abs(P[:, top]).mean(1)
    return err, mag, row_fold


def main():
    de = load_de_stats(layers=("log_fc", "baseMean"))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB); coexpr = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    Ecoexpr = np.array([coexpr.get(g, np.full(emb_dim, np.nan)) for g in gene_of_row])
    # top-1000 by baseMean control expression (reference-correct, leakage-safe expression level, not an
    # effect-derived quantity). Was previously a dead-code fallback to mean|log_fc|, fixed per audit.
    expr = np.nan_to_num(de.layers["baseMean"]).mean(0)
    top = np.argsort(-expr)[:N_TOP]

    # the diverse routing roster: the CLEAN predictors with a defined delta-Pearson error (no_change predicts
    # all zeros, so its delta-Pearson is undefined and it cannot participate in delta-based routing) plus the
    # six frozen-adapted foundation predictors. Eleven predictors total.
    def clean_roster():
        return [("mean_condition", MeanPredictor("condition")), ("mean_global", MeanPredictor("global")),
                ("knn_coexpr_k25", KNNSimilarityPredictor(coexpr, k=25)),
                ("ridge_embed", RidgeEmbeddingPredictor(coexpr, alpha=100.0))]

    def foundation_roster():
        out = []
        for ef in sorted(glob.glob(os.path.join("results", "features", "foundation_*.npz"))):
            name = os.path.basename(ef)[len("foundation_"):-len(".npz")]
            z = np.load(ef); ge = {str(g): v for g, v in zip(z["gene_ids"], z["embedding"])}
            out.append((f"{name}_ridge", RidgeEmbeddingPredictor(ge, alpha=100.0)))
            out.append((f"{name}_knn", KNNSimilarityPredictor(ge, k=25)))
        return out

    names = [n for n, _ in clean_roster()] + [n for n, _ in foundation_roster()]
    rows = []
    for seed in SEEDS:
        err = {}; mag = {}; row_fold = None
        for nm, pred in clean_roster() + foundation_roster():
            e, m, rf = per_row_error_and_mag(pred, lfc, obs, gene_of_row, folds, seed, top)
            err[nm] = e; mag[nm] = m; row_fold = rf
        valid = np.isfinite(np.column_stack([err[n] for n in names])).all(1)
        vi = np.where(valid)[0]
        ERR = np.column_stack([err[n] for n in names])
        # per-predictor PertEMA reliability estimator (gene-disjoint OOF), predicts each predictor's error
        pred_err = {}
        for nm in names:
            X = np.column_stack([Ecoexpr, mag[nm]])
            y = err[nm]; pe = np.full(len(y), np.nan)
            for k in np.unique(row_fold[row_fold >= 0]):
                tr = np.where((row_fold != k) & np.isfinite(y))[0]; te = np.where(row_fold == k)[0]
                if tr.size < 50 or te.size == 0:
                    continue
                pe[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
            pred_err[nm] = pe
        PRED = np.column_stack([pred_err[n] for n in names])
        both_valid = valid & np.isfinite(PRED).all(1)
        bvi = np.where(both_valid)[0]
        chosen = np.argmin(PRED[bvi], axis=1)
        routed = ERR[bvi, chosen]
        mean_fixed = {n: float(np.nanmean(err[n][vi])) for n in names}
        best_name = min(mean_fixed, key=mean_fixed.get)
        best_fixed = ERR[bvi, names.index(best_name)]
        oracle = ERR[bvi].min(1)
        rng = np.random.default_rng(seed)
        diffs = np.array([np.mean(best_fixed[rng.integers(0, len(bvi), len(bvi))]
                                  - routed[rng.integers(0, len(bvi), len(bvi))]) for _ in range(N_BOOT)])
        rows.append(dict(seed=seed, n=len(bvi), n_predictors=len(names), best_fixed=best_name,
                         err_best_fixed=mean_fixed[best_name], err_routed=float(routed.mean()),
                         err_oracle=float(oracle.mean()), oracle_headroom=float(best_fixed.mean() - oracle.mean()),
                         routed_minus_bestfixed=float(routed.mean() - best_fixed.mean()),
                         boot_frac_routed_better=float(np.mean(diffs > 0))))
        print(f"seed {seed}: best_fixed {best_name} {mean_fixed[best_name]:.4f} | routed {routed.mean():.4f} "
              f"(delta {routed.mean()-best_fixed.mean():+.4f}) | oracle {oracle.mean():.4f} "
              f"(headroom {best_fixed.mean()-oracle.mean():+.4f}) | frac-better {rows[-1]['boot_frac_routed_better']:.3f}",
              flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "router_diverse_gladstone.csv"), index=False)
    mr, mb = res["err_routed"].mean(), res["err_best_fixed"].mean()
    mo, mf = res["err_oracle"].mean(), res["boot_frac_routed_better"].mean()
    verdict = "POSITIVE" if (mr < mb and mf > 0.95) else "NEGATIVE"
    print(f"\n=== D6 routing with the diverse {res['n_predictors'].iloc[0]}-predictor roster (Gladstone, 3 seeds) ===")
    print(f"best fixed {mb:.4f} | routed {mr:.4f} (delta {mr-mb:+.4f}, frac-better {mf:.3f}) | "
          f"oracle {mo:.4f} (headroom {mb-mo:+.4f})")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}/router_diverse_gladstone.csv")


if __name__ == "__main__":
    main()
