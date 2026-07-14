"""X1 hardening (X7-style): is the Adamson routing failure a real small-sample router limit, or an artifact of
an overfit xgboost estimator? Probe a small PRE-SPECIFIED set of standard estimators (not cherry-picked, all
reported) plus the ORACLE router (route by true error, the capturable ceiling). If NO standard estimator's
routed error beats best-fixed, the "feasible geometry, insufficient router" conclusion is robust rather than
an artifact of one overfit model. If a standard estimator DOES beat best-fixed, routing is feasible on Adamson
with an adequate estimator and the directional prediction is confirmed. Reported plainly either way.

Run: pixi run python src/eval/run_router_adamson_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("predictors", "eval"):
    sys.path.insert(0, os.path.join(SRC, _p))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402

EFF = "results/features/adamson_effects.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_FOLDS = 5


def pearson_delta(P, T, cols):
    Pc = P[:, cols] - P[:, cols].mean(1, keepdims=True)
    Tc = T[:, cols] - T[:, cols].mean(1, keepdims=True)
    num = (Pc * Tc).sum(1)
    den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gene_disjoint_folds(n, k, seed):
    order = np.random.RandomState(seed).permutation(n)
    fold = np.empty(n, int)
    fold[order] = np.arange(n) % k
    return fold


ESTIMATORS = {
    "xgb_standard": lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, n_jobs=8, tree_method="hist", random_state=0),
    "xgb_shallow": lambda: xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                            n_jobs=8, tree_method="hist", random_state=0),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "rf_shallow": lambda: RandomForestRegressor(n_estimators=300, max_depth=3, n_jobs=8, random_state=0),
    "knn_feat": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
}


def main():
    d = np.load(EFF, allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff"].astype(np.float32)
    emb_of = {g: v for g, v in zip(d["emb_genes"].astype(str), d["embedding"].astype(np.float32))}
    in_panel = np.array([g in emb_of and np.isfinite(eff_all[i, 0]) for i, g in enumerate(pgenes)])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]; eff = eff_all[idx]; n = len(idx)
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": "adamson"})
    gene_emb = {g: emb_of[g] for g in genes}
    Egene = np.array([emb_of[g] for g in genes])
    top = np.argsort(-eff.var(0))[:N_TOP]

    def build():
        return [("mean_global", MeanPredictor("global")),
                ("ridge_embed", RidgeEmbeddingPredictor(gene_emb, alpha=100.0)),
                ("knn_coexpr_k25", KNNSimilarityPredictor(gene_emb, k=25))]
    names = [nm for nm, _ in build()]

    # cache per-seed OOF errors and magnitudes (predictor layer is estimator-independent)
    cache = {}
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        err = {nm: np.full(n, np.nan) for nm in names}
        mag = {nm: np.full(n, np.nan) for nm in names}
        for nm, pred in build():
            for k in range(N_FOLDS):
                tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
                if te.size == 0:
                    continue
                pred.fit(tr, eff, obs); P = pred.predict(te, obs)
                err[nm][te] = 1.0 - pearson_delta(P, eff[te], top)
                mag[nm][te] = np.abs(P[:, top]).mean(1)
        cache[seed] = (fold, err, mag)

    rows = []
    for est_name, ctor in list(ESTIMATORS.items()) + [("ORACLE_true_error", None)]:
        for seed in SEEDS:
            fold, err, mag = cache[seed]
            valid = np.isfinite(np.column_stack([err[nm] for nm in names])).all(1)
            ERR = np.column_stack([err[nm] for nm in names])
            if est_name == "ORACLE_true_error":
                PRED = ERR.copy()   # route by true error: the capturable ceiling
                rlist = [1.0]
            else:
                pred_err = {}; rlist = []
                for nm in names:
                    X = np.column_stack([Egene, mag[nm]]); y = err[nm]
                    pe = np.full(n, np.nan)
                    for k in np.unique(fold):
                        tr = np.where((fold != k) & np.isfinite(y))[0]; te = np.where(fold == k)[0]
                        if tr.size < 20 or te.size == 0:
                            continue
                        pe[te] = ctor().fit(X[tr], y[tr]).predict(X[te])
                    pred_err[nm] = pe
                    m = valid & np.isfinite(pe)
                    if m.sum() > 5:
                        rlist.append(spearmanr(pe[m], y[m]).statistic)
                PRED = np.column_stack([pred_err[nm] for nm in names])
            bv = valid & np.isfinite(PRED).all(1); bvi = np.where(bv)[0]
            chosen = np.argmin(PRED[bvi], axis=1)
            routed = ERR[bvi, chosen]
            mean_fixed = {nm: float(np.nanmean(err[nm][np.where(valid)[0]])) for nm in names}
            best_name = min(mean_fixed, key=mean_fixed.get)
            best_fixed = ERR[bvi, names.index(best_name)]
            oracle = ERR[bvi].min(1)
            rng = np.random.default_rng(seed)
            diffs = np.array([np.mean(best_fixed[rng.integers(0, len(bvi), len(bvi))]
                                      - routed[rng.integers(0, len(bvi), len(bvi))]) for _ in range(2000)])
            rows.append(dict(estimator=est_name, seed=seed, n=len(bvi),
                             err_best_fixed=mean_fixed[best_name], err_routed=float(routed.mean()),
                             err_oracle=float(oracle.mean()),
                             routed_minus_bestfixed=float(routed.mean() - best_fixed.mean()),
                             r=float(np.mean(rlist)), frac_better=float(np.mean(diffs > 0))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "router_adamson_sensitivity.csv"), index=False)
    agg = df.groupby("estimator").agg(err_best_fixed=("err_best_fixed", "mean"),
                                      err_routed=("err_routed", "mean"), err_oracle=("err_oracle", "mean"),
                                      delta=("routed_minus_bestfixed", "mean"), r=("r", "mean"),
                                      frac_better=("frac_better", "mean")).reset_index()
    print("=== X1 estimator sensitivity (Adamson routing, 3 seeds) ===")
    print(agg.to_string(index=False))
    beats = agg[(agg.estimator != "ORACLE_true_error") & (agg.delta < 0) & (agg.frac_better > 0.95)]
    print(f"\nstandard estimators that beat best-fixed at frac>0.95: {len(beats)} of {len(ESTIMATORS)}")
    print("CONCLUSION:", "ROUTING FEASIBLE with an adequate estimator" if len(beats) else
          "insufficient-router robust across standard estimators (routing fails despite real oracle headroom)")
    print(f"wrote {OUT}/router_adamson_sensitivity.csv")


if __name__ == "__main__":
    main()
