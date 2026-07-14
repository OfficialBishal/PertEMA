"""PertEMA gradient-boosted-tree estimator: predict per-perturbation predictor error from
prediction-time features, and test it against heuristics and negative controls.

Design (leakage-safe):
  - Target: the out-of-fold predictor error err_1mp_hvg from results/predictor_errors (already OOF from
    gene-disjoint predictor CV).
  - Features (prediction-time only, no true-effect quantity): predicted effect magnitude; control baseline
    expression, dropout, and cross-donor variance of the perturbed gene in the query condition; the gene's
    control co-expression embedding (50 dims); condition one-hot; and training-set similarity = distance to
    the nearest TRAINING gene in embedding space, computed inside each outer fold so no test gene informs it.
  - Outer gene-disjoint CV uses the same fixed folds, so the estimator never trains on the error of a gene
    it scores (invariant 1). Second-order note: predictor OOF labels for training genes were produced by
    predictors that saw test genes in their own training pool; a fully nested recompute is deferred.

Evaluation per predictor and seed:
  - Calibration: Spearman(predicted error, true error).
  - Selective prediction: area under the risk-coverage curve (AURC, lower better), abstaining by predicted
    error, compared to the effect-magnitude heuristic, the similarity heuristic, an oracle, and no-selection.
  - Negative controls: random-feature estimator (must not beat real), label-shuffle (calibration must die).
Headline is the family mean over seeds 42/43/44 with a 95% normal CI.

Run: pixi run python src/pertema/run_estimator.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
from metrics import aurc, reliability_spearman  # noqa: E402

ERR = "results/predictor_errors"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEEDS = [42, 43, 44]
PREDICTORS = ["mean_condition", "knn_coexpr_k25", "ridge_embed"]
CONDS = ["Rest", "Stim8hr", "Stim48hr"]


def gbt():
    # xgboost hist; handles NaN features natively. Threads capped (128-core host oversubscribes).
    return xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=8, tree_method="hist", random_state=0)


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return reliability_spearman(a[m], b[m]) if m.sum() > 3 else np.nan


def best_aurc(true_err, score):
    """AURC using score as reliability; try both sign conventions and keep the better (lower) one,
    so a heuristic is given its most favorable orientation."""
    m = np.isfinite(true_err) & np.isfinite(score)
    if m.sum() < 10:
        return np.nan
    return float(min(aurc(true_err[m], score[m]), aurc(true_err[m], -score[m])))


def build_base_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim):
    n = df.shape[0]
    cond_idx = {c: i for i, c in enumerate(CONDS)}
    ci = df["condition"].map(cond_idx).to_numpy()
    gi = df["gene"].map(gene_col).to_numpy()          # column of gene in baseline arrays, or nan
    feats = {"pred_magnitude": df["pred_magnitude"].to_numpy()}
    for name, arr in [("baseline", baseline), ("dropout", dropout), ("donor_var", donor_var)]:
        v = np.full(n, np.nan)
        ok = ~np.isnan(gi)
        idx = gi[ok].astype(int)
        v[ok] = arr[ci[ok], idx]
        feats[name] = v
    E = np.full((n, emb_dim), np.nan, dtype=np.float32)
    for j, g in enumerate(df["gene"].to_numpy()):
        e = emb_map.get(g)
        if e is not None:
            E[j] = e
    for c in CONDS:
        feats[f"is_{c}"] = (df["condition"] == c).to_numpy().astype(float)
    base = np.column_stack([feats[k] for k in feats])
    return np.column_stack([base, E]), df["gene"].to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    bz = np.load(BASE)
    genes_b = [str(g) for g in bz["genes"]]
    gene_col = {g: i for i, g in enumerate(genes_b)}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in SEEDS:
        err = pd.read_csv(os.path.join(ERR, f"errors_seed{seed}.csv"), dtype={"gene": str})
        for pname in PREDICTORS:
            df = err[err["predictor"] == pname].reset_index(drop=True)
            Xb, gene_arr = build_base_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
            y = df["err_1mp_hvg"].to_numpy()
            fold = df["fold"].to_numpy()
            emb_arr = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in gene_arr])

            # per-unique-gene embeddings; folds are gene-disjoint so a gene is wholly train or test
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)
            fold_u = np.full(len(genes_u), -1)
            fold_u[inv] = fold

            oof = {k: np.full(len(df), np.nan) for k in ["est", "rand", "sim"]}
            rng = np.random.default_rng(seed)
            for k in np.unique(fold):
                tr = np.where(fold != k)[0]
                te = np.where(fold == k)[0]
                # training-set similarity: nearest training gene in embedding (brute, threaded, per gene)
                tr_u = np.where((fold_u != k) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
                sim_u = np.full(len(genes_u), np.nan)
                q = np.where(good_u)[0]
                d, _ = nn.kneighbors(emb_u[q])
                sim_u[q] = d.ravel()
                sim = sim_u[inv]
                oof["sim"][te] = sim[te]

                Xtr = np.column_stack([Xb[tr], sim[tr]])
                Xte = np.column_stack([Xb[te], sim[te]])
                est = gbt()
                est.fit(Xtr, y[tr])
                oof["est"][te] = est.predict(Xte)
                # random-feature control: permute each training feature independently
                Xtr_r = np.column_stack([rng.permutation(Xtr[:, j]) for j in range(Xtr.shape[1])])
                est_r = gbt()
                est_r.fit(Xtr_r, y[tr])
                oof["rand"][te] = est_r.predict(Xte)

            # label-shuffle control: fit on shuffled y, measure calibration collapse (single split proxy)
            tr0, te0 = np.where(fold != 0)[0], np.where(fold == 0)[0]
            simcol = oof["sim"]
            est_ls = gbt()
            est_ls.fit(np.column_stack([Xb[tr0], simcol[tr0]]), rng.permutation(y[tr0]))
            ls_pred = est_ls.predict(np.column_stack([Xb[te0], simcol[te0]]))
            ls_spear = sp(ls_pred, y[te0])

            r = dict(seed=seed, predictor=pname, n=len(df),
                     spearman_est=sp(oof["est"], y),
                     spearman_rand=sp(oof["rand"], y),
                     spearman_labelshuffle=ls_spear,
                     aurc_est=best_aurc(y, -oof["est"]),
                     aurc_magnitude=best_aurc(y, df["pred_magnitude"].to_numpy()),
                     aurc_similarity=best_aurc(y, -oof["sim"]),
                     aurc_random_feat=best_aurc(y, -oof["rand"]),
                     aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]),
                     aurc_noselect=float(np.nanmean(y)))
            rows.append(r)
            print(f"seed {seed} {pname}: spearman_est={r['spearman_est']:.3f} "
                  f"AURC est={r['aurc_est']:.3f} mag={r['aurc_magnitude']:.3f} "
                  f"sim={r['aurc_similarity']:.3f} rand={r['aurc_random_feat']:.3f} "
                  f"oracle={r['aurc_oracle']:.3f} noselect={r['aurc_noselect']:.3f} "
                  f"| labelshuffle_spearman={ls_spear:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "estimator_metrics_per_seed.csv"), index=False)

    # family mean +/- 95% normal CI across seeds
    agg = []
    metric_cols = [c for c in res.columns if c not in ("seed", "predictor", "n")]
    for pname in PREDICTORS:
        sub = res[res["predictor"] == pname]
        row = {"predictor": pname}
        for c in metric_cols:
            v = sub[c].to_numpy()
            m, sd = np.nanmean(v), np.nanstd(v, ddof=1) if v.size > 1 else 0.0
            row[c] = f"{m:.3f} +/- {1.96*sd/np.sqrt(len(v)):.3f}"
        agg.append(row)
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(os.path.join(OUT, "estimator_metrics_summary.csv"), index=False)
    print("\n=== family mean +/- 95% CI (seeds 42/43/44) ===")
    print(aggdf.to_string(index=False))
    print(f"\nwrote {OUT}/estimator_metrics_*.csv")


if __name__ == "__main__":
    main()
