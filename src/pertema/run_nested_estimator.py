"""Fully nested PertEMA estimator: remove the documented second-order label dependency.

The single-level estimator (run_estimator.py) trains on predictor out-of-fold (OOF) error labels whose
predictors, in their own OOF, were trained on pools that included the estimator's test-fold genes. This is
a second-order dependency: a test gene's OWN error never informs its estimate (invariant 1 holds), but a
TRAINING gene's error label was produced by a predictor that had seen test-fold genes. An internal review flagged this as non-fatal and deferred a nested recompute.

This script performs that nested recompute. For each outer estimator-test fold k:
  - The estimator's TRAINING-gene labels (and the predictor-derived pred_magnitude feature) are recomputed
    by an inner leave-one-fold-out over the folds != k, so no predictor that produced a training label or
    predictor-derived feature ever saw a gene in fold k.
  - The TEST-fold labels are already nested-clean (their predictors were trained on folds != k by
    construction) and are reused from results/predictor_errors.
  - All other features (control baseline expression, dropout, cross-donor variance, co-expression
    embedding, within-fold training similarity) never see any true effect and need no recompute.

The standard and nested estimator OOF are computed in the SAME run over identical features and folds, so
the only difference is the training labels/pred_magnitude. We report the paired delta on AURC and Spearman.
If the delta is within the 3-seed noise, the second-order dependency is negligible and the caveat is
resolved with evidence rather than argument.

Run: pixi run python src/pertema/run_nested_estimator.py [seed ...]   (default seeds 42 43 44)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("eval", "predictors", "data", "pertema"):
    sys.path.insert(0, os.path.join(SRC, _p))

from load_de import load_de_stats                                          # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_predictors import N_HVG, row_pearson                             # noqa: E402
from run_estimator import (BASE, CONDS, EMB, ERR, OUT, best_aurc,         # noqa: E402
                           build_base_features, gbt, sp)
from metrics import aurc                                                  # noqa: E402
from sklearn.neighbors import NearestNeighbors                            # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
PREDICTORS = ["mean_condition", "knn_coexpr_k25", "ridge_embed"]


def make_predictors(gene_emb):
    return {
        "mean_condition": MeanPredictor("condition"),
        "knn_coexpr_k25": KNNSimilarityPredictor(gene_emb, k=25),
        "ridge_embed": RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
    }


def nested_labels(k, n_folds, row_fold, lfc, obs, gene_emb):
    """Inner leave-one-fold-out over folds != k. Returns pname -> {obs_index: (err, pmag)} for every
    row whose gene is in a training fold (fold != k), computed blind to fold k."""
    preds = make_predictors(gene_emb)
    out = {pn: {} for pn in preds}
    for j in range(n_folds):
        if j == k:
            continue
        inner_train = np.where((row_fold != k) & (row_fold != j))[0]
        inner_test = np.where(row_fold == j)[0]
        var = lfc[inner_train].var(axis=0)               # leakage-safe HVG from inner-train only
        hvg = np.argsort(-var)[:N_HVG]
        true_hvg = lfc[inner_test][:, hvg]
        for pn, pred in preds.items():
            pred.fit(inner_train, lfc, obs)
            P = pred.predict(inner_test, obs)[:, hvg]
            err = 1.0 - row_pearson(P, true_hvg)
            pmag = np.abs(P).mean(1)
            for t, r in enumerate(inner_test):
                out[pn][int(r)] = (float(err[t]), float(pmag[t]))
    return out


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [42, 43, 44]
    os.makedirs(OUT, exist_ok=True)

    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()

    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    bz = np.load(BASE)
    genes_b = [str(g) for g in bz["genes"]]
    gene_col = {g: i for i, g in enumerate(genes_b)}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_map = gene_emb
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in seeds:
        fold_of_gene = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fold_of_gene.get(g, -1) for g in gene_of_row])
        assert (row_fold >= 0).all()
        n_folds = int(row_fold.max()) + 1

        # nested training labels per outer fold (shared across the 3 predictors)
        print(f"[seed {seed}] computing nested labels over {n_folds} outer folds ...", flush=True)
        nested = {k: nested_labels(k, n_folds, row_fold, lfc, obs, gene_emb) for k in range(n_folds)}

        err = pd.read_csv(os.path.join(ERR, f"errors_seed{seed}.csv"), dtype={"gene": str})
        for pname in PREDICTORS:
            df = err[err["predictor"] == pname].reset_index(drop=True)
            oi = df["obs_index"].to_numpy()
            fold = df["fold"].to_numpy()
            y_std = df["err_1mp_hvg"].to_numpy()             # realized error (evaluation target, both estimators)
            gene_arr = df["gene"].to_numpy()

            # per-unique-gene embeddings for the similarity feature (same as run_estimator)
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)

            oof_std = np.full(len(df), np.nan)
            oof_nested = np.full(len(df), np.nan)
            Xb_std, _ = build_base_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
            fold_u = np.full(len(genes_u), -1)
            fold_u[inv] = fold
            for k in np.unique(fold):
                tr = np.where(fold != k)[0]
                te = np.where(fold == k)[0]
                # within-fold training similarity (identical for both estimators)
                tr_u = np.where((fold_u != k) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
                sim_u = np.full(len(genes_u), np.nan)
                qq = np.where(good_u)[0]
                d, _ = nn.kneighbors(emb_u[qq])
                sim_u[qq] = d.ravel()
                sim = sim_u[inv]

                # standard features/labels
                Xtr_std = np.column_stack([Xb_std[tr], sim[tr]])
                Xte = np.column_stack([Xb_std[te], sim[te]])
                m = gbt(); m.fit(Xtr_std, y_std[tr]); oof_std[te] = m.predict(Xte)

                # nested features/labels for TRAINING rows only (test rows use standard, unchanged)
                dfn = df.copy()
                nl = nested[k]
                pm = df["pred_magnitude"].to_numpy().copy()
                yn = y_std.copy()
                for idx in tr:
                    e_p = nl[pname].get(int(oi[idx]))
                    if e_p is not None:
                        yn[idx] = e_p[0]
                        pm[idx] = e_p[1]
                dfn["pred_magnitude"] = pm
                Xb_n, _ = build_base_features(dfn, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
                Xtr_n = np.column_stack([Xb_n[tr], sim[tr]])
                # test features use the standard pred_magnitude (Xte from Xb_std), matched to y_std target
                mn = gbt(); mn.fit(Xtr_n, yn[tr]); oof_nested[te] = mn.predict(Xte)

            r = dict(
                seed=seed, predictor=pname, n=len(df),
                spearman_std=sp(oof_std, y_std), spearman_nested=sp(oof_nested, y_std),
                aurc_std=best_aurc(y_std, -oof_std), aurc_nested=best_aurc(y_std, -oof_nested),
                aurc_noselect=float(np.nanmean(y_std)),
                aurc_oracle=float(aurc(y_std[np.isfinite(y_std)], -y_std[np.isfinite(y_std)])),
            )
            r["delta_aurc_nested_minus_std"] = r["aurc_nested"] - r["aurc_std"]
            r["delta_spearman_nested_minus_std"] = r["spearman_nested"] - r["spearman_std"]
            rows.append(r)
            print(f"[seed {seed}] {pname}: spearman std={r['spearman_std']:.3f} nested={r['spearman_nested']:.3f} "
                  f"(d={r['delta_spearman_nested_minus_std']:+.3f}) | AURC std={r['aurc_std']:.3f} "
                  f"nested={r['aurc_nested']:.3f} (d={r['delta_aurc_nested_minus_std']:+.4f})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "nested_estimator_comparison.csv"), index=False)

    # family mean over seeds
    print("\n=== nested vs standard, family mean over seeds ===")
    agg = []
    for pname in PREDICTORS:
        sub = res[res["predictor"] == pname]
        agg.append(dict(
            predictor=pname,
            spearman_std=sub["spearman_std"].mean(), spearman_nested=sub["spearman_nested"].mean(),
            aurc_std=sub["aurc_std"].mean(), aurc_nested=sub["aurc_nested"].mean(),
            delta_aurc=sub["delta_aurc_nested_minus_std"].mean(),
            delta_spearman=sub["delta_spearman_nested_minus_std"].mean(),
        ))
    aggdf = pd.DataFrame(agg)
    print(aggdf.round(4).to_string(index=False))
    aggdf.to_csv(os.path.join(OUT, "nested_estimator_summary.csv"), index=False)
    print(f"\nwrote {OUT}/nested_estimator_comparison.csv and nested_estimator_summary.csv")


if __name__ == "__main__":
    main()
