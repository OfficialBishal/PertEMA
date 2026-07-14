"""Fully nested PertEMA transfer estimator: close the second-order caveat for the PRIMARY result.

run_nested_estimator.py closed the second-order label dependency for the single-context estimator. The
primary contribution is the transfer estimator, which has the same structure: a training gene's
transfer_err label was produced by a predictor trained on (fold != that gene's fold) and (cond == src),
whose pool includes the estimator's outer test-fold genes. This script removes that dependency for the
transfer estimator. For each outer estimator-test fold k, the training-gene transfer_err labels and the
predictor-derived magnitude feature are recomputed by an inner leave-one-fold-out over the folds != k, so
no predictor that produced a training label ever saw a gene in fold k. Test-fold labels are already
nested-clean (their predictors were trained on folds != k) and are reused from results/transfer.

Standard and nested transfer estimators are computed in one run over identical features and folds, so the
only difference is the training labels. We report the paired delta on AURC and Spearman. If it is within the
seed noise, the second-order dependency is negligible for the primary result too.

Run: pixi run python src/pertema/run_nested_transfer_estimator.py [seed ...]   (default 42 43 44)
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
from predictors import KNNSimilarityPredictor, MeanPredictor              # noqa: E402
from run_transfer import N_HVG, row_pearson                               # noqa: E402
from run_transfer_estimator import CONDS, build_transfer_features         # noqa: E402
from run_estimator import best_aurc, gbt, sp                              # noqa: E402
from metrics import aurc                                                  # noqa: E402
from sklearn.neighbors import NearestNeighbors                            # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
PREDICTORS = ["mean_condition", "knn_coexpr_k25"]


def make_predictors(gene_emb):
    return {"mean_condition": MeanPredictor("condition"),
            "knn_coexpr_k25": KNNSimilarityPredictor(gene_emb, k=25)}


def nested_transfer_labels(k, n_folds, row_fold, lfc, obs, gene_emb, gene, cond, row_of, folds, seedcol):
    """Inner LOO over folds != k. Returns pname -> {(gene, src, dst): (transfer_err, pmag)} for every
    training gene (fold != k) and cross-context pair, computed blind to fold k."""
    preds = make_predictors(gene_emb)
    out = {pn: {} for pn in preds}
    for j in range(n_folds):
        if j == k:
            continue
        inner_test_genes = folds.loc[folds[seedcol] == j, "gene"].tolist()
        for src in CONDS:
            for dst in CONDS:
                if src == dst:
                    continue
                train_A = np.where((row_fold != k) & (row_fold != j) & (cond == src))[0]
                if train_A.size == 0:
                    continue
                var = lfc[train_A].var(0)
                hvg = np.argsort(-var)[:N_HVG]
                pairs = [(g, row_of.get((g, src)), row_of.get((g, dst))) for g in inner_test_genes]
                pairs = [(g, a, b) for g, a, b in pairs if a is not None and b is not None]
                if not pairs:
                    continue
                a_rows = np.array([a for _, a, _ in pairs])
                b_rows = np.array([b for _, _, b in pairs])
                true_dst = lfc[b_rows][:, hvg]
                for pn, pred in preds.items():
                    pred.fit(train_A, lfc, obs)
                    P = pred.predict(a_rows, obs)[:, hvg]
                    err = 1.0 - row_pearson(P, true_dst)
                    pmag = np.abs(P).mean(1)
                    for t, (g, _, _) in enumerate(pairs):
                        out[pn][(g, src, dst)] = (float(err[t]), float(pmag[t]))
    return out


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [42, 43, 44]
    os.makedirs(OUT, exist_ok=True)

    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene = obs["target_contrast"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    row_of = {(gene[i], cond[i]): i for i in range(len(obs))}

    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_map = gene_emb
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in seeds:
        seedcol = f"fold_seed{seed}"
        fog = dict(zip(folds["gene"], folds[seedcol]))
        row_fold = np.array([fog.get(g, -1) for g in gene])
        n_folds = int(row_fold.max()) + 1

        print(f"[seed {seed}] computing nested transfer labels over {n_folds} outer folds ...", flush=True)
        nested = {k: nested_transfer_labels(k, n_folds, row_fold, lfc, obs, gene_emb, gene, cond,
                                            row_of, folds, seedcol) for k in range(n_folds)}

        tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{seed}.csv"), dtype={"gene": str})
        tr = tr[tr["transfer"]].reset_index(drop=True)
        for pname in PREDICTORS:
            df = tr[tr["predictor"] == pname].reset_index(drop=True)
            Xb_std, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
            y = df["transfer_err"].to_numpy()
            fold = df["fold"].to_numpy()
            keys = list(zip(df["gene"].to_numpy(), df["src"].to_numpy(), df["dst"].to_numpy()))
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)
            fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold

            oof_std = np.full(len(df), np.nan)
            oof_nested = np.full(len(df), np.nan)
            for k in np.unique(fold):
                trn = np.where(fold != k)[0]
                te = np.where(fold == k)[0]
                tr_u = np.where((fold_u != k) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
                sim_u = np.full(len(genes_u), np.nan); qq = np.where(good_u)[0]
                sim_u[qq] = nn.kneighbors(emb_u[qq])[0].ravel()
                sim = sim_u[inv]

                Xtr_std = np.column_stack([Xb_std[trn], sim[trn]])
                Xte = np.column_stack([Xb_std[te], sim[te]])
                m = gbt(); m.fit(Xtr_std, y[trn]); oof_std[te] = m.predict(Xte)

                dfn = df.copy()
                pm = df["pred_magnitude"].to_numpy().copy()
                yn = y.copy()
                nl = nested[k][pname]
                for idx in trn:
                    e_p = nl.get(keys[idx])
                    if e_p is not None:
                        yn[idx] = e_p[0]; pm[idx] = e_p[1]
                dfn["pred_magnitude"] = pm
                Xb_n, _ = build_transfer_features(dfn, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
                Xtr_n = np.column_stack([Xb_n[trn], sim[trn]])
                mn = gbt(); mn.fit(Xtr_n, yn[trn]); oof_nested[te] = mn.predict(Xte)

            r = dict(
                seed=seed, predictor=pname, n=len(df),
                spearman_std=sp(oof_std, y), spearman_nested=sp(oof_nested, y),
                aurc_std=best_aurc(y, -oof_std), aurc_nested=best_aurc(y, -oof_nested),
                aurc_noselect=float(np.nanmean(y)),
                aurc_oracle=float(aurc(y[np.isfinite(y)], -y[np.isfinite(y)])),
            )
            r["delta_aurc_nested_minus_std"] = r["aurc_nested"] - r["aurc_std"]
            r["delta_spearman_nested_minus_std"] = r["spearman_nested"] - r["spearman_std"]
            rows.append(r)
            print(f"[seed {seed}] {pname}: spearman std={r['spearman_std']:.3f} nested={r['spearman_nested']:.3f} "
                  f"(d={r['delta_spearman_nested_minus_std']:+.3f}) | AURC std={r['aurc_std']:.3f} "
                  f"nested={r['aurc_nested']:.3f} (d={r['delta_aurc_nested_minus_std']:+.4f})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "nested_transfer_estimator_comparison.csv"), index=False)
    print("\n=== nested vs standard transfer estimator, family mean over seeds ===")
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
    aggdf.to_csv(os.path.join(OUT, "nested_transfer_estimator_summary.csv"), index=False)
    print(f"\nwrote {OUT}/nested_transfer_estimator_comparison.csv and nested_transfer_estimator_summary.csv")


if __name__ == "__main__":
    main()
