"""PertEMA on context transfer (the primary contribution).

Predicts the per-perturbation cross-context transfer error from prediction-time features, so PertEMA
flags which of a predictor's outputs stay trustworthy when the predictor trained in one activation state
is applied in another. Features (leakage-safe): predicted effect magnitude, control baseline expression,
dropout, and cross-donor variance of the perturbed gene in BOTH the source and destination conditions,
the control co-expression embedding, source and destination one-hots (the transfer pair), and training-set
similarity to the nearest training gene. Target: transfer_err on cross-context rows (src != dst). Outer
gene-disjoint CV, seeds 42/43/44. Compared to the magnitude and similarity heuristics and to random-feature
and label-shuffle negative controls.

Run: pixi run python src/pertema/run_transfer_estimator.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
from metrics import aurc                                   # noqa: E402
from run_estimator import best_aurc, gbt, sp               # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEEDS = [42, 43, 44]
PREDICTORS = ["mean_condition", "knn_coexpr_k25"]
CONDS = ["Rest", "Stim8hr", "Stim48hr"]


def build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim):
    n = df.shape[0]
    cidx = {c: i for i, c in enumerate(CONDS)}
    gi = df["gene"].map(gene_col).to_numpy()
    feats = {"pred_magnitude": df["pred_magnitude"].to_numpy()}
    for tag in ("src", "dst"):
        ci = df[tag].map(cidx).to_numpy()
        for name, arr in [("baseline", baseline), ("dropout", dropout), ("donor_var", donor_var)]:
            v = np.full(n, np.nan)
            ok = ~np.isnan(gi)
            v[ok] = arr[ci[ok], gi[ok].astype(int)]
            feats[f"{name}_{tag}"] = v
    for c in CONDS:
        feats[f"src_{c}"] = (df["src"] == c).to_numpy().astype(float)
        feats[f"dst_{c}"] = (df["dst"] == c).to_numpy().astype(float)
    E = np.full((n, emb_dim), np.nan, dtype=np.float32)
    for j, g in enumerate(df["gene"].to_numpy()):
        e = emb_map.get(g)
        if e is not None:
            E[j] = e
    base = np.column_stack([feats[k] for k in feats])
    return np.column_stack([base, E]), df["gene"].to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in SEEDS:
        tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{seed}.csv"), dtype={"gene": str})
        tr = tr[tr["transfer"]].reset_index(drop=True)     # cross-context only
        for pname in PREDICTORS:
            df = tr[tr["predictor"] == pname].reset_index(drop=True)
            Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var,
                                                   gene_col, emb_map, emb_dim)
            y = df["transfer_err"].to_numpy()
            fold = df["fold"].to_numpy()
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)
            fold_u = np.full(len(genes_u), -1)
            fold_u[inv] = fold

            oof = {k: np.full(len(df), np.nan) for k in ["est", "rand", "sim"]}
            rng = np.random.default_rng(seed)
            for k in np.unique(fold):
                trn = np.where(fold != k)[0]
                te = np.where(fold == k)[0]
                tr_u = np.where((fold_u != k) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
                sim_u = np.full(len(genes_u), np.nan)
                q = np.where(good_u)[0]
                sim_u[q] = nn.kneighbors(emb_u[q])[0].ravel()
                sim = sim_u[inv]
                oof["sim"][te] = sim[te]

                Xtr = np.column_stack([Xb[trn], sim[trn]])
                Xte = np.column_stack([Xb[te], sim[te]])
                est = gbt().fit(Xtr, y[trn])
                oof["est"][te] = est.predict(Xte)
                Xtr_r = np.column_stack([rng.permutation(Xtr[:, j]) for j in range(Xtr.shape[1])])
                oof["rand"][te] = gbt().fit(Xtr_r, y[trn]).predict(Xte)

            tr0, te0 = np.where(fold != 0)[0], np.where(fold == 0)[0]
            simcol = oof["sim"]
            est_ls = gbt().fit(np.column_stack([Xb[tr0], simcol[tr0]]), rng.permutation(y[tr0]))
            ls_spear = sp(est_ls.predict(np.column_stack([Xb[te0], simcol[te0]])), y[te0])

            r = dict(seed=seed, predictor=pname, n=len(df),
                     spearman_est=sp(oof["est"], y), spearman_rand=sp(oof["rand"], y),
                     spearman_labelshuffle=ls_spear,
                     aurc_est=best_aurc(y, -oof["est"]),
                     aurc_magnitude=best_aurc(y, df["pred_magnitude"].to_numpy()),
                     aurc_similarity=best_aurc(y, -oof["sim"]),
                     aurc_random_feat=best_aurc(y, -oof["rand"]),
                     aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]),
                     aurc_noselect=float(np.nanmean(y)))
            rows.append(r)
            print(f"seed {seed} {pname}: spearman_est={r['spearman_est']:.3f} "
                  f"AURC est={r['aurc_est']:.3f} mag={r['aurc_magnitude']:.3f} sim={r['aurc_similarity']:.3f} "
                  f"rand={r['aurc_random_feat']:.3f} oracle={r['aurc_oracle']:.3f} noselect={r['aurc_noselect']:.3f} "
                  f"| labelshuffle={ls_spear:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "transfer_estimator_per_seed.csv"), index=False)
    agg = []
    for pname in PREDICTORS:
        sub = res[res["predictor"] == pname]
        row = {"predictor": pname}
        for c in [c for c in res.columns if c not in ("seed", "predictor", "n")]:
            v = sub[c].to_numpy()
            row[c] = f"{np.nanmean(v):.3f} +/- {1.96*np.nanstd(v, ddof=1)/np.sqrt(len(v)):.3f}"
        agg.append(row)
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(os.path.join(OUT, "transfer_estimator_summary.csv"), index=False)
    print("\n=== TRANSFER PertEMA: family mean +/- 95% CI (seeds 42/43/44) ===")
    print(aggdf.to_string(index=False))


if __name__ == "__main__":
    main()
