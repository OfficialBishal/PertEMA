"""PertEMA on the donor transfer axis (P2, second transfer axis, independent of activation state).

Uses recomputed per-donor-group effects (src/data/donor_effects.py). A mean predictor is trained on one
donor group and scored against the other group's truth, per gene and condition, gene-disjoint. PertEMA
then predicts the per-perturbation donor-transfer error from prediction-time features. Compared to the
training-set-similarity heuristic and no-selection and to random-feature and label-shuffle controls. The
effect-magnitude heuristic is degenerate for a mean predictor (constant prediction) and is not reported.

Run: pixi run python src/pertema/run_donor_transfer.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, HERE)
from metrics import aurc                       # noqa: E402
from run_estimator import best_aurc, gbt, sp   # noqa: E402

DEFF = "results/features/donor_effects.npz"
SPLITS = "results/splits/gene_folds.csv"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
N_HVG = 2000


def row_pearson(P, T):
    Pc = P - P.mean(1, keepdims=True); Tc = T - T.mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    dz = np.load(DEFF, allow_pickle=False)
    genes = np.array([str(g) for g in dz["genes"]])
    effA, effB = dz["effect_A"], dz["effect_B"]     # (ng, 3, n_meas)
    gpos = {g: i for i, g in enumerate(genes)}
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline = bz["baseline"]
    ez = np.load(EMB); emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        recs = []
        for (sgrp, dgrp) in [(0, 1), (1, 0)]:           # A->B and B->A (cross-donor)
            eff_s = effA if sgrp == 0 else effB
            eff_d = effA if dgrp == 0 else effB
            for c in range(3):
                gidx = np.array([i for i, g in enumerate(genes)
                                 if g in fog and np.isfinite(eff_s[i, c, 0]) and np.isfinite(eff_d[i, c, 0])])
                if gidx.size < 100:
                    continue
                gf = np.array([fog[genes[i]] for i in gidx])
                for k in np.unique(gf):
                    tr = gidx[gf != k]; te = gidx[gf == k]
                    mean_s = eff_s[tr, c].mean(0)
                    hvg = np.argsort(-eff_s[tr, c].var(0))[:N_HVG]
                    P = np.tile(mean_s[hvg], (te.size, 1))
                    err = 1.0 - row_pearson(P, eff_d[te, c][:, hvg])
                    for j, i in enumerate(te):
                        recs.append(dict(gene=genes[i], cond=c, src=sgrp, dst=dgrp,
                                         fold=int(k), err=float(err[j])))
        df = pd.DataFrame.from_records(recs)

        # features (prediction-time): baseline expr of gene in condition, embedding, condition + src one-hot
        n = len(df)
        gi = df["gene"].map(gene_col).to_numpy()
        ci = df["cond"].to_numpy()
        bexpr = np.full(n, np.nan); ok = ~np.isnan(gi)
        bexpr[ok] = baseline[ci[ok], gi[ok].astype(int)]
        E = np.full((n, emb_dim), np.nan, dtype=np.float32)
        for j, g in enumerate(df["gene"].to_numpy()):
            e = emb_map.get(g)
            if e is not None:
                E[j] = e
        onehot = np.column_stack([(ci == 0), (ci == 1), (ci == 2),
                                  df["src"].to_numpy() == 0]).astype(float)
        Xb = np.column_stack([bexpr, onehot, E])
        y = df["err"].to_numpy()
        fold = df["fold"].to_numpy()
        gene_arr = df["gene"].to_numpy()
        genes_u, inv = np.unique(gene_arr, return_inverse=True)
        emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
        good_u = ~np.isnan(emb_u).any(1)
        fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold

        oof = {k: np.full(n, np.nan) for k in ["est", "rand", "sim"]}
        rng = np.random.default_rng(seed)
        for k in np.unique(fold):
            trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            tr_u = np.where((fold_u != k) & good_u)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
            su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
            su[q] = nn.kneighbors(emb_u[q])[0].ravel()
            sim = su[inv]; oof["sim"][te] = sim[te]
            Xtr = np.column_stack([Xb[trn], sim[trn]]); Xte = np.column_stack([Xb[te], sim[te]])
            oof["est"][te] = gbt().fit(Xtr, y[trn]).predict(Xte)
            Xtr_r = np.column_stack([rng.permutation(Xtr[:, j]) for j in range(Xtr.shape[1])])
            oof["rand"][te] = gbt().fit(Xtr_r, y[trn]).predict(Xte)
        tr0, te0 = np.where(fold != 0)[0], np.where(fold == 0)[0]
        ls = gbt().fit(np.column_stack([Xb[tr0], oof["sim"][tr0]]), rng.permutation(y[tr0]))
        ls_sp = sp(ls.predict(np.column_stack([Xb[te0], oof["sim"][te0]])), y[te0])

        r = dict(seed=seed, n=n, spearman_est=sp(oof["est"], y), spearman_rand=sp(oof["rand"], y),
                 spearman_labelshuffle=ls_sp, aurc_est=best_aurc(y, -oof["est"]),
                 aurc_similarity=best_aurc(y, -oof["sim"]), aurc_random_feat=best_aurc(y, -oof["rand"]),
                 aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]),
                 aurc_noselect=float(np.nanmean(y)), mean_transfer_err=float(np.nanmean(y)))
        rows.append(r)
        print(f"seed {seed}: n={n} spearman_est={r['spearman_est']:.3f} AURC est={r['aurc_est']:.3f} "
              f"sim={r['aurc_similarity']:.3f} rand={r['aurc_random_feat']:.3f} oracle={r['aurc_oracle']:.3f} "
              f"noselect={r['aurc_noselect']:.3f} labelshuffle={ls_sp:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "donor_transfer_per_seed.csv"), index=False)
    agg = {"axis": "donor"}
    for c in [c for c in res.columns if c not in ("seed", "n")]:
        v = res[c].to_numpy()
        agg[c] = f"{np.nanmean(v):.3f} +/- {1.96*np.nanstd(v, ddof=1)/np.sqrt(len(v)):.3f}"
    pd.DataFrame([agg]).to_csv(os.path.join(OUT, "donor_transfer_summary.csv"), index=False)
    print("\n=== DONOR transfer PertEMA (family mean +/- 95% CI) ===")
    print(pd.DataFrame([agg]).to_string(index=False))


if __name__ == "__main__":
    main()
