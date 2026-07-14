"""PertEMA on the external K562 vs RPE1 transfer axis (P2, external cross-cell-line testbed).

Uses Replogle 2022 per-perturbation effects (src/data/replogle_effects.py). A mean predictor is trained on
one cell line and scored against the other over the 849 shared essential-gene perturbations, gene-disjoint.
PertEMA predicts the per-perturbation cross-cell-line transfer error from prediction-time features (control
co-expression embedding of the perturbed gene, source and destination cell-line indicators, and training-set
similarity). Compared to the similarity heuristic and no-selection and to random-feature and label-shuffle
controls. This is a third, fully external transfer axis independent of the Gladstone activation-state and
donor axes.

Run: pixi run python src/pertema/run_replogle_transfer.py
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

EFF = "results/features/replogle_effects.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
N_HVG = 1000
K = 5


def row_pearson(P, T):
    Pc = P - P.mean(1, keepdims=True); Tc = T - T.mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    z = np.load(EFF, allow_pickle=False)
    perts = np.array([str(p) for p in z["perts"]])
    pgenes = np.array([str(g) for g in z["pgenes"]])
    eff = {"k562": z["eff_k562"], "rpe1": z["eff_rpe1"]}
    emb_genes = np.array([str(g) for g in z["emb_genes"]])
    gene_pos = {g: i for i, g in enumerate(emb_genes)}
    embmat = z["embedding"]
    emb_dim = embmat.shape[1]

    def pert_emb(pg):
        return embmat[gene_pos[pg]] if pg in gene_pos else np.full(emb_dim, np.nan)
    Egene = np.array([pert_emb(g) for g in pgenes])

    rows = []
    for seed in SEEDS:
        uniqg = np.array(sorted(set(pgenes)))
        rng = np.random.default_rng(seed)
        fold_of = {g: i % K for i, g in enumerate(rng.permutation(uniqg))}
        recs = []
        for (src, dst) in [("k562", "rpe1"), ("rpe1", "k562")]:
            es, ed = eff[src], eff[dst]
            keep = np.where(np.isfinite(es[:, 0]) & np.isfinite(ed[:, 0]))[0]
            gf = np.array([fold_of[pgenes[i]] for i in keep])
            for k in range(K):
                tr = keep[gf != k]; te = keep[gf == k]
                if tr.size < 50 or te.size == 0:
                    continue
                hvg = np.argsort(-es[tr].var(0))[:N_HVG]
                mean_src = es[tr].mean(0)[hvg]
                P = np.tile(mean_src, (te.size, 1))
                err = 1.0 - row_pearson(P, ed[te][:, hvg])
                for j, i in enumerate(te):
                    recs.append(dict(pert=perts[i], pgene=pgenes[i], src=src, dst=dst,
                                     fold=int(k), err=float(err[j])))
        df = pd.DataFrame.from_records(recs)
        n = len(df)
        # features: perturbed-gene embedding + src one-hot + (fold-wise) similarity
        gi = df["pgene"].to_numpy()
        E = np.array([pert_emb(g) for g in gi])
        srcflag = (df["src"].to_numpy() == "k562").astype(float)[:, None]
        Xb = np.column_stack([srcflag, E])
        y = df["err"].to_numpy()
        fold = df["fold"].to_numpy()
        genes_u, inv = np.unique(gi, return_inverse=True)
        emb_u = np.array([pert_emb(g) for g in genes_u])
        good_u = ~np.isnan(emb_u).any(1)
        fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold

        oof = {k: np.full(n, np.nan) for k in ["est", "rand", "sim"]}
        rng2 = np.random.default_rng(seed)
        for k in np.unique(fold):
            trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            tr_u = np.where((fold_u != k) & good_u)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
            su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
            su[q] = nn.kneighbors(emb_u[q])[0].ravel()
            sim = su[inv]; oof["sim"][te] = sim[te]
            Xtr = np.column_stack([Xb[trn], sim[trn]]); Xte = np.column_stack([Xb[te], sim[te]])
            oof["est"][te] = gbt().fit(Xtr, y[trn]).predict(Xte)
            Xtr_r = np.column_stack([rng2.permutation(Xtr[:, j]) for j in range(Xtr.shape[1])])
            oof["rand"][te] = gbt().fit(Xtr_r, y[trn]).predict(Xte)
        # label-shuffle control: full-OOF calibration under permuted labels, averaged over R repeats.
        # The old single-fold single-shuffle version was unstable (+/-0.4): with small n a finite
        # permutation leaves a random mean difference in shuffled y between the two transfer directions,
        # the GBT latches on the clean srcflag feature, and direction (which truly ranks y) drives a large
        # random-signed Spearman. Full-OOF over all folds, averaged over R permutations, estimates the null
        # stably. The random-feature control (spearman_rand) is the load-bearing negative control here.
        R_LS = 25
        ls_vals = []
        for _ in range(R_LS):
            oof_ls = np.full(n, np.nan)
            for k in np.unique(fold):
                trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]
                m = gbt().fit(np.column_stack([Xb[trn], oof["sim"][trn]]), rng2.permutation(y[trn]))
                oof_ls[te] = m.predict(np.column_stack([Xb[te], oof["sim"][te]]))
            ls_vals.append(sp(oof_ls, y))
        ls_sp = float(np.nanmean(ls_vals))

        r = dict(seed=seed, n=n, spearman_est=sp(oof["est"], y), spearman_rand=sp(oof["rand"], y),
                 spearman_labelshuffle=ls_sp, aurc_est=best_aurc(y, -oof["est"]),
                 aurc_similarity=best_aurc(y, -oof["sim"]), aurc_random_feat=best_aurc(y, -oof["rand"]),
                 aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]), aurc_noselect=float(np.nanmean(y)),
                 mean_transfer_err=float(np.nanmean(y)))
        rows.append(r)
        print(f"seed {seed}: n={n} spearman_est={r['spearman_est']:.3f} AURC est={r['aurc_est']:.3f} "
              f"sim={r['aurc_similarity']:.3f} rand={r['aurc_random_feat']:.3f} oracle={r['aurc_oracle']:.3f} "
              f"noselect={r['aurc_noselect']:.3f} labelshuffle={ls_sp:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "replogle_transfer_per_seed.csv"), index=False)
    agg = {"axis": "external_k562_rpe1"}
    for c in [c for c in res.columns if c not in ("seed", "n")]:
        v = res[c].to_numpy()
        agg[c] = f"{np.nanmean(v):.3f} +/- {1.96*np.nanstd(v, ddof=1)/np.sqrt(len(v)):.3f}"
    pd.DataFrame([agg]).to_csv(os.path.join(OUT, "replogle_transfer_summary.csv"), index=False)
    print("\n=== EXTERNAL K562<->RPE1 transfer PertEMA (family mean +/- 95% CI) ===")
    print(pd.DataFrame([agg]).to_string(index=False))


if __name__ == "__main__":
    main()
