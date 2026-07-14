"""R2 (transfer arm): compare PertEMA's post-hoc reliability against an INTRINSIC-uncertainty method on the
context-transfer task, where PRESCRIBE cannot be run.

PRESCRIBE is single-context by design (a NatPN uncertainty head on a GEARS backbone trained per context in
the Norman/GEARS data format); applying it to the Gladstone Rest->Stim transfer would require retraining its
whole pipeline on this data, which its released code does not support. The fair substitute is an intrinsic-
uncertainty baseline that IS applicable to the transfer predictor: a bootstrap deep ensemble. We resample the
training genes B times within each gene-disjoint fold, refit the kNN transfer prediction, and take the
per-perturbation standard deviation of the predicted effect across ensemble members as the intrinsic
uncertainty (higher = less reliable). We then compare, on the SAME held-out transfer errors, how well three
signals rank realized error: PertEMA post-hoc reliability, the intrinsic ensemble uncertainty, and the
effect-magnitude heuristic. This tests whether a cheap post-hoc estimator matches an intrinsic-uncertainty
method under context transfer.

Run: pixi run python src/pertema/run_intrinsic_uncertainty.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("eval", "predictors", "data", "pertema"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                  # noqa: E402
from metrics import aurc, reliability_spearman                     # noqa: E402
from run_estimator import best_aurc, gbt                           # noqa: E402
from run_transfer import N_HVG, row_pearson                        # noqa: E402
from run_transfer_estimator import build_transfer_features         # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
BASE = "results/features/control_baseline.npz"
OUT = "results/pertema"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
SEEDS = [42, 43, 44]
B_ENS = 10          # bootstrap ensemble members
K = 25


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return reliability_spearman(a[m], b[m]) if m.sum() > 3 else np.nan


def main():
    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene = obs["target_contrast"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    row_of = {(gene[i], cond[i]): i for i in range(len(obs))}
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB); gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]

    def knn_predict(train_rows, a_rows, hvg, rng=None):
        """kNN transfer prediction over the hvg gene set, optionally bootstrapping the training genes."""
        g_tr = gene[train_rows]
        if rng is not None:
            bi = rng.integers(0, len(train_rows), len(train_rows)); train_rows = train_rows[bi]; g_tr = gene[train_rows]
        keep = [i for i, r in enumerate(train_rows) if g_tr[i] in gene_emb]
        if len(keep) < K + 1:
            return np.tile(lfc[train_rows][:, hvg].mean(0), (len(a_rows), 1))
        tr_emb = np.stack([gene_emb[g_tr[i]] for i in keep]); tr_rows_k = train_rows[keep]
        nn = NearestNeighbors(n_neighbors=K, algorithm="brute", n_jobs=8).fit(tr_emb)
        qmask = [g in gene_emb for g in gene[a_rows]]
        P = np.tile(lfc[train_rows][:, hvg].mean(0), (len(a_rows), 1))
        qi = np.where(qmask)[0]
        if len(qi):
            _, idx = nn.kneighbors(np.stack([gene_emb[g] for g in gene[a_rows[qi]]]))
            for t, j in enumerate(qi):
                P[j] = lfc[tr_rows_k[idx[t]]][:, hvg].mean(0)
        return P

    rows = []
    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fog.get(g, -1) for g in gene]); n_folds = int(row_fold.max()) + 1
        recs = []
        for src in CONDS:
            for dst in CONDS:
                if src == dst:
                    continue
                for k in range(n_folds):
                    tr = np.where((row_fold != k) & (cond == src))[0]
                    if tr.size < 50:
                        continue
                    hvg = np.argsort(-lfc[tr].var(0))[:N_HVG]
                    tgenes = folds.loc[folds[f"fold_seed{seed}"] == k, "gene"].tolist()
                    pairs = [(g, row_of.get((g, src)), row_of.get((g, dst))) for g in tgenes]
                    pairs = [(g, a, b) for g, a, b in pairs if a is not None and b is not None]
                    if not pairs:
                        continue
                    a_rows = np.array([a for _, a, _ in pairs]); b_rows = np.array([b for _, _, b in pairs])
                    true_dst = lfc[b_rows][:, hvg]
                    P = knn_predict(tr, a_rows, hvg)
                    err = 1.0 - row_pearson(P, true_dst); pmag = np.abs(P).mean(1)
                    # intrinsic uncertainty: std of predicted effect across bootstrap ensemble members
                    rng = np.random.default_rng(1000 * seed + k)
                    ens = np.stack([knn_predict(tr, a_rows, hvg, rng=rng) for _ in range(B_ENS)])
                    unc = ens.std(0).mean(1)     # mean per-gene std across the hvg set
                    for j, (g, _, _) in enumerate(pairs):
                        recs.append(dict(gene=g, src=src, dst=dst, fold=k, err=float(err[j]),
                                         pmag=float(pmag[j]), intrinsic_unc=float(unc[j])))
        df = pd.DataFrame.from_records(recs)
        # PertEMA post-hoc reliability on the SAME rows (pooled transfer estimator OOF)
        Xb, gene_arr = build_transfer_features(df.rename(columns={"pmag": "pred_magnitude"}),
                                               baseline, dropout, donor_var, gene_col, gene_emb, emb_dim)
        y = df["err"].to_numpy(); fold = df["fold"].to_numpy()
        gu, inv = np.unique(gene_arr, return_inverse=True)
        emu = np.array([gene_emb.get(g, np.full(emb_dim, np.nan)) for g in gu]); good = ~np.isnan(emu).any(1)
        fu = np.full(len(gu), -1); fu[inv] = fold
        rel = np.full(len(df), np.nan)
        for k in np.unique(fold):
            trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]; tru = np.where((fu != k) & good)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emu[tru])
            su = np.full(len(gu), np.nan); q = np.where(good)[0]; su[q] = nn.kneighbors(emu[q])[0].ravel(); sim = su[inv]
            rel[te] = -gbt().fit(np.column_stack([Xb[trn], sim[trn]]), y[trn]).predict(np.column_stack([Xb[te], sim[te]]))
        r = dict(seed=seed, n=len(df),
                 spearman_pertema=sp(rel, -y), spearman_intrinsic=sp(-df["intrinsic_unc"], -y),
                 spearman_magnitude=sp(df["pmag"], -y),
                 aurc_pertema=best_aurc(y, rel), aurc_intrinsic=best_aurc(y, -df["intrinsic_unc"].to_numpy()),
                 aurc_magnitude=best_aurc(y, df["pmag"].to_numpy()),
                 aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]), aurc_noselect=float(np.nanmean(y)))
        rows.append(r)
        print(f"seed {seed}: n={r['n']} | Spearman pertema={r['spearman_pertema']:.3f} "
              f"intrinsic={r['spearman_intrinsic']:.3f} mag={r['spearman_magnitude']:.3f} | "
              f"AURC pertema={r['aurc_pertema']:.3f} intrinsic={r['aurc_intrinsic']:.3f} "
              f"mag={r['aurc_magnitude']:.3f} oracle={r['aurc_oracle']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    agg = {"comparison": "pertema_vs_intrinsic_uncertainty_transfer"}
    for c in [c for c in res.columns if c not in ("seed", "n")]:
        v = res[c].to_numpy(); agg[c] = f"{np.nanmean(v):.3f} +/- {1.96*np.nanstd(v, ddof=1)/np.sqrt(len(v)):.3f}"
    pd.DataFrame([agg]).to_csv(os.path.join(OUT, "intrinsic_uncertainty_transfer.csv"), index=False)
    res.to_csv(os.path.join(OUT, "intrinsic_uncertainty_transfer_per_seed.csv"), index=False)
    print("\n=== PertEMA vs intrinsic ensemble uncertainty on transfer (family mean +/- 95% CI) ===")
    print(pd.DataFrame([agg]).to_string(index=False))


if __name__ == "__main__":
    main()
