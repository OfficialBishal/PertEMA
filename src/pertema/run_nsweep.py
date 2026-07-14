"""D7 robustness: a training-set-size sweep for the PertEMA transfer estimator. How much labelled
training data does the reliability estimator need before its ranking quality degrades? Within each
gene-disjoint fold we subsample a fraction of the TRAINING GENES (keeping the estimator gene-disjoint),
refit, and measure out-of-fold reliability quality (reliability-Spearman and risk-coverage AUC) on the full
held-out set. Reuses the pre-computed transfer errors and features, so it needs no primary-data reload.

Reported across fractions, averaged over three seeds, so a reviewer sees the data-efficiency curve and the
smallest training fraction that still recovers most of the signal.

Run: pixi run python src/pertema/run_nsweep.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
sys.path.insert(0, HERE)
from metrics import aurc                                             # noqa: E402
from run_estimator import best_aurc, gbt, sp                         # noqa: E402
from run_transfer_estimator import build_transfer_features          # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEEDS = [42, 43, 44]
PREDICTOR = "mean_condition"
FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def main():
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB); emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in SEEDS:
        tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{seed}.csv"), dtype={"gene": str})
        df = tr[(tr["transfer"]) & (tr["predictor"] == PREDICTOR)].reset_index(drop=True)
        Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
        # training-set similarity, computed once per fold (gene-disjoint), reused across fractions
        y = df["transfer_err"].to_numpy(); fold = df["fold"].to_numpy()
        genes_u, inv = np.unique(gene_arr, return_inverse=True)
        emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
        good_u = ~np.isnan(emb_u).any(1); fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold
        sim = np.full(len(df), np.nan)
        for k in np.unique(fold):
            te = np.where(fold == k)[0]; tru = np.where((fold_u != k) & good_u)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tru])
            su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
            su[q] = nn.kneighbors(emb_u[q])[0].ravel(); sim[te] = su[inv][te]
        X = np.column_stack([Xb, sim])
        for frac in FRACTIONS:
            rng = np.random.default_rng(1000 * seed + int(frac * 100))
            oof = np.full(len(df), np.nan)
            for k in np.unique(fold):
                te = np.where(fold == k)[0]
                tr_genes = np.array(sorted(set(gene_arr[fold != k])))
                m = max(1, int(round(frac * len(tr_genes))))
                keep = set(rng.choice(tr_genes, m, replace=False))
                trn = np.where((fold != k) & np.array([g in keep for g in gene_arr]))[0]
                if trn.size < 20:
                    continue
                oof[te] = gbt().fit(X[trn], y[trn]).predict(X[te])
            rows.append(dict(seed=seed, fraction=frac,
                             n_train_genes=int(round(frac * len(set(gene_arr)))),
                             spearman=sp(oof, y), aurc=best_aurc(y, -oof),
                             aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]),
                             aurc_noselect=float(np.nanmean(y))))
            print(f"seed {seed} frac {frac:.2f}: spearman {rows[-1]['spearman']:.3f} "
                  f"AURC {rows[-1]['aurc']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "nsweep_transfer_per_seed.csv"), index=False)
    agg = res.groupby("fraction").agg(spearman=("spearman", "mean"), spearman_sd=("spearman", "std"),
                                      aurc=("aurc", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "nsweep_transfer.csv"), index=False)
    full = agg.loc[agg["fraction"] == 1.0, "spearman"].iloc[0]
    print("\n=== D7 training-set-size sweep, PertEMA transfer estimator (3 seeds) ===")
    print(agg.round(4).to_string(index=False))
    recovers = agg[agg["spearman"] >= 0.9 * full]["fraction"].min()
    print(f"full-data Spearman {full:.3f}; the smallest fraction recovering >=90 percent of it is {recovers}")
    print(f"wrote {OUT}/nsweep_transfer.csv")


if __name__ == "__main__":
    main()
