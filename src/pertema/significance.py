"""Paired cluster-bootstrap significance for PertEMA vs the magnitude heuristic (addresses review B1).

The three-seed non-overlapping CI is a weak significance argument because it under-represents the dominant
variance component, which is the gene composition of the folds. Here we resample the primitive unit
(perturbed genes, as clusters) with replacement and recompute both PertEMA and the heuristic AURC on the
SAME resample, giving a paired delta-AURC distribution that includes gene-level variance. We report the
mean paired delta (heuristic AURC minus PertEMA AURC, positive means PertEMA better), a 95 percent
percentile CI, and a bootstrap p-value (fraction of resamples where PertEMA did not win).

Run: pixi run python src/pertema/significance.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, HERE)
from metrics import aurc                                   # noqa: E402
from run_transfer_estimator import build_transfer_features, gbt  # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEED = 42
B = 2000


def paired_boot(y, est_score, heur_score, gene_arr, n_boot, rng):
    """Cluster bootstrap over genes; delta = AURC(heuristic) - AURC(PertEMA) on each resample."""
    genes = np.unique(gene_arr)
    by_gene = {g: np.where(gene_arr == g)[0] for g in genes}
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(genes, size=genes.size, replace=True)
        idx = np.concatenate([by_gene[g] for g in samp])
        yy = y[idx]
        m = np.isfinite(yy)
        a_est = aurc(yy[m], est_score[idx][m])
        a_heur = aurc(yy[m], heur_score[idx][m])
        deltas.append(a_heur - a_est)
    return np.array(deltas)


def main():
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB); emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{SEED}.csv"), dtype={"gene": str})
    df = tr[tr["transfer"] & (tr["predictor"] == "mean_condition")].reset_index(drop=True)
    Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
    y = df["transfer_err"].to_numpy()
    fold = df["fold"].to_numpy()
    genes_u, inv = np.unique(gene_arr, return_inverse=True)
    emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
    good_u = ~np.isnan(emb_u).any(1)
    fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold

    est = np.full(len(df), np.nan); sim = np.full(len(df), np.nan)
    for k in np.unique(fold):
        trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        tr_u = np.where((fold_u != k) & good_u)[0]
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
        su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
        su[q] = nn.kneighbors(emb_u[q])[0].ravel()
        s = su[inv]; sim[te] = s[te]
        est[te] = gbt().fit(np.column_stack([Xb[trn], s[trn]]), y[trn]).predict(np.column_stack([Xb[te], s[te]]))

    rng = np.random.default_rng(SEED)
    mag = df["pred_magnitude"].to_numpy()
    # give each heuristic its best sign orientation on the full data, then bootstrap paired vs PertEMA
    def best_sign(score):
        m = np.isfinite(y) & np.isfinite(score)
        return score if aurc(y[m], score[m]) < aurc(y[m], -score[m]) else -score
    d_mag = paired_boot(y, -est, best_sign(mag), gene_arr, B, rng)
    d_sim = paired_boot(y, -est, best_sign(sim), gene_arr, B, rng)

    res = []
    for name, d in [("vs_magnitude", d_mag), ("vs_similarity", d_sim)]:
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(np.mean(d <= 0))   # fraction where PertEMA did not beat the heuristic
        res.append(dict(comparison=name, mean_delta_aurc=float(d.mean()),
                        ci_lo=float(lo), ci_hi=float(hi), boot_p_value=p, n_boot=B))
        print(f"PertEMA {name}: paired delta-AURC {d.mean():.4f} (95% CI {lo:.4f} to {hi:.4f}), "
              f"bootstrap p={p:.4f} (over gene resamples)")
    pd.DataFrame(res).to_csv(os.path.join(OUT, "significance_paired_bootstrap.csv"), index=False)
    print(f"\nwrote {OUT}/significance_paired_bootstrap.csv")


if __name__ == "__main__":
    main()
