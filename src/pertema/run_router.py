"""D6 routing swing: does routing each perturbation to the predictor PertEMA deems most reliable beat the
best fixed single predictor (the flat baseline the reference papers crown)?

Leakage-safe: the router selects using only PertEMA's out-of-fold PREDICTED reliability (prediction-time
features), never the true error. Per perturbation g we pick p*(g) = argmax_p reliability_p(g) and take that
predictor's realized error. We compare the routed error against the best fixed predictor, each individual
predictor, and an ORACLE router (per-perturbation argmin true error, the ceiling). Significance by a paired
cluster bootstrap over perturbed genes.

The routing headroom is bounded by predictor diversity: if the predictors have correlated errors, even the
oracle router barely beats the best fixed predictor. This script reports the headroom so the result is
interpretable, and the KILL CRITERION is explicit: routed must beat best-fixed with significance to count as
a positive, else it is reported as a negative and the contribution rests on the benchmark resource (D8).

Run: pixi run python src/pertema/run_router.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, HERE)
from run_estimator import BASE, EMB, ERR, build_base_features, gbt   # noqa: E402

OUT = "results/pertema"
SEEDS = [42, 43, 44]
PREDICTORS = ["mean_condition", "mean_global", "knn_coexpr_k25", "ridge_embed"]


def oof_reliability(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim, seed):
    """Per-perturbation OOF PertEMA reliability (minus predicted error) for one predictor's error table."""
    Xb, gene_arr = build_base_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
    y = df["err_1mp_hvg"].to_numpy(); fold = df["fold"].to_numpy()
    genes_u, inv = np.unique(gene_arr, return_inverse=True)
    emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
    good_u = ~np.isnan(emb_u).any(1); fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold
    rel = np.full(len(df), np.nan)
    for k in np.unique(fold):
        trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]; tru = np.where((fold_u != k) & good_u)[0]
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tru])
        su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]; su[q] = nn.kneighbors(emb_u[q])[0].ravel()
        sim = su[inv]
        rel[te] = -gbt().fit(np.column_stack([Xb[trn], sim[trn]]), y[trn]).predict(np.column_stack([Xb[te], sim[te]]))
    return rel


def main():
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB); emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    rows = []
    for seed in SEEDS:
        err = pd.read_csv(os.path.join(ERR, f"errors_seed{seed}.csv"), dtype={"gene": str})
        # align all predictors on the common obs_index
        base = err[err["predictor"] == PREDICTORS[0]][["obs_index", "gene"]].reset_index(drop=True)
        E = np.full((len(base), len(PREDICTORS)), np.nan)     # true error per predictor
        R = np.full((len(base), len(PREDICTORS)), np.nan)     # PertEMA reliability per predictor
        for pi, p in enumerate(PREDICTORS):
            dfp = err[err["predictor"] == p].set_index("obs_index").reindex(base["obs_index"]).reset_index()
            E[:, pi] = dfp["err_1mp_hvg"].to_numpy()
            R[:, pi] = oof_reliability(dfp, baseline, dropout, donor_var, gene_col, emb_map, emb_dim, seed)
        ok = np.isfinite(E).all(1) & np.isfinite(R).all(1)
        E, R, genes = E[ok], R[ok], base["gene"].to_numpy()[ok]
        mean_err = E.mean(0)
        bf = int(np.argmin(mean_err))                          # best fixed predictor
        routed = E[np.arange(len(E)), np.argmax(R, 1)]         # PertEMA-routed error
        oracle = E.min(1)                                       # per-perturbation best (ceiling)
        # paired cluster bootstrap over genes for routed - best_fixed
        ug = np.unique(genes); rng = np.random.default_rng(seed); B = 2000
        d_boot = np.empty(B)
        gidx = {g: np.where(genes == g)[0] for g in ug}
        routed_minus_bf = routed - E[:, bf]
        for b in range(B):
            samp = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
            d_boot[b] = routed_minus_bf[samp].mean()
        rows.append(dict(seed=seed, n=len(E),
                         best_fixed=PREDICTORS[bf], err_best_fixed=float(mean_err[bf]),
                         err_routed=float(routed.mean()), err_oracle=float(oracle.mean()),
                         routed_minus_bestfixed=float(routed.mean() - mean_err[bf]),
                         oracle_headroom=float(mean_err[bf] - oracle.mean()),
                         boot_frac_routed_better=float((d_boot < 0).mean()),
                         **{f"err_{p}": float(mean_err[i]) for i, p in enumerate(PREDICTORS)}))
        print(f"seed {seed}: best_fixed={PREDICTORS[bf]} {mean_err[bf]:.4f} | routed {routed.mean():.4f} "
              f"(delta {routed.mean()-mean_err[bf]:+.4f}, boot frac better {rows[-1]['boot_frac_routed_better']:.3f}) "
              f"| oracle {oracle.mean():.4f} (headroom {mean_err[bf]-oracle.mean():.4f})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "router_single_context.csv"), index=False)
    print("\n=== D6 routing swing, single context (family mean over seeds) ===")
    print(f"best fixed error   {res['err_best_fixed'].mean():.4f}")
    print(f"routed error       {res['err_routed'].mean():.4f}  (delta {res['routed_minus_bestfixed'].mean():+.4f})")
    print(f"oracle error       {res['err_oracle'].mean():.4f}  (headroom {res['oracle_headroom'].mean():.4f})")
    print(f"bootstrap fraction routed beats best-fixed: {res['boot_frac_routed_better'].mean():.3f}")
    verdict = "POSITIVE" if (res['routed_minus_bestfixed'].mean() < 0 and res['boot_frac_routed_better'].mean() > 0.975) else "NEGATIVE"
    print(f"KILL CRITERION verdict on this roster: {verdict}")
    print(f"wrote {OUT}/router_single_context.csv")


if __name__ == "__main__":
    main()
