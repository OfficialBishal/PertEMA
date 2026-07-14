"""X1 (the falsification): the routing test on Adamson, the one dataset the phase-boundary law predicts is
routing-FEASIBLE (sim capture +0.26, rho 0.462, N_eff 2.05 of 3, f_noise 0.071). Pre-registered before this
run in results/PREREGISTRATION_adamson_routing.md.

Identical protocol to the three committed routing negatives (run_router_diverse.py, run_router_replogle.py):
the three Adamson predictors with a defined delta-Pearson error (mean_global, ridge_embed, knn_coexpr_k25, the
same roster for which rho and N_eff were measured), gene-disjoint 5-fold OOF, seeds 42/43/44, error =
1 - Pearson-delta on the top-1000-by-effect-variance gene set used for the placement. A per-predictor PertEMA
reliability estimator (xgboost, gene-disjoint OOF, prediction-time features only: the perturbed gene embedding
plus the predictor's predicted magnitude) predicts each predictor's error. The router picks the lowest
predicted error per perturbation. Leakage-safe: routing never sees held-out truth (invariant 1).

Kill criterion (pre-registered): routed beats best-fixed with paired-bootstrap fraction-better > 0.95 ->
CONFIRMED. Else we report, alongside, the realized oracle headroom (best-fixed minus oracle on the actual
errors) and the realized router rank quality r (Spearman of predicted vs true per-perturbation error), which
distinguish "feasible geometry but insufficient router" (Result 3, positive headroom, low r) from "collapsed
headroom" (law falsified). Reported plainly either way.

Run: pixi run python src/eval/run_router_adamson.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval", "pertema"):
    sys.path.insert(0, os.path.join(SRC, _p))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_estimator import gbt                                                   # noqa: E402

EFF = "results/features/adamson_effects.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_FOLDS = 5
N_BOOT = 2000


def pearson_delta(P, T, cols):
    Pc = P[:, cols] - P[:, cols].mean(1, keepdims=True)
    Tc = T[:, cols] - T[:, cols].mean(1, keepdims=True)
    num = (Pc * Tc).sum(1)
    den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gene_disjoint_folds(n, k, seed):
    order = np.random.RandomState(seed).permutation(n)
    fold = np.empty(n, int)
    fold[order] = np.arange(n) % k
    return fold


def per_pred_oof(pred, eff, obs, fold, top):
    """Gene-disjoint OOF per-row error (1 - Pearson-delta over top) and predicted magnitude for one predictor."""
    n = eff.shape[0]
    err = np.full(n, np.nan); mag = np.full(n, np.nan)
    for k in range(int(fold.max()) + 1):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        if te.size == 0:
            continue
        pred.fit(tr, eff, obs)
        P = pred.predict(te, obs)
        err[te] = 1.0 - pearson_delta(P, eff[te], top)
        mag[te] = np.abs(P[:, top]).mean(1)
    return err, mag


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(EFF, allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff"].astype(np.float32)
    emb_of = {g: v for g, v in zip(d["emb_genes"].astype(str), d["embedding"].astype(np.float32))}
    in_panel = np.array([g in emb_of and np.isfinite(eff_all[i, 0]) for i, g in enumerate(pgenes)])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]
    eff = eff_all[idx]
    n = len(idx)
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": "adamson"})
    gene_emb = {g: emb_of[g] for g in genes}
    emb_dim = d["embedding"].shape[1]
    Egene = np.array([emb_of[g] for g in genes])
    top = np.argsort(-eff.var(0))[:N_TOP]

    def build():
        return [("mean_global", MeanPredictor("global")),
                ("ridge_embed", RidgeEmbeddingPredictor(gene_emb, alpha=100.0)),
                ("knn_coexpr_k25", KNNSimilarityPredictor(gene_emb, k=25))]
    names = [nm for nm, _ in build()]

    rows = []
    r_rows = []
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        err = {}; mag = {}
        for nm, pred in build():
            e, m = per_pred_oof(pred, eff, obs, fold, top)
            err[nm] = e; mag[nm] = m
        valid = np.isfinite(np.column_stack([err[nm] for nm in names])).all(1)
        vi = np.where(valid)[0]
        ERR = np.column_stack([err[nm] for nm in names])

        # per-predictor PertEMA reliability estimator (gene-disjoint OOF), prediction-time features only.
        pred_err = {}
        for nm in names:
            X = np.column_stack([Egene, mag[nm]])
            y = err[nm]; pe = np.full(len(y), np.nan)
            for k in np.unique(fold):
                tr = np.where((fold != k) & np.isfinite(y))[0]; te = np.where(fold == k)[0]
                if tr.size < 20 or te.size == 0:
                    continue
                pe[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
            pred_err[nm] = pe
            # realized router rank quality for this predictor (Spearman of predicted vs true error)
            m = valid & np.isfinite(pe)
            if m.sum() > 5:
                rr = spearmanr(pe[m], y[m]).statistic
                r_rows.append(dict(seed=seed, predictor=nm, r=float(rr), n=int(m.sum())))
        PRED = np.column_stack([pred_err[nm] for nm in names])
        both_valid = valid & np.isfinite(PRED).all(1)
        bvi = np.where(both_valid)[0]
        chosen = np.argmin(PRED[bvi], axis=1)
        routed = ERR[bvi, chosen]
        mean_fixed = {nm: float(np.nanmean(err[nm][vi])) for nm in names}
        best_name = min(mean_fixed, key=mean_fixed.get)
        best_fixed = ERR[bvi, names.index(best_name)]
        oracle = ERR[bvi].min(1)
        rng = np.random.default_rng(seed)
        diffs = np.array([np.mean(best_fixed[rng.integers(0, len(bvi), len(bvi))]
                                  - routed[rng.integers(0, len(bvi), len(bvi))]) for _ in range(N_BOOT)])
        rows.append(dict(seed=seed, n=len(bvi), n_predictors=len(names), best_fixed=best_name,
                         err_best_fixed=mean_fixed[best_name], err_routed=float(routed.mean()),
                         err_oracle=float(oracle.mean()),
                         oracle_headroom=float(best_fixed.mean() - oracle.mean()),
                         routed_minus_bestfixed=float(routed.mean() - best_fixed.mean()),
                         boot_frac_routed_better=float(np.mean(diffs > 0))))
        print(f"seed {seed}: best_fixed {best_name} {mean_fixed[best_name]:.4f} | routed {routed.mean():.4f} "
              f"(delta {routed.mean()-best_fixed.mean():+.4f}) | oracle {oracle.mean():.4f} "
              f"(headroom {best_fixed.mean()-oracle.mean():+.4f}) | frac-better "
              f"{rows[-1]['boot_frac_routed_better']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "router_adamson.csv"), index=False)
    rdf = pd.DataFrame(r_rows)
    rdf.to_csv(os.path.join(OUT, "router_adamson_rquality.csv"), index=False)
    mr, mb = res["err_routed"].mean(), res["err_best_fixed"].mean()
    mo, mf = res["err_oracle"].mean(), res["boot_frac_routed_better"].mean()
    mhead = res["oracle_headroom"].mean()
    mean_r = float(rdf["r"].mean())
    confirmed = (mr < mb and mf > 0.95)
    print(f"\n=== X1 Adamson routing test (3 predictors, 3 seeds, n={res['n'].iloc[0]}) ===")
    print(f"best fixed {mb:.4f} | routed {mr:.4f} (delta {mr-mb:+.4f}, frac-better {mf:.3f}) | "
          f"oracle {mo:.4f} (realized headroom {mhead:+.4f})")
    print(f"realized router rank quality r (Spearman pred vs true error, mean over predictors/seeds): {mean_r:+.3f}")
    if confirmed:
        verdict = "CONFIRMED (routing beats best-fixed, law confirmed out of sample)"
    elif mhead > 0.01:
        verdict = ("FEASIBLE-GEOMETRY-INSUFFICIENT-ROUTER (positive oracle headroom, router r too low to "
                   "capture it on n=88; Result 3 regime, geometry law survives, deployable claim qualified)")
    else:
        verdict = "FALSIFIED (oracle headroom collapsed on the actual errors; law falsified, pivot the paper)"
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}/router_adamson.csv, router_adamson_rquality.csv")


if __name__ == "__main__":
    main()
