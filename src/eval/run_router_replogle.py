"""D6 routing swing, re-tested on the cleaner external axis (Replogle K562).

The routing swing failed on the Gladstone primary data because the transfer error is noise-dominated
(~0.9 ceiling), so the oracle headroom for choosing the best-of-N predictor per perturbation is only
~0.010 and is shared across predictors (results/pertema/ROUTING_FINDINGS.md). The Replogle axis carries a
much stronger reliability signal (transfer Spearman 0.567 vs Gladstone 0.146), so the honest question is
whether predictor routing adds over the best fixed predictor WHERE the signal exists. This script answers
it on the identical 213-perturbation in-panel K562 cohort, splits, and predictors used in the parity table
(src/eval/run_parity_replogle.py), so the routing result sits directly beside the parity result.

Construction (leakage-safe, mirrors run_replogle_transfer's nested OOF):
  - Gene-disjoint 5-fold, seeds 42/43/44 (a perturbation targets a distinct gene, so a seeded partition is
    gene-disjoint). Outer CV: each predictor is fit on the training folds and predicts the held-out fold,
    giving every perturbation one honest held-out error (1 - Pearson-delta on the top-1000 variance genes,
    the same evaluation gene set as the parity table) under each predictor.
  - For each predictor, a PertEMA GBT reliability estimator is trained gene-disjoint (same folds) to predict
    that predictor's per-perturbation error from prediction-time features only: the co-expression embedding
    of the perturbed gene, the training-set similarity, and the predictor's own predicted effect magnitude.
    No test perturbation's realized error ever enters the estimator (invariant 1).
  - Routing: each perturbation is routed to the predictor with the LOWEST predicted error; the routed error
    is that chosen predictor's realized held-out error. Compared to the best fixed predictor (the lowest
    mean realized error) and to the oracle (per-perturbation best-of-N realized error) for headroom.

Kill criterion (D6): routing must beat the best fixed predictor with significance (paired cluster bootstrap
over genes, fraction-better > 0.95). Reported plainly either way.

Run: pixi run python src/eval/run_router_replogle.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "predictors"))
sys.path.insert(0, os.path.join(SRC, "pertema"))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_estimator import gbt                                                          # noqa: E402

EFFECTS = "results/features/replogle_effects.npz"
OUT = "results/pertema"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_FOLDS = 5
CONTEXT = "K562"
N_BOOT = 2000


def pearson_delta(P, T, cols):
    Pc = P[:, cols] - P[:, cols].mean(1, keepdims=True)
    Tc = T[:, cols] - T[:, cols].mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gene_disjoint_folds(n_items, n_folds, seed):
    order = np.random.RandomState(seed).permutation(n_items)
    fold = np.empty(n_items, dtype=int)
    fold[order] = np.arange(n_items) % n_folds
    return fold


def main():
    d = np.load(EFFECTS, allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff_k562"].astype(np.float32)
    emb_genes = d["emb_genes"].astype(str)
    embedding = d["embedding"].astype(np.float32)
    emb_of = {g: v for g, v in zip(emb_genes, embedding)}

    in_panel = np.array([g in emb_of for g in pgenes])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]
    eff = eff_all[idx]
    n = len(idx)
    assert len(set(genes.tolist())) == n, "perturbed genes must be unique for gene-disjoint folds"
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": CONTEXT})
    gene_emb = {g: emb_of[g] for g in genes}
    Eperturb = np.array([emb_of[g] for g in genes])                 # (n, 50) prediction-time feature
    top = np.argsort(-eff.var(0))[:N_TOP]

    def build():
        return [MeanPredictor("global"),
                RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
                KNNSimilarityPredictor(gene_emb, k=25)]
    names = [p.name for p in build()]

    rows = []
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        # outer CV: honest held-out per-predictor error, predicted magnitude, and the predicted effect
        # vectors (for a prediction-disagreement feature) for every perturbation. All prediction-time only.
        err = {nm: np.full(n, np.nan) for nm in names}
        mag = {nm: np.full(n, np.nan) for nm in names}
        Pstack = {nm: np.full((n, N_TOP), np.nan, np.float32) for nm in names}
        for k in range(N_FOLDS):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            T = eff[te]
            for p in build():
                p.fit(tr, eff, obs)
                P = p.predict(te, obs)
                err[p.name][te] = 1.0 - pearson_delta(P, T, top)
                mag[p.name][te] = np.abs(P[:, top]).mean(1)
                Pstack[p.name][te] = P[:, top]
        # prediction disagreement: per-perturbation std across the 3 predictors' predicted effects, averaged
        # over the evaluation genes. Large where predictors disagree, which is exactly where routing matters.
        # Computed from predictions only (no held-out truth), so leakage-safe (invariant 1).
        Pcube = np.stack([Pstack[nm] for nm in names], axis=0)      # (3, n, N_TOP)
        disagree = Pcube.std(0).mean(1)                             # (n,)
        magmat = np.column_stack([mag[nm] for nm in names])        # (n, 3), shared magnitude context
        # per-predictor reliability estimator, gene-disjoint OOF on the same folds. Each estimator sees the
        # gene embedding, ALL three predicted magnitudes, and the disagreement, so it can tell predictors apart.
        pred_err = {nm: np.full(n, np.nan) for nm in names}
        for nm in names:
            X = np.column_stack([Eperturb, magmat, disagree])
            y = err[nm]
            for k in range(N_FOLDS):
                tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
                m = np.isfinite(y[tr])
                pred_err[nm][te] = gbt().fit(X[tr][m], y[tr][m]).predict(X[te])
        # route each perturbation to the lowest predicted error; realize that predictor's held-out error
        ERR = np.column_stack([err[nm] for nm in names])            # (n, 3) realized
        PRED = np.column_stack([pred_err[nm] for nm in names])      # (n, 3) predicted
        valid = np.isfinite(ERR).all(1) & np.isfinite(PRED).all(1)
        chosen = np.argmin(PRED, axis=1)
        routed = ERR[np.arange(n), chosen]
        mean_fixed = {nm: float(np.nanmean(err[nm][valid])) for nm in names}
        best_fixed_name = min(mean_fixed, key=mean_fixed.get)
        best_fixed = ERR[:, names.index(best_fixed_name)]
        oracle = ERR.min(1)
        # paired cluster bootstrap over genes (each gene is its own cluster here): fraction-better vs best-fixed
        vi = np.where(valid)[0]
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(N_BOOT):
            bs = rng.choice(vi, size=vi.size, replace=True)
            diffs.append(np.mean(best_fixed[bs] - routed[bs]))      # positive => routing better
        diffs = np.array(diffs)
        rows.append(dict(seed=seed, n=int(valid.sum()),
                         **{f"fixed_{nm}": mean_fixed[nm] for nm in names},
                         best_fixed=best_fixed_name, best_fixed_err=mean_fixed[best_fixed_name],
                         routed_err=float(np.nanmean(routed[valid])),
                         oracle_err=float(np.nanmean(oracle[valid])),
                         delta_vs_bestfixed=float(np.nanmean(best_fixed[valid] - routed[valid])),
                         frac_boot_better=float(np.mean(diffs > 0)),
                         route_agree_bestfixed=float(np.mean(chosen[valid] == names.index(best_fixed_name)))))
        print(f"seed {seed}: best-fixed {best_fixed_name} {mean_fixed[best_fixed_name]:.4f} | "
              f"routed {rows[-1]['routed_err']:.4f} | oracle {rows[-1]['oracle_err']:.4f} | "
              f"delta {rows[-1]['delta_vs_bestfixed']:+.4f} | frac-better {rows[-1]['frac_boot_better']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "router_replogle_per_seed.csv"), index=False)
    num = res.select_dtypes(include=[np.number])
    agg = {"axis": "replogle_k562_within"}
    for c in num.columns:
        v = num[c].to_numpy()
        agg[c] = f"{np.mean(v):.4f} +/- {1.96*np.std(v, ddof=1)/np.sqrt(len(v)):.4f}"
    agg["best_fixed"] = res["best_fixed"].mode().iloc[0]
    pd.DataFrame([agg]).to_csv(os.path.join(OUT, "router_replogle_summary.csv"), index=False)

    mean_routed = res["routed_err"].mean(); mean_best = res["best_fixed_err"].mean()
    mean_oracle = res["oracle_err"].mean(); mean_frac = res["frac_boot_better"].mean()
    print("\n=== D6 predictor routing on Replogle K562 (within-context, 3 seeds) ===")
    print(f"best fixed predictor: {agg['best_fixed']}  mean err {mean_best:.4f}")
    print(f"routed (PertEMA):     {mean_routed:.4f}  delta {mean_best - mean_routed:+.4f}  "
          f"frac-boot-better {mean_frac:.3f}")
    print(f"oracle best-of-N:     {mean_oracle:.4f}  (headroom {mean_best - mean_oracle:+.4f})")
    verdict = ("POSITIVE" if (mean_routed < mean_best and mean_frac > 0.95)
               else "NEGATIVE (routing does not beat the best fixed predictor with significance)")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}/router_replogle_summary.csv and router_replogle_per_seed.csv")


if __name__ == "__main__":
    main()
