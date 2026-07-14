"""E2 extension for the single-context effect panels (Norman, Adamson), to place them on the phase plane as
PREDICTIONS of the routing-feasibility boundary (these carry no empirical routing test). Mirrors
run_error_correlation_replogle.py with the single-context 'eff' key and each dataset's full measured panel,
same gene-disjoint OOF, same top-1000-by-effect-variance error metric, same roster (mean, ridge, kNN).

Usage: pixi run python src/eval/run_error_correlation_dataset.py norman|adamson
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "predictors"))
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402

CFG = {"norman": "results/features/norman_effects.npz",
       "adamson": "results/features/adamson_effects.npz"}
OUT = "results/error_correlation"
SEEDS = [42, 43, 44]
N_TOP = 1000
N_FOLDS = 5


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


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = sys.argv[1]
    d = np.load(CFG[ds], allow_pickle=True)
    pgenes = d["pgenes"].astype(str)
    eff_all = d["eff"].astype(np.float32)
    emb_of = {g: v for g, v in zip(d["emb_genes"].astype(str), d["embedding"].astype(np.float32))}
    in_panel = np.array([g in emb_of and np.isfinite(eff_all[i, 0]) for i, g in enumerate(pgenes)])
    idx = np.where(in_panel)[0]
    genes = pgenes[idx]
    eff = eff_all[idx]
    n = len(idx)
    obs = pd.DataFrame({"target_contrast": genes, "culture_condition": ds})
    gene_emb = {g: emb_of[g] for g in genes}
    top = np.argsort(-eff.var(0))[:N_TOP]

    def build():
        return [MeanPredictor("global"), RidgeEmbeddingPredictor(gene_emb, alpha=100.0),
                KNNSimilarityPredictor(gene_emb, k=25)]
    names = [p.name for p in build()]

    err_stack = {nm: [] for nm in names}
    for seed in SEEDS:
        fold = gene_disjoint_folds(n, N_FOLDS, seed)
        preds = build()
        err = {p.name: np.full(n, np.nan) for p in preds}
        for k in range(N_FOLDS):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            if te.size == 0:
                continue
            for p in preds:
                p.fit(tr, eff, obs)
                P = p.predict(te, obs)
                err[p.name][te] = 1.0 - pearson_delta(P, eff[te], top)
        for nm in names:
            err_stack[nm].append(err[nm])
    mean_err = {nm: np.nanmean(np.column_stack(err_stack[nm]), axis=1) for nm in names}
    ME = np.column_stack([mean_err[nm] for nm in names])
    valid = np.isfinite(ME).all(1)
    ERR = ME[valid]

    Cp = np.corrcoef(ERR.T)
    Rank = np.column_stack([rankdata(ERR[:, j]) for j in range(ERR.shape[1])])
    Cs = np.corrcoef(Rank.T)
    eig = np.clip(np.linalg.eigvalsh(Cp), 0, None)
    neff = float((eig.sum() ** 2) / (eig ** 2).sum())
    iu = np.triu_indices(len(names), 1)
    offp = float(np.nanmean(Cp[iu])); offs = float(np.nanmean(Cs[iu]))
    gmean = {nm: float(np.nanmean(mean_err[nm][valid])) for nm in names}
    best = min(gmean, key=gmean.get)
    oracle = ERR.min(1).mean()
    headroom = gmean[best] - oracle

    dfp = pd.DataFrame({"pert": genes[valid]})
    for j, nm in enumerate(names):
        dfp[f"{nm}_err"] = ERR[:, j]
    dfp["oracle_err"] = ERR.min(1)
    dfp["best_fixed_err"] = mean_err[best][valid]
    dfp.to_csv(os.path.join(OUT, f"per_perturbation_errors_{ds}.csv"), index=False)
    s_between = float(np.std([gmean[nm] for nm in names]))
    s_within = float(np.mean([np.nanstd(mean_err[nm][valid]) for nm in names]))
    pd.DataFrame([dict(dataset=ds, n=int(valid.sum()), p=len(names), mean_offdiag_pearson=offp,
                       mean_offdiag_spearman=offs, N_eff=neff, best_fixed=best, err_best_fixed=gmean[best],
                       err_oracle=float(oracle), oracle_headroom=float(headroom),
                       s_between=s_between, s_within=s_within)]).to_csv(
        os.path.join(OUT, f"error_correlation_{ds}.csv"), index=False)
    print(f"{ds}: n {int(valid.sum())} | rho {offp:.3f} Spearman {offs:.3f} | N_eff {neff:.2f} of {len(names)} "
          f"| best {best} {gmean[best]:.3f} | headroom {headroom:+.4f} | skill ratio {s_between/s_within:.3f}")


if __name__ == "__main__":
    main()
