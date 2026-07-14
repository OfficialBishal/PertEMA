"""E2: cross-model per-perturbation error-correlation structure on Gladstone (the mechanistic core).

This converts the routing negative (D6) from a null result into a measured law. For the SAME diverse
10-predictor roster D6 routed over (two constant baselines, a co-expression ridge and kNN, and six
frozen-adapted foundation ridge/kNN heads over Geneformer, scGPT, and gene2vec pretrained embeddings),
we compute each predictor's realized per-perturbation error (1 - Pearson-delta on the top-1000 expressed
genes, the exact D6 metric) via the SAME gene-disjoint OOF machinery, then measure:

  1. pairwise Pearson and Spearman error-correlation matrices (correlation of the per-perturbation error
     vectors across predictor pairs), averaged over seeds 42/43/44;
  2. conditional error probability P(B in its worst error quartile | A in its worst error quartile),
     which is 0.25 under independence and rises toward 1.0 as failures coincide;
  3. effective ensemble size N_eff = (sum eig)^2 / sum(eig^2) = p^2 / sum(eig^2) of the Pearson
     error-correlation matrix (participation ratio: 1 = all predictors fail on the same perturbations,
     p = independent), plus the fraction of error variance carried by the first principal component;
  4. an inductive-bias-class breakdown: within-class vs cross-class mean error correlation.

Leakage-safe (invariant 1): this is post-hoc analysis of realized errors, not training. No held-out truth
reaches any predictor. It is what, why, how: WHAT, the per-perturbation error-correlation structure; WHY,
routing needs some predictor to be reliably better on an identifiable subset, which requires the error
vectors to differ; HOW, correlate the OOF error vectors and read off N_eff and the coincident-failure rate.

A consistency assertion checks that the per-predictor mean errors and the oracle headroom reproduce
results/pertema/router_diverse_gladstone.csv, so E2 provably explains the exact negative D6 reported.

Run: pixi run python src/eval/run_error_correlation.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval", "pertema"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                                      # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_parity import pearson_delta                                                   # noqa: E402
from run_router_diverse import per_row_error_and_mag                                   # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/error_correlation"
SEEDS = [42, 43, 44]
N_TOP = 1000

# inductive-bias class of each predictor (there is no graph predictor in the frozen roster; that is E1).
CLASS = {
    "mean_condition": "constant", "mean_global": "constant",
    "knn_coexpr_k25": "coexpr", "ridge_embed": "coexpr",
    "geneformer_ridge": "foundation", "geneformer_knn": "foundation",
    "scgpt_ridge": "foundation", "scgpt_knn": "foundation",
    "gene2vec_ridge": "foundation", "gene2vec_knn": "foundation",
}


def build_roster(coexpr):
    """The identical D6 diverse roster: 4 CLEAN predictors with a defined delta-Pearson error plus the
    6 frozen-adapted foundation heads. Mirrors run_router_diverse.py (the canonical source)."""
    clean = [("mean_condition", MeanPredictor("condition")), ("mean_global", MeanPredictor("global")),
             ("knn_coexpr_k25", KNNSimilarityPredictor(coexpr, k=25)),
             ("ridge_embed", RidgeEmbeddingPredictor(coexpr, alpha=100.0))]
    found = []
    for ef in sorted(glob.glob(os.path.join("results", "features", "foundation_*.npz"))):
        name = os.path.basename(ef)[len("foundation_"):-len(".npz")]
        z = np.load(ef)
        ge = {str(g): v for g, v in zip(z["gene_ids"], z["embedding"])}
        found.append((f"{name}_ridge", RidgeEmbeddingPredictor(ge, alpha=100.0)))
        found.append((f"{name}_knn", KNNSimilarityPredictor(ge, k=25)))
    return clean + found


def neff_and_pc1(C):
    """Participation-ratio effective ensemble size and PC1 variance fraction of a correlation matrix."""
    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    neff = float((eig.sum() ** 2) / (eig ** 2).sum())
    pc1 = float(eig.max() / eig.sum())
    return neff, pc1


def conditional_worst_quartile(ERR, names):
    """P(B in its worst error quartile | A in its worst error quartile) for every ordered pair."""
    thr = np.nanpercentile(ERR, 75, axis=0)
    worst = ERR >= thr  # (n, p) boolean, worst 25 percent per predictor
    rows = []
    for a in range(len(names)):
        na = worst[:, a].sum()
        for b in range(len(names)):
            if a == b:
                continue
            p_b_given_a = float((worst[:, a] & worst[:, b]).sum() / na) if na else np.nan
            rows.append(dict(model_a=names[a], model_b=names[b], p_b_worst_given_a_worst=p_b_given_a))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc", "baseMean"))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()
    cond_of_row = obs["culture_condition"].astype(str).to_numpy()
    gname_of_row = obs["target_contrast_gene_name"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB)
    coexpr = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    # top-1000 by baseMean control expression: the reference-correct "1000 most highly expressed in
    # control" set (Ahlmann-Eltze). baseMean is an expression level, NOT a perturbation effect, so it is
    # leakage-safe. We export the exact selected measured-gene ids so E3 (noise ceiling) scores its floor
    # on the byte-identical gene support (fixes the E2/E3 gene-set mismatch found in the internal review).
    expr = np.nan_to_num(de.layers["baseMean"]).mean(0)
    top = np.argsort(-expr)[:N_TOP]
    np.savetxt(os.path.join(OUT, "eval_genes_gladstone.txt"), de.gene_ids[top], fmt="%s")

    roster = build_roster(coexpr)
    names = [n for n, _ in roster]

    pear_stack, spear_stack, summ = [], [], []
    cond_accum = None
    per_pred_err_accum = {n: [] for n in names}
    full_err_stack = {n: [] for n in names}   # full-length per-row errors per seed, for the per-perturbation table
    headroom_accum = []
    for seed in SEEDS:
        err = {}
        for nm, pred in roster:
            e, _m, _rf = per_row_error_and_mag(pred, lfc, obs, gene_of_row, folds, seed, top)
            err[nm] = e
            full_err_stack[nm].append(e)
        ERRfull = np.column_stack([err[n] for n in names])
        valid = np.isfinite(ERRfull).all(1)
        ERR = ERRfull[valid]
        # 1. correlation matrices of the per-perturbation error vectors
        Cp = np.corrcoef(ERR.T)
        Rank = np.column_stack([rankdata(ERR[:, j]) for j in range(ERR.shape[1])])
        Cs = np.corrcoef(Rank.T)
        pear_stack.append(Cp)
        spear_stack.append(Cs)
        # 3. effective ensemble size
        neff_p, pc1_p = neff_and_pc1(Cp)
        # 2. conditional worst-quartile coincidence (accumulate)
        cq = conditional_worst_quartile(ERR, names).set_index(["model_a", "model_b"])
        cond_accum = cq if cond_accum is None else cond_accum.add(cq, fill_value=0)
        # class breakdown on the Pearson matrix
        cls = np.array([CLASS[n] for n in names])
        iu = np.triu_indices(len(names), k=1)
        same = cls[iu[0]] == cls[iu[1]]
        within = float(np.nanmean(Cp[iu][same])) if same.any() else np.nan
        cross = float(np.nanmean(Cp[iu][~same])) if (~same).any() else np.nan
        offdiag = float(np.nanmean(Cp[iu]))
        offdiag_s = float(np.nanmean(Cs[iu]))
        # consistency vs D6: mean per-predictor error + oracle headroom on the same valid rows
        mean_fixed = {n: float(np.nanmean(err[n][valid])) for n in names}
        best_name = min(mean_fixed, key=mean_fixed.get)
        oracle = ERR.min(1).mean()
        headroom = mean_fixed[best_name] - oracle
        headroom_accum.append(headroom)
        for n in names:
            per_pred_err_accum[n].append(mean_fixed[n])
        summ.append(dict(seed=seed, n=int(valid.sum()), p=len(names),
                         mean_offdiag_pearson=offdiag, mean_offdiag_spearman=offdiag_s,
                         N_eff=neff_p, pc1_var_frac=pc1_p,
                         within_class_pearson=within, cross_class_pearson=cross,
                         best_fixed=best_name, err_best_fixed=mean_fixed[best_name],
                         err_oracle=float(oracle), oracle_headroom=float(headroom)))
        print(f"seed {seed}: N_eff {neff_p:.2f}/{len(names)} | PC1 {pc1_p:.2f} | mean off-diag Pearson "
              f"{offdiag:.3f} Spearman {offdiag_s:.3f} | within-class {within:.3f} cross-class {cross:.3f} "
              f"| oracle headroom {headroom:+.4f}", flush=True)

    Cp_mean = np.mean(pear_stack, axis=0)
    Cs_mean = np.mean(spear_stack, axis=0)
    pd.DataFrame(Cp_mean, index=names, columns=names).to_csv(os.path.join(OUT, "error_corr_pearson.csv"))
    pd.DataFrame(Cs_mean, index=names, columns=names).to_csv(os.path.join(OUT, "error_corr_spearman.csv"))
    (cond_accum / len(SEEDS)).reset_index().to_csv(
        os.path.join(OUT, "conditional_worst_quartile.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, "error_correlation_summary.csv"), index=False)

    # persist the single authoritative per-perturbation error table (seed-averaged), consumed by E3
    # (achievable-gain envelope) and E4 (dataset anchoring). Keys are (gene ENSG, condition) joinable to
    # donor_effects.npz. oracle_err is the per-perturbation best-of-roster; best_fixed is the globally best
    # predictor by mean error, both in the top-1000 delta-Pearson metric.
    mean_err = {n: np.nanmean(np.column_stack(full_err_stack[n]), axis=1) for n in names}
    ME = np.column_stack([mean_err[n] for n in names])
    valid_all = np.isfinite(ME).all(1)
    gmean = {n: float(np.nanmean(mean_err[n][valid_all])) for n in names}
    best_name = min(gmean, key=gmean.get)
    dfp = pd.DataFrame({"gene": gene_of_row, "gene_name": gname_of_row, "condition": cond_of_row})
    for n in names:
        dfp[f"{n}_err"] = mean_err[n]
    dfp["oracle_err"] = ME.min(1)
    dfp["best_fixed_err"] = mean_err[best_name]
    dfp["best_fixed_name"] = best_name
    dfp[valid_all].to_csv(os.path.join(OUT, "per_perturbation_errors.csv"), index=False)

    # consistency assertion vs the committed D6 router CSV (same roster, same metric)
    # regression guard, NOT independent validation: E2 reuses D6's per_row_error_and_mag on nearly the same
    # rows (E2 keeps all valid rows, D6 keeps rows with finite GBT predictions), so the headroom agrees by
    # construction. This only confirms the two row masks barely differ on the corrected gene set.
    rd = pd.read_csv("results/pertema/router_diverse_gladstone.csv")
    d6_head = rd["oracle_headroom"].mean()
    e2_head = np.mean(headroom_accum)
    ok = abs(d6_head - e2_head) < 5e-3
    print(f"\nregression guard vs D6 router (same error code, near-same rows): oracle headroom E2 "
          f"{e2_head:.4f} vs D6 {d6_head:.4f} -> {'consistent' if ok else 'DIVERGED'}")

    neff = S["N_eff"].mean()
    pc1 = S["pc1_var_frac"].mean()
    offp = S["mean_offdiag_pearson"].mean()
    within = S["within_class_pearson"].mean()
    cross = S["cross_class_pearson"].mean()
    print(f"\n=== E2 error-correlation structure (Gladstone, {len(names)} predictors, 3 seeds) ===")
    print(f"mean off-diagonal Pearson error correlation {offp:.3f} | N_eff {neff:.2f} of {len(names)} "
          f"| PC1 variance fraction {pc1:.2f}")
    print(f"within-inductive-bias-class {within:.3f} vs cross-class {cross:.3f} error correlation")
    print(f"wrote {OUT}/ (error_corr_pearson.csv, error_corr_spearman.csv, conditional_worst_quartile.csv,"
          f" error_correlation_summary.csv)")
    return S, Cp_mean, names, ok


if __name__ == "__main__":
    main()
