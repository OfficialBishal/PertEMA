"""Head-to-head: PRESCRIBE's intrinsic uncertainty vs PertEMA's post-hoc reliability on Norman (A4, P1).

PRESCRIBE was trained and evaluated on Norman 2019 in a container (its own reported dataset). Its test
produced, per perturbation, an error (pearson_delta, higher is worse), an intrinsic uncertainty (n), and a
predicted effect size (pred_de). Here we score three reliability signals on the SAME perturbations and the
SAME PRESCRIBE errors: PRESCRIBE's own uncertainty, the effect-magnitude heuristic, and PertEMA (a
gradient-boosted tree over prediction-time features only, gene-disjoint). This is a fair post-hoc wrapper
comparison: does a cheap model-agnostic estimator match PRESCRIBE's built-in uncertainty at flagging which
of PRESCRIBE's own predictions to trust.

Run: pixi run python src/pertema/prescribe_compare.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, HERE)
from metrics import aurc                                   # noqa: E402
from run_estimator import gbt                              # noqa: E402
from sklearn.neighbors import NearestNeighbors            # noqa: E402

CSV = "baselines/PRESCRIBE/prescribe_norman_perturb.csv"
EMB = "baselines/PRESCRIBE/norman_gene_embedding.npz"
OUT = "results/pertema"
K = 5


def pert_genes(cond):
    return [g.strip().upper() for g in str(cond).split("+") if g.strip() and g.strip() != "ctrl"]


def main():
    df = pd.read_csv(CSV, index_col=0)
    print("CSV columns:", list(df.columns), "| n_perturbations:", len(df))
    # locate columns robustly
    err_col = "pearson_delta"
    unc_col = "n" if "n" in df.columns else ("epi" if "epi" in df.columns else None)
    mag_col = "pred_de" if "pred_de" in df.columns else None
    conds = df.index.astype(str).to_numpy()
    y = df[err_col].to_numpy(float)                        # error, higher worse
    unc = df[unc_col].to_numpy(float) if unc_col else np.full(len(df), np.nan)
    mag = df[mag_col].to_numpy(float) if mag_col else np.full(len(df), np.nan)

    ez = np.load(EMB)
    emb = {str(g): v for g, v in zip(ez["genes"], ez["embedding"])}
    d = ez["embedding"].shape[1]

    def perturb_embedding(c):
        gs = [emb[g] for g in pert_genes(c) if g in emb]
        return np.mean(gs, 0) if gs else np.full(d, np.nan)
    E = np.array([perturb_embedding(c) for c in conds])
    primary = np.array([(pert_genes(c)[0] if pert_genes(c) else "NA") for c in conds])

    def spcc(score):
        m = np.isfinite(score) & np.isfinite(y)
        return float(spearmanr(score[m], y[m]).correlation) if m.sum() > 3 else np.nan

    def au(score_reliab):  # reliability score, higher = more trustworthy; AURC on error y
        m = np.isfinite(score_reliab) & np.isfinite(y)
        return aurc(y[m], score_reliab[m])

    # PertEMA: predict error from prediction-time features (magnitude + embedding + train-set similarity),
    # gene-disjoint over the primary perturbed gene.
    genes_u = np.array(sorted(set(primary) - {"NA"}))
    rng = np.random.default_rng(42)
    fold_of = {g: i % K for i, g in enumerate(rng.permutation(genes_u))}
    fold = np.array([fold_of.get(g, -1) for g in primary])
    Xb = np.column_stack([mag, E])
    pert_pred = np.full(len(df), np.nan)
    for k in range(K):
        tr = np.where((fold != k) & (fold >= 0))[0]
        te = np.where(fold == k)[0]
        if len(tr) < 20 or len(te) == 0:
            continue
        okg = ~np.isnan(E[tr]).any(1)
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(E[tr][okg])
        sim = np.full(len(df), np.nan)
        for grp in (tr, te):
            g_ok = ~np.isnan(E[grp]).any(1)
            sim[grp[g_ok]] = nn.kneighbors(E[grp][g_ok])[0].ravel()
        Xtr = np.column_stack([Xb[tr], sim[tr]])
        Xte = np.column_stack([Xb[te], sim[te]])
        m = np.isfinite(y[tr])
        pert_pred[te] = gbt().fit(Xtr[m], y[tr][m]).predict(Xte)

    res = {
        "n_perturbations": int(np.isfinite(y).sum()),
        "prescribe_uncertainty_SPCC": spcc(unc),
        "magnitude_heuristic_SPCC": spcc(mag),
        "pertema_SPCC": spcc(pert_pred),
        "prescribe_uncertainty_AURC": au(-unc),
        "magnitude_heuristic_AURC": au(-mag),
        "pertema_AURC": au(-pert_pred),
        "oracle_AURC": au(-y),
        "noselect_AURC": float(np.nanmean(y)),
    }
    pd.DataFrame([res]).to_csv(os.path.join(OUT, "prescribe_vs_pertema_norman.csv"), index=False)
    print("\n=== PRESCRIBE vs PertEMA on Norman (rank PRESCRIBE's own errors) ===")
    for k, v in res.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    print(f"\nwrote {OUT}/prescribe_vs_pertema_norman.csv")


if __name__ == "__main__":
    main()
