"""E5: cross-dataset transfer of the reliability estimator (decides the venue, P5 / P14).

Tests whether a gene's perturbation-effect predictability is a gene-intrinsic property that transfers across
screens. Dataset-agnostic feature: the perturbed gene's foundation-model embedding (a shared, gene-identity
representation), reduced by PCA fit on the TRAIN set only (no leakage). We test THREE shared embeddings
(Geneformer, scGPT, gene2vec) so the negative does not rest on a single feature. Target: the kNN predictor's
per-perturbation error (present on all four datasets). Gladstone is aggregated per-gene (mean over its three
conditions). We train on one dataset and apply it to another without refitting, and report the reliability
Spearman for every train-test pair, with a bootstrap 95 percent CI on each off-diagonal (cross-dataset) cell.

Decision rule stated in advance (P14): portability is SUPPORTED only if the cross-dataset reliability
Spearman reaches about 0.4 on multiple independent screens (point estimate, with a CI that excludes the
null). Otherwise portability is UNSUPPORTED and PertEMA is a within-screen method plus the portable
error-correlation law, targeting Genome Biology or Cell Systems. This is a conservative rule: absence of
support (not proof of zero transfer) is what the venue decision acts on.

CAVEATS reported plainly: the single-context K562 screens are SMALL (n 59 to 123, only the perturbed genes
that map to the CD4 measured-gene namespace), so cross-dataset cells are underpowered and their CIs are wide.
Gladstone uses the top-1000 expressed-gene error metric while the K562 screens use top-1000 variance, so the
metric is not identical across datasets (Spearman is rank-based, which mitigates but does not remove this);
the metric-consistent same-cell-line K562 3x3 block is reported separately.

Run: pixi run python src/eval/run_estimator_transfer.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src/data")
from load_de import load_de_stats  # noqa: E402

TARGET = "knn_coexpr_k25_err"
N_PCA = 20
SEEDS = [42, 43, 44]
B = 1000
OUT = "results/pertema"
DATASETS = ["gladstone", "replogle", "norman", "adamson"]
EMBEDDINGS = ["geneformer", "scgpt", "gene2vec"]


def gbt():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, max_iter=200,
                                         min_samples_leaf=5, random_state=0)


def load_dataset(ds, sym2ens, emb):
    if ds == "gladstone":
        p = pd.read_csv("results/error_correlation/per_perturbation_errors.csv")
        g = p.groupby("gene")[TARGET].mean().reset_index()
        ens = g["gene"].astype(str).to_numpy(); y = g[TARGET].to_numpy()
    else:
        p = pd.read_csv(f"results/error_correlation/per_perturbation_errors_{ds}.csv")
        ens = np.array([sym2ens.get(s, "") for s in p["pert"].astype(str).str.upper()])
        y = p[TARGET].to_numpy()
    X, yy = [], []
    for e, v in zip(ens, y):
        if e in emb and np.isfinite(v):
            X.append(emb[e]); yy.append(v)
    return np.asarray(X, dtype=np.float32), np.asarray(yy, dtype=np.float64)


def transfer_cell(XA, yA, XB, yB):
    pca = PCA(min(N_PCA, XA.shape[1]), random_state=0).fit(XA)
    m = gbt().fit(pca.transform(XA), yA)
    pB = m.predict(pca.transform(XB))
    val = float(spearmanr(pB, yB).correlation)
    rng = np.random.default_rng(0); bs = np.empty(B)
    for b in range(B):
        bi = rng.integers(0, len(yB), len(yB))
        bs[b] = spearmanr(pB[bi], yB[bi]).correlation
    return val, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def diag_cell(XA, yA):
    sp = []
    for s in SEEDS:
        idx = np.random.default_rng(s).permutation(len(yA)); h = len(idx) // 2
        tr, te = idx[:h], idx[h:]
        pca = PCA(min(N_PCA, XA.shape[1]), random_state=0).fit(XA[tr])
        m = gbt().fit(pca.transform(XA[tr]), yA[tr])
        c = spearmanr(m.predict(pca.transform(XA[te])), yA[te]).correlation
        if np.isfinite(c):
            sp.append(c)
    return float(np.mean(sp)) if sp else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc",))
    sym2ens = {str(s).upper(): str(e) for s, e in zip(de.gene_names, de.gene_ids)}

    rows, summ = [], []
    for ename in EMBEDDINGS:
        z = np.load(f"results/features/foundation_{ename}.npz", allow_pickle=True)
        emb = {str(k): v for k, v in zip(z["gene_ids"], z["embedding"])}
        data = {ds: load_dataset(ds, sym2ens, emb) for ds in DATASETS}
        for A in DATASETS:
            XA, yA = data[A]
            for Bd in DATASETS:
                XB, yB = data[Bd]
                if A == Bd:
                    val = diag_cell(XA, yA); lo = hi = np.nan
                else:
                    val, lo, hi = transfer_cell(XA, yA, XB, yB)
                rows.append(dict(embedding=ename, train=A, test=Bd, reliability_spearman=val,
                                 ci_lo=lo, ci_hi=hi, n_test=len(yB)))
        off = pd.DataFrame([r for r in rows if r["embedding"] == ename and r["train"] != r["test"]])
        k562 = off[off["train"].isin(["replogle", "norman", "adamson"]) & off["test"].isin(["replogle", "norman", "adamson"])]
        diag = pd.DataFrame([r for r in rows if r["embedding"] == ename and r["train"] == r["test"]])
        summ.append(dict(embedding=ename,
                         offdiag_mean=float(off["reliability_spearman"].mean()),
                         offdiag_max=float(off["reliability_spearman"].max()),
                         offdiag_frac_ci_reaches_0p4=float((off["ci_hi"] >= 0.4).mean()),
                         offdiag_frac_pointest_ge_0p4=float((off["reliability_spearman"] >= 0.4).mean()),
                         k562_offdiag_mean=float(k562["reliability_spearman"].mean()),
                         gladstone_diag=float(diag[diag["train"] == "gladstone"]["reliability_spearman"].iloc[0]),
                         n_diag_defined=int(diag["reliability_spearman"].notna().sum())))
    M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT, "estimator_transfer_matrix.csv"), index=False)
    S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "estimator_transfer_summary.csv"), index=False)

    print("=== E5 cross-dataset reliability transfer, three shared embeddings ===")
    print(S.round(3).to_string(index=False))
    any_portable = ((S["offdiag_frac_pointest_ge_0p4"] >= 0.5) & (S["offdiag_mean"] >= 0.4)).any()
    print(f"\nAcross all three embeddings, no off-diagonal mean reaches 0.4 "
          f"(max off-diagonal means {S['offdiag_max'].max():.3f}). Small K562 screens are underpowered "
          f"(some single-cell CIs overlap 0.4), so portability is UNSUPPORTED by current evidence, not proven zero.")
    print(f"DECISION (P14): {'PORTABLE, Nature Methods defensible' if any_portable else 'portability UNSUPPORTED, within-screen tool + portable law, Genome Biology / Cell Systems'}")
    print(f"wrote {OUT}/estimator_transfer_matrix.csv, estimator_transfer_summary.csv")


if __name__ == "__main__":
    main()
