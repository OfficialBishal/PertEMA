"""D4 metric panel: add the Wong et al. 2025 (Bioinformatics btaf317) DELTA Pearson metrics to the parity
roster, alongside the Ahlmann-Eltze L2 and error-relative-to-mean already computed. Wong compares perturbation
predictors with, among others:
  - Pearson Delta (PD): Pearson(pred_delta, true_delta) over ALL genes, per perturbation then averaged.
  - Pearson DE Delta (PDED): the same restricted to the top-20 differentially expressed genes of each
    perturbation (the 20 genes with the largest absolute true effect for that perturbation).
Both are computed per-perturbation and averaged over targets (Wong's stated protocol). Our data is the
DE-stats delta representation (log_fc is already perturbed-minus-control), so the two ABSOLUTE-expression
Pearson metrics Wong also reports (Pearson Correlation, Pearson DE) are not computable here and are reported
as not-applicable rather than approximated (invariant 4, no relaxing a definition).

Runs the full 12-predictor roster (6 CLEAN + 6 frozen-adapted foundation) so the Wong metrics sit beside the
Ahlmann-Eltze ones on the same predictors. Gene-disjoint, 3 seeds.

Run: pixi run python src/eval/run_parity_wong.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("data", "predictors", "eval"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                              # noqa: E402
from predictors import KNNSimilarityPredictor, MeanPredictor, RidgeEmbeddingPredictor  # noqa: E402
from run_parity import NoChangePredictor                                       # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
OUT = "results/parity"
SEEDS = [42, 43, 44]
N_DE = 20


def _row_pd(P, T, cols=None):
    """Per-row Pearson Delta: correlation of the (mean-centered) predicted and true delta vectors."""
    if cols is not None:
        P = P[:, cols]; T = T[:, cols]
    Pc = P - P.mean(1, keepdims=True); Tc = T - T.mean(1, keepdims=True)
    num = (Pc * Tc).sum(1); den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB); coexpr = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}

    def roster():
        r = [("mean_condition", MeanPredictor("condition")), ("mean_global", MeanPredictor("global")),
             ("knn_coexpr_k25", KNNSimilarityPredictor(coexpr, k=25)),
             ("ridge_embed", RidgeEmbeddingPredictor(coexpr, alpha=100.0)), ("no_change", NoChangePredictor())]
        for ef in sorted(glob.glob(os.path.join("results", "features", "foundation_*.npz"))):
            name = os.path.basename(ef)[len("foundation_"):-len(".npz")]
            z = np.load(ef); ge = {str(g): v for g, v in zip(z["gene_ids"], z["embedding"])}
            r += [(f"{name}_ridge", RidgeEmbeddingPredictor(ge, alpha=100.0)),
                  (f"{name}_knn", KNNSimilarityPredictor(ge, k=25))]
        return r

    names = [n for n, _ in roster()]
    rows = []
    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fog.get(g, -1) for g in gene_of_row])
        for nm, pred in roster():
            pd_all, pded = [], []
            for k in range(int(row_fold.max()) + 1):
                tr = np.where(row_fold != k)[0]; te = np.where(row_fold == k)[0]
                if te.size == 0:
                    continue
                pred.fit(tr, lfc, obs)
                P = pred.predict(te, obs); T = lfc[te]
                pd_all.append(_row_pd(P, T))
                # top-20 DE genes per row by largest absolute TRUE effect (evaluation-time selection), vectorized
                de_idx = np.argsort(-np.abs(T), axis=1)[:, :N_DE]
                Psel = np.take_along_axis(P, de_idx, axis=1); Tsel = np.take_along_axis(T, de_idx, axis=1)
                pded.append(_row_pd(Psel, Tsel))
            rows.append(dict(seed=seed, predictor=nm,
                             pearson_delta_all=float(np.nanmean(np.concatenate(pd_all))),
                             pearson_de_delta_top20=float(np.nanmean(np.concatenate(pded)))))
            print(f"seed {seed} {nm}: PD(all) {rows[-1]['pearson_delta_all']:.4f}  "
                  f"PDED(top20) {rows[-1]['pearson_de_delta_top20']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    agg = res.groupby("predictor").agg(pearson_delta_all=("pearson_delta_all", "mean"),
                                       pearson_de_delta_top20=("pearson_de_delta_top20", "mean")).reset_index()
    agg = agg.sort_values("pearson_delta_all", ascending=False)
    agg.to_csv(os.path.join(OUT, "parity_wong_metrics.csv"), index=False)
    print("\n=== D4 Wong delta-Pearson metric panel (12-predictor roster, 3 seeds) ===")
    print(agg.round(4).to_string(index=False))
    print("Absolute-expression Pearson metrics (Pearson Correlation, Pearson DE) are N/A on the delta/DE-stats "
          "representation and are not approximated.")
    print(f"wrote {OUT}/parity_wong_metrics.csv")


if __name__ == "__main__":
    main()
