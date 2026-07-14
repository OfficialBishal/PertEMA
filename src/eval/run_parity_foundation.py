"""D1/D5 parity with the FOUNDATION-MODEL roster (frozen-adapted). For each pretrained gene embedding
extracted by extract_foundation_embeddings.py (Geneformer, scGPT, gene2vec), fit a light ridge and kNN head
that maps the frozen embedding of a perturbed gene to its effect vector, exactly parallel to the CLEAN
co-expression ridge_embed / knn predictors in run_parity.py. This answers the parity question with foundation
models: do foundation-model gene priors beat the trivial per-condition mean baseline?

Runs in the DEFAULT env: it consumes only the plain npz embedding files (produced in the isolated deep env)
plus the same Gladstone effects, splits, and Ahlmann-Eltze metrics as run_parity.py. The frozen-adapted head
is trained only on the training folds (gene-disjoint, 3 seeds), so the ADAPTED head is CLEAN, but the frozen
embedding comes from a model pretrained on public corpora that may overlap the benchmark cell types, so every
foundation predictor carries the UNKNOWN-OVERLAP provenance flag (invariant 2).

Run: pixi run python src/eval/run_parity_foundation.py
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
from run_parity import pearson_delta, rmse                                     # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
OUT = "results/parity"
SEEDS = [42, 43, 44]
N_TOP = 1000


def main():
    os.makedirs(OUT, exist_ok=True)
    de = load_de_stats(layers=("log_fc", "baseMean"))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene_of_row = obs["target_contrast"].astype(str).to_numpy()
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    # top-1000 most highly expressed in control (Ahlmann-Eltze), ranked by baseMean expression, not the
    # mean |log_fc| the dead-code fallback previously used. Leakage-safe, matches run_parity.py. (Audit fix.)
    expr = np.nan_to_num(de.layers["baseMean"]).mean(0)
    top = np.argsort(-expr)[:N_TOP]

    emb_files = sorted(glob.glob(os.path.join("results", "features", "foundation_*.npz")))
    assert emb_files, "no foundation_*.npz; run extract_foundation_embeddings.py in the deep env first"

    # the shared mean_condition baseline (so error_rel_to_mean is exact and comparable to parity_gladstone.csv)
    def eval_predictor(pred):
        per = {"l2": [], "pdelta": []}
        for seed in SEEDS:
            fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
            row_fold = np.array([fog.get(g, -1) for g in gene_of_row])
            for k in range(int(row_fold.max()) + 1):
                tr = np.where(row_fold != k)[0]; te = np.where(row_fold == k)[0]
                if te.size == 0:
                    continue
                pred.fit(tr, lfc, obs)
                P = pred.predict(te, obs)
                per["l2"].append(rmse(P, lfc[te], top))
                per["pdelta"].append(pearson_delta(P, lfc[te], top))
        return float(np.nanmean(np.concatenate(per["l2"]))), float(np.nanmean(np.concatenate(per["pdelta"])))

    base_l2, _ = eval_predictor(MeanPredictor("condition"))
    rows = [dict(predictor="mean_condition", model="baseline", head="constant",
                 l2=base_l2, pearson_delta=_, error_rel_to_mean=1.0, provenance="CLEAN",
                 n_genes_covered=len(set(gene_of_row)))]

    for ef in emb_files:
        name = os.path.basename(ef)[len("foundation_"):-len(".npz")]
        z = np.load(ef)
        gene_emb = {str(g): v for g, v in zip(z["gene_ids"], z["embedding"])}
        covered = len(set(gene_of_row) & set(gene_emb))
        for head, pred in (("ridge", RidgeEmbeddingPredictor(gene_emb, alpha=100.0)),
                           ("knn", KNNSimilarityPredictor(gene_emb, k=25))):
            l2, pd_ = eval_predictor(pred)
            rows.append(dict(predictor=f"{name}_{head}", model=name, head=head, l2=l2, pearson_delta=pd_,
                             error_rel_to_mean=l2 / base_l2, provenance="UNKNOWN-OVERLAP",
                             n_genes_covered=covered))
            print(f"{name}_{head}: L2 {l2:.4f}  Pearson-delta {pd_:.4f}  rel-to-mean {l2/base_l2:.4f}  "
                  f"({covered} genes with a pretrained embedding)")

    res = pd.DataFrame(rows).sort_values("error_rel_to_mean")
    res.to_csv(os.path.join(OUT, "parity_gladstone_foundation.csv"), index=False)
    beats = res[(res["error_rel_to_mean"] < 1.0) & (res["predictor"] != "mean_condition")]
    print("\n=== D1/D5 foundation-model parity on Gladstone (frozen-adapted, Ahlmann-Eltze metrics) ===")
    print(res[["predictor", "l2", "pearson_delta", "error_rel_to_mean", "provenance"]].round(4).to_string(index=False))
    print(f"\nFrozen-adapted foundation predictors beating the per-condition mean (rel-to-mean < 1): "
          f"{beats['predictor'].tolist() if len(beats) else 'NONE - the mean baseline is not beaten even by foundation-model priors'}")
    print(f"wrote {OUT}/parity_gladstone_foundation.csv")


if __name__ == "__main__":
    main()
