"""D8 open benchmark resource: consolidate every reliability-estimation result into one standardized,
long-format table so PertEMA and the UQ baselines can be compared across datasets and settings, and so
others can extend it. Also emits the predictor provenance table (CLEAN / UNKNOWN-OVERLAP / KNOWN-OVERLAP).

This is a reliability benchmark for perturbation prediction: for each setting (dataset and context), each UQ
method's ability to rank a predictor's realized error is scored by the area under the risk-coverage curve
(lower is better) and the Spearman of predicted reliability against realized accuracy, against the oracle and
no-selection references. It reads only committed result files.

Run: pixi run python src/pertema/build_benchmark.py
"""
import os

import numpy as np
import pandas as pd

R = "results/pertema"
OUT = "results/benchmark"

# Predictor pretraining provenance (invariant 2). All current predictors are trained only on the
# gene-disjoint splits with no pretraining, so CLEAN. Foundation models added later would be UNKNOWN-OVERLAP.
PROVENANCE = {
    "mean_condition": "CLEAN", "mean_global": "CLEAN", "knn_coexpr_k25": "CLEAN", "ridge_embed": "CLEAN",
    "mlp_decoder": "CLEAN", "no_change": "CLEAN",
    "prescribe_gears_backbone": "UNKNOWN-OVERLAP",   # GEARS backbone; Norman may be in any pretraining corpus
    # Frozen-adapted foundation-model roster (D1): a light head over PRETRAINED gene embeddings. The head is
    # CLEAN (trained gene-disjoint) but the embedding is pretrained on public corpora that may overlap the
    # benchmark cell types, so the predictor is UNKNOWN-OVERLAP (invariant 2).
    "geneformer_ridge": "UNKNOWN-OVERLAP", "geneformer_knn": "UNKNOWN-OVERLAP",
    "scgpt_ridge": "UNKNOWN-OVERLAP", "scgpt_knn": "UNKNOWN-OVERLAP",
    "gene2vec_ridge": "UNKNOWN-OVERLAP", "gene2vec_knn": "UNKNOWN-OVERLAP",
}


def mc(s):
    """Parse a 'mean +/- ci' string into (mean, ci)."""
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s)
    if "+/-" in s:
        a, b = s.split("+/-")
        return float(a), float(b)
    return float(s), np.nan


def emit(rows, dataset, setting, predictor, method, metric, mean, ci, n, provenance):
    rows.append(dict(dataset=dataset, setting=setting, predictor=predictor, uq_method=method,
                     metric=metric, value=round(mean, 4) if np.isfinite(mean) else np.nan,
                     ci95=round(ci, 4) if np.isfinite(ci) else np.nan, n=n, provenance=provenance))


def parse_estimator_file(rows, path, dataset, setting, predictor_col="predictor", n=None):
    """estimator/transfer summary: one row per predictor, UQ methods in columns."""
    df = pd.read_csv(path)
    method_cols = {"pertema": "aurc_est", "magnitude_heuristic": "aurc_magnitude",
                   "similarity_heuristic": "aurc_similarity", "random_feature_control": "aurc_random_feat",
                   "oracle": "aurc_oracle", "no_selection": "aurc_noselect"}
    sp_cols = {"pertema": "spearman_est", "random_feature_control": "spearman_rand",
               "label_shuffle_control": "spearman_labelshuffle"}
    for _, r in df.iterrows():
        pred = r[predictor_col] if predictor_col in df.columns else "mean_condition"
        prov = PROVENANCE.get(pred, "CLEAN")
        for method, col in method_cols.items():
            if col in df.index or col in df.columns:
                m, c = mc(r[col])
                emit(rows, dataset, setting, pred, method, "aurc", m, c, n, prov)
        for method, col in sp_cols.items():
            if col in df.columns:
                m, c = mc(r[col])
                emit(rows, dataset, setting, pred, method, "reliability_spearman", m, c, n, prov)


def build_accuracy():
    """D8 prediction-accuracy layer: consolidate the parity metrics (Ahlmann-Eltze L2, Pearson-delta,
    error-relative-to-mean, and the Wong delta-Pearson panel) across the predictor roster and datasets into
    one long-format table with provenance, so the benchmark resource covers prediction accuracy AND
    reliability. Reads only committed parity CSVs; skips any not yet built."""
    P = "results/parity"
    arows = []

    def add(dataset, df, cols):
        for _, r in df.iterrows():
            prov = r.get("provenance", PROVENANCE.get(r["predictor"], "CLEAN"))
            for metric, col in cols.items():
                if col in r and pd.notna(r[col]):
                    arows.append(dict(dataset=dataset, predictor=r["predictor"], metric=metric,
                                      value=float(r[col]), provenance=prov))

    ae = {"l2": "l2", "pearson_delta": "pearson_delta", "error_rel_to_mean": "error_rel_to_mean"}
    for ds, fn in [("Gladstone_CD4", "parity_gladstone.csv"), ("Gladstone_CD4", "parity_gladstone_foundation.csv"),
                   ("Replogle_K562", "parity_replogle.csv"), ("Norman_K562", "parity_norman.csv"),
                   ("Adamson_K562", "parity_adamson.csv")]:
        p = os.path.join(P, fn)
        if os.path.exists(p):
            add(ds, pd.read_csv(p), ae)
    wong = os.path.join(P, "parity_wong_metrics.csv")
    if os.path.exists(wong):
        add("Gladstone_CD4", pd.read_csv(wong),
            {"wong_pearson_delta_all": "pearson_delta_all", "wong_pearson_de_delta_top20": "pearson_de_delta_top20"})
    if arows:
        acc = pd.DataFrame(arows).drop_duplicates(["dataset", "predictor", "metric"]).reset_index(drop=True)
        acc.to_csv(f"{OUT}/accuracy_metrics.csv", index=False)
        print(f"wrote {OUT}/accuracy_metrics.csv ({len(acc)} rows, "
              f"{acc['predictor'].nunique()} predictors, {acc['dataset'].nunique()} datasets)")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # Gladstone single-context (3 predictors)
    parse_estimator_file(rows, f"{R}/estimator_metrics_summary.csv", "Gladstone_CD4", "single_context", n=33983)
    # Gladstone activation-state transfer (mean, kNN)
    parse_estimator_file(rows, f"{R}/transfer_estimator_summary.csv", "Gladstone_CD4", "activation_transfer", n=135932)
    # Gladstone donor transfer (axis-level)
    parse_estimator_file(rows, f"{R}/donor_transfer_summary.csv", "Gladstone_CD4", "donor_transfer",
                         predictor_col="axis", n=None)
    # External Replogle K562<->RPE1 transfer
    parse_estimator_file(rows, f"{R}/replogle_transfer_summary.csv", "Replogle_K562_RPE1", "cross_cell_line_transfer",
                         predictor_col="axis", n=1696)

    # Deep-ensemble intrinsic uncertainty on Gladstone transfer (extra UQ method, same setting)
    iu = pd.read_csv(f"{R}/intrinsic_uncertainty_transfer.csv").iloc[0]
    for method, sp, au in [("pertema", "spearman_pertema", "aurc_pertema"),
                           ("deep_ensemble_intrinsic", "spearman_intrinsic", "aurc_intrinsic"),
                           ("magnitude_heuristic", "spearman_magnitude", "aurc_magnitude"),
                           ("oracle", None, "aurc_oracle"), ("no_selection", None, "aurc_noselect")]:
        m, c = mc(iu[au]); emit(rows, "Gladstone_CD4", "activation_transfer_pooled", "knn_coexpr_k25",
                                method, "aurc", m, c, 67086, "CLEAN")
        if sp:
            m, c = mc(iu[sp]); emit(rows, "Gladstone_CD4", "activation_transfer_pooled", "knn_coexpr_k25",
                                    method, "reliability_spearman", m, c, 67086, "CLEAN")

    # MC-dropout intrinsic uncertainty on Gladstone transfer (D2 second deep intrinsic method), if built
    mcp = f"{R}/mc_dropout_transfer.csv"
    if os.path.exists(mcp):
        md = pd.read_csv(mcp).iloc[0]
        for method, sp, au in [("mc_dropout_intrinsic", "spearman_mcdropout", "aurc_mcdropout")]:
            m, c = mc(md[au]); emit(rows, "Gladstone_CD4", "activation_transfer_pooled", "mlp_decoder",
                                    method, "aurc", m, c, int(md.get("n", 0)) or None, "CLEAN")
            m, c = mc(md[sp]); emit(rows, "Gladstone_CD4", "activation_transfer_pooled", "mlp_decoder",
                                    method, "reliability_spearman", m, c, None, "CLEAN")

    # Norman single-context PRESCRIBE head-to-head
    pv = pd.read_csv(f"{R}/prescribe_vs_pertema_norman.csv").iloc[0]
    npn = int(pv["n_perturbations"])
    for method, sp, au, prov in [("pertema", "pertema_SPCC", "pertema_AURC", "CLEAN"),
                                 ("prescribe_intrinsic", "prescribe_uncertainty_SPCC", "prescribe_uncertainty_AURC", "UNKNOWN-OVERLAP"),
                                 ("magnitude_heuristic", "magnitude_heuristic_SPCC", "magnitude_heuristic_AURC", "CLEAN"),
                                 ("oracle", None, "oracle_AURC", "CLEAN"), ("no_selection", None, "noselect_AURC", "CLEAN")]:
        emit(rows, "Norman_K562", "single_context", "prescribe_gears_backbone", method, "aurc",
             float(pv[au]), np.nan, npn, prov)
        if sp and pd.notna(pv[sp]):
            emit(rows, "Norman_K562", "single_context", "prescribe_gears_backbone", method, "reliability_spearman",
                 float(pv[sp]), np.nan, npn, prov)

    bench = pd.DataFrame(rows)
    bench = bench.dropna(subset=["value"])
    bench.to_csv(f"{OUT}/benchmark_reliability.csv", index=False)

    prov = pd.DataFrame([{"predictor": k, "provenance": v} for k, v in PROVENANCE.items()])
    prov.to_csv(f"{OUT}/provenance.csv", index=False)

    build_accuracy()      # D8 prediction-accuracy layer (parity metrics), reads committed parity CSVs

    print(f"wrote {OUT}/benchmark_reliability.csv ({len(bench)} rows), {OUT}/provenance.csv")
    print("\n=== benchmark coverage ===")
    print("datasets:", sorted(bench["dataset"].unique()))
    print("settings:", sorted(bench["setting"].unique()))
    print("UQ methods:", sorted(bench["uq_method"].unique()))
    print("\n=== PertEMA vs baselines, AURC (lower better) per setting ===")
    piv = (bench[(bench["metric"] == "aurc") & (bench["uq_method"].isin(
        ["pertema", "magnitude_heuristic", "similarity_heuristic", "deep_ensemble_intrinsic",
         "prescribe_intrinsic", "no_selection", "oracle"]))]
        .pivot_table(index=["dataset", "setting", "predictor"], columns="uq_method", values="value"))
    print(piv.to_string())


if __name__ == "__main__":
    main()
