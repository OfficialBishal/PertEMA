"""U13: build the seven main tables (T1..T7) as markdown from committed result artifacts.

The manuscript previously had ZERO tables (disqualifying at the target tier). This assembles them from the
result CSVs so they regenerate with the pipeline. T2..T5 read result CSVs at runtime (so they reflect the
current, audit-verified numbers); T1, T6, T7 carry assembled descriptive content whose every quantitative
entry is traced to a result file or a recorded fact.

Run: pixi run python src/pertema/build_tables.py
"""
import os

import numpy as np
import pandas as pd

OUT = "tables"
B = "results/benchmark"
EC = "results/error_correlation"
CE = "results/ceiling"


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def df_to_md(df, index=False):
    """Minimal DataFrame-to-markdown (tabulate is not in the pinned env)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def w(f, text):
    with open(os.path.join(OUT, f), "w") as fh:
        fh.write(text)


def t1_datasets():
    rows = [
        ("Gladstone_CD4", "primary human CD4 T cells", "CRISPRi Perturb-seq (10x)", "12731", "10282",
         "3 activation states (Rest, Stim8hr, Stim48hr) + 4 donors", "4", "primary",
         "Marson and Pritchard labs (marson2025)"),
        ("Replogle_K562", "K562 (and RPE1) cells", "CRISPRi Perturb-seq", "213 (parity roster)", "~8000",
         "2 cell lines (K562, RPE1), external transfer axis", "n/a", "external transfer",
         "Replogle et al. 2022"),
        ("Norman_K562", "K562 cells", "CRISPRa Perturb-seq", "102-105", "~8500",
         "single-context (combinatorial screen)", "n/a", "cross-dataset", "Norman et al. 2019"),
        ("Adamson_K562", "K562 cells", "CRISPRi Perturb-seq", "88", "~8000",
         "single-context", "n/a", "cross-dataset (routing test)", "Adamson et al. 2016"),
    ]
    hdr = "| dataset | cell type | technology | perturbations | measured genes | contexts | donors | role | reference |\n"
    hdr += "|---|---|---|---|---|---|---|---|---|\n"
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows)
    w("T1_datasets.md", "## Table 1. Datasets\n\n" + hdr + body)


def t2_metric_battery():
    df = read(f"{B}/accuracy_metrics.csv")
    if df is None:
        return
    piv = df.pivot_table(index=["dataset", "predictor"], columns="metric", values="value", aggfunc="first")
    txt = "## Table 2. Accuracy metric battery (relative to the per-condition mean baseline)\n\n"
    txt += "Every predictor is scored on multiple field-standard accuracy metrics. error_rel_to_mean = 1.0 is\n"
    txt += "the per-condition mean baseline; values near or above 1.0 mean the predictor does not beat the mean.\n\n"
    txt += df_to_md(piv.round(4), index=True)
    w("T2_metric_battery.md", txt)


def t3_reliability():
    df = read(f"{B}/benchmark_reliability.csv")
    if df is None:
        return
    txt = "## Table 3. Reliability estimation (per dataset x setting x predictor x UQ method)\n\n"
    txt += "Reliability-Spearman and AURC (lower better) for PertEMA vs the magnitude and similarity heuristics,\n"
    txt += "no selection, and the oracle, with 95 percent CIs and n. Negative controls collapse to near zero.\n\n"
    keep = df[["dataset", "setting", "predictor", "uq_method", "metric", "value", "ci95", "n", "provenance"]]
    txt += df_to_md(keep.round(4))
    w("T3_reliability.md", txt)


def t4_mechanism():
    txt = "## Table 4. Mechanism (per dataset)\n\n"
    txt += ("rho (mean off-diagonal error correlation), N_eff (participation-ratio effective ensemble size),\n"
            "f_noise (shared-noise fraction), noise floor, best-fixed and oracle error, and the winner's-curse\n"
            "DEBIASED oracle headroom with a bootstrap CI (Gladstone). model-specific rho is the leave-2-out\n"
            "partial correlation (X3) showing the raw rho is largely shared per-perturbation difficulty.\n\n")
    rows = []
    # Gladstone from the seed-averaged summary
    g = read(f"{EC}/error_correlation_summary.csv")
    art = read(f"{EC}/rho_artifact_control.csv")
    wc = read(f"{CE}/winner_curse_debiased.csv")
    theory = read("results/theory/R4_datasets_theory_plane.csv")

    def art_val(name, col):
        if art is None:
            return np.nan
        r = art[art.dataset == name]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def theory_val(name, col):
        if theory is None:
            return np.nan
        r = theory[theory.dataset == name]
        return float(r[col].iloc[0]) if len(r) else np.nan

    if g is not None:
        gm = g.mean(numeric_only=True)
        nc = read(f"{CE}/noise_ceiling_summary.csv")
        rows.append(dict(dataset="Gladstone_CD4", rho=round(gm["mean_offdiag_pearson"], 3),
                         N_eff=round(gm["N_eff"], 2), PC1=round(gm["pc1_var_frac"], 3),
                         within_class=round(gm["within_class_pearson"], 3),
                         cross_class=round(gm["cross_class_pearson"], 3),
                         model_specific_rho=round(art_val("Gladstone_CD4", "model_specific_rho"), 3),
                         f_noise=round(float(nc["f_noise_shared"].iloc[0]), 3) if nc is not None else np.nan,
                         floor=round(float(nc["mean_floor_sb"].iloc[0]), 3) if nc is not None else np.nan,
                         best_fixed=round(float(nc["mean_best_fixed_err"].iloc[0]), 3) if nc is not None else np.nan,
                         raw_headroom=round(float(wc[wc.quantity == "raw_oracle_headroom"]["estimate"].iloc[0]), 4)
                         if wc is not None else np.nan,
                         debiased_headroom=(f"{float(wc[wc.quantity=='debiased_headroom_above_floor']['estimate'].iloc[0]):.4f} "
                                            f"[{float(wc[wc.quantity=='debiased_headroom_above_floor']['ci_lo'].iloc[0]):.4f}, "
                                            f"{float(wc[wc.quantity=='debiased_headroom_above_floor']['ci_hi'].iloc[0]):.4f}]")
                         if wc is not None else "n/a",
                         break_even_r=round(theory_val("Gladstone_CD4", "break_even_r"), 3)))
    for ds, name in [("replogle", "Replogle_K562"), ("norman", "Norman_K562"), ("adamson", "Adamson_K562")]:
        e = read(f"{EC}/error_correlation_{ds}.csv")
        nc = read(f"{CE}/noise_ceiling_{ds}_summary.csv")
        if e is None:
            continue
        er = e.iloc[0]
        rows.append(dict(dataset=name, rho=round(er["mean_offdiag_pearson"], 3), N_eff=round(er["N_eff"], 2),
                         PC1=np.nan, within_class=np.nan, cross_class=np.nan,
                         model_specific_rho=round(art_val(name, "model_specific_rho"), 3),
                         f_noise=round(float(nc["f_noise_shared"].iloc[0]), 3) if nc is not None else np.nan,
                         floor=round(float(nc["mean_floor_sb"].iloc[0]), 3) if nc is not None else np.nan,
                         best_fixed=round(er["err_best_fixed"], 3),
                         raw_headroom=round(er["oracle_headroom"], 4), debiased_headroom="n/a (Gladstone only)",
                         break_even_r=round(theory_val(name, "break_even_r"), 3)))
    txt += df_to_md(pd.DataFrame(rows))
    w("T4_mechanism.md", txt)


def t5_routing():
    txt = "## Table 5. Routing feasibility (per dataset)\n\n"
    txt += ("Empirical routing tests and phase-boundary placement. simulated_capture is the fraction of oracle\n"
            "headroom the simulation predicts is capturable at the dataset's measured mean-skill spread;\n"
            "break_even_r is the router quality routing requires; realized_r is the achieved router quality.\n"
            "Adamson carries the pre-registered routing test (X1): positive oracle headroom but realized r below\n"
            "break-even, so routing hurts (feasible geometry, insufficient router).\n\n")
    mech = read(f"{B}/mechanism_metrics.csv")
    theory = read("results/theory/R4_datasets_theory_plane.csv")
    ad = read("results/pertema/router_adamson.csv")
    rows = []
    if mech is not None:
        for _, m in mech.iterrows():
            name = m["dataset"]
            tr = theory[theory.dataset == name] if theory is not None else None
            rstar = round(float(tr["break_even_r"].iloc[0]), 3) if tr is not None and len(tr) else np.nan
            realized = round(float(tr["realized_r"].iloc[0]), 3) if tr is not None and len(tr) and not pd.isna(tr["realized_r"].iloc[0]) else "not tested"
            emp = m["empirical_routing"]
            if name == "Adamson_K562" and ad is not None:
                emp = (f"tested: routed {ad['err_routed'].mean():.3f} vs best-fixed {ad['err_best_fixed'].mean():.3f} "
                       f"(delta +{ad['err_routed'].mean()-ad['err_best_fixed'].mean():.3f}, frac-better "
                       f"{ad['boot_frac_routed_better'].mean():.3f}), headroom +{ad['oracle_headroom'].mean():.3f}")
            rows.append(dict(dataset=name, rho=m["error_correlation_rho"], N_eff=m["N_eff"],
                             f_noise=m["shared_noise_fraction_f"],
                             simulated_capture=m.get("simulated_geometry_capture", m.get("simulated_routing_capture")),
                             break_even_r=rstar, realized_r=realized,
                             feasibility=m.get("deployable_routing", m.get("routing_feasibility")),
                             empirical_routing=emp))
    txt += df_to_md(pd.DataFrame(rows))
    w("T5_routing.md", txt)


def t6_stats():
    txt = "## Table 6. Statistical tests for every headline claim\n\n"
    txt += ("| claim | test | statistic | n | p or CI | effect size | correction |\n"
            "|---|---|---|---|---|---|---|\n")
    rows = [
        ("PertEMA beats magnitude heuristic (AURC, single-context)", "paired cluster bootstrap over genes",
         "area 0.0031", "33983", "95% CI [0.0023, 0.0039], p < 5e-4", "small", "gene-cluster resampling"),
        ("Adamson routing does not beat best-fixed (X1)", "paired bootstrap (2000)", "delta +0.016",
         "88", "frac-better 0.339", "routing hurts", "3 seeds"),
        ("Adamson oracle headroom positive (X1)", "3-seed mean", "+0.088", "88", "n/a", "large", "3 seeds"),
        ("Winner's-curse selection-bias fraction (U6)", "bootstrap (2000)", "0.76", "33972",
         "95% CI [0.70, 0.82]", "large", "perturbation resampling"),
        ("rho is shared-difficulty (X3, model-specific < 0.5)", "leave-2-out partial correlation",
         "0.16-0.42", "88-33983", "n/a", "shared-difficulty dominated", "n/a"),
        ("Noise floor robust to replicate (X4)", "guide vs donor split-half", "f_noise 1.00 vs 0.97",
         "35747", "n/a", "robust", "n/a"),
        ("Analytic oracle premium exact (X2 R1)", "numeric vs simulation", "max rel err 0.7%", "20 rho points",
         "n/a", "validated", "3 sim seeds"),
        ("Reproducibility shortlist uplift over magnitude (E6)", "bootstrap (2000)", "+0.049 to +0.063",
         "~200-400", "frac>0 0.99+", "modest", "per destination"),
        ("Reliability != regulator recovery (E6)", "degree-matched permutation null", "AUROC 0.46 vs 0.54",
         "248", "p 0.007", "negative", "degree-matched"),
    ]
    txt += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    txt += ("\nNote: headline reliability and mechanism numbers are family means over seeds 42/43/44. "
            "Negative controls (random-feature, label-shuffle) collapse to near zero (STATUS_MASTER R4). "
            "A Benjamini-Hochberg FDR correction across the ten headline directional tests (alpha 0.05, "
            "results/stats_multiple_comparison.csv) leaves the four core claims significant, PertEMA beats "
            "magnitude (q 0.005), magnitude anti-selects reproducibility on the primary destination (q 0.013), "
            "magnitude below chance at reproducibility precision (q 0.013), and reliability underperforms "
            "magnitude at regulator recovery (q 0.018), while the marginal Stim8hr effects and the at-chance "
            "reliability-enrichment do not survive, consistent with how they are reported.\n")
    w("T6_statistical_tests.md", txt)


def t7_compute():
    txt = "## Table 7. Compute and runtime\n\n"
    txt += ("| stage | hardware | wall-clock | peak memory | notes |\n"
            "|---|---|---|---|---|\n")
    rows = [
        ("Predictor OOF errors (per dataset)", "CPU (128-core host)", "minutes", "< 8 GB",
         "mean/ridge/kNN, gene-disjoint 5-fold, 3 seeds"),
        ("PertEMA estimator training (per predictor)", "CPU", "seconds-minutes", "< 4 GB",
         "xgboost hist, n_jobs=8"),
        ("E2 error correlation (Gladstone, 10 predictors)", "CPU", "minutes", "< 16 GB", "3 seeds"),
        ("E3 / X4 noise floor (Gladstone)", "CPU", "few minutes (44 GB pseudobulk streamed)", "< 8 GB streamed",
         "CSR chunked, CP1e6+log1p"),
        ("E4 phase simulation + X2 theory", "CPU", "1-2 minutes", "< 4 GB", "25x25 grid, 3 sim seeds"),
        ("X1 Adamson routing + sensitivity", "CPU", "< 2 minutes", "< 4 GB", "5 estimators, 3 seeds"),
        ("Foundation embeddings (Geneformer/scGPT/gene2vec)", "GPU 0/1 (A100-40GB)", "as recorded", "< 40 GB",
         "frozen, embedding extraction only"),
        ("One-command reproduce (headline numbers)", "GPU 0 + CPU", "~953 s (PUB core path)", "n/a",
         "results/reproduce_report.md, determinism at float32 epsilon"),
    ]
    txt += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    txt += "\nGPU policy: CUDA_VISIBLE_DEVICES restricted to GPU 0 and GPU 1 (A100-SXM4-40GB). Seeds 42/43/44.\n"
    w("T7_compute.md", txt)


def main():
    os.makedirs(OUT, exist_ok=True)
    t1_datasets(); t2_metric_battery(); t3_reliability(); t4_mechanism(); t5_routing(); t6_stats(); t7_compute()
    # combined file
    combined = "# PertEMA main tables (T1-T7)\n\nRegenerated by src/pertema/build_tables.py from result artifacts.\n\n"
    for f in ["T1_datasets.md", "T2_metric_battery.md", "T3_reliability.md", "T4_mechanism.md",
              "T5_routing.md", "T6_statistical_tests.md", "T7_compute.md"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            combined += open(p).read() + "\n\n"
    w("ALL_TABLES.md", combined)
    print("wrote tables/T1..T7 and ALL_TABLES.md")
    for f in sorted(os.listdir(OUT)):
        print(" ", f)


if __name__ == "__main__":
    main()
