"""U13 supplementary figures: build supplementary figures S1-S8 from the committed ULT analysis CSVs.

Run: pixi run python src/pertema/build_supp_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "figures"
OK = "#0072B2"
RED = "#D55E00"
GRN = "#009E73"


def read(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p} ({os.path.getsize(p)//1024} KB)")


def s1_mode_collapse():
    d = read("results/diagnostics/mode_collapse.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(d["predictor"], d["dispersion_ratio"], color=OK)
    ax.axhline(1.0, color=RED, ls="--", lw=1, label="true dispersion (ratio = 1)")
    ax.set_ylabel("dispersion ratio\n(pred magnitude SD / true SD)")
    ax.set_title("S1. Predictors under-disperse (mode collapse toward the mean)", fontsize=9, loc="left")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7)
    save(fig, "S1_mode_collapse.png")


def s2_effect_size():
    d = read("results/diagnostics/effect_size_stratified.csv")
    if d is None:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    a1.bar(d["effect_stratum"], d["mean_error"], color=OK)
    a1.set_ylim(0.79, 0.83); a1.set_ylabel("mean error (1 - Pearson)")
    a1.set_title("S2a. Error is flat across effect-size strata", fontsize=9, loc="left")
    a2.bar(d["effect_stratum"], d["magnitude_reliability_spearman"],
           color=[RED if v < 0 else GRN for v in d["magnitude_reliability_spearman"]])
    a2.axhline(0, color="#888", lw=0.8)
    a2.set_ylabel("magnitude reliability Spearman")
    a2.set_title("S2b. Magnitude anti-selects at small effect", fontsize=9, loc="left")
    save(fig, "S2_effect_size.png")


def s3_transfer_confound():
    d = read("results/diagnostics/transfer_confound.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(d)); w = 0.35
    ax.bar(x - w / 2, d["feature_transfer_diag_max"], w, label="feature-transfer (refit, max)", color=GRN)
    ax.bar(x + w / 2, d["weight_transfer_offdiag_max"], w, label="weight-transfer (off-diag, max)", color=RED)
    ax.axhline(0.4, color="#888", ls="--", lw=1, label="portability bar 0.4")
    ax.set_xticks(x); ax.set_xticklabels(d["embedding"])
    ax.set_ylabel("reliability Spearman")
    ax.set_title("S3. Strong refit estimator, weights do not transfer (genuine non-portability)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, "S3_transfer_confound.png")


def s4_gnn_bound():
    d = read("results/theory/gnn_bound_add_decorrelated.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(d["rho_new"], d["break_even_r"], "o-", color=OK, label="required break-even r*")
    ax.axhline(0.068, color=RED, ls="--", lw=1.2, label="achievable router quality 0.068")
    ax.plot(d["rho_new"], d["N_eff"] / 3.0, "s--", color=GRN, ms=4, label="N_eff / 3 (diversity)")
    ax.set_xlabel("GNN correlation with the pack (rho_new)")
    ax.set_ylabel("router quality")
    ax.set_title("S4. Even a decorrelated GNN cannot make routing feasible", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, "S4_gnn_bound.png")


def s5_fdr():
    d = read("results/stats_multiple_comparison.csv")
    if d is None:
        return
    d = d.sort_values("p_value")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))
    ax.scatter(d["p_value"], y, color=[GRN if s else RED for s in d["significant_at_fdr_005"]], zorder=3)
    ax.axvline(0.05, color="#888", ls="--", lw=1, label="alpha 0.05")
    ax.set_yticks(y); ax.set_yticklabels([c[:42] for c in d["claim"]], fontsize=6)
    ax.set_xlabel("p-value"); ax.set_xscale("log")
    ax.set_title("S5. Benjamini-Hochberg FDR (green survives at 0.05)", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, "S5_multiple_comparison.png")


def s6_theory_validation():
    r1 = read("results/theory/R1_oracle_headroom_validation.csv")
    r2 = read("results/theory/R2_meanskill_suppression_validation.csv")
    if r1 is None:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(r1["rho"], r1["head_sim"], "o", color=OK, ms=4, label="simulation")
    a1.plot(r1["rho"], r1["head_analytic"], "-", color=RED, label="analytic sqrt(1-rho) a_K")
    a1.set_xlabel("error correlation rho"); a1.set_ylabel("oracle headroom")
    a1.set_title("S6a. R1 premium exact (< 1% error)", fontsize=9, loc="left"); a1.legend(fontsize=7)
    if r2 is not None:
        a2.plot(r2["Delta"], r2["head_sim"], "o", color=OK, ms=4, label="simulation")
        a2.plot(r2["Delta"], r2["head_corrected"], "-", color=GRN, label="corrected form")
        a2.plot(r2["Delta"], r2["head_prompt"], "--", color=RED, label="prompt form (sqrt2 slip)")
        a2.set_xlabel("mean-skill gap Delta"); a2.set_ylabel("oracle headroom")
        a2.set_title("S6b. R2 corrected vs the sqrt(2) slip", fontsize=9, loc="left"); a2.legend(fontsize=7)
    save(fig, "S6_theory_validation.png")


def s7_rho_artifact():
    d = read("results/error_correlation/rho_artifact_control.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(d)); w = 0.35
    ax.bar(x - w / 2, d["raw_rho"], w, label="raw error correlation", color=OK)
    ax.bar(x + w / 2, d["model_specific_rho"], w, label="model-specific residual (leave-2-out)", color=RED)
    ax.axhline(0.5, color="#888", ls="--", lw=1, label="threshold 0.5")
    ax.set_xticks(x); ax.set_xticklabels([s.split("_")[0] for s in d["dataset"]])
    ax.set_ylabel("correlation")
    ax.set_title("S7. Raw correlation is largely shared per-perturbation difficulty", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    save(fig, "S7_rho_artifact.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    s1_mode_collapse(); s2_effect_size(); s3_transfer_confound(); s4_gnn_bound()
    s5_fdr(); s6_theory_validation(); s7_rho_artifact()
    print("supplementary figures S1-S7 built")


if __name__ == "__main__":
    main()
