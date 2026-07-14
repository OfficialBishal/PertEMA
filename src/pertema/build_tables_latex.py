"""Clean, portrait-fitting booktabs LaTeX tables for the manuscript (no rotated sidewaystables).

Wide source tables are condensed to their decision-relevant columns so each fits \\textwidth. Writes
manuscript/sn-pertema/part_tables.tex. Run: pixi run python src/pertema/build_tables_latex.py
"""
import os

import numpy as np
import pandas as pd

OUT = "manuscript/sn-pertema/part_tables.tex"
DS = {"Gladstone_CD4": "Gladstone", "Replogle_K562": "Replogle", "Norman_K562": "Norman",
      "Adamson_K562": "Adamson"}
PRED = {"mean_condition": "cond.\\ mean", "mean_global": "global mean", "knn_coexpr_k25": "kNN",
        "ridge_embed": "ridge", "gene2vec_ridge": "gene2vec", "geneformer_ridge": "Geneformer",
        "scgpt_ridge": "scGPT", "no_change": "no-change", "mlp_decoder": "MLP",
        "geneformer_knn": "Geneformer$_k$", "scgpt_knn": "scGPT$_k$", "gene2vec_knn": "gene2vec$_k$"}


def read(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def num(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).split("+/-")[0].split("(")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def tbl(caption, label, colspec, header, rows, note=None, small=True):
    s = "\\begin{table}[htbp]\n\\centering\n"
    s += "\\caption{%s}\\label{%s}\n" % (caption, label)
    if small:
        s += "\\footnotesize\n"
    s += "\\begin{tabular}{%s}\n\\toprule\n" % colspec
    s += " & ".join(header) + " \\\\\n\\midrule\n"
    for r in rows:
        s += " & ".join(str(c) for c in r) + " \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    if note:
        s += "\\begin{tablenotes}\\footnotesize\\item %s\\end{tablenotes}\n" % note if False else \
             "\n{\\footnotesize %s\\par}\n" % note
    s += "\\end{table}\n\n"
    return s


def main():
    out = []

    # T1 Datasets (all screens are Perturb-seq; stated in the caption to keep the table within \textwidth)
    rows = [
        ["Gladstone", "human CD4 T", "CRISPRi", "11{,}526", "3 states, 4 donors", "primary"],
        ["Replogle", "K562 / RPE1", "CRISPRi", "213", "2 cell lines", "ext.\\ transfer"],
        ["Norman", "K562", "CRISPRa", "102", "single context", "cross-dataset"],
        ["Adamson", "K562", "CRISPRi", "88", "single context", "routing test"],
    ]
    out.append(tbl("Datasets used in this study. All four are single-cell Perturb-seq screens. The primary "
                   "screen is the genome-scale CRISPRi screen in primary human CD4 T cells, and the other three "
                   "are external screens used for transfer, cross-dataset, and routing tests.", "tab:datasets",
                   "llllll", ["Dataset", "Cell type", "Modality", "Perturbations", "Contexts", "Role"],
                   rows, small=True))

    # T4 Mechanism (condensed to decision-relevant columns; the operative headline table)
    mech = read("results/benchmark/mechanism_metrics.csv")
    ra = read("results/error_correlation/rho_artifact_control.csv")
    ms = {r["dataset"]: r["model_specific_rho"] for _, r in ra.iterrows()} if ra is not None else {}
    rows = []
    for _, m in mech.iterrows():
        d = m["dataset"]
        rows.append([DS[d], f"{m['error_correlation_rho']:.2f}", f"{m['N_eff']:.2f}",
                     f"{m['shared_noise_fraction_f']:.2f}", f"{ms.get(d, float('nan')):.2f}",
                     f"{m['break_even_router_r']:.2f}",
                     ("%.2f" % m["realized_router_r"]) if not pd.isna(m["realized_router_r"]) else "n/a",
                     m["geometry_feasible"]])
    out.append(tbl("Cross-model error structure and routing-feasibility mechanism, per screen. "
                   "$\\rho$ is the mean off-diagonal error correlation, $N_{\\mathrm{eff}}$ the effective "
                   "ensemble size, $f_{\\mathrm{noise}}$ the shared-noise fraction, model-specific $\\rho$ the "
                   "leave-two-out residual correlation, $r^\\ast$ the break-even router quality, and realized "
                   "$r$ the achieved router quality (where an empirical routing test exists).",
                   "tab:mechanism", "lccccccc",
                   ["Screen", "$\\rho$", "$N_{\\mathrm{eff}}$", "$f_{\\mathrm{noise}}$",
                    "model-spec.\\ $\\rho$", "$r^\\ast$", "realized $r$", "geom.\\ feasible"],
                   rows, small=True))

    # T5 Routing outcomes (condensed)
    rows = []
    verdict = {"Gladstone_CD4": "negative (tested): noise-dominated",
               "Replogle_K562": "negative (tested): dominant predictor",
               "Adamson_K562": "negative (tested): weak router",
               "Norman_K562": "predicted infeasible"}
    for _, m in mech.iterrows():
        d = m["dataset"]
        rows.append([DS[d], f"{m['simulated_geometry_capture']:+.2f}", f"{m['break_even_router_r']:.2f}",
                     ("%.2f" % m["realized_router_r"]) if not pd.isna(m["realized_router_r"]) else "n/a",
                     verdict[d]])
    out.append(tbl("Routing outcomes. Simulated geometry capture is the fraction of oracle headroom captured "
                   "at each screen's measured mean-skill spread. Per-instance routing does not beat the best "
                   "fixed predictor on any screen with an empirical test.",
                   "tab:routing", "lcccl",
                   ["Screen", "sim.\\ capture", "$r^\\ast$", "realized $r$", "empirical outcome"],
                   rows, small=True))

    # T3 Reliability (condensed: single-context transfer estimator, key metrics per predictor)
    te = read("results/pertema/transfer_estimator_summary.csv")
    rows = []
    if te is not None:
        for _, r in te.iterrows():
            rows.append([PRED.get(r["predictor"], r["predictor"]), f"{num(r['spearman_est']):.3f}",
                         f"{num(r['aurc_est']):.3f}", f"{num(r['aurc_magnitude']):.3f}",
                         f"{num(r['aurc_noselect']):.3f}", f"{num(r['aurc_oracle']):.3f}"])
    out.append(tbl("Reliability estimation on the activation-transfer axis, per predictor. Reliability "
                   "Spearman and the area under the risk-coverage curve (AURC, lower is better) for PertEMA "
                   "against the effect-magnitude heuristic, no selection, and the oracle.",
                   "tab:reliability", "lccccc",
                   ["Predictor", "reliab.\\ $\\rho$", "AURC PertEMA", "AURC magn.", "AURC no-sel.",
                    "AURC oracle"], rows, small=True))

    # T6 Statistical tests (condensed to the FDR-surviving headline set)
    rows = [
        ["PertEMA beats magnitude (AURC)", "paired cluster bootstrap", "$p<5\\times10^{-4}$", "0.005"],
        ["Magnitude anti-selects reproducibility", "threshold-free Spearman", "$p=0.003$", "0.013"],
        ["Magnitude below chance at precision", "hypergeometric", "$p=0.004$", "0.013"],
        ["Reliability $\\neq$ regulator recovery", "degree-matched null", "$p=0.007$", "0.018"],
    ]
    out.append(tbl("Headline statistical tests surviving Benjamini-Hochberg FDR control at $0.05$ "
                   "(four of ten directional tests, and the marginal effects do not survive and are reported as "
                   "such). $q$ is the FDR-adjusted value.",
                   "tab:stats", "llcc",
                   ["Claim", "Test", "$p$", "$q$"], rows, small=True))

    open(OUT, "w").write("".join(out))
    print("wrote", OUT, "(5 portrait tables: datasets, mechanism, routing, reliability, stats)")


if __name__ == "__main__":
    main()
