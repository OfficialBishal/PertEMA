"""Publication-quality figures for the PertEMA manuscript, one coherent module, one consistent style.

All figures are generated from the committed result CSVs (no composing of pre-rendered rasters) with the
scienceplots 'science'+'nature' style and LaTeX text rendering, and saved as VECTOR PDF sized to journal
column widths. Run with LaTeX on PATH:
  PATH=pixi run python src/pertema/figures_paper.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401  (registers the styles)

OUT = "figures/paper"
# scienceplots colour cycle
C = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]

DS = {"Gladstone_CD4": "Gladstone", "Replogle_K562": "Replogle", "Norman_K562": "Norman",
      "Adamson_K562": "Adamson"}
PRED = {"mean_condition": r"cond.\ mean", "mean_global": "global mean", "knn_coexpr_k25": "kNN",
        "ridge_embed": "ridge", "gene2vec_ridge": "gene2vec", "gene2vec_knn": r"gene2vec$_{k}$",
        "geneformer_ridge": "Geneformer", "geneformer_knn": r"Geneformer$_{k}$", "scgpt_ridge": "scGPT",
        "scgpt_knn": r"scGPT$_{k}$", "no_change": "no-change", "mlp_decoder": "MLP"}


def setup():
    plt.style.use(["science", "nature"])
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.prop_cycle": plt.cycler("color", C),
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
    os.makedirs(OUT, exist_ok=True)


def panel(ax, letter, dx=-0.20, dy=1.05):
    ax.text(dx, dy, r"\textbf{" + letter + "}", transform=ax.transAxes, fontsize=9,
            va="top", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def read(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def num(x):
    """Parse a float that may be stored as '0.871 +/- 0.000' or '0.871 (0.001)'."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).split("+/-")[0].split("(")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


SRC = os.path.join(OUT, "source_data")


def source_data(name, df):
    """Emit the plotted values next to the figure, per the reproducibility invariant."""
    os.makedirs(SRC, exist_ok=True)
    df.to_csv(os.path.join(SRC, name), index=False)
    print("  source-data", os.path.join(SRC, name))


def boot_ratio_ci(numer, denom, B=2000, seed=42):
    """95 percent CI for mean(numer)/mean(denom) resampling the paired per-perturbation vector."""
    rng = np.random.default_rng(seed)
    n = len(numer)
    idx = rng.integers(0, n, size=(B, n))
    rb = numer[idx].mean(axis=1) / denom[idx].mean(axis=1)
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return float(numer.mean() / denom.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------------------------------------
def fig1_concept_parity():
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)

    # (a) schematic
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def box(x, y, w, h, text, fc="#eaf1fb", ec=C[0]):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333"), zorder=1)
    box(0.2, 3.6, 2.6, 1.3, r"frozen\\predictor")
    box(3.6, 3.6, 2.2, 1.3, r"prediction $\hat{y}$")
    box(3.6, 0.5, 2.2, 1.3, r"true effect $y$")
    box(7.0, 2.05, 2.8, 1.3, r"error $1-r(\hat{y},y)$", fc="#fdecea", ec=C[3])
    box(0.2, 0.5, 2.6, 1.3, r"prediction-time\\features", fc="#eafaf0", ec=C[1])
    box(3.4, -1.2, 3.2, 0.0, "", fc="none", ec="none")
    ax.add_patch(plt.Rectangle((3.5, -1.15), 3.2, 1.15, fc="#f3ecfa", ec=C[4], lw=0.8))
    ax.text(5.1, -0.58, r"PertEMA\\meta-model", ha="center", va="center", fontsize=7)
    arrow(2.8, 4.25, 3.6, 4.25); arrow(4.7, 3.6, 4.7, 1.85); arrow(5.8, 2.7, 7.0, 2.7)
    arrow(2.8, 1.15, 3.5, 0.2); arrow(5.1, -0.0, 6.0, 1.4)
    arrow(8.4, 2.05, 6.7, -0.3)
    ax.text(9.85, 0.4, r"predicted\\reliability", ha="right", va="center", fontsize=6.5, color=C[4])
    panel(ax, "a", dx=-0.02, dy=1.08)

    # (b) parity forest: per-perturbation error relative to the per-condition mean, 95% bootstrap CIs
    #     (Ref A Fig 2c idiom: pointrange, null at 1.0, CIs that cross the null greyed out)
    ax = fig.add_subplot(gs[0, 1])
    pe = read("results/error_correlation/per_perturbation_errors.csv")
    ref = pe["mean_condition_err"].to_numpy()
    predcols = [("mean_global_err", "mean_global"), ("knn_coexpr_k25_err", "knn_coexpr_k25"),
                ("ridge_embed_err", "ridge_embed"), ("gene2vec_ridge_err", "gene2vec_ridge"),
                ("gene2vec_knn_err", "gene2vec_knn"), ("geneformer_ridge_err", "geneformer_ridge"),
                ("geneformer_knn_err", "geneformer_knn"), ("scgpt_ridge_err", "scgpt_ridge"),
                ("scgpt_knn_err", "scgpt_knn")]
    recs = [(key,) + boot_ratio_ci(pe[col].to_numpy(), ref) for col, key in predcols]
    recs.sort(key=lambda t: t[1], reverse=True)  # worst at top, best (nearest null) at bottom
    keys = [r[0] for r in recs]
    est = np.array([r[1] for r in recs]); lo = np.array([r[2] for r in recs]); hi = np.array([r[3] for r in recs])
    y = np.arange(len(keys))
    crosses = (lo <= 1.0) & (hi >= 1.0)  # CI overlaps the null: indistinguishable from the mean baseline
    for i in range(len(keys)):
        col = C[6] if crosses[i] else C[0]
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=col, lw=1.2, zorder=2, solid_capstyle="round")
        ax.plot([est[i]], [y[i]], "o", color=col, ms=3.4, zorder=3)
    ax.axvline(1.0, color=C[3], lw=1.0, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([PRED.get(k, k) for k in keys])
    ax.set_xlim(0.998, hi.max() + 0.006)
    ax.set_xlabel(r"per-perturbation error relative to the mean")
    ax.text(1.0, -0.55, r"mean baseline", color=C[3], fontsize=6, rotation=90, va="bottom", ha="right")
    ax.set_ylim(-0.7, len(keys) - 0.3)
    panel(ax, "b", dx=-0.55)
    source_data("F1b_parity.csv", pd.DataFrame({"predictor": keys, "error_rel_to_mean": est,
                                                "ci_lo": lo, "ci_hi": hi, "ci_crosses_null": crosses}))
    save(fig, "F1.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig2_reliability():
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.5)
    # (a) estimator forest: reliability Spearman per predictor against the two negative controls
    #     (Ref A Fig 2c forest idiom; error bars are +/- 1 SD across three seeds)
    ax = fig.add_subplot(gs[0, 0])
    es = read("results/pertema/estimator_metrics_per_seed.csv")
    preds = ["mean_condition", "ridge_embed", "knn_coexpr_k25"]
    ticks, ticklab, srows = [], [], []
    for yy, p in enumerate(preds):
        g = es[es.predictor == p]
        for m, c, mk in [("spearman_est", C[0], "o"), ("spearman_rand", C[6], "s"),
                         ("spearman_labelshuffle", C[5], "^")]:
            me, sd = g[m].mean(), g[m].std(ddof=1)
            ax.errorbar(me, yy, xerr=sd, fmt=mk, color=c, ms=3.8 if mk == "o" else 3.0,
                        lw=1.0, capsize=2, zorder=3 if mk == "o" else 2, alpha=1.0 if mk == "o" else 0.85)
            srows.append((p, m, me, sd))
        ticks.append(yy); ticklab.append(PRED.get(p, p))
    ax.axvline(0, color=C[3], ls="--", lw=0.9, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklab); ax.set_ylim(-0.6, len(preds) - 0.4)
    ax.set_xlabel(r"reliability Spearman $\rho$")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C[0], lw=0, label=r"PertEMA"),
                       Line2D([0], [0], marker="s", color=C[6], lw=0, label=r"random feat."),
                       Line2D([0], [0], marker="^", color=C[5], lw=0, label=r"label shuffle")],
              loc="lower right", frameon=False, fontsize=6, handletextpad=0.2)
    panel(ax, "a", dx=-0.42)
    source_data("F2a_estimator_forest.csv", pd.DataFrame(srows, columns=["predictor", "metric", "mean", "sd"]))
    # (b) calibration: ECE raw vs isotonic per predictor, +/- 1 SD across seeds
    ax = fig.add_subplot(gs[0, 1])
    cps = read("results/pertema/calibration_per_seed.csv")
    preds2 = list(dict.fromkeys(cps["predictor"]))
    x = np.arange(len(preds2)); w = 0.34
    raw_m = [cps[cps.predictor == p]["ece_raw"].mean() for p in preds2]
    raw_s = [cps[cps.predictor == p]["ece_raw"].std(ddof=1) for p in preds2]
    iso_m = [cps[cps.predictor == p]["ece_isotonic"].mean() for p in preds2]
    iso_s = [cps[cps.predictor == p]["ece_isotonic"].std(ddof=1) for p in preds2]
    ax.bar(x - w / 2, raw_m, w, yerr=raw_s, capsize=2, color=C[6], label=r"raw")
    ax.bar(x + w / 2, iso_m, w, yerr=iso_s, capsize=2, color=C[0], label=r"isotonic")
    ax.set_xticks(x); ax.set_xticklabels([PRED.get(p, p) for p in preds2])
    ax.set_ylabel(r"expected calibration error")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    panel(ax, "b", dx=-0.42)
    save(fig, "F2.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig3_transfer():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    conds = ["Rest", "Stim8hr", "Stim48hr"]
    tp = read("results/pertema/transfer_pair_errors.csv")
    M = np.full((3, 3), np.nan)
    for _, r in tp.iterrows():
        if r["src"] in conds and r["dst"] in conds:
            M[conds.index(r["src"]), conds.index(r["dst"])] = r["mean_1_minus_pearson"]
    im = a1.imshow(M, cmap="RdBu_r", vmin=0.82, vmax=0.90)
    a1.set_xticks(range(3)); a1.set_xticklabels([r"Rest", r"Stim8h", r"Stim48h"])
    a1.set_yticks(range(3)); a1.set_yticklabels([r"Rest", r"Stim8h", r"Stim48h"])
    a1.set_xlabel(r"destination context"); a1.set_ylabel(r"source context")
    for i in range(3):
        for j in range(3):
            if np.isfinite(M[i, j]):
                a1.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.87 else "black")
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04, label=r"transfer error $1-r$")
    panel(a1, "a", dx=-0.30)
    # (b) reliability Spearman across axes with negative controls
    te = read("results/pertema/transfer_estimator_summary.csv")
    rp = read("results/pertema/replogle_transfer_summary.csv")
    labels, est, ctl = [], [], []
    if te is not None:
        for c in ["spearman_est", "spearman_rand", "spearman_labelshuffle"]:
            te[c] = te[c].map(num)
        row = te.iloc[te["spearman_est"].idxmax()]
        labels.append(r"activation\\(Gladstone)"); est.append(num(row["spearman_est"]))
        ctl.append(max(abs(num(row["spearman_rand"])), abs(num(row["spearman_labelshuffle"]))))
    if rp is not None:
        r0 = rp.iloc[0]
        labels.append(r"cross-line\\(Replogle)"); est.append(num(r0["spearman_est"]))
        ctl.append(abs(num(r0.get("spearman_rand", 0.0))))
    x = np.arange(len(labels)); w = 0.35
    a2.bar(x - w / 2, est, w, label=r"PertEMA reliability", color=C[0])
    a2.bar(x + w / 2, ctl, w, label=r"negative controls", color=C[6])
    a2.axhline(0, color="#888", lw=0.6)
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel(r"reliability Spearman $\rho$")
    a2.legend(loc="upper right", frameon=False)
    panel(a2, "b")
    save(fig, "F3.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig4_corrlaw():
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.5)
    # (a) heatmap
    ax = fig.add_subplot(gs[0, 0])
    cm = read("results/error_correlation/error_corr_pearson.csv")
    Mtx = cm.set_index(cm.columns[0]).to_numpy()
    im = ax.imshow(Mtx, cmap="viridis", vmin=0.85, vmax=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"error correlation (10 predictors)", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    panel(ax, "a", dx=-0.05, dy=1.10)
    # (b) rho and N_eff across datasets
    ax = fig.add_subplot(gs[0, 1])
    mech = read("results/benchmark/mechanism_metrics.csv")
    ds = [DS[d] for d in mech["dataset"]]
    x = np.arange(len(ds))
    ax.bar(x, mech["error_correlation_rho"], color=C[0], width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=25, ha="right")
    ax.set_ylabel(r"error correlation $\rho$"); ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(x, mech["N_eff"], "o-", color=C[3], ms=3, lw=0.9)
    ax2.set_ylabel(r"$N_{\mathrm{eff}}$", color=C[3]); ax2.set_ylim(1, 2.2)
    ax2.tick_params(axis="y", colors=C[3])
    panel(ax, "b", dx=-0.42, dy=1.10)
    # (c) raw vs model-specific
    ax = fig.add_subplot(gs[0, 2])
    ra = read("results/error_correlation/rho_artifact_control.csv")
    ds = [DS[d] for d in ra["dataset"]]
    x = np.arange(len(ds)); w = 0.35
    ax.bar(x - w / 2, ra["raw_rho"], w, label=r"raw $\rho$", color=C[0])
    ax.bar(x + w / 2, ra["model_specific_rho"], w, label=r"model-specific", color=C[2])
    ax.axhline(0.5, color=C[3], ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ds, rotation=25, ha="right")
    ax.set_ylabel(r"correlation"); ax.legend(loc="upper right", frameon=False)
    panel(ax, "c", dx=-0.42, dy=1.10)
    save(fig, "F4.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig5_noise():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    mech = read("results/benchmark/mechanism_metrics.csv")
    ds = [DS[d] for d in mech["dataset"]]
    x = np.arange(len(ds))
    a1.bar(x, mech["shared_noise_fraction_f"], color=C[0], width=0.6)
    a1.set_xticks(x); a1.set_xticklabels(ds, rotation=25, ha="right")
    a1.set_ylabel(r"shared-noise fraction $f_{\mathrm{noise}}$"); a1.set_ylim(0, 1.05)
    panel(a1, "a")
    # (b) winner's curse
    wc = read("results/ceiling/winner_curse_debiased.csv").set_index("quantity")
    def q(n): return wc.loc[n]
    raw = q("raw_oracle_headroom"); deb = q("debiased_headroom_above_floor")
    a2.bar([0], [raw["estimate"]], yerr=[[raw["estimate"] - raw["ci_lo"]], [raw["ci_hi"] - raw["estimate"]]],
           color=C[6], width=0.5, capsize=2, label=r"raw oracle headroom")
    a2.bar([1], [deb["estimate"]], yerr=[[deb["estimate"] - deb["ci_lo"]], [deb["ci_hi"] - deb["estimate"]]],
           color=C[1], width=0.5, capsize=2, label=r"debiased (above floor)")
    a2.set_xticks([0, 1]); a2.set_xticklabels([r"raw", r"debiased"])
    a2.set_ylabel(r"oracle routing headroom")
    sb = q("selection_bias_fraction")
    a2.text(0.5, raw["estimate"] * 0.9, r"$%d\%%$ selection bias" % round(sb["estimate"] * 100),
            ha="center", fontsize=6.5, color=C[3])
    panel(a2, "b")
    save(fig, "F5.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig6_routing():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.38)
    # (a) phase diagram
    ax = fig.add_subplot(gs[0, 0])
    ph = read("results/simulation/routing_phase_grid.csv")
    piv = ph.pivot(index="f_noise", columns="rho", values="frac_captured")
    rhos = piv.columns.values; fs = piv.index.values
    vmax = float(np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdBu",
                   extent=[rhos.min(), rhos.max(), fs.min(), fs.max()], vmin=-vmax, vmax=vmax)
    ax.contour(rhos, fs, piv.values, levels=[0.0], colors="k", linewidths=0.8)
    mech = read("results/benchmark/mechanism_metrics.csv")
    for _, r in mech.iterrows():
        cap = r["simulated_geometry_capture"]
        ax.scatter([r["error_correlation_rho"]], [r["shared_noise_fraction_f"]], s=28,
                   marker="*", color="#111", edgecolor="white", lw=0.5, zorder=5)
        ax.annotate(DS[r["dataset"]], (r["error_correlation_rho"], r["shared_noise_fraction_f"]),
                    fontsize=5.5, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel(r"error correlation $\rho$"); ax.set_ylabel(r"noise fraction $f_{\mathrm{noise}}$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"oracle headroom captured")
    panel(ax, "a", dx=-0.28, dy=1.12)
    # (b) router-quality attenuation + break-even
    ax = fig.add_subplot(gs[0, 1])
    r3 = read("results/theory/R3_router_attenuation.csv")
    for i, (g, sub) in enumerate(r3.groupby("geometry")):
        ax.plot(sub["r"], sub["frac_captured"], "-", lw=1.0, color=C[i],
                label=r"%s ($\rho=%.2f$)" % (g.replace("-like", "").replace("mid", "mid"), sub["rho"].iloc[0]))
    ax.axhline(0, color="#888", lw=0.6)
    ax.axvline(0.215, color=C[3], ls=":", lw=0.9)
    ax.set_xlabel(r"router quality $r$"); ax.set_ylabel(r"fraction captured")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.22, ax.get_ylim()[0] * 0.8, r"Adamson $r=0.22$", color=C[3], fontsize=5.5, rotation=90)
    panel(ax, "b", dx=-0.30, dy=1.12)
    # (c) Adamson test bars
    ax = fig.add_subplot(gs[1, 0])
    ra = read("results/pertema/router_adamson.csv")
    bf, ro, orc = ra["err_best_fixed"].mean(), ra["err_routed"].mean(), ra["err_oracle"].mean()
    ax.bar([0, 1, 2], [bf, ro, orc], color=[C[0], C[3], C[1]], width=0.6)
    ax.axhline(bf, color=C[0], ls="--", lw=0.7)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"best-fixed", r"routed", r"oracle"])
    ax.set_ylim(0.42, 0.58); ax.set_ylabel(r"prediction error")
    for xi, vi in zip([0, 1, 2], [bf, ro, orc]):
        ax.text(xi, vi + 0.004, f"{vi:.3f}", ha="center", fontsize=6)
    ax.set_title(r"pre-registered Adamson test", fontsize=7)
    panel(ax, "c", dx=-0.28, dy=1.14)
    # (d) sensitivity: realized r across estimators vs break-even
    ax = fig.add_subplot(gs[1, 1])
    rs = read("results/pertema/router_adamson_sensitivity.csv")
    agg = rs[rs.estimator != "ORACLE_true_error"].groupby("estimator")["r"].mean().sort_values()
    ax.barh(np.arange(len(agg)), agg.values, color=C[5], height=0.6)
    ax.axvline(0.457, color=C[3], ls="--", lw=0.9)
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels([e.replace("_", r"\_") for e in agg.index])
    ax.set_xlabel(r"achieved router quality $r$")
    ax.text(0.46, len(agg) - 0.5, r"break-even $r^\ast=0.46$", color=C[3], fontsize=5.5, rotation=90, va="top")
    panel(ax, "d", dx=-0.42, dy=1.14)
    save(fig, "F6.pdf")


# ---------------------------------------------------------------------------------------------------------
def fig7_utility():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    ru = read("results/utility/reproducibility_uplift.csv")
    if ru is not None:
        lab = [r"%s/%s" % (DS.get(r["dst"], r["dst"]).replace("Gladstone", ""), r["target"]) for _, r in ru.iterrows()]
        lab = [r"%s (%s)" % (r["dst"].replace("Stim", "Stim "), r["target"]) for _, r in ru.iterrows()]
        x = np.arange(len(ru)); w = 0.35
        rel_err = np.vstack([np.clip(ru["uplift_reliability"] - ru["rel_ci_lo"], 0, None),
                             np.clip(ru["rel_ci_hi"] - ru["uplift_reliability"], 0, None)])
        a1.bar(x - w / 2, ru["uplift_reliability"], w, yerr=rel_err, capsize=2,
               label=r"PertEMA reliability", color=C[0])
        a1.bar(x + w / 2, ru["uplift_magnitude"], w, label=r"effect magnitude", color=C[2])
        a1.axhline(0, color="#888", lw=0.6)
        source_data("F7a_uplift.csv", ru[["dst", "target", "uplift_reliability", "rel_ci_lo",
                                          "rel_ci_hi", "uplift_magnitude", "rel_frac_pos"]])
        a1.set_xticks(x); a1.set_xticklabels(lab, rotation=25, ha="right")
        a1.set_ylabel(r"reproducibility uplift")
        a1.legend(loc="upper right", frameon=False)
    panel(a1, "a", dx=-0.26)
    # (b) estimator non-transfer matrix (geneformer)
    em = read("results/pertema/estimator_transfer_matrix.csv")
    gf = em[em.embedding == "geneformer"]
    order = ["gladstone", "replogle", "norman", "adamson"]
    Mtx = np.full((4, 4), np.nan)
    for _, r in gf.iterrows():
        if r["train"] in order and r["test"] in order:
            Mtx[order.index(r["train"]), order.index(r["test"])] = r["reliability_spearman"]
    im = a2.imshow(Mtx, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    a2.set_xticks(range(4)); a2.set_xticklabels([o.capitalize()[:4] for o in order])
    a2.set_yticks(range(4)); a2.set_yticklabels([o.capitalize()[:4] for o in order])
    a2.set_xlabel(r"applied to"); a2.set_ylabel(r"trained on")
    for i in range(4):
        for j in range(4):
            if np.isfinite(Mtx[i, j]):
                a2.text(j, i, f"{Mtx[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(Mtx[i, j]) > 0.25 else "black")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04, label=r"reliability Spearman")
    panel(a2, "b", dx=-0.36)
    save(fig, "F7.pdf")


def main():
    setup()
    fig1_concept_parity()
    fig2_reliability()
    fig3_transfer()
    fig4_corrlaw()
    fig5_noise()
    fig6_routing()
    fig7_utility()
    print("all figures written to", OUT)


if __name__ == "__main__":
    main()
