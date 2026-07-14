"""F3, the headline mechanism figure: the error-correlation structure (E2), the noise-ceiling and
achievable-gain envelope (E3), and the routing-feasibility phase diagram (E4). Reads only committed
result files. Colorblind-safe (Okabe-Ito accents, perceptually-uniform / diverging maps for the fields).

Run (after E2, E3, E4): pixi run python src/pertema/figures_mechanism.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "figures"
CLASS_ORDER = {"constant": 0, "coexpr": 1, "foundation": 2}
CLASS = {
    "mean_condition": "constant", "mean_global": "constant",
    "knn_coexpr_k25": "coexpr", "ridge_embed": "coexpr",
    "geneformer_ridge": "foundation", "geneformer_knn": "foundation",
    "scgpt_ridge": "foundation", "scgpt_knn": "foundation",
    "gene2vec_ridge": "foundation", "gene2vec_knn": "foundation",
}


def panel_error_corr(ax):
    C = pd.read_csv("results/error_correlation/error_corr_pearson.csv", index_col=0)
    S = pd.read_csv("results/error_correlation/error_correlation_summary.csv")
    order = sorted(C.columns, key=lambda n: (CLASS_ORDER.get(CLASS.get(n, "z"), 9), n))
    C = C.loc[order, order]
    im = ax.imshow(C.values, cmap="magma", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6)
    # class separators
    cls = [CLASS.get(n, "z") for n in order]
    bounds = [i for i in range(1, len(order)) if cls[i] != cls[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.2); ax.axvline(b - 0.5, color="white", lw=1.2)
    neff = S["N_eff"].mean(); offp = S["mean_offdiag_pearson"].mean()
    ax.set_title(f"A. Cross-model error correlation\nN_eff {neff:.2f} of {len(order)}; "
                 f"mean off-diagonal {offp:.2f}", fontsize=9, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson error correlation")


def panel_noise_ceiling(ax):
    """The achievable-gain envelope, shown distributionally (the honest aggregate claim; the per-perturbation
    floor-vs-error coupling is weak, so we do not imply a tight per-perturbation relationship)."""
    m = pd.read_csv("results/ceiling/noise_ceiling_per_perturbation.csv")
    s = pd.read_csv("results/ceiling/noise_ceiling_summary.csv").iloc[0]
    bins = np.linspace(0.3, 1.05, 46)
    for col, color, lab in [("floor_sb", "#009E73", "noise floor (irreducible)"),
                            ("oracle_err", "#0072B2", "oracle predictor error"),
                            ("best_fixed_err", "#D55E00", "best fixed predictor")]:
        ax.hist(m[col], bins=bins, histtype="step", lw=1.8, color=color, density=True, label=lab)
        ax.axvline(float(m[col].mean()), color=color, ls="--", lw=1.0)
    ax.set_xlabel("1 - Pearson error (top-1000 expressed genes)")
    ax.set_ylabel("density")
    ax.set_title(f"B. The deployable predictor is noise-limited\nfloor {s['mean_floor_sb']:.2f}, best-fixed "
                 f"{s['mean_best_fixed_err']:.2f} (f_noise {s['f_noise_shared']:.2f}), oracle "
                 f"{s['mean_oracle_err']:.2f} sits below the floor (selection bias)", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left")


def panel_phase(ax):
    df = pd.read_csv("results/simulation/routing_phase_grid.csv")
    pts = pd.read_csv("results/simulation/datasets_on_phase.csv") if \
        os.path.exists("results/simulation/datasets_on_phase.csv") else pd.DataFrame()
    rhos = np.sort(df["rho"].unique()); fs = np.sort(df["f_noise"].unique())
    grid = df.pivot(index="f_noise", columns="rho", values="frac_captured").values
    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu",
                   extent=[rhos.min(), rhos.max(), fs.min(), fs.max()], vmin=-vmax, vmax=vmax)
    cs = ax.contour(rhos, fs, grid, levels=[0.0], colors="#000000", linewidths=1.6)
    ax.clabel(cs, fmt={0.0: "routing breaks even"}, fontsize=7)
    for _, r in pts.iterrows():
        cap = r.get("sim_capture_exact", 0.0)
        col = "#D55E00" if cap < 0 else "#009E73"        # red if routing hurts at its measured skill spread
        pred = "prediction" in str(r.get("routing_verdict", ""))
        mk, sz = ("D", 80) if pred else ("*", 195)       # diamond = prediction, star = empirical routing test
        ax.scatter([r["rho"]], [r["f_noise"]], s=sz, marker=mk, color=col,
                   edgecolor="white", linewidth=1.2, zorder=6)
        dx, ha = (-12, "right") if r["rho"] > 0.55 else (12, "left")
        ax.annotate(f"{r['dataset']} ({cap:+.2f})", (r["rho"], r["f_noise"]), textcoords="offset points",
                    xytext=(dx, -18), fontsize=7, color="#111111", ha=ha, zorder=6,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
                    bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.9))
    ax.set_xlabel("cross-predictor error correlation  rho  (E2)")
    ax.set_ylabel("shared measurement-noise fraction  f_noise  (E3)")
    ax.set_title("C. Routing-feasibility phase diagram (stars = empirical routing test, diamonds = prediction)\n"
                 "both empirical failures land infeasible (red), each at its measured mean-skill spread",
                 fontsize=8, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of oracle headroom captured")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panel_error_corr(axes[0])
    panel_noise_ceiling(axes[1])
    panel_phase(axes[2])
    fig.suptitle("PertEMA F3: the mechanism and the law. Errors are correlated (A) and noise-dominated (B), "
                 "so routing is infeasible where the real data sit (C), while abstention still works.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIG, "mechanism_F3.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
