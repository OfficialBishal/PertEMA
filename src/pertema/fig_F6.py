"""F6 THE HEADLINE: routing feasibility.

Composes existing high-DPI panels (phase diagram, router attenuation) and
regenerates the Adamson pre-registered test panel from CSVs.

Panels:
  (a) routing phase diagram across four screens (composed PNG)
  (b) router-quality attenuation and break-even r* (composed PNG)
  (c) Adamson pre-registered test: best-fixed vs routed vs oracle bars, plus
      the 5-estimator sensitivity strip (0 of 5 beat best-fixed)

Run: CUDA_VISIBLE_DEVICES=0 pixi run python src/pertema/fig_F6.py
Output: figures/F6_routing_phase.png
ASCII only. Agg backend. 200 DPI.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

PHASE_PNG = os.path.join(FIG, "routing_phase_diagram.png")
ATTEN_PNG = os.path.join(FIG, "theory_router_attenuation.png")
ADAMSON_CSV = os.path.join(RES, "pertema", "router_adamson.csv")
SENS_CSV = os.path.join(RES, "pertema", "router_adamson_sensitivity.csv")
OUT = os.path.join(FIG, "F6_routing_phase.png")

# ---- style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
})
C_BEST = "#4C72B0"    # best-fixed (the line to beat)
C_ROUTED = "#C44E52"  # routed (the tested router) -- worse
C_ORACLE = "#55A868"  # oracle (headroom exists)
C_PT = "#8172B3"      # estimator sensitivity points


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compose_panel(ax, png_path, letter):
    """Show an existing PNG panel, axis off, with a bold panel letter."""
    img = mpimg.imread(png_path)
    ax.imshow(img)
    ax.set_anchor("NW")
    ax.axis("off")
    ax.text(-0.01, 1.03, letter, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left")


def build_adamson_panel(ax, letter):
    rows = read_csv_rows(ADAMSON_CSV)
    best = np.array([float(r["err_best_fixed"]) for r in rows])
    routed = np.array([float(r["err_routed"]) for r in rows])
    oracle = np.array([float(r["err_oracle"]) for r in rows])
    frac = np.array([float(r["boot_frac_routed_better"]) for r in rows])
    headroom = np.array([float(r["oracle_headroom"]) for r in rows])

    means = [best.mean(), routed.mean(), oracle.mean()]
    stds = [best.std(ddof=0), routed.std(ddof=0), oracle.std(ddof=0)]
    labels = ["Best-fixed", "Routed", "Oracle"]
    colors = [C_BEST, C_ROUTED, C_ORACLE]
    xpos = [0.0, 1.0, 2.0]

    bars = ax.bar(xpos, means, width=0.66, yerr=stds, capsize=4,
                  color=colors, edgecolor="#222222", linewidth=0.9,
                  error_kw=dict(elinewidth=1.1, ecolor="#333333"), zorder=3)

    # "line to beat": best-fixed level (lower error is better)
    ax.axhline(best.mean(), color=C_BEST, ls="--", lw=1.3, zorder=2, alpha=0.9)

    # sensitivity strip: mean routed error per estimator (5 non-oracle)
    srows = read_csv_rows(SENS_CSV)
    est_err = {}
    for r in srows:
        e = r["estimator"]
        if e == "ORACLE_true_error":
            continue
        est_err.setdefault(e, []).append(float(r["err_routed"]))
    est_means = {e: np.mean(v) for e, v in est_err.items()}
    xs_x = 3.25
    rng = np.random.default_rng(0)
    n_beat = 0
    for e, m in est_means.items():
        jx = xs_x + rng.uniform(-0.14, 0.14)
        beat = m < best.mean()
        n_beat += int(beat)
        ax.plot(jx, m, "o", ms=8, color=C_PT, mec="#222222", mew=0.7,
                zorder=4)
    ax.text(xs_x, max(est_means.values()) + 0.009,
            "5 estimators", ha="center", va="bottom", fontsize=9.5,
            color="#333333")

    # value labels on bars
    for x, m in zip(xpos, means):
        ax.text(x, m + 0.008, "%.3f" % m, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(xpos + [xs_x])
    ax.set_xticklabels(labels + ["Sensitivity"], fontsize=10)
    ax.set_ylabel("prediction error  (lower = better)", fontsize=10.5)
    ax.set_ylim(0.40, 0.66)
    ax.set_xlim(-0.55, 3.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", color="#cccccc", lw=0.7, zorder=0)

    # annotation box with the pre-registered verdict
    txt = (
        "Pre-registered Adamson test\n"
        "frac(routed better) = %.3f  (< 0.50)\n"
        "%d of 5 estimators beat best-fixed\n"
        "oracle headroom = %.3f (unclaimed)"
        % (frac.mean(), n_beat, headroom.mean())
    )
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.2, family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fbf3d8",
                      ec="#b8a15a", lw=1.0))

    # arrow marking the "line to beat"
    ax.annotate("line to beat", xy=(2.55, best.mean()),
                xytext=(2.55, best.mean() - 0.055), fontsize=8.8,
                color=C_BEST, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.1))

    ax.text(-0.055, 1.03, letter, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title("Adamson: geometry confirmed, router too weak",
                 fontsize=10.5, pad=8)


def main():
    for p in (PHASE_PNG, ATTEN_PNG, ADAMSON_CSV, SENS_CSV):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    fig = plt.figure(figsize=(13.0, 10.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.32, 1.0],
                  width_ratios=[1.16, 1.0], hspace=0.16, wspace=0.14,
                  left=0.055, right=0.985, top=0.90, bottom=0.055)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])

    # (a) phase diagram -- compose, fall back to a text note if it fails
    try:
        compose_panel(ax_a, PHASE_PNG, "a")
    except Exception as exc:  # pragma: no cover
        ax_a.axis("off")
        ax_a.text(0.5, 0.5, "phase diagram unavailable\n%s" % exc,
                  ha="center", va="center")
        ax_a.text(-0.01, 1.03, "a", transform=ax_a.transAxes, fontsize=17,
                  fontweight="bold")

    # (c) Adamson bar + sensitivity from CSV
    build_adamson_panel(ax_c, "c")

    # (b) router attenuation strip -- compose, fall back to a text note
    try:
        compose_panel(ax_b, ATTEN_PNG, "b")
    except Exception as exc:  # pragma: no cover
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "attenuation panel unavailable\n%s" % exc,
                  ha="center", va="center")
        ax_b.text(-0.01, 1.03, "b", transform=ax_b.transAxes, fontsize=17,
                  fontweight="bold")

    title = ("Routing is infeasible on all four screens (three tested, Norman predicted). The "
             "pre-registered\nAdamson test confirms the geometry but the router is too weak")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)

    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)

    size_kb = os.path.getsize(OUT) / 1024.0
    print("wrote %s (%.1f KB)" % (OUT, size_kb))
    if size_kb <= 20:
        raise SystemExit("output too small: %.1f KB" % size_kb)


if __name__ == "__main__":
    main()
