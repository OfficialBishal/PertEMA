"""E4: routing-feasibility phase boundary by simulation (the generalization move).

A negative routing result on real data is a null result. A negative plus a simulation that shows exactly
when routing can and cannot work is a mechanism. We build a controlled generative model of predictor
errors governed by two parameters and sweep both:

  rho     = cross-predictor per-perturbation error correlation (E2 measures this on real data). High rho
            means predictors err on the same perturbations, so the oracle-router headroom collapses.
  f_noise = fraction of each predictor's per-perturbation error that is irreducible measurement noise
            (E3 measures this on real data). The estimator can only rank by the reducible part, so its
            effective ranking quality is q_eff = q_max*sqrt(1-f_noise). High f_noise blinds the router.

Generative model (what, why, how): WHAT, latent per-perturbation errors e[i,m] for M predictors with
controllable correlation rho and per-predictor mean skill mu[m]; WHY, this reproduces the two forces that
decide routing (headroom from rho, ranking quality from f_noise); HOW,
  e[i,m]    = sqrt(rho)*z[i] + sqrt(1-rho)*w[i,m] + mu[m]      (z shared, w independent, unit variance)
  ehat[i,m] = q_eff*globalstd(e)[i,m] + sqrt(1-q_eff^2)*xi     (router's noisy view, keeps mean skill)
The router picks argmin_m ehat; routing gain = mean(best_fixed) - mean(routed), reported as the fraction
of the oracle headroom captured (negative = routing hurts).

Two design choices the internal review forced, both applied here:
  - The mean-skill amplitude mu_amp is CALIBRATED from the real per-perturbation errors (S_between/S_within
    from results/error_correlation/per_perturbation_errors.csv), not hardcoded, because a hardcoded value is
    outcome-determining. A mu_amp sensitivity table is printed.
  - The router's view is standardized GLOBALLY (one scalar mean/std) so relative predictor mean skill is
    preserved in the ranking signal, consistent with best_fixed/oracle which use the raw errors.

Real datasets are located on the plane by their measured coordinates and evaluated at their EXACT
coordinates (not grid-snapped). Deterministic: numpy Generator seeded per cell.

Run (after E2 and E3): pixi run python src/eval/run_routing_phase.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/simulation"
FIG = "figures"
M = 6            # predictors in the routing roster
N = 4000         # perturbations per simulation
SIM_SEEDS = [0, 1, 2]
Q_MAX = 0.6      # estimator competence on the fully-reducible part of the error (realistic, < 1)
GRID = 25


def calibrate_mu_amp(path="results/error_correlation/per_perturbation_errors.csv"):
    """Derive the simulated mean-skill amplitude from a dataset's real per-perturbation errors. WHAT, mu_amp
    for linspace(-a, a, M); WHY, the audit showed a hardcoded mu is outcome-determining, and the mean-skill
    spread is a genuine dataset property (how dominant the best predictor is); HOW, measure S_between (spread
    of predictor mean errors) and S_within (typical per-perturbation error spread) and set std(mu) =
    S_between/S_within (the sim's per-perturbation error std is 1). Measured, not tuned."""
    p = path
    if not os.path.exists(p):
        return 0.30, None
    df = pd.read_csv(p)
    ecols = [c for c in df.columns if c.endswith("_err") and c not in ("oracle_err", "best_fixed_err")]
    E = df[ecols].to_numpy()
    s_between = float(np.nanstd(np.nanmean(E, axis=0)))
    s_within = float(np.nanmean(np.nanstd(E, axis=0)))
    ratio = s_between / s_within if s_within > 1e-12 else 0.30
    std_unit = np.sqrt((M + 1) / (3 * (M - 1)))          # std of linspace(-1, 1, M)
    return float(ratio / std_unit), dict(s_between=s_between, s_within=s_within, ratio=ratio)


def simulate_cell(rho, f_noise, q_max, mu_amp, sim_seed):
    """One (rho, f_noise) cell: return (routing_gain, oracle_headroom, frac_captured)."""
    rng = np.random.default_rng(1000 * sim_seed + int(1e4 * rho) + int(1e2 * f_noise) + int(1e3 * mu_amp))
    mu = np.linspace(-mu_amp, mu_amp, M)                  # a genuine best-fixed predictor exists
    z = rng.standard_normal(N)
    w = rng.standard_normal((N, M))
    e = np.sqrt(rho) * z[:, None] + np.sqrt(max(0.0, 1 - rho)) * w + mu[None, :]
    e_std = (e - e.mean()) / (e.std() + 1e-9)             # GLOBAL standardization keeps relative mean skill
    q_eff = q_max * np.sqrt(max(0.0, 1 - f_noise))
    xi = rng.standard_normal((N, M))
    ehat = q_eff * e_std + np.sqrt(max(0.0, 1 - q_eff ** 2)) * xi
    choice = ehat.argmin(1)
    routed = e[np.arange(N), choice]
    best_fixed = e[:, e.mean(0).argmin()]
    oracle = e.min(1)
    gain = float(best_fixed.mean() - routed.mean())
    head = float(best_fixed.mean() - oracle.mean())
    frac = gain / head if head > 1e-9 else 0.0
    return gain, head, frac


def analytic_frac(rho, f_noise, q_max):
    """Tractable approximation overlaid for comparison (the simulation is primary; if they disagree, the
    simulation governs). Assumes Gaussian errors and selection dominated by the top-2 predictors."""
    q_eff = q_max * np.sqrt(max(0.0, 1 - f_noise))
    return float(np.clip(q_eff ** 2 * (1 - rho) ** 0.5, 0, 1))


def build_grid(mu_amp):
    rhos = np.linspace(0.0, 0.95, GRID)
    fs = np.linspace(0.0, 0.95, GRID)
    rows = []
    for rho in rhos:
        for f in fs:
            gs, hs, frs = zip(*[simulate_cell(rho, f, Q_MAX, mu_amp, s) for s in SIM_SEEDS])
            rows.append(dict(rho=rho, f_noise=f, routing_gain=np.mean(gs),
                             oracle_headroom=np.mean(hs), frac_captured=np.mean(frs),
                             analytic_frac=analytic_frac(rho, f, Q_MAX)))
    return pd.DataFrame(rows), rhos, fs


def place_datasets():
    """Locate the real datasets by measured coordinates. Gladstone from E2 rho + E3 f_noise (rigorous);
    Replogle rho from its error-correlation extension, f_noise from its cell-level split-half if present."""
    pts = []
    e2 = "results/error_correlation/error_correlation_summary.csv"
    e3 = "results/ceiling/noise_ceiling_summary.csv"
    if os.path.exists(e2) and os.path.exists(e3):
        pts.append(dict(dataset="Gladstone_CD4",
                        rho=float(pd.read_csv(e2)["mean_offdiag_pearson"].mean()),
                        f_noise=float(pd.read_csv(e3)["f_noise_shared"].iloc[0]),
                        errors_path="results/error_correlation/per_perturbation_errors.csv",
                        routing_verdict="empirical NEGATIVE (D6)"))
    # single-context datasets placed by their own error-correlation rho + cell-level split-half f_noise.
    # Replogle carries an empirical routing test (D6 Replogle); Norman and Adamson are boundary predictions.
    for ds, name, verdict in [("replogle", "Replogle_K562", "empirical NEGATIVE (D6 Replogle)"),
                              ("norman", "Norman_K562", "prediction (no routing test)"),
                              ("adamson", "Adamson_K562", "prediction (no routing test)")]:
        ec = f"results/error_correlation/error_correlation_{ds}.csv"
        nc = f"results/ceiling/noise_ceiling_{ds}_summary.csv"
        if os.path.exists(ec) and os.path.exists(nc):
            pts.append(dict(dataset=name,
                            rho=float(pd.read_csv(ec)["mean_offdiag_pearson"].iloc[0]),
                            f_noise=float(pd.read_csv(nc)["f_noise_shared"].iloc[0]),
                            errors_path=f"results/error_correlation/per_perturbation_errors_{ds}.csv",
                            routing_verdict=verdict))
    return pd.DataFrame(pts)


def render(df, rhos, fs, pts):
    grid = df.pivot(index="f_noise", columns="rho", values="frac_captured").values
    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu",
                   extent=[rhos.min(), rhos.max(), fs.min(), fs.max()], vmin=-vmax, vmax=vmax)
    cs = ax.contour(rhos, fs, grid, levels=[0.0], colors="#000000", linewidths=1.6)
    ax.clabel(cs, fmt={0.0: "routing breaks even"}, fontsize=8)
    for _, r in pts.iterrows():
        cap = r.get("sim_capture_exact", 0.0)
        col = "#D55E00" if cap < 0 else "#009E73"       # red if routing hurts at its own mean-skill spread
        ax.scatter([r["rho"]], [r["f_noise"]], s=175, marker="*", color=col,
                   edgecolor="white", linewidth=1.3, zorder=6)
        ax.annotate(f"{r['dataset']} ({cap:+.2f})", (r["rho"], r["f_noise"]), textcoords="offset points",
                    xytext=(-14, -22), fontsize=8, color="#111111", ha="right", zorder=6,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
                    bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.9))
    ax.set_xlabel("cross-predictor error correlation  rho  (E2)")
    ax.set_ylabel("shared measurement-noise fraction  f_noise  (E3)")
    ax.set_title("Routing-feasibility phase diagram (background at a representative mean-skill spread,\n"
                 "stars use each dataset's measured spread, red = routing hurts)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, label="fraction of oracle headroom captured")
    fig.tight_layout()
    p = os.path.join(FIG, "routing_phase_diagram.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    mu_amp, mu_info = calibrate_mu_amp()          # representative spread for the background grid (Gladstone)
    df, rhos, fs = build_grid(mu_amp)
    df.to_csv(os.path.join(OUT, "routing_phase_grid.csv"), index=False)
    pts = place_datasets()
    # each dataset is evaluated at its exact coordinates with ITS OWN measured mean-skill spread (mu_amp),
    # a third measured dataset property, not the representative grid value.
    for i, r in pts.iterrows():
        ep = r.get("errors_path")
        mu_ds = calibrate_mu_amp(ep)[0] if isinstance(ep, str) and os.path.exists(ep) else mu_amp
        pts.loc[i, "mu_amp"] = mu_ds
        pts.loc[i, "sim_capture_exact"] = float(np.mean(
            [simulate_cell(r["rho"], r["f_noise"], Q_MAX, mu_ds, s)[2] for s in SIM_SEEDS]))
    pts.to_csv(os.path.join(OUT, "datasets_on_phase.csv"), index=False)
    render(df, rhos, fs, pts)

    print("\n=== E4 routing-feasibility phase boundary (simulation, M=%d, N=%d, q_max=%.2f) ===" % (M, N, Q_MAX))
    if mu_info:
        print(f"mean-skill amplitude mu_amp={mu_amp:.3f} CALIBRATED from data: S_between {mu_info['s_between']:.4f}"
              f" / S_within {mu_info['s_within']:.4f} = ratio {mu_info['ratio']:.3f} (not hardcoded)")
    feas = df[df["frac_captured"] > 0.0]
    print(f"cells where routing helps (capture > 0): {len(feas)}/{len(df)}, at low rho and low f_noise")
    for _, r in pts.iterrows():
        region = "INFEASIBLE (routing hurts)" if r["sim_capture_exact"] < 0.0 else "feasible"
        print(f"{r['dataset']}: rho={r['rho']:.3f} f_noise={r['f_noise']:.3f} mu_amp={r['mu_amp']:.3f} -> "
              f"exact-coord capture {r['sim_capture_exact']:+.3f} ({region}), empirical {r['routing_verdict']}")
    if len(pts):
        g = pts.iloc[0]
        print(f"mu_amp sensitivity at {g['dataset']} (rho {g['rho']:.3f}, f_noise {g['f_noise']:.3f}):")
        for a in sorted(set([0.0, 0.03, 0.10, round(mu_amp, 3), 0.40])):
            v = np.mean([simulate_cell(g["rho"], g["f_noise"], Q_MAX, a, s)[2] for s in SIM_SEEDS])
            tag = " <- calibrated" if abs(a - round(mu_amp, 3)) < 1e-9 else ""
            print(f"  mu_amp {a:.3f}: capture {v:+.3f}{tag}")
    print(f"wrote {OUT}/routing_phase_grid.csv, datasets_on_phase.csv")
    return df, pts


if __name__ == "__main__":
    main()
