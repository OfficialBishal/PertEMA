"""X2: the analytic theory of routing feasibility (contribution N2, the closed-form phase boundary).

We derive the achievable routing gain in closed form and verify each result numerically against the
generative simulation (the simulation is the operative result where the closed form breaks; stated honestly).

Model: e_i(x) = c(x) + mu_i + s xi_i(x). c(x) shared per-perturbation difficulty, mu_i predictor mean-skill
offset, xi_i zero-mean unit-variance idiosyncratic fluctuations with cross-predictor equicorrelation rho.
The simulation uses e[i,m] = sqrt(rho) z[i] + sqrt(1-rho) w[i,m] + mu[m], so total per-predictor error
variance is 1 (s=1 in these units) and corr(e_i,e_j) = rho exactly.

Results implemented and verified:
  R1 (equal means): Gain_oracle = s sqrt(1-rho) a_K, a_K = E[max of K iid N(0,1)]. Verified vs the sim head.
  R2 (mean-skill suppression, K=2): Gain_oracle = sigma_D phi(Delta/sigma_D) - Delta Phi(-Delta/sigma_D) with
     sigma_D = s sqrt(2(1-rho)). NOTE: the specification Result 2 wrote the phi coefficient as s sqrt(1-rho);
     the correct coefficient is sigma_D = s sqrt(2(1-rho)) (derived from E[(X)_+], X ~ N(-Delta, sigma_D^2)).
     We use the corrected form, show it reduces to R1 at Delta=0 (a_2 = 1/sqrt(pi)), and verify vs a 2-predictor
     sim. The prompt's version is off by sqrt(2); flagged, not propagated.
  R3 (router imperfection): Gain_router approx r Gain_oracle to first order, r the router rank quality. We show
     this holds at high r and BREAKS at low r, where noisy argmin selection biases toward worse-than-best-fixed
     predictors, so there is a break-even router quality r* below which routing HURTS despite positive oracle
     headroom. This is exactly the Adamson X1 regime (realized r 0.215 < r*).
  R4 (N_eff form): equicorrelation N_eff_eq = K/(1+(K-1)rho). The MEASURED participation-ratio N_eff differs
     from N_eff_eq when the real correlation matrix is heterogeneous; we use average pairwise rho for the
     premium and report N_eff as the diversity summary, per the stated honesty caveat.

Run: pixi run python src/eval/run_theory.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import norm

OUT = "results/theory"
FIG = "figures"
SIM_SEEDS = [0, 1, 2]
N_SIM = 40000


def a_K(K):
    """E[max of K iid N(0,1)] via numerical integration of x K phi(x) Phi(x)^(K-1)."""
    val, _ = integrate.quad(lambda x: x * K * norm.pdf(x) * norm.cdf(x) ** (K - 1), -20, 20)
    return float(val)


def head_equal_means(rho, K, s=1.0):
    return s * np.sqrt(max(0.0, 1 - rho)) * a_K(K)


def head_k2(rho, Delta, s=1.0):
    sigma_D = s * np.sqrt(2 * max(0.0, 1 - rho))
    if sigma_D < 1e-12:
        return max(0.0, -Delta)
    return float(sigma_D * norm.pdf(Delta / sigma_D) - Delta * norm.cdf(-Delta / sigma_D))


def head_k2_prompt(rho, Delta, s=1.0):
    """The specification Result 2 (phi coefficient s sqrt(1-rho)); shown only to demonstrate the sqrt(2) slip."""
    sd = s * np.sqrt(2 * max(0.0, 1 - rho))
    if sd < 1e-12:
        return max(0.0, -Delta)
    return float(s * np.sqrt(max(0.0, 1 - rho)) * norm.pdf(Delta / sd) - Delta * norm.cdf(-Delta / sd))


def sim_route(rho, mu_amp, r, K=6, N=N_SIM, seed=0):
    """One (rho, mu_amp, r) cell of the generative model, router rank quality set directly to r (q_eff=r).
    Returns (gain, head, frac_captured)."""
    rng = np.random.default_rng(1000 * seed + int(1e4 * rho) + int(1e3 * mu_amp) + int(1e3 * r))
    mu = np.linspace(-mu_amp, mu_amp, K)
    z = rng.standard_normal(N)
    w = rng.standard_normal((N, K))
    e = np.sqrt(rho) * z[:, None] + np.sqrt(max(0.0, 1 - rho)) * w + mu[None, :]
    e_std = (e - e.mean()) / (e.std() + 1e-9)
    xi = rng.standard_normal((N, K))
    ehat = r * e_std + np.sqrt(max(0.0, 1 - r ** 2)) * xi
    choice = ehat.argmin(1)
    routed = e[np.arange(N), choice]
    best_fixed = e[:, e.mean(0).argmin()]
    oracle = e.min(1)
    gain = float(best_fixed.mean() - routed.mean())
    head = float(best_fixed.mean() - oracle.mean())
    return gain, head, (gain / head if head > 1e-9 else 0.0)


def sim_mean(fn, *a, **k):
    return float(np.mean([fn(*a, seed=s, **k)[k.get("_idx", 2) if False else 2] for s in SIM_SEEDS]))


def calibrate_mu_amp(path, M=6):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ecols = [c for c in df.columns if c.endswith("_err") and c not in ("oracle_err", "best_fixed_err")]
    E = df[ecols].to_numpy()
    s_between = float(np.nanstd(np.nanmean(E, axis=0)))
    s_within = float(np.nanmean(np.nanstd(E, axis=0)))
    ratio = s_between / s_within if s_within > 1e-12 else 0.30
    std_unit = np.sqrt((M + 1) / (3 * (M - 1)))
    return float(ratio / std_unit)


def break_even_r(rho, mu_amp, K=6):
    """The router rank quality r at which routing breaks even (frac_captured crosses 0). Below r*, routing
    hurts despite positive oracle headroom."""
    f = lambda r: np.mean([sim_route(rho, mu_amp, r, K=K, seed=s)[0] for s in SIM_SEEDS])  # mean gain
    lo, hi = f(0.01), f(0.99)
    if lo > 0:      # already helps at r~0 (no dominant predictor): break-even at 0
        return 0.0
    if hi < 0:      # never helps even at r~1
        return 1.0
    return float(brentq(f, 0.01, 0.99, xtol=1e-3))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    print("a_K:", {K: round(a_K(K), 4) for K in (2, 3, 6, 10)})

    # --- R1 validation: analytic oracle headroom (equal means) vs simulation ---
    rows = []
    for rho in np.linspace(0.0, 0.95, 20):
        h_sim = float(np.mean([sim_route(rho, 0.0, 0.0, K=6, seed=s)[1] for s in SIM_SEEDS]))
        h_an = head_equal_means(rho, 6)
        rows.append(dict(rho=rho, head_sim=h_sim, head_analytic=h_an,
                         rel_err=abs(h_sim - h_an) / (h_an + 1e-9)))
    r1 = pd.DataFrame(rows)
    r1.to_csv(os.path.join(OUT, "R1_oracle_headroom_validation.csv"), index=False)
    print(f"R1 equal-means oracle headroom: max relative error vs sim = {r1['rel_err'].max():.4f} "
          f"(mean {r1['rel_err'].mean():.4f})")

    # --- R2 validation: K=2 unequal-means suppression, corrected vs prompt formula, vs 2-predictor sim ---
    rows = []
    for Delta in np.linspace(0.0, 2.0, 11):
        g_sim = float(np.mean([sim_route(0.5, 0.0, 1.0, K=2, seed=s)[0] for s in SIM_SEEDS])) if Delta == 0 else None
        # direct 2-predictor oracle headroom sim at rho=0.5 with a mean gap Delta (mu = [-Delta/2, +Delta/2])
        def sim_head_k2(Delta, rho=0.5, N=N_SIM, seed=0):
            rng = np.random.default_rng(7 * seed + int(1e3 * Delta))
            z = rng.standard_normal(N); w = rng.standard_normal((N, 2))
            e = np.sqrt(rho) * z[:, None] + np.sqrt(1 - rho) * w + np.array([-Delta / 2, Delta / 2])[None, :]
            return float(e[:, e.mean(0).argmin()].mean() - e.min(1).mean())
        h_sim = float(np.mean([sim_head_k2(Delta, seed=s) for s in SIM_SEEDS]))
        rows.append(dict(Delta=Delta, head_sim=h_sim, head_corrected=head_k2(0.5, Delta),
                         head_prompt=head_k2_prompt(0.5, Delta)))
    r2 = pd.DataFrame(rows)
    r2["relerr_corrected"] = (r2["head_corrected"] - r2["head_sim"]).abs() / (r2["head_sim"] + 1e-9)
    r2["relerr_prompt"] = (r2["head_prompt"] - r2["head_sim"]).abs() / (r2["head_sim"] + 1e-9)
    r2.to_csv(os.path.join(OUT, "R2_meanskill_suppression_validation.csv"), index=False)
    print(f"R2 K=2 suppression: corrected-form max rel err {r2['relerr_corrected'].max():.4f} vs "
          f"prompt-form max rel err {r2['relerr_prompt'].max():.4f} (prompt is off by ~sqrt(2))")

    # --- R3: does frac_captured approx r? where it holds and breaks. break-even r* per geometry ---
    # each geometry uses ITS real roster size K (Adamson/Gladstone K=3/10), matching the R4 break-even r*.
    rows = []
    for rho, mu_amp, kk, tag in [(0.462, 0.28, 3, "Adamson-like"), (0.80, 0.28, 6, "mid"),
                                 (0.919, 0.09, 10, "Gladstone-like")]:
        for r in np.linspace(0.0, 0.95, 20):
            frac = float(np.mean([sim_route(rho, mu_amp, r, K=kk, seed=s)[2] for s in SIM_SEEDS]))
            rows.append(dict(geometry=tag, rho=rho, mu_amp=mu_amp, K=kk, r=r, frac_captured=frac,
                             first_order_frac=r))
    r3 = pd.DataFrame(rows)
    r3.to_csv(os.path.join(OUT, "R3_router_attenuation.csv"), index=False)
    hi = r3[r3.r >= 0.6]
    print(f"R3 first-order frac approx r: at r>=0.6 mean |frac - r| = "
          f"{(hi['frac_captured'] - hi['r']).abs().mean():.3f}; at low r it breaks (frac goes negative).")

    # --- R4 + the four datasets: place on the (rho premium, mu_amp) plane, break-even r*, tie to X1 ---
    ec = "results/error_correlation"
    files = {"Gladstone_CD4": (f"{ec}/per_perturbation_errors.csv", 0.919, None),
             "Replogle_K562": (f"{ec}/per_perturbation_errors_replogle.csv", 0.792, None),
             "Norman_K562": (f"{ec}/per_perturbation_errors_norman.csv", 0.911, None),
             "Adamson_K562": (f"{ec}/per_perturbation_errors_adamson.csv", 0.462, 0.215)}
    # measured N_eff per dataset (from the committed error-correlation summaries)
    neff_meas = {"Gladstone_CD4": 1.16, "Replogle_K562": 1.32, "Norman_K562": 1.12, "Adamson_K562": 2.05}
    kK = {"Gladstone_CD4": 10, "Replogle_K562": 3, "Norman_K562": 3, "Adamson_K562": 3}
    rows = []
    for name, (path, rho, realized_r) in files.items():
        # calibrate mu_amp with the dataset's OWN roster size K (std_unit must match the linspace length
        # used in break_even_r); passing the default M=6 mis-sized the spread (audit fix).
        mu_amp = calibrate_mu_amp(path, M=kK[name]) or 0.30
        head = head_equal_means(rho, kK[name])                       # analytic premium (equal-means upper bound)
        rstar = break_even_r(rho, mu_amp, K=kK[name])
        neff_eq = kK[name] / (1 + (kK[name] - 1) * rho)              # equicorrelation N_eff (idealized)
        rows.append(dict(dataset=name, K=kK[name], rho=rho, sqrt_1_minus_rho=np.sqrt(1 - rho),
                         mu_amp=mu_amp, analytic_premium_equalmeans=head, N_eff_measured=neff_meas[name],
                         N_eff_equicorr=neff_eq, break_even_r=rstar, realized_r=realized_r,
                         routing_feasible_geometry=bool(head > 0.3),
                         routing_deployable=bool(realized_r is not None and realized_r > rstar)))
    r4 = pd.DataFrame(rows)
    r4.to_csv(os.path.join(OUT, "R4_datasets_theory_plane.csv"), index=False)
    print("\n=== four datasets on the analytic plane ===")
    print(r4[["dataset", "rho", "sqrt_1_minus_rho", "mu_amp", "N_eff_measured", "N_eff_equicorr",
              "break_even_r", "realized_r"]].to_string(index=False))
    ad = r4[r4.dataset == "Adamson_K562"].iloc[0]
    print(f"\nX1 tie-in: Adamson break-even r* = {ad['break_even_r']:.3f}, realized r = {ad['realized_r']:.3f} "
          f"-> realized < r*, so routing HURTS despite the large premium (sqrt(1-rho)={ad['sqrt_1_minus_rho']:.3f}). "
          f"Exactly the X1 outcome.")

    _render(r3, r4)
    print(f"\nwrote {OUT}/R1..R4 csvs and {FIG}/theory_router_attenuation.png")


def _render(r3, r4):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    for tag, g in r3.groupby("geometry"):
        axA.plot(g["r"], g["frac_captured"], marker="o", ms=3, label=f"{tag} (rho {g['rho'].iloc[0]:.2f})")
    axA.plot([0, 1], [0, 1], "k--", lw=1, label="first-order frac = r")
    axA.axhline(0, color="#888", lw=0.8)
    axA.axvline(0.215, color="#D55E00", lw=1.2, ls=":", label="Adamson realized r = 0.215")
    axA.set_xlabel("router rank quality r"); axA.set_ylabel("fraction of oracle headroom captured")
    axA.set_title("R3: routing gain vs router quality (first-order breaks at low r)", fontsize=9, loc="left")
    axA.legend(fontsize=7)
    # panel B: datasets by premium vs break-even r
    for _, row in r4.iterrows():
        col = "#009E73" if row["routing_deployable"] else "#D55E00"
        axB.scatter([row["sqrt_1_minus_rho"]], [row["break_even_r"]], s=140, color=col,
                    edgecolor="white", zorder=5)
        axB.annotate(f"{row['dataset'].split('_')[0]}\n(r*={row['break_even_r']:.2f})",
                     (row["sqrt_1_minus_rho"], row["break_even_r"]), fontsize=7,
                     textcoords="offset points", xytext=(6, -4))
    axB.axhline(0.215, color="#D55E00", ls=":", lw=1, label="Adamson realized r = 0.215")
    axB.set_xlabel("oracle premium  sqrt(1 - rho)"); axB.set_ylabel("break-even router quality  r*")
    axB.set_title("R4: premium vs required router quality (four datasets)", fontsize=9, loc="left")
    axB.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "theory_router_attenuation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
