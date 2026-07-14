"""P15: add the error-correlation / routing-feasibility mechanism layer to the open benchmark resource.

Consolidates the per-dataset mechanism metrics (E2 error correlation and effective ensemble size, E3
shared-noise fraction, the E4 mean-skill spread and simulated routing capture, and the routing verdict)
into one table, results/benchmark/mechanism_metrics.csv, so the open benchmark includes the error-correlation
layer and the simulation alongside the reliability and accuracy layers.

Run: pixi run python src/pertema/build_mechanism_layer.py
"""
import os

import numpy as np
import pandas as pd

OUT = "results/benchmark"
NEFF = {
    "Gladstone_CD4": ("results/error_correlation/error_correlation_summary.csv", "N_eff", "mean"),
    "Replogle_K562": ("results/error_correlation/error_correlation_replogle.csv", "N_eff", "iloc0"),
    "Norman_K562": ("results/error_correlation/error_correlation_norman.csv", "N_eff", "iloc0"),
    "Adamson_K562": ("results/error_correlation/error_correlation_adamson.csv", "N_eff", "iloc0"),
}


def get_neff(spec):
    path, col, how = spec
    if not os.path.exists(path):
        return np.nan
    v = pd.read_csv(path)[col]
    return float(v.mean()) if how == "mean" else float(v.iloc[0])


def main():
    os.makedirs(OUT, exist_ok=True)
    pts = pd.read_csv("results/simulation/datasets_on_phase.csv")
    # theory layer: break-even router quality r* and realized r (X2), keyed by dataset
    theory = pd.read_csv("results/theory/R4_datasets_theory_plane.csv") if \
        os.path.exists("results/theory/R4_datasets_theory_plane.csv") else None
    # empirical routing test outcomes (X1 tested Adamson; Gladstone and Replogle were tested; Norman predicted)
    EMPIRICAL = {
        "Gladstone_CD4": "NEGATIVE (tested, D6): noise-dominated, no headroom",
        "Replogle_K562": "NEGATIVE (tested, D6): dominant best predictor, no headroom",
        "Adamson_K562": "NEGATIVE (tested, X1): feasible geometry (headroom +0.088), insufficient router (r 0.215 < r* 0.457)",
        "Norman_K562": "prediction (no routing test)",
    }

    def theory_val(ds, col):
        if theory is None:
            return np.nan
        t = theory[theory.dataset == ds]
        if not len(t) or pd.isna(t[col].iloc[0]):
            return np.nan
        return float(t[col].iloc[0])

    rows = []
    for _, r in pts.iterrows():
        ds = r["dataset"]
        rstar = theory_val(ds, "break_even_r")
        realized = theory_val(ds, "realized_r")
        rows.append(dict(
            dataset=ds,
            error_correlation_rho=round(float(r["rho"]), 4),
            N_eff=round(get_neff(NEFF.get(ds, (None, None, None))) if ds in NEFF else np.nan, 3),
            shared_noise_fraction_f=round(float(r["f_noise"]), 4),
            mean_skill_spread_mu=round(float(r["mu_amp"]), 4),
            simulated_geometry_capture=round(float(r["sim_capture_exact"]), 4),
            break_even_router_r=round(rstar, 3) if not np.isnan(rstar) else np.nan,
            realized_router_r=round(realized, 3) if not np.isnan(realized) else np.nan,
            geometry_feasible=("yes" if r["sim_capture_exact"] > 0 else "no"),
            deployable_routing=("infeasible" if EMPIRICAL.get(ds, "").startswith("NEGATIVE") else "not tested"),
            empirical_routing=EMPIRICAL.get(ds, "prediction"),
        ))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, "mechanism_metrics.csv"), index=False)
    print("=== open-benchmark mechanism layer (E2 error corr, E3 noise, E4 phase, X1 test, X2 theory) ===")
    print(m.to_string(index=False))
    print(f"\nwrote {OUT}/mechanism_metrics.csv")


if __name__ == "__main__":
    main()
