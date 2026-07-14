"""X3: is the headline error correlation rho a shared-per-perturbation-difficulty artifact?

Concern (review G3): error = 1 - Pearson-delta on a shared gene set can induce cross-predictor correlation
purely because some perturbations are hard for everyone (a noisy or small-effect target), not because
predictors share model-specific failure modes. If the correlation is mostly shared difficulty, the routing-
infeasibility law should be restated on the model-specific residual correlation.

Two decompositions on each dataset's committed per-perturbation error matrix (no re-run needed):
1. Model-specific residual correlation via LEAVE-TWO-OUT partial correlation. For each predictor pair (a, b),
   control for the mean error of the OTHER predictors (an independent estimate of that perturbation's shared
   difficulty), then correlate the residuals of e_a and e_b. This is unbiased for small M, unlike naive
   double-centering, which forces row sums to zero and injects a spurious -1/(M-1) correlation.
2. Two-way variance components: fraction of error variance attributable to the shared per-perturbation
   difficulty (row effect) vs model skill (column effect) vs model-specific residual.

Threshold (pre-stated, review G3): if the model-specific residual correlation falls below about 0.5, the
rho headline is substantially a shared-difficulty artifact and the mechanism is restated as
shared-difficulty-dominated (which is consistent with, not contrary to, the noise-limited thesis).

Run: pixi run python src/eval/run_rho_artifact.py
"""
import os

import numpy as np
import pandas as pd

EC = "results/error_correlation"
OUT = "results/error_correlation"
FILES = {
    "Gladstone_CD4": f"{EC}/per_perturbation_errors.csv",
    "Replogle_K562": f"{EC}/per_perturbation_errors_replogle.csv",
    "Norman_K562": f"{EC}/per_perturbation_errors_norman.csv",
    "Adamson_K562": f"{EC}/per_perturbation_errors_adamson.csv",
}


def err_cols(df):
    return [c for c in df.columns if c.endswith("_err") and c not in ("oracle_err", "best_fixed_err")]


def partial_corr(x, y, z):
    """corr(x, y | z): correlation of residuals after regressing each on z (z can be 1-D or 2-D)."""
    Z = np.column_stack([np.ones(len(x)), z]) if z.ndim == 1 else np.column_stack([np.ones(len(x)), z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []
    for name, path in FILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols = err_cols(df)
        E = df[cols].to_numpy()
        valid = np.isfinite(E).all(1)
        E = E[valid]
        n, M = E.shape

        # 1. raw off-diagonal Pearson (should match the committed rho)
        C = np.corrcoef(E.T)
        iu = np.triu_indices(M, 1)
        raw_rho = float(np.nanmean(C[iu]))

        # 2. leave-two-out partial correlation (model-specific residual correlation)
        partials = []
        for a in range(M):
            for b in range(a + 1, M):
                others = [c for c in range(M) if c not in (a, b)]
                if not others:
                    continue
                ctrl = E[:, others].mean(1) if len(others) > 1 else E[:, others[0]]
                pc = partial_corr(E[:, a], E[:, b], ctrl)
                if np.isfinite(pc):
                    partials.append(pc)
        model_specific_rho = float(np.mean(partials)) if partials else np.nan

        # 3. two-way variance components (shared difficulty vs model skill vs model-specific residual)
        g = E.mean()
        a_i = E.mean(1) - g                      # perturbation (shared difficulty) effect
        b_m = E.mean(0) - g                       # model (skill) effect
        resid = E - g - a_i[:, None] - b_m[None, :]
        v_shared = float(a_i.var())
        v_model = float(b_m.var())
        v_resid = float(resid.var())
        frac_shared = v_shared / (v_shared + v_resid) if (v_shared + v_resid) > 0 else np.nan

        # 4. does the shared difficulty a_i track an independent noise/difficulty proxy? use the oracle error
        #    per perturbation (a lower bound on that perturbation's irreducible difficulty) if present.
        corr_ai_oracle = np.nan
        if "oracle_err" in df.columns:
            oe = df["oracle_err"].to_numpy()[valid]
            if np.isfinite(oe).all() and oe.std() > 1e-12:
                corr_ai_oracle = float(np.corrcoef(a_i, oe)[0, 1])

        rows.append(dict(dataset=name, n=n, M=M, raw_rho=raw_rho,
                         model_specific_rho=model_specific_rho, frac_variance_shared=frac_shared,
                         corr_shared_vs_oracle=corr_ai_oracle,
                         artifact_verdict=("shared-difficulty-dominated" if model_specific_rho < 0.5
                                           else "genuine model-specific co-failure")))
        print(f"{name}: n {n} M {M} | raw rho {raw_rho:.3f} | model-specific (leave-2-out partial) "
              f"{model_specific_rho:.3f} | var-shared frac {frac_shared:.3f} | "
              f"corr(shared,oracle) {corr_ai_oracle:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "rho_artifact_control.csv"), index=False)
    print("\n=== X3 verdict ===")
    for _, r in res.iterrows():
        print(f"{r['dataset']}: {r['artifact_verdict']} (model-specific rho {r['model_specific_rho']:.3f})")
    print(f"wrote {OUT}/rho_artifact_control.csv")


if __name__ == "__main__":
    main()
