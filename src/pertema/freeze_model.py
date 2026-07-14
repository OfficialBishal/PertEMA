"""Freeze the deployable PertEMA inference artifact (Section 3 bridge to the application).

Trains the gradient-boosted-tree reliability estimator on ALL Gladstone activation-transfer data (no
held-out, since this is the shipped model, not an evaluation), fits the isotonic recalibration, and writes a
versioned, self-describing bundle the application loads read-only. The bundle records the exact feature
specification, the seeds, the measured accuracy ceiling, the calibration coverage achieved in evaluation, a
training-data hash, and the model version, so every scored result is traceable and reproducible.

This artifact is for scoring the reliability of a user's predictor outputs. It is NEVER retrained on user
ground truth (invariant 2).

Run: pixi run python src/pertema/freeze_model.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
sys.path.insert(0, HERE)
from run_estimator import gbt                                       # noqa: E402
from run_transfer_estimator import CONDS, build_transfer_features   # noqa: E402

VERSION = "0.1.0"
TRE = "results/transfer"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
OUTDIR = f"app/model/pertema_model_v{VERSION}"
PREDICTOR = "mean_condition"   # the shipped estimator is fit on the mean predictor's transfer errors
SEED = 42

FEATURE_SPEC = {
    "description": "prediction-time features only, no true-effect quantity (leakage-safe)",
    "features": ["pred_magnitude",
                 "baseline_src", "dropout_src", "donor_var_src",
                 "baseline_dst", "dropout_dst", "donor_var_dst",
                 "src_onehot(Rest,Stim8hr,Stim48hr)", "dst_onehot(Rest,Stim8hr,Stim48hr)",
                 "coexpr_embedding(50d)", "training_set_similarity"],
    "target": "per-perturbation transfer error, 1 - Pearson on the source-training high-variance gene set",
}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB); emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{SEED}.csv"), dtype={"gene": str})
    df = tr[(tr["predictor"] == PREDICTOR) & (tr["transfer"])].reset_index(drop=True)
    Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
    y = df["transfer_err"].to_numpy()

    # training-set similarity feature over the full training set (deployable: nearest embedded gene)
    genes_u, inv = np.unique(gene_arr, return_inverse=True)
    emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
    good = ~np.isnan(emb_u).any(1)
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=8).fit(emb_u[good])
    su = np.full(len(genes_u), np.nan)
    d, _ = nn.kneighbors(emb_u[good]); su[np.where(good)[0]] = d[:, 1]     # nearest OTHER gene
    sim = su[inv]
    X = np.column_stack([Xb, sim])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, ytr = X[ok], y[ok]

    est = gbt(); est.fit(X, ytr)
    pred = est.predict(X)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred, ytr)

    est.save_model(os.path.join(OUTDIR, "estimator.json"))
    # isotonic map saved as (x, y) knots
    xk = np.linspace(float(pred.min()), float(pred.max()), 200)
    yk = iso.predict(xk)
    np.savez(os.path.join(OUTDIR, "calibration.npz"), iso_x=xk, iso_y=yk)

    # provenance
    data_hash = hashlib.sha256(open(os.path.join(TRE, f"transfer_errors_seed{SEED}.csv"), "rb").read()).hexdigest()[:16]
    cal = pd.read_csv("results/pertema/calibration.csv").set_index("predictor")
    prov = {
        "model_version": VERSION,
        "estimator": "xgboost gradient-boosted tree (n_estimators=300, max_depth=6, lr=0.05)",
        "trained_on": "Gladstone CD4 activation-transfer errors of the per-condition mean predictor, all folds",
        "n_training_perturbations": int(ok.sum()),
        "seeds": [42, 43, 44],
        "training_data_hash_sha256_16": data_hash,
        "feature_spec": FEATURE_SPEC,
        "measured_accuracy_ceiling_1minus_pearson": {"hit_genes": 0.731, "all_genes": 0.111},
        "evaluation_calibration": {
            "ece_raw": float(cal.loc[PREDICTOR, "ece_raw"]),
            "ece_isotonic": float(cal.loc[PREDICTOR, "ece_isotonic"]),
            "conformal_coverage": float(cal.loc[PREDICTOR, "conformal_coverage"]),
            "conformal_target": float(cal.loc[PREDICTOR, "conformal_target"]),
        },
        "honest_scope": ("Reliability estimator. It scores which predictions to trust, not which genes are "
                         "important. Gains are modest on noisy data (see manuscript). Never retrain on user truth."),
        "provenance_flag": "CLEAN",
    }
    with open(os.path.join(OUTDIR, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    with open(os.path.join(OUTDIR, "feature_spec.json"), "w") as f:
        json.dump(FEATURE_SPEC, f, indent=2)

    # verify round-trip load reproduces predictions to float epsilon
    import xgboost as xgb
    est2 = xgb.XGBRegressor(); est2.load_model(os.path.join(OUTDIR, "estimator.json"))
    maxdiff = float(np.max(np.abs(est2.predict(X) - pred)))
    print(f"froze pertema_model_v{VERSION}: n_train={int(ok.sum())}, "
          f"round-trip max abs diff {maxdiff:.2e}, ECE {prov['evaluation_calibration']['ece_isotonic']:.4f}, "
          f"coverage {prov['evaluation_calibration']['conformal_coverage']:.3f}")
    print(f"wrote {OUTDIR}/ (estimator.json, calibration.npz, provenance.json, feature_spec.json)")


if __name__ == "__main__":
    main()
