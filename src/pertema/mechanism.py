"""Mechanism and feature-group ablation for the context-transfer estimator (A5, P6, P4).

Three analyses on the mean-condition transfer task (seed 42, gene-disjoint OOF):
  1. Feature-group importance: mean xgboost gain per group across folds.
  2. Drop-one-group ablation: AURC when each group is removed (bigger AURC rise = more useful group).
  3. Mechanistic test: does transfer error track how much a gene's control baseline expression shifts
     between the source and destination activation states? Spearman of transfer error against
     |baseline_dst - baseline_src|, and against control-state donor variance.

Run: pixi run python src/pertema/mechanism.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
sys.path.insert(0, HERE)
from metrics import aurc                                              # noqa: E402
from run_estimator import best_aurc, gbt                              # noqa: E402
from run_transfer_estimator import build_transfer_features, CONDS     # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEED = 42

FEATNAMES = (["pred_magnitude", "baseline_src", "dropout_src", "donor_var_src",
              "baseline_dst", "dropout_dst", "donor_var_dst"]
             + [f"{p}_{c}" for c in CONDS for p in ("src", "dst")]
             + [f"emb_{i}" for i in range(50)] + ["similarity"])
GROUPS = {
    "predicted_magnitude": ["pred_magnitude"],
    "source_baseline": ["baseline_src", "dropout_src", "donor_var_src"],
    "dest_baseline": ["baseline_dst", "dropout_dst", "donor_var_dst"],
    "transfer_pair": [f"{p}_{c}" for c in CONDS for p in ("src", "dst")],
    "coexpr_embedding": [f"emb_{i}" for i in range(50)],
    "training_similarity": ["similarity"],
}


def oof_with_columns(Xfull, y, fold, keep_cols):
    pred = np.full(len(y), np.nan)
    imp = np.zeros(len(keep_cols))
    for k in np.unique(fold):
        tr = np.where(fold != k)[0]
        te = np.where(fold == k)[0]
        m = gbt().fit(Xfull[tr][:, keep_cols], y[tr])
        pred[te] = m.predict(Xfull[te][:, keep_cols])
        imp += m.feature_importances_
    return pred, imp / len(np.unique(fold))


def main():
    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    cidx = {c: i for i, c in enumerate(CONDS)}

    tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{SEED}.csv"), dtype={"gene": str})
    df = tr[tr["transfer"] & (tr["predictor"] == "mean_condition")].reset_index(drop=True)
    Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
    y = df["transfer_err"].to_numpy()
    fold = df["fold"].to_numpy()

    # per-fold similarity (leakage-safe), appended as the last column
    genes_u, inv = np.unique(gene_arr, return_inverse=True)
    emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
    good_u = ~np.isnan(emb_u).any(1)
    fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold
    sim = np.full(len(df), np.nan)
    for k in np.unique(fold):
        tr_u = np.where((fold_u != k) & good_u)[0]
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
        su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
        su[q] = nn.kneighbors(emb_u[q])[0].ravel()
        te = np.where(fold == k)[0]
        sim[te] = su[inv][te]
    Xfull = np.column_stack([Xb, sim])

    # 1. full model importance
    all_cols = np.arange(Xfull.shape[1])
    _, imp = oof_with_columns(Xfull, y, fold, all_cols)
    grp_imp = {g: float(sum(imp[FEATNAMES.index(f)] for f in fs)) for g, fs in GROUPS.items()}
    tot = sum(grp_imp.values())
    grp_imp = {g: v / tot for g, v in grp_imp.items()}

    # 2. drop-one-group ablation
    full_aurc = best_aurc(y, -oof_with_columns(Xfull, y, fold, all_cols)[0])
    abl = {}
    for g, fs in GROUPS.items():
        drop = set(FEATNAMES.index(f) for f in fs)
        keep = np.array([c for c in all_cols if c not in drop])
        p, _ = oof_with_columns(Xfull, y, fold, keep)
        abl[g] = best_aurc(y, -p)

    # 3. mechanistic test
    gi = df["gene"].map(gene_col).to_numpy()
    si = df["src"].map(cidx).to_numpy(); di = df["dst"].map(cidx).to_numpy()
    ok = ~np.isnan(gi)
    state_change = np.full(len(df), np.nan)
    dvar = np.full(len(df), np.nan)
    idx = gi[ok].astype(int)
    state_change[ok] = np.abs(baseline[di[ok], idx] - baseline[si[ok], idx])
    dvar[ok] = 0.5 * (donor_var[si[ok], idx] + donor_var[di[ok], idx])
    mfin = ok & np.isfinite(y)
    r_state = spearmanr(state_change[mfin], y[mfin])
    r_dvar = spearmanr(dvar[mfin], y[mfin])

    print("=== feature-group importance (share of xgboost gain) ===")
    for g, v in sorted(grp_imp.items(), key=lambda x: -x[1]):
        print(f"  {g:22s} {v:.3f}")
    print(f"\n=== drop-one-group ablation (AURC; full={full_aurc:.4f}, higher after drop = group helps) ===")
    for g, v in sorted(abl.items(), key=lambda x: -x[1]):
        print(f"  drop {g:22s} AURC={v:.4f}  (delta {v-full_aurc:+.4f})")
    print("\n=== mechanism: transfer error vs biological drivers (Spearman) ===")
    print(f"  |baseline_dst - baseline_src| (activation-state expression shift): "
          f"rho={r_state.correlation:.3f} p={r_state.pvalue:.1e}")
    print(f"  control-state donor variance: rho={r_dvar.correlation:.3f} p={r_dvar.pvalue:.1e}")

    pd.DataFrame([{"group": g, "gain_share": grp_imp[g], "aurc_drop": abl[g],
                   "delta_vs_full": abl[g] - full_aurc} for g in GROUPS]
                 ).to_csv(os.path.join(OUT, "mechanism_feature_groups.csv"), index=False)
    pd.DataFrame([{"driver": "state_expression_shift", "spearman": r_state.correlation, "p": r_state.pvalue},
                  {"driver": "control_donor_variance", "spearman": r_dvar.correlation, "p": r_dvar.pvalue}]
                 ).to_csv(os.path.join(OUT, "mechanism_drivers.csv"), index=False)
    print(f"\nwrote {OUT}/mechanism_feature_groups.csv and mechanism_drivers.csv")


if __name__ == "__main__":
    main()
