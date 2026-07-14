"""Control-state baseline features from non-targeting-control pseudobulks.

These are legitimate prediction-time features: they describe the unperturbed reference state and use
no perturbation's true effect. Per culture condition we compute, over 18,129 genes:
  baseline   : mean log1p CPM across NTC pseudobulks (baseline expression level)
  dropout    : fraction of NTC pseudobulks with zero counts for the gene
  donor_var  : variance across the 4 donors of the per-donor mean log1p CPM (control-state instability)

Saved to results/features/control_baseline.npz. Run: pixi run python src/data/control_baseline.py
"""
import os

import anndata
import h5py
import numpy as np
import scipy.sparse as sp

PATH = "data/raw/marson2025/GWCD4i.pseudobulk_merged.h5ad"
OUT = "results/features"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]


def read_csr_rows(path, rows):
    """Gather specific rows of the on-disk CSR X into an in-memory csr_matrix (rows must be sorted)."""
    with h5py.File(path, "r") as f:
        g = f["X"]
        ncols = int(g.attrs["shape"][1])
        indptr = g["indptr"][:]
        d, idx = g["data"], g["indices"]
        data_parts, ind_parts, new_indptr = [], [], [0]
        for r in rows:
            s, e = int(indptr[r]), int(indptr[r + 1])
            data_parts.append(d[s:e])
            ind_parts.append(idx[s:e])
            new_indptr.append(new_indptr[-1] + (e - s))
    X = sp.csr_matrix((np.concatenate(data_parts), np.concatenate(ind_parts),
                       np.array(new_indptr)), shape=(len(rows), ncols))
    return X


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(PATH, backed="r")
    obs = a.obs
    ntc_mask = (obs["guide_type"].astype(str) == "non-targeting").to_numpy()
    print(f"total pseudobulks {a.n_obs}, non-targeting {ntc_mask.sum()}")
    ntc_rows = np.where(ntc_mask)[0]
    var_names = np.asarray(a.var_names)
    ntc_obs = obs.iloc[ntc_rows].reset_index(drop=True)
    a.file.close()

    X = read_csr_rows(PATH, ntc_rows).astype(np.float64)

    class _S:  # light holder to keep the rest of main unchanged
        pass
    sub = _S()
    sub.X = X
    sub.obs = ntc_obs
    sub.var_names = var_names
    tc = sub.obs["total_counts"].to_numpy()
    inv = np.where(tc > 0, 1e6 / tc, 0.0)
    X = X.multiply(inv[:, None]).tocsr()      # CPM per pseudobulk
    X.data = np.log1p(X.data)                  # log1p CPM
    cond = sub.obs["culture_condition"].astype(str).to_numpy()
    donor = sub.obs["donor_id"].astype(str).to_numpy()
    genes = np.asarray(sub.var_names)
    n_genes = genes.shape[0]

    baseline = np.zeros((len(CONDS), n_genes))
    dropout = np.zeros((len(CONDS), n_genes))
    donor_var = np.zeros((len(CONDS), n_genes))
    for ci, c in enumerate(CONDS):
        m = cond == c
        Xc = X[m]
        baseline[ci] = np.asarray(Xc.mean(0)).ravel()
        nz = np.asarray((Xc > 0).sum(0)).ravel()
        dropout[ci] = 1.0 - nz / max(1, m.sum())
        dmeans = [np.asarray(X[m & (donor == d)].mean(0)).ravel() for d in np.unique(donor[m])]
        donor_var[ci] = np.var(np.stack(dmeans), axis=0)
        print(f"{c}: n_ntc={m.sum()} donors={np.unique(donor[m]).size} "
              f"baseline median={np.median(baseline[ci]):.3f} dropout median={np.median(dropout[ci]):.3f}")

    np.savez(os.path.join(OUT, "control_baseline.npz"),
             genes=np.asarray(genes).astype(str), conditions=np.array(CONDS),
             baseline=baseline.astype(np.float32),
             dropout=dropout.astype(np.float32),
             donor_var=donor_var.astype(np.float32))
    print(f"\nwrote {OUT}/control_baseline.npz  baseline{baseline.shape}")


if __name__ == "__main__":
    main()
