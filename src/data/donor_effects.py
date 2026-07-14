"""Recompute per-donor-group perturbation effect vectors from the pseudobulk (donor transfer axis, P2).

The 44 GB pseudobulk is streamed in row chunks so peak memory stays bounded (a shared 1 TB host with
~140 GB free). Donors are split into two disjoint groups, A = {CE0006864, CE0008162} and
B = {CE0008678, CE0010866}. For each group and culture condition we compute, per perturbed gene, the mean
log1p CPM across that gene's targeting pseudobulks minus the mean log1p CPM across the group's
non-targeting-control pseudobulks. That difference is a log-fold-change effect vector over the measured
genes. Training a predictor on group A and scoring it against group B truth gives a donor transfer axis
that is independent of the activation-state axis.

Saves results/features/donor_effects.npz. Run: pixi run python src/data/donor_effects.py
"""
import os

import anndata
import h5py
import numpy as np
import scipy.sparse as sp

PSEUDO = "data/raw/marson2025/GWCD4i.pseudobulk_merged.h5ad"
DE = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
OUT = "results/features"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
GROUP_A = {"CE0006864", "CE0008162"}
CHUNK = 20000


def main():
    os.makedirs(OUT, exist_ok=True)
    a = anndata.read_h5ad(PSEUDO, backed="r")
    obs = a.obs
    var_ids = np.asarray(a.var["gene_ids"]).astype(str)
    pgene = obs["perturbed_gene_id"].astype(str).to_numpy()
    donor = obs["donor_id"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    is_targ = (obs["guide_type"].astype(str) == "targeting").to_numpy()
    tc = obs["total_counts"].to_numpy().astype(np.float64)
    a.file.close()

    # measured-gene columns: align pseudobulk var to the DE_stats measured genes
    da = anndata.read_h5ad(DE, backed="r")
    measured = np.asarray(da.var_names).astype(str)
    da.file.close()
    col_of = {g: i for i, g in enumerate(var_ids)}
    meas_cols = np.array([col_of[g] for g in measured if g in col_of])
    meas_ids = np.array([g for g in measured if g in col_of])
    n_meas = meas_cols.size
    print(f"measured genes aligned: {n_meas}")

    group = np.where(np.isin(donor, list(GROUP_A)), 0, 1)   # 0=A, 1=B
    cidx = {c: i for i, c in enumerate(CONDS)}
    ci = np.array([cidx[c] for c in cond])

    genes = sorted(set(pgene[is_targ]) - {"nan"})
    gidx = {g: i for i, g in enumerate(genes)}
    ng = len(genes)
    print(f"perturbed genes: {ng}")

    # accumulators over measured genes only (bounded): targeting cells = gene*6 + group*3 + cond
    n_cells = ng * 6
    acc = np.zeros((n_cells, n_meas), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.float64)
    ntc = np.zeros((6, n_meas), dtype=np.float64)     # group*3 + cond
    ntc_cnt = np.zeros(6, dtype=np.float64)

    cell_id = np.full(len(obs), -1, dtype=np.int64)
    for r in range(len(obs)):
        if is_targ[r] and pgene[r] in gidx:
            cell_id[r] = gidx[pgene[r]] * 6 + group[r] * 3 + ci[r]

    f = h5py.File(PSEUDO, "r")
    g = f["X"]
    indptr = g["indptr"][:]
    dset_d, dset_i = g["data"], g["indices"]
    n_rows = len(obs)
    for r0 in range(0, n_rows, CHUNK):
        r1 = min(r0 + CHUNK, n_rows)
        s, e = int(indptr[r0]), int(indptr[r1])
        data = dset_d[s:e]
        idx = dset_i[s:e]
        iptr = indptr[r0:r1 + 1] - indptr[r0]
        X = sp.csr_matrix((data, idx, iptr), shape=(r1 - r0, len(var_ids)))
        inv = np.where(tc[r0:r1] > 0, 1e6 / tc[r0:r1], 0.0)
        X = X.multiply(inv[:, None]).tocsr()
        X.data = np.log1p(X.data)
        Xm = X[:, meas_cols]                            # restrict to measured genes
        # targeting accumulation
        cids = cell_id[r0:r1]
        tmask = cids >= 0
        if tmask.any():
            local = cids[tmask]
            uniq, invu = np.unique(local, return_inverse=True)
            S = sp.csr_matrix((np.ones(tmask.sum()), (invu, np.where(tmask)[0])),
                              shape=(uniq.size, r1 - r0))
            acc[uniq] += np.asarray((S @ Xm).todense())
            np.add.at(cnt, local, 1.0)
        # NTC accumulation
        nmask = (~is_targ[r0:r1])
        if nmask.any():
            ncell = group[r0:r1][nmask] * 3 + ci[r0:r1][nmask]
            uniq, invu = np.unique(ncell, return_inverse=True)
            S = sp.csr_matrix((np.ones(nmask.sum()), (invu, np.where(nmask)[0])),
                              shape=(uniq.size, r1 - r0))
            ntc[uniq] += np.asarray((S @ Xm).todense())
            np.add.at(ntc_cnt, ncell, 1.0)
        if (r0 // CHUNK) % 3 == 0:
            print(f"  streamed {r1}/{n_rows} rows", flush=True)
    f.close()

    ntc_mean = np.where(ntc_cnt[:, None] > 0, ntc / np.maximum(ntc_cnt[:, None], 1), np.nan)
    # effect[gene, group, cond] = mean targeting log-CPM - NTC baseline log-CPM (over measured genes)
    eff = np.full((ng, 2, 3, n_meas), np.nan, dtype=np.float32)
    for gi in range(ng):
        for grp in range(2):
            for c in range(3):
                cell = gi * 6 + grp * 3 + c
                if cnt[cell] > 0:
                    base = ntc_mean[grp * 3 + c]
                    eff[gi, grp, c] = (acc[cell] / cnt[cell] - base).astype(np.float32)
    np.savez(os.path.join(OUT, "donor_effects.npz"),
             genes=np.array(genes), meas_ids=meas_ids, conditions=np.array(CONDS),
             effect_A=eff[:, 0], effect_B=eff[:, 1])
    valid = np.isfinite(eff[:, 0, 0, 0]) & np.isfinite(eff[:, 1, 0, 0])
    print(f"\nwrote {OUT}/donor_effects.npz  effect_A{eff[:,0].shape}  genes with both groups (Rest): {valid.sum()}")


if __name__ == "__main__":
    main()
