"""X4: guide-level split-half noise floor on Gladstone (the cleanest technical replicate).

The committed Gladstone floor uses a DONOR split-half (donor variation is BIOLOGICAL), which likely
OVER-estimates the irreducible error and thus f_noise. Here we split each perturbed gene's GUIDES into two
halves, POOLED across all four donors, so donor biological variance is common to both halves and cancels in
the split-half correlation, isolating technical/guide variance. If the guide-level floor is materially lower
than the donor floor, f_noise on Gladstone is lower and the noise-limited claim must be re-attributed.

Threshold (pre-stated, review X4): if the guide-level floor drops f_noise below about 0.8 on Gladstone, the
noise-limited claim weakens and Gladstone routing infeasibility is re-attributed to correlation alone.

Same normalization as src/data/donor_effects.py (CP-1e6 + log1p, streamed CSR chunks), same top-1000 ENSG
eval genes as E2/E3 (byte-identical support), same effect definition (mean targeting log-CPM minus the
non-targeting-control baseline). Only the split axis changes: guides instead of donors.

Run: pixi run python src/eval/run_noise_ceiling_guide_gladstone.py
"""
import os

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

PSEUDO = "data/raw/marson2025/GWCD4i.pseudobulk_merged.h5ad"
EVAL = "results/error_correlation/eval_genes_gladstone.txt"
DONORF = "results/ceiling/noise_ceiling_summary.csv"
OUT = "results/ceiling"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
CHUNK = 20000
BEST_FIXED = 0.7037095542557176   # E2 best-fixed error (unchanged; f_noise = floor / best_fixed)


def main():
    a = anndata.read_h5ad(PSEUDO, backed="r")
    obs = a.obs
    var_ids = np.asarray(a.var["gene_ids"]).astype(str)
    pgene = obs["perturbed_gene_id"].astype(str).to_numpy()
    gene_name = obs["perturbed_gene_name"].astype(str).to_numpy()
    guide = obs["guide_id"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    is_targ = (obs["guide_type"].astype(str) == "targeting").to_numpy()
    tc = obs["total_counts"].to_numpy().astype(np.float64)
    a.file.close()

    eval_ids = [ln.strip() for ln in open(EVAL) if ln.strip()]
    col_of = {g: i for i, g in enumerate(var_ids)}
    top_cols = np.array([col_of[g] for g in eval_ids if g in col_of])
    print(f"top eval-gene columns mapped: {top_cols.size} of {len(eval_ids)}", flush=True)
    n_top = top_cols.size
    cidx = {c: i for i, c in enumerate(CONDS)}
    ci = np.array([cidx.get(c, -1) for c in cond])

    # assign each targeting guide of a gene to half 0/1 (pooled across donors); need >=2 guides per gene
    from collections import defaultdict
    g_guides = defaultdict(set)
    for r in range(len(obs)):
        if is_targ[r] and pgene[r] != "nan":
            g_guides[pgene[r]].add(guide[r])
    guide_half = {}
    genes = []
    for g, gs in g_guides.items():
        gl = sorted(gs)
        if len(gl) < 2:
            continue
        genes.append(g)
        for j, gd in enumerate(gl):
            guide_half[(g, gd)] = j % 2
    genes = sorted(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    ng = len(genes)
    name_of = {pgene[r]: gene_name[r] for r in range(len(obs)) if pgene[r] in gidx}
    print(f"perturbed genes with >=2 guides: {ng}", flush=True)

    # per-row cell id: gene*6 + half*3 + cond (targeting only, gene has >=2 guides)
    cell_id = np.full(len(obs), -1, dtype=np.int64)
    for r in range(len(obs)):
        if is_targ[r] and pgene[r] in gidx and ci[r] >= 0:
            h = guide_half.get((pgene[r], guide[r]))
            if h is not None:
                cell_id[r] = gidx[pgene[r]] * 6 + h * 3 + ci[r]

    acc = np.zeros((ng * 6, n_top), dtype=np.float64)
    cnt = np.zeros(ng * 6, dtype=np.float64)
    ntc = np.zeros((3, n_top), dtype=np.float64)
    ntc_cnt = np.zeros(3, dtype=np.float64)

    f = h5py.File(PSEUDO, "r")
    g = f["X"]
    indptr = g["indptr"][:]
    dset_d, dset_i = g["data"], g["indices"]
    n_rows = len(obs)
    for r0 in range(0, n_rows, CHUNK):
        r1 = min(r0 + CHUNK, n_rows)
        s, e = int(indptr[r0]), int(indptr[r1])
        X = sp.csr_matrix((dset_d[s:e], dset_i[s:e], indptr[r0:r1 + 1] - indptr[r0]),
                          shape=(r1 - r0, len(var_ids)))
        inv = np.where(tc[r0:r1] > 0, 1e6 / tc[r0:r1], 0.0)
        X = X.multiply(inv[:, None]).tocsr()
        X.data = np.log1p(X.data)
        Xm = np.asarray(X[:, top_cols].todense())
        cids = cell_id[r0:r1]
        tmask = cids >= 0
        if tmask.any():
            local = cids[tmask]
            np.add.at(acc, local, Xm[tmask])
            np.add.at(cnt, local, 1.0)
        nmask = (~is_targ[r0:r1]) & (ci[r0:r1] >= 0)
        if nmask.any():
            np.add.at(ntc, ci[r0:r1][nmask], Xm[nmask])
            np.add.at(ntc_cnt, ci[r0:r1][nmask], 1.0)
        if (r0 // CHUNK) % 3 == 0:
            print(f"  streamed {r1}/{n_rows}", flush=True)
    f.close()

    ntc_mean = np.where(ntc_cnt[:, None] > 0, ntc / np.maximum(ntc_cnt[:, None], 1), np.nan)
    rows = []
    for gi in range(ng):
        for c in range(3):
            cellA = gi * 6 + 0 * 3 + c
            cellB = gi * 6 + 1 * 3 + c
            if cnt[cellA] > 0 and cnt[cellB] > 0:
                eA = acc[cellA] / cnt[cellA] - ntc_mean[c]
                eB = acc[cellB] / cnt[cellB] - ntc_mean[c]
                m = np.isfinite(eA) & np.isfinite(eB)
                if m.sum() >= 10 and eA[m].std() > 1e-9 and eB[m].std() > 1e-9:
                    r = float(np.corrcoef(eA[m], eB[m])[0, 1])
                    rows.append(dict(gene=name_of.get(genes[gi], genes[gi]), condition=CONDS[c], r_AB=r))
    nf = pd.DataFrame(rows)
    rr = nf["r_AB"].clip(1e-6, 0.999)
    r_full = (2 * rr) / (1 + rr)
    nf["floor_sb"] = 1 - np.sqrt(r_full.clip(0, 1))
    nf.to_csv(os.path.join(OUT, "noise_ceiling_guide_gladstone_per_pert.csv"), index=False)

    mean_floor = float(nf["floor_sb"].mean())
    mean_rab = float(nf["r_AB"].mean())
    f_noise = mean_floor / BEST_FIXED
    donor = pd.read_csv(DONORF).iloc[0]
    summ = pd.DataFrame([dict(dataset="Gladstone_CD4", split="guide_level_pooled_donors", n=len(nf),
                              mean_r_AB=mean_rab, mean_floor_sb=mean_floor, best_fixed=BEST_FIXED,
                              f_noise_guide=f_noise, f_noise_donor=float(donor["f_noise_shared"]),
                              donor_floor_sb=float(donor["mean_floor_sb"]))])
    summ.to_csv(os.path.join(OUT, "noise_ceiling_guide_gladstone_summary.csv"), index=False)
    print("\n=== X4 guide-level noise floor (Gladstone) ===")
    print(f"guide-level: mean r_AB {mean_rab:.3f}, floor_sb {mean_floor:.3f}, f_noise {f_noise:.3f} (n {len(nf)})")
    print(f"donor-level (committed): floor_sb {donor['mean_floor_sb']:.3f}, f_noise {donor['f_noise_shared']:.3f}")
    verdict = ("f_noise still >= 0.8: the noise-limited claim is ROBUST to the replicate definition"
               if f_noise >= 0.8 else
               "f_noise < 0.8 under the guide-level (technical) replicate: RE-ATTRIBUTE Gladstone infeasibility")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}/noise_ceiling_guide_gladstone_summary.csv")


if __name__ == "__main__":
    main()
