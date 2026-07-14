"""Loader for the genome-scale CD4 T cell DE_stats AnnData.

Schema verified from ground truth (src/data/inspect_h5ad.py on GWCD4i.DE_stats.h5ad):
  obs  : 33,983 rows, one per (target gene, culture_condition). Key columns:
         target_contrast_gene_name (symbol), target_contrast (Ensembl), culture_condition
         (Rest / Stim8hr / Stim48hr), ontarget_effect_size, ontarget_significant, n_downstream,
         guide_correlation_all / guide_correlation_signif (cross-guide reproducibility),
         donor_correlation_all_mean / donor_correlation_hits_mean (cross-donor reproducibility).
  var  : 10,282 measured genes (gene_ids = Ensembl, gene_name = symbol).
  layers: log_fc, zscore, p_value, adj_p_value, baseMean, lfcSE, each (33983, 10282) float64.
         The true perturbation effect vector is layers['log_fc'][i] (delta LFC) or layers['zscore'][i].
  X is empty; all signal is in layers.

X is never loaded. Only the requested layers are read, cast to float32 (~1.4 GB each) to keep memory sane.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata
import h5py
import numpy as np
import pandas as pd

DEFAULT_PATH = "data/raw/marson2025/GWCD4i.DE_stats.h5ad"
CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")


@dataclass
class DEData:
    obs: pd.DataFrame            # 33983 x metadata
    gene_ids: np.ndarray         # (10282,) Ensembl ids of measured genes
    gene_names: np.ndarray       # (10282,) symbols
    layers: dict                 # name -> (33983, 10282) float32

    @property
    def n_obs(self):
        return self.obs.shape[0]

    @property
    def n_genes(self):
        return self.gene_ids.shape[0]


def load_de_stats(path=DEFAULT_PATH, layers=("log_fc", "zscore"), dtype="float32"):
    a = anndata.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    gene_ids = np.asarray(a.var_names)
    gene_names = np.asarray(a.var["gene_name"]) if "gene_name" in a.var else gene_ids
    a.file.close()
    out = {}
    with h5py.File(path, "r") as f:
        for name in layers:
            out[name] = f["layers/" + name][:].astype(dtype)
    return DEData(obs=obs, gene_ids=gene_ids, gene_names=gene_names, layers=out)


def ontarget_index(de):
    """For each obs row, the column index of its own perturbed gene among measured genes, or -1.

    Lets us read the on-target knockdown effect (should be strongly negative for CRISPRi).
    """
    id_to_col = {g: i for i, g in enumerate(de.gene_ids)}
    return np.array([id_to_col.get(t, -1) for t in de.obs["target_contrast"].astype(str)])
