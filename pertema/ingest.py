"""Ingest and validate a user's perturbation predictions. Supports the long CSV format (one row per
perturbation and gene). Groups into per-perturbation predicted effect vectors and computes the predicted
effect magnitude the featurizer needs. Precise errors on a bad schema. Wide CSV, Parquet, and AnnData h5ad
are documented extensions that reduce to the same per-perturbation representation.
"""
import io

import numpy as np
import pandas as pd

LONG_REQUIRED = ["perturbed_gene", "gene", "predicted_lfc"]
LONG_TEMPLATE = (
    "perturbed_gene,gene,predicted_lfc,src_context,dst_context\n"
    "IL2,IL2RA,0.83,Rest,Stim48hr\n"
    "IL2,IFNG,-0.21,Rest,Stim48hr\n"
    "IL2,FOXP3,0.11,Rest,Stim48hr\n"
    "STAT5A,IL2RA,0.44,Rest,Stim48hr\n"
    "STAT5A,MYC,0.30,Rest,Stim48hr\n"
    "# one row per (perturbed_gene, gene). predicted_lfc is your model's predicted delta log-fold-change.\n"
    "# src_context/dst_context are optional and must be Rest, Stim8hr, or Stim48hr (default Rest -> Stim48hr).\n"
)


def parse_long_csv(csv_text, src_default="Rest", dst_default="Stim48hr", max_rows=5_000_000):
    """Parse a long-format predictions CSV into per-perturbation records. Returns (predictions, report).
    Raises ValueError with a precise message on a bad schema."""
    try:
        df = pd.read_csv(io.StringIO(csv_text), comment="#")
    except Exception as e:
        raise ValueError(f"could not parse CSV: {e}")
    if len(df) > max_rows:
        raise ValueError(f"too many rows: {len(df)} > limit {max_rows}")
    missing = [c for c in LONG_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}; got {list(df.columns)}. "
                         f"Long format needs {LONG_REQUIRED} (+ optional src_context, dst_context).")
    df["predicted_lfc"] = pd.to_numeric(df["predicted_lfc"], errors="coerce")
    n_bad = int(df["predicted_lfc"].isna().sum())
    df = df.dropna(subset=["predicted_lfc"])
    src_col = "src_context" if "src_context" in df.columns else None
    dst_col = "dst_context" if "dst_context" in df.columns else None
    keys = ["perturbed_gene"] + ([src_col] if src_col else []) + ([dst_col] if dst_col else [])
    preds = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        pg = key[0]
        src = key[keys.index(src_col)] if src_col else src_default
        dst = key[keys.index(dst_col)] if dst_col else dst_default
        pred_magnitude = float(np.abs(g["predicted_lfc"].to_numpy()).mean())
        preds.append({"perturbed_gene": str(pg), "src": str(src), "dst": str(dst),
                      "pred_magnitude": pred_magnitude, "n_genes": int(len(g))})
    report = {"n_rows": int(len(df)), "n_perturbations": len(preds),
              "n_dropped_nonnumeric_lfc": n_bad,
              "contexts_from": "columns" if src_col else f"defaults ({src_default} -> {dst_default})"}
    return preds, report
