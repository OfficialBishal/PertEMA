"""Inspect the on-disk structure of an h5ad file without loading matrices into memory.

Verifies the real .obs / .var / .layers / .varm schema (anndata HDF5 encoding) so loaders are
written against ground truth, not assumptions. Usage: pixi run python src/data/inspect_h5ad.py <path>
"""
import sys

import h5py
import numpy as np


def col_summary(grp, name):
    """Summarize one obs/var column, decoding anndata categorical encoding."""
    obj = grp[name]
    if isinstance(obj, h5py.Group):  # categorical: datasets 'categories' + 'codes'
        cats = obj.get("categories")
        codes = obj.get("codes")
        ncat = cats.shape[0] if cats is not None else "?"
        sample = [c.decode() if isinstance(c, bytes) else c for c in cats[:6]] if cats is not None else []
        return f"categorical n_cat={ncat} e.g.={sample} n={codes.shape[0] if codes is not None else '?'}"
    else:
        dt = obj.dtype
        n = obj.shape[0] if obj.ndim else 1
        try:
            vals = obj[:5]
            vals = [v.decode() if isinstance(v, bytes) else v for v in vals]
        except Exception:
            vals = "?"
        return f"{dt} n={n} e.g.={vals}"


def dump_group_columns(f, path):
    if path not in f:
        print(f"  ({path} absent)")
        return
    g = f[path]
    order = g.attrs.get("column-order")
    idxkey = g.attrs.get("_index")
    if idxkey is not None:
        idxkey = idxkey.decode() if isinstance(idxkey, bytes) else idxkey
        print(f"  _index = '{idxkey}': {col_summary(g, idxkey)}")
    cols = list(order) if order is not None else [k for k in g.keys()]
    for c in cols:
        c = c.decode() if isinstance(c, bytes) else c
        if c in g:
            print(f"  {c}: {col_summary(g, c)}")


def main(path):
    with h5py.File(path, "r") as f:
        print(f"===== {path} =====")
        print("root attrs:", dict(f.attrs))
        # X
        if "X" in f:
            x = f["X"]
            if isinstance(x, h5py.Group):
                print("X: sparse", dict(x.attrs), {k: x[k].shape for k in x.keys()})
            else:
                print("X: dense", x.shape, x.dtype)
        for sect in ["layers", "varm", "obsm", "obsp", "varp"]:
            if sect in f:
                print(f"\n{sect}:")
                for k in f[sect].keys():
                    o = f[sect][k]
                    shp = o.shape if isinstance(o, h5py.Dataset) else f"group{list(o.keys())}"
                    dt = o.dtype if isinstance(o, h5py.Dataset) else ""
                    print(f"  {k}: {shp} {dt}")
        print("\nobs columns:")
        dump_group_columns(f, "obs")
        print("\nvar columns:")
        dump_group_columns(f, "var")
        if "uns" in f:
            print("\nuns keys:", list(f["uns"].keys()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/marson2025/GWCD4i.DE_stats.h5ad")
