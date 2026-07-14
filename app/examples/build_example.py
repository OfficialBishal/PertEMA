"""Build the compact bundled example (F12): a small slice of Gladstone activation-transfer perturbations with
their prediction-time features (matching the frozen model's feature_spec), the perturbed gene, the transfer
pair, and the true realized error (held for evaluate-mode only). Under a strict size budget so the demo,
docs, and acceptance test run without the multi-gigabyte primary data.

Run: pixi run python app/examples/build_example.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, os.path.join(ROOT, "src", "pertema"))
from run_transfer_estimator import build_transfer_features   # noqa: E402

N_EXAMPLE = 400
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    bz = np.load(os.path.join(ROOT, "results/features/control_baseline.npz"))
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(os.path.join(ROOT, "results/features/gene_embedding.npz"))
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    gf = pd.read_csv(os.path.join(ROOT, "results/splits/gene_folds.csv"), dtype=str)
    ens2sym = dict(zip(gf["gene"], gf["gene_name"]))

    tr = pd.read_csv(os.path.join(ROOT, "results/transfer/transfer_errors_seed42.csv"), dtype={"gene": str})
    df = tr[(tr["predictor"] == "mean_condition") & (tr["src"] == "Rest") & (tr["dst"] == "Stim48hr")]
    df = df.reset_index(drop=True)
    Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
    genes_u, inv = np.unique(gene_arr, return_inverse=True)
    emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
    good = ~np.isnan(emb_u).any(1)
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=8).fit(emb_u[good])
    su = np.full(len(genes_u), np.nan); d, _ = nn.kneighbors(emb_u[good]); su[np.where(good)[0]] = d[:, 1]
    sim = su[inv]
    X = np.column_stack([Xb, sim])
    ok = np.isfinite(X).all(1)
    idx = np.where(ok)[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(idx, size=min(N_EXAMPLE, len(idx)), replace=False)
    sel.sort()

    feats = X[sel].astype(np.float32)
    genes = np.array([ens2sym.get(g, g) for g in df["gene"].to_numpy()[sel]])
    true_err = df["transfer_err"].to_numpy()[sel].astype(np.float32)

    np.savez_compressed(os.path.join(OUT, "example_gladstone.npz"),
                        features=feats, genes=genes, src=np.array(["Rest"] * len(sel)),
                        dst=np.array(["Stim48hr"] * len(sel)), true_error=true_err)
    # a small human-readable preview
    pd.DataFrame({"gene": genes, "src": "Rest", "dst": "Stim48hr", "true_error": true_err}).head(20).to_csv(
        os.path.join(OUT, "example_preview.csv"), index=False)
    size_kb = os.path.getsize(os.path.join(OUT, "example_gladstone.npz")) // 1024
    print(f"wrote {OUT}/example_gladstone.npz: {len(sel)} perturbations, {feats.shape[1]} features, {size_kb} KB")


if __name__ == "__main__":
    main()
