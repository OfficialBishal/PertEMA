"""GNN estimator for context-transfer reliability (P4: GNN versus gradient-boosted trees).

A graph convolutional network over the control co-expression graph. Nodes are perturbed genes with
control-state node features (baseline expression, dropout, cross-donor variance per condition, and the
co-expression embedding). Two GCN layers propagate features along co-expression edges to produce
graph-aware gene representations. A per-instance head combines the gene representation with the
predicted effect magnitude, the source and destination condition one-hots, and the training-set
similarity to predict the cross-context transfer error. The GCN is transductive: it uses node features
and graph structure for all genes but the loss is taken only on training-fold instances, so no test
gene's error is ever used (invariant 1). Pure torch, no torch-geometric (its compiled extensions do not
build on this EL7 host).

Reports the mean-condition transfer AURC for the GNN against the GBT baseline. Single seed for speed.
Run: pixi run python src/pertema/gnn_estimator.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
from metrics import aurc                                   # noqa: E402

TRE = "results/transfer"
OUT = "results/pertema"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEED = 42
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
K_GRAPH = 10
torch.manual_seed(0)


def norm_adj(emb, k):
    """Symmetric-normalized adjacency of a kNN co-expression graph with self-loops (sparse)."""
    n = emb.shape[0]
    nn_ = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=8).fit(emb)
    _, idx = nn_.kneighbors(emb)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    ii = np.concatenate([rows, cols, np.arange(n)])
    jj = np.concatenate([cols, rows, np.arange(n)])
    A = torch.sparse_coo_tensor(np.stack([ii, jj]), torch.ones(len(ii)), (n, n)).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1)
    dinv = deg.pow(-0.5)
    v = A.values() * dinv[A.indices()[0]] * dinv[A.indices()[1]]
    return torch.sparse_coo_tensor(A.indices(), v, (n, n)).coalesce()


class GCN(nn.Module):
    def __init__(self, in_dim, hid, gdim, inst_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid)
        self.w2 = nn.Linear(hid, gdim)
        self.head = nn.Sequential(nn.Linear(gdim + inst_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, A, X, gene_idx, inst):
        h = torch.relu(torch.sparse.mm(A, self.w1(X)))
        z = torch.sparse.mm(A, self.w2(h))          # (n_genes, gdim)
        zi = z[gene_idx]
        return self.head(torch.cat([zi, inst], 1)).squeeze(1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]

    tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{SEED}.csv"), dtype={"gene": str})
    df = tr[tr["transfer"] & (tr["predictor"] == "mean_condition")].reset_index(drop=True)
    df = df[df["gene"].isin(emb_map)].reset_index(drop=True)      # need an embedding for graph nodes

    genes = sorted(df["gene"].unique())
    gidx = {g: i for i, g in enumerate(genes)}
    emb = np.stack([emb_map[g] for g in genes]).astype(np.float32)
    # node features: baseline/dropout/donor_var per condition + embedding
    nf = []
    for g in genes:
        c = gene_col.get(g)
        stats = ([baseline[j, c] for j in range(3)] + [dropout[j, c] for j in range(3)]
                 + [donor_var[j, c] for j in range(3)]) if c is not None else [0] * 9
        nf.append(stats)
    X = np.concatenate([np.nan_to_num(np.array(nf, np.float32)), emb], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    A = norm_adj(emb, K_GRAPH).to(dev)
    Xt = torch.tensor(X, device=dev)

    cidx = {c: i for i, c in enumerate(CONDS)}
    gi_inst = df["gene"].map(gidx).to_numpy()
    src1h = np.eye(3)[df["src"].map(cidx).to_numpy()]
    dst1h = np.eye(3)[df["dst"].map(cidx).to_numpy()]
    y = df["transfer_err"].to_numpy().astype(np.float32)
    fold = df["fold"].to_numpy()

    # per-fold training-set similarity
    fold_of_gene = {g: df.loc[df["gene"] == g, "fold"].iloc[0] for g in genes}
    fold_g = np.array([fold_of_gene[g] for g in genes])
    pred_oof = np.full(len(df), np.nan)
    for k in np.unique(fold):
        tr_g = np.where(fold_g != k)[0]
        nn_ = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb[tr_g])
        simg = nn_.kneighbors(emb)[0].ravel()
        sim = simg[gi_inst]
        inst = np.column_stack([df["pred_magnitude"].to_numpy(), src1h, dst1h, sim]).astype(np.float32)
        inst = np.nan_to_num((inst - np.nanmean(inst, 0)) / (np.nanstd(inst, 0) + 1e-6))
        tr_i = np.where(fold != k)[0]
        te_i = np.where(fold == k)[0]
        ok = np.isfinite(y)
        tr_i = tr_i[np.isfinite(y[tr_i])]
        model = GCN(X.shape[1], 64, 32, inst.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        gi_t = torch.tensor(gi_inst, device=dev)
        inst_t = torch.tensor(inst, device=dev)
        y_t = torch.tensor(y, device=dev)
        tr_t = torch.tensor(tr_i, device=dev)
        model.train()
        for ep in range(200):
            opt.zero_grad()
            p = model(A, Xt, gi_t[tr_t], inst_t[tr_t])
            loss = ((p - y_t[tr_t]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            te_t = torch.tensor(te_i, device=dev)
            pred_oof[te_i] = model(A, Xt, gi_t[te_t], inst_t[te_t]).cpu().numpy()
        print(f"  fold {k}: train loss {loss.item():.4f}")

    m = np.isfinite(y) & np.isfinite(pred_oof)
    gnn_aurc = aurc(y[m], -pred_oof[m])
    from scipy.stats import spearmanr
    gnn_spear = spearmanr(pred_oof[m], y[m]).correlation
    # GBT reference from the committed transfer summary
    gbt_aurc = 0.871
    print(f"\n=== GNN vs GBT (transfer, mean_condition, seed {SEED}) ===")
    print(f"  GNN  AURC {gnn_aurc:.4f}  Spearman {gnn_spear:.3f}")
    print(f"  GBT  AURC {gbt_aurc:.4f}  (from results/pertema/transfer_estimator_summary.csv)")
    print("  " + ("GNN wins" if gnn_aurc < gbt_aurc else "GBT wins or ties - honest baseline stands"))
    pd.DataFrame([{"estimator": "GNN", "aurc": gnn_aurc, "spearman": gnn_spear},
                  {"estimator": "GBT", "aurc": gbt_aurc, "spearman": 0.132}]
                 ).to_csv(os.path.join(OUT, "gnn_vs_gbt.csv"), index=False)
    print(f"wrote {OUT}/gnn_vs_gbt.csv")


if __name__ == "__main__":
    main()
