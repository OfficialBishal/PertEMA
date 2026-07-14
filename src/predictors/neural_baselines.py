"""Simple neural baselines (Wu et al. 2024, as used by Wong) behind the common fit/predict interface. Kept
separate from the validated predictors.py so the frozen reliability pipeline is untouched. Trained per
dataset from scratch on the gene-disjoint splits, so CLEAN (no pretraining).

MLPDecoderPredictor is the Decoder-Only baseline: a small multilayer perceptron from the perturbed gene's
control co-expression embedding to its effect vector. A nonlinear inductive bias, distinct from the constant
mean, the local kNN average, and the linear ridge. GPU 0 if available, CPU otherwise.
"""
from __future__ import annotations

import numpy as np


class MLPDecoderPredictor:
    def __init__(self, gene_emb, hidden=256, epochs=30, lr=1e-3, seed=0, target_cols=None):
        self.gene_emb = gene_emb
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.target_cols = target_cols   # predict only these gene columns (memory), fill rest with the mean
        self.name = "mlp_decoder"

    def fit(self, train_rows, effect, obs):
        import torch
        torch.manual_seed(self.seed)
        # GPU 0 is shared; fall back to CPU on out-of-memory rather than crash.
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        gene = obs["target_contrast"].astype(str).to_numpy()
        self._global = effect[train_rows].mean(0)
        cols = self.target_cols if self.target_cols is not None else np.arange(effect.shape[1])
        self._cols = cols
        X, Y = [], []
        for r in train_rows:
            e = self.gene_emb.get(gene[r])
            if e is not None:
                X.append(e); Y.append(effect[r][cols])
        if len(X) < 100:
            self.net = None; return self
        Xn, Yn = np.stack(X), np.stack(Y)
        try:
            return self._train(torch, Xn, Yn)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); self.dev = "cpu"
                return self._train(torch, Xn, Yn)
            raise

    def _train(self, torch, Xn, Yn):
        X = torch.tensor(Xn, dtype=torch.float32, device=self.dev)
        Y = torch.tensor(Yn, dtype=torch.float32, device=self.dev)
        d_in, d_out = X.shape[1], Y.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, d_out)).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        bs = 2048
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), Y[idx])
                loss.backward(); opt.step()
        self.net.eval()
        return self

    def predict(self, query_rows, obs):
        import torch
        gene = obs["target_contrast"].astype(str).to_numpy()
        out = np.tile(self._global, (len(query_rows), 1)).astype(np.float32)
        if getattr(self, "net", None) is None:
            return out
        embs, pos = [], []
        for j, r in enumerate(query_rows):
            e = self.gene_emb.get(gene[r])
            if e is not None:
                embs.append(e); pos.append(j)
        if embs:
            with torch.no_grad():
                pred = self.net(torch.tensor(np.stack(embs), dtype=torch.float32, device=self.dev)).cpu().numpy()
            for t, j in enumerate(pos):
                out[j, self._cols] = pred[t]      # MLP predicts only self._cols; rest stays the global mean
        return out
