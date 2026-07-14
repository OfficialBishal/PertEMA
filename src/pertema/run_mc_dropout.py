"""D2 UQ panel: add an MC-dropout intrinsic-uncertainty baseline to the transfer comparison, so the panel has
a second deep intrinsic-uncertainty method beside the bootstrap deep ensemble (run_intrinsic_uncertainty.py).

Method: a dropout multilayer-perceptron transfer predictor (perturbed-gene co-expression embedding -> effect
on the source high-variance gene set) is trained on the source-context training genes of each gene-disjoint
fold. At inference we keep dropout ACTIVE and take T stochastic forward passes; the mean per-gene standard
deviation across passes is the MC-dropout uncertainty (higher = less reliable). On the SAME held-out transfer
errors we compare how well three signals rank realized error: PertEMA post-hoc reliability, the MC-dropout
intrinsic uncertainty, and the effect-magnitude heuristic (reliability-Spearman and risk-coverage AUC).

Leakage-safe: the uncertainty uses only the model's own stochasticity at prediction time, never the held-out
truth. GPU 0 only, CPU fallback on OOM. Gene-disjoint, 3 seeds.

Run: pixi run python src/pertema/run_mc_dropout.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
for _p in ("eval", "predictors", "data", "pertema"):
    sys.path.insert(0, os.path.join(SRC, _p))
from load_de import load_de_stats                                  # noqa: E402
from metrics import aurc, reliability_spearman                     # noqa: E402
from run_estimator import best_aurc, gbt                           # noqa: E402
from run_transfer import N_HVG, row_pearson                        # noqa: E402
from run_transfer_estimator import build_transfer_features         # noqa: E402

SPLITS = "results/splits/gene_folds.csv"
EMB = "results/features/gene_embedding.npz"
BASE = "results/features/control_baseline.npz"
OUT = "results/pertema"
CONDS = ["Rest", "Stim8hr", "Stim48hr"]
SEEDS = [42, 43, 44]
T_MC = 20          # MC-dropout stochastic forward passes
EPOCHS = 40
P_DROP = 0.2


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return reliability_spearman(a[m], b[m]) if m.sum() > 3 else np.nan


def mc_dropout_predict(Xtr, Ytr, Xte, seed, dev):
    """Train a dropout MLP (emb -> hvg effect) and return (mean prediction, MC-dropout uncertainty per row)."""
    import torch
    torch.manual_seed(seed)
    # best-effort CUDA determinism for reproducibility (MC-dropout is a stochastic GPU baseline, so it
    # reproduces to within Monte Carlo error rather than byte-identically without these).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(P_DROP),
        torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(P_DROP),
        torch.nn.Linear(256, Ytr.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    bs, n = 2048, Xtr.shape[0]
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = loss_fn(net(Xtr_t[idx]), Ytr_t[idx]); loss.backward(); opt.step()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    net.train()                                    # keep dropout ACTIVE for MC sampling
    with torch.no_grad():
        samples = np.stack([net(Xte_t).cpu().numpy() for _ in range(T_MC)])   # (T, n_te, hvg)
    return samples.mean(0), samples.std(0).mean(1)


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    de = load_de_stats(layers=("log_fc",))
    lfc = de.layers["log_fc"]
    obs = de.obs.reset_index(drop=True)
    gene = obs["target_contrast"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    row_of = {(gene[i], cond[i]): i for i in range(len(obs))}
    folds = pd.read_csv(SPLITS, dtype={"gene": str})
    ez = np.load(EMB); gene_emb = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    bz = np.load(BASE); gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]

    rows = []
    for seed in SEEDS:
        fog = dict(zip(folds["gene"], folds[f"fold_seed{seed}"]))
        row_fold = np.array([fog.get(g, -1) for g in gene]); n_folds = int(row_fold.max()) + 1
        recs = []
        for src in CONDS:
            for dst in CONDS:
                if src == dst:
                    continue
                for k in range(n_folds):
                    tr = np.where((row_fold != k) & (cond == src))[0]
                    tr = np.array([r for r in tr if gene[r] in gene_emb])
                    if tr.size < 100:
                        continue
                    hvg = np.argsort(-lfc[tr].var(0))[:N_HVG]
                    tgenes = folds.loc[folds[f"fold_seed{seed}"] == k, "gene"].tolist()
                    pairs = [(g, row_of.get((g, src)), row_of.get((g, dst))) for g in tgenes]
                    pairs = [(g, a, b) for g, a, b in pairs if a is not None and b is not None and g in gene_emb]
                    if not pairs:
                        continue
                    b_rows = np.array([b for _, _, b in pairs])
                    Xtr = np.stack([gene_emb[gene[r]] for r in tr]); Ytr = lfc[tr][:, hvg]
                    Xte = np.stack([gene_emb[g] for g, _, _ in pairs])
                    try:
                        P, unc = mc_dropout_predict(Xtr, Ytr, Xte, 1000 * seed + k, dev)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache(); P, unc = mc_dropout_predict(Xtr, Ytr, Xte, 1000 * seed + k, "cpu")
                        else:
                            raise
                    err = 1.0 - row_pearson(P, lfc[b_rows][:, hvg]); pmag = np.abs(P).mean(1)
                    for j, (g, _, _) in enumerate(pairs):
                        recs.append(dict(gene=g, src=src, dst=dst, fold=k, err=float(err[j]),
                                         pmag=float(pmag[j]), mc_unc=float(unc[j])))
        df = pd.DataFrame.from_records(recs)
        # PertEMA post-hoc reliability on the SAME rows (pooled transfer estimator OOF)
        Xb, gene_arr = build_transfer_features(df.rename(columns={"pmag": "pred_magnitude"}),
                                               baseline, dropout, donor_var, gene_col, gene_emb, emb_dim)
        y = df["err"].to_numpy(); fold = df["fold"].to_numpy()
        gu, inv = np.unique(gene_arr, return_inverse=True)
        emu = np.array([gene_emb.get(g, np.full(emb_dim, np.nan)) for g in gu]); good = ~np.isnan(emu).any(1)
        fu = np.full(len(gu), -1); fu[inv] = fold
        rel = np.full(len(df), np.nan)
        for k in np.unique(fold):
            trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]; tru = np.where((fu != k) & good)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emu[tru])
            su = np.full(len(gu), np.nan); q = np.where(good)[0]; su[q] = nn.kneighbors(emu[q])[0].ravel(); sim = su[inv]
            rel[te] = -gbt().fit(np.column_stack([Xb[trn], sim[trn]]), y[trn]).predict(np.column_stack([Xb[te], sim[te]]))
        r = dict(seed=seed, n=len(df),
                 spearman_pertema=sp(rel, -y), spearman_mcdropout=sp(-df["mc_unc"], -y),
                 spearman_magnitude=sp(df["pmag"], -y),
                 aurc_pertema=best_aurc(y, rel), aurc_mcdropout=best_aurc(y, -df["mc_unc"].to_numpy()),
                 aurc_magnitude=best_aurc(y, df["pmag"].to_numpy()),
                 aurc_oracle=aurc(y[np.isfinite(y)], -y[np.isfinite(y)]), aurc_noselect=float(np.nanmean(y)))
        rows.append(r)
        print(f"seed {seed}: n={r['n']} | Spearman pertema={r['spearman_pertema']:.3f} "
              f"mcdropout={r['spearman_mcdropout']:.3f} mag={r['spearman_magnitude']:.3f} | "
              f"AURC pertema={r['aurc_pertema']:.3f} mcdropout={r['aurc_mcdropout']:.3f} "
              f"oracle={r['aurc_oracle']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    agg = {"comparison": "pertema_vs_mcdropout_transfer"}
    for c in [c for c in res.columns if c not in ("seed", "n")]:
        v = res[c].to_numpy(); agg[c] = f"{np.nanmean(v):.3f} +/- {1.96*np.nanstd(v, ddof=1)/np.sqrt(len(v)):.3f}"
    pd.DataFrame([agg]).to_csv(os.path.join(OUT, "mc_dropout_transfer.csv"), index=False)
    res.to_csv(os.path.join(OUT, "mc_dropout_transfer_per_seed.csv"), index=False)
    print("\n=== PertEMA vs MC-dropout intrinsic uncertainty on transfer (family mean +/- 95% CI) ===")
    print(pd.DataFrame([agg]).to_string(index=False))


if __name__ == "__main__":
    main()
