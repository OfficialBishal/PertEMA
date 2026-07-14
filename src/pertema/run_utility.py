"""R3 downstream-utility: does gating a predictor's discovery list by PertEMA reliability recover known
CD4-activation / IL-2 regulators better than the magnitude and similarity heuristics and no selection?

Design (hardened after internal review):
  - Discovery predictor: knn_coexpr_k25 (gene-specific predictions). The mean predictor is degenerate here
    (it predicts one vector for every gene, so a magnitude ranking is undefined), so it is not used for the
    discovery pool.
  - Setting: context transfer Rest -> Stim48hr (predict the strongly activated state from the resting
    reference), the deployment target for an activation screen. Robustness: Rest -> Stim8hr.
  - Pool: the predictor's large-effect calls, the top q of genes by predicted effect magnitude m(g). This is
    the shortlist a biologist actually triages.
  - Scores, all leakage-free and available at prediction time:
      r(g)  = minus out-of-fold PertEMA predicted transfer error (reliability, the protagonist)
      m(g)  = pred_magnitude (the effect-magnitude heuristic and the pool-defining discovery signal)
      s(g)  = minus nearest-training-gene distance in the co-expression embedding (similarity heuristic)
      r_rand(g)        = random-feature-gate estimator (features permuted; a control gate)
      r_lshuf(g)       = label-shuffle-gate estimator (PertEMA trained on shuffled errors; a control gate)
  - Ground truth: model-independent regulator sets built from prior literature only (a programmatic pathway
    union and an orthogonal functional-screen set), never from this screen. results/utility/regulators_*.csv.
  - Metric: within the pool, select the top-K by each score and measure precision (fraction that are
    regulators) and enrichment (precision / pool base rate). Delta-Precision = precision(pertema) minus
    precision(magnitude). Also the precision-coverage curve area.
  - Confound-matched nulls (the panel's demand): the study-bias/hubness confound is controlled by permuting
    the regulator labels within strata of predicted magnitude AND within strata of co-expression degree
    (hubness). If the PertEMA gain survives the degree-matched null it is not explained by annotating hubs.
  - Controls: random-feature-gate and label-shuffle-gate Delta-Precision must be about 0.
  - Significance: B-permutation label test (two-sided p), reported per seed and as a 3-seed mean.

Run: pixi run python src/pertema/run_utility.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SRC, "eval"))
sys.path.insert(0, HERE)
from run_estimator import gbt, sp                                # noqa: E402
from run_transfer_estimator import build_transfer_features       # noqa: E402

TRE = "results/transfer"
SPLITS = "results/splits/gene_folds.csv"   # carries Ensembl gene id and HGNC gene_name
OUT = "results/utility"
BASE = "results/features/control_baseline.npz"
EMB = "results/features/gene_embedding.npz"
SEEDS = [42, 43, 44]
PREDICTOR = "knn_coexpr_k25"          # discovery predictor (gene-specific; mean is degenerate)
POOL_Q = float(os.environ.get("PERTEMA_POOLQ", "1.0"))  # 1.0 = evaluate over all perturbations (powered);
#   a smaller q restricts to the predictor's top-q discovery shortlist. Most regulators are not in the top
#   10 percent by predicted magnitude, so the full-set rank enrichment (AUROC) is the primary, powered metric.
K_LIST = [25, 50, 100]                 # validation budgets
B_PERM = int(os.environ.get("PERTEMA_BPERM", "10000"))   # label-permutation repeats (env-overridable)
DEGREE_K = 25                          # kNN graph degree for the hubness proxy


def load_regulators(path, present_col="present_in_library"):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    sym_col = "symbol" if "symbol" in df.columns else df.columns[0]
    if present_col in df.columns:
        df = df[df[present_col].astype(str).str.lower().isin(["true", "1", "yes"])]
    return set(str(s).strip() for s in df[sym_col])


def coexpr_degree(gene_emb, k=DEGREE_K):
    """Hubness proxy: in-degree of each gene in a kNN graph on the co-expression embedding."""
    genes = np.array(sorted(gene_emb.keys()))
    X = np.array([gene_emb[g] for g in genes])
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=8).fit(X)
    _, idx = nn.kneighbors(X)
    indeg = np.zeros(len(genes), dtype=int)
    for row in idx[:, 1:]:               # skip self
        indeg[row] += 1
    return {g: int(d) for g, d in zip(genes, indeg)}


def precision_at_k(is_reg, score, K):
    """Fraction of the top-K by score that are regulators. Ties broken by score then index (stable)."""
    order = np.argsort(-score, kind="stable")
    return float(is_reg[order[:K]].mean())


def auroc(is_reg, score):
    """Rank AUROC of regulator-ness by score (probability a random regulator outranks a random non-regulator).
    Uses the rank-sum identity; ties get midranks. Higher = the score ranks regulators above non-regulators."""
    pos = is_reg > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def make_stat(kind):
    if kind == "auroc":
        return lambda ir, sc: auroc(ir, sc)
    K = int(kind.split("@")[1])
    return lambda ir, sc: precision_at_k(ir, sc, K)


def strata_labels(x, n_bins=10):
    """Assign each element to a quantile stratum of x (for matched permutation nulls)."""
    ranks = pd.Series(x).rank(method="first").to_numpy()
    return np.floor((ranks - 1) / len(x) * n_bins).astype(int)


def perm_pvalue(is_reg, r_score, m_score, stat, B, strata=None, rng=None):
    """Two-sided p for Delta(pertema - magnitude) of an arbitrary statistic under label permutation.
    If strata is given, labels are permuted only within strata (a matched null)."""
    rng = rng or np.random.default_rng(0)
    obs = stat(is_reg, r_score) - stat(is_reg, m_score)
    null = np.empty(B)
    idx_by_stratum = [np.where(strata == s)[0] for s in np.unique(strata)] if strata is not None else None
    for b in range(B):
        if strata is None:
            perm = rng.permutation(is_reg)
        else:
            perm = is_reg.copy()
            for grp in idx_by_stratum:
                perm[grp] = is_reg[grp][rng.permutation(len(grp))]
        null[b] = stat(perm, r_score) - stat(perm, m_score)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1)
    return float(obs), float(p)


def main():
    os.makedirs(OUT, exist_ok=True)
    reg_func = load_regulators(os.path.join(OUT, "regulators_functional.csv"))
    reg_path = load_regulators(os.path.join(OUT, "regulators_pathway.csv"))
    reg_any = reg_func | reg_path
    print(f"regulator sets: functional={len(reg_func)}, pathway={len(reg_path)}, union={len(reg_any)}")
    if not reg_any:
        print("NO regulator sets found under results/utility/. Run the ground-truth step first.")
        return

    bz = np.load(BASE)
    gene_col = {str(g): i for i, g in enumerate(bz["genes"])}
    baseline, dropout, donor_var = bz["baseline"], bz["dropout"], bz["donor_var"]
    ez = np.load(EMB)
    emb_map = {str(g): v for g, v in zip(ez["gene_ids"], ez["embedding"])}
    emb_dim = ez["embedding"].shape[1]
    degree = coexpr_degree(emb_map)
    # transfer_errors 'gene' is an Ensembl id; regulator sets are HGNC symbols. Map Ensembl -> symbol.
    gf = pd.read_csv(SPLITS, dtype=str)
    ens2sym = dict(zip(gf["gene"], gf["gene_name"]))

    def gate_oof(df, y, fold, Xb, genes_u, inv, emb_u, good_u, fold_u, seed, shuffle=False, randfeat=False):
        oof = np.full(len(df), np.nan)
        rng = np.random.default_rng(seed + (1 if shuffle else 0) + (2 if randfeat else 0))
        for k in np.unique(fold):
            trn = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            tr_u = np.where((fold_u != k) & good_u)[0]
            nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
            su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
            su[q] = nn.kneighbors(emb_u[q])[0].ravel()
            sim = su[inv]
            Xtr = np.column_stack([Xb[trn], sim[trn]]); Xte = np.column_stack([Xb[te], sim[te]])
            ytr = rng.permutation(y[trn]) if shuffle else y[trn]
            if randfeat:
                Xtr = np.column_stack([rng.permutation(Xtr[:, j]) for j in range(Xtr.shape[1])])
            oof[te] = gbt().fit(Xtr, ytr).predict(Xte)
        return oof

    settings = [("Rest", "Stim48hr"), ("Rest", "Stim8hr")]
    rows = []
    score_rows = []   # per-perturbation scores, saved for figures (risk-coverage curves, calibration)
    for (src, dst) in settings:
        for seed in SEEDS:
            tr = pd.read_csv(os.path.join(TRE, f"transfer_errors_seed{seed}.csv"), dtype={"gene": str})
            df = tr[(tr["predictor"] == PREDICTOR) & (tr["src"] == src) & (tr["dst"] == dst)].reset_index(drop=True)
            if df.empty:
                continue
            Xb, gene_arr = build_transfer_features(df, baseline, dropout, donor_var, gene_col, emb_map, emb_dim)
            y = df["transfer_err"].to_numpy()
            m = df["pred_magnitude"].to_numpy()
            fold = df["fold"].to_numpy()
            genes_u, inv = np.unique(gene_arr, return_inverse=True)
            emb_u = np.array([emb_map.get(g, np.full(emb_dim, np.nan)) for g in genes_u])
            good_u = ~np.isnan(emb_u).any(1)
            fold_u = np.full(len(genes_u), -1); fold_u[inv] = fold

            # similarity score s(g): nearest-training-gene distance, OOF (higher reliability = smaller dist)
            sim_full = np.full(len(df), np.nan)
            for k in np.unique(fold):
                te = np.where(fold == k)[0]
                tr_u = np.where((fold_u != k) & good_u)[0]
                nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=8).fit(emb_u[tr_u])
                su = np.full(len(genes_u), np.nan); q = np.where(good_u)[0]
                su[q] = nn.kneighbors(emb_u[q])[0].ravel()
                sim_full[te] = su[inv][te]

            r = -gate_oof(df, y, fold, Xb, genes_u, inv, emb_u, good_u, fold_u, seed)           # reliability
            r_rand = -gate_oof(df, y, fold, Xb, genes_u, inv, emb_u, good_u, fold_u, seed, randfeat=True)
            r_lshuf = -gate_oof(df, y, fold, Xb, genes_u, inv, emb_u, good_u, fold_u, seed, shuffle=True)
            s = -sim_full

            g_ens = gene_arr
            g_sym = np.array([ens2sym.get(x, x) for x in gene_arr])
            is_reg_func = np.array([1.0 if x in reg_func else 0.0 for x in g_sym])
            is_reg_path = np.array([1.0 if x in reg_path else 0.0 for x in g_sym])
            deg = np.array([degree.get(x, 0) for x in g_ens], float)

            for i in range(len(df)):
                score_rows.append(dict(src=src, dst=dst, seed=seed, gene=g_sym[i], gene_ens=g_ens[i],
                                       true_error=float(y[i]), reliability=float(r[i]),
                                       pred_magnitude=float(m[i]), similarity=float(s[i]),
                                       coexpr_degree=float(deg[i]), is_reg_functional=int(is_reg_func[i]),
                                       is_reg_pathway=int(is_reg_path[i])))

            # pool = top q by predicted magnitude
            pool_n = max(K_LIST[-1] * 2, int(np.ceil(POOL_Q * len(df))))
            pool = np.argsort(-m, kind="stable")[:pool_n]
            rng = np.random.default_rng(seed)
            metrics = ["auroc"] + [f"prec@{K}" for K in K_LIST]
            scores = {"pertema": r[pool], "magnitude": m[pool], "similarity": s[pool],
                      "rand_gate": r_rand[pool], "lshuf_gate": r_lshuf[pool]}
            mstr = strata_labels(m[pool]); dstr = strata_labels(deg[pool])
            for gtname, is_reg_all in [("functional", is_reg_func), ("pathway", is_reg_path)]:
                if is_reg_all.sum() == 0:
                    continue
                ir = is_reg_all[pool]
                base_rate = float(ir.mean())
                for mk in metrics:
                    if mk.startswith("prec@") and int(mk.split("@")[1]) > len(pool):
                        continue
                    stat = make_stat(mk)
                    vals = {rule: stat(ir, sc) for rule, sc in scores.items()}
                    _, p_plain = perm_pvalue(ir, r[pool], m[pool], stat, B_PERM, rng=rng)
                    _, p_mag = perm_pvalue(ir, r[pool], m[pool], stat, B_PERM, strata=mstr, rng=rng)
                    _, p_deg = perm_pvalue(ir, r[pool], m[pool], stat, B_PERM, strata=dstr, rng=rng)
                    rows.append(dict(
                        src=src, dst=dst, seed=seed, ground_truth=gtname, metric=mk,
                        pool_n=int(len(pool)), base_rate=base_rate, n_reg_in_pool=int(ir.sum()),
                        val_pertema=vals["pertema"], val_magnitude=vals["magnitude"],
                        val_similarity=vals["similarity"], val_rand_gate=vals["rand_gate"],
                        val_lshuf_gate=vals["lshuf_gate"],
                        delta_pertema_minus_magnitude=vals["pertema"] - vals["magnitude"],
                        enrichment_pertema=(vals["pertema"] / base_rate if (mk.startswith("prec@") and base_rate > 0) else np.nan),
                        p_perm_label=p_plain, p_perm_magnitude_matched=p_mag, p_perm_degree_matched=p_deg,
                    ))
            print(f"[{src}->{dst} seed {seed}] pool={len(pool)} "
                  f"func_reg_in_pool={int(is_reg_func[pool].sum())} path_reg_in_pool={int(is_reg_path[pool].sum())}",
                  flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "utility_per_seed.csv"), index=False)
    pd.DataFrame(score_rows).to_csv(os.path.join(OUT, "pertema_scores.csv"), index=False)

    # family mean over seeds per (setting, ground_truth, metric)
    agg = []
    for (src, dst, gt, mk), sub in res.groupby(["src", "dst", "ground_truth", "metric"]):
        row = dict(src=src, dst=dst, ground_truth=gt, metric=mk, n_seeds=len(sub),
                   base_rate=sub["base_rate"].mean(), n_reg_in_pool=sub["n_reg_in_pool"].mean())
        for c in ["val_pertema", "val_magnitude", "val_similarity", "val_rand_gate", "val_lshuf_gate",
                  "delta_pertema_minus_magnitude", "enrichment_pertema",
                  "p_perm_label", "p_perm_magnitude_matched", "p_perm_degree_matched"]:
            row[c] = sub[c].mean()
            if c == "delta_pertema_minus_magnitude":
                v = sub[c].to_numpy()
                row["delta_ci95"] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        agg.append(row)
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(os.path.join(OUT, "utility_summary.csv"), index=False)
    print("\n=== R3 downstream utility (family mean over seeds) ===")
    show = ["src", "dst", "ground_truth", "metric", "base_rate", "val_pertema", "val_magnitude",
            "delta_pertema_minus_magnitude", "p_perm_label", "p_perm_degree_matched"]
    print(aggdf[show].round(4).to_string(index=False))
    print(f"\nwrote {OUT}/utility_summary.csv and utility_per_seed.csv")


if __name__ == "__main__":
    main()
