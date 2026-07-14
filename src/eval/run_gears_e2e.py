"""E1: GEARS end-to-end on Norman as a real predictor (the vendored GNN, its own training loop and GO
graph). TRACKED reproducibility copy. The live copy runs from baselines/PRESCRIBE/ (which is gitignored),
so to run: copy this file there and execute it INSIDE the pytorch241 container:

  cp src/eval/run_gears_e2e.py baselines/PRESCRIBE/
  bash baselines/run_in_container.sh env CUDA_VISIBLE_DEVICES=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_gears_e2e.py --epochs 15

STATUS (2026-07-11): DOCUMENTED FALLBACK, E1 not yet producing OOF predictions. E1 is environment-feasible
(container + GPU 1 + GEARS API + data all load, model constructs) but the PRESCRIBE-preprocessed Norman
h5ad (baselines/PRESCRIBE/data/norman/perturb_processed.h5ad) diverges from GEARS's native from-scratch
format across at least six points, fixed in sequence here as they surfaced:
  1. gears.py:87 indexed a scipy sparse matrix with a pandas boolean Series -> needs .values (patched in
     the vendored gears.py).
  2. GEARS init's first CUDA allocation transiently OOM'd on the co-tenant-contended GPU 0 -> run on GPU 1.
  3. adata.uns lacked GEARS's DE-gene keys (rank_genes_groups_cov_all, non_zeros_gene_idx) -> compute them.
  4. get_DE_genes rebuilds condition_name from raw cell_type/dose_val columns absent here -> set cell_type.
  5. condition_name here is 2-part (K562_GENE+ctrl) vs GEARS's native 3-part (K562_..._1) -> rebuild it.
  6. model_initialize's co-expression network wants adata.var.gene_name, absent here -> set it (below).
Each fix revealed the next, which is the honest signal that this preprocessed h5ad is not GEARS-native.
CLEAN PATH (the documented next step, not a patch cascade): re-process Norman through GEARS's own
PertData.new_data_process from raw counts with GEARS's expected obs/var schema, then train. Until then the
routing negative and the E2 error-correlation are reported as holding "across the roster we could run" (two
constant baselines, co-expression ridge/kNN, six frozen-adapted foundation heads), with the end-to-end GNN
arm as future work, stated in the abstract per the specification.

GEARS trains on a held-out-perturbation split, so its test predictions are leakage-free out-of-fold.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gears import GEARS, PertData  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", default="norman")
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--split", default="simulation")
    ap.add_argument("--split_seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--out", default="../../results/parity/gears_norman_e2e.csv")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[E1] loading PertData {args.data_name} from {args.data_path} ...", flush=True)
    pert_data = PertData(args.data_path)
    pert_data.load(data_name=args.data_name)
    # rebuild GEARS's native 3-part condition_name + DE-gene keys from raw conditions BEFORE splitting.
    from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes  # noqa: E402
    if "non_zeros_gene_idx" not in pert_data.adata.uns:
        print("[E1] rebuilding GEARS condition_name + DE-gene keys from raw conditions ...", flush=True)
        pert_data.adata.obs["cell_type"] = "K562"                    # covariate get_DE_genes needs
        pert_data.adata = get_DE_genes(pert_data.adata, skip_calc_de=False)
        pert_data.adata = get_dropout_non_zero_genes(pert_data.adata)
    # fix 6: the co-expression network builder reads adata.var.gene_name, absent in this h5ad.
    if "gene_name" not in pert_data.adata.var.columns:
        pert_data.adata.var["gene_name"] = pert_data.adata.var_names
    pert_data.prepare_split(split=args.split, seed=args.split_seed)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=128)
    set2c = getattr(pert_data, "set2conditions", {})
    n_test = len(set2c.get("test", [])) if isinstance(set2c, dict) else "?"
    n_train = len(set2c.get("train", [])) if isinstance(set2c, dict) else "?"
    print(f"[E1] loaded: {pert_data.adata.shape}, train perts {n_train}, test perts {n_test} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[E1] initializing GEARS (hidden {args.hidden_size}) on {args.device} ...", flush=True)
    gears = GEARS(pert_data, device=args.device)
    gears.model_initialize(hidden_size=args.hidden_size)
    print(f"[E1] GEARS model initialized ({time.time()-t0:.0f}s)", flush=True)
    if args.validate_only:
        print("[E1] VALIDATE-ONLY: pipeline works up to model construction.", flush=True)
        return

    print(f"[E1] training GEARS for {args.epochs} epochs ...", flush=True)
    gears.train(epochs=args.epochs)
    print(f"[E1] training done ({time.time()-t0:.0f}s). Evaluating held-out perturbations ...", flush=True)

    # held-out per-perturbation predicted vs true delta on the top-1000 expressed genes, parity vs the mean
    ctrl_mean = np.asarray(pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"].X.mean(0)).ravel()
    expr_rank = np.argsort(-ctrl_mean)[:1000]
    rows = []
    for cond in [c for c in set2c.get("test", []) if c != "ctrl"]:
        genes = [g for g in cond.split("+") if g != "ctrl"]
        if not genes:
            continue
        try:
            pred_expr = np.asarray(list(gears.predict([genes]).values())[0]).ravel()
        except Exception as e:
            print(f"[E1] predict failed for {cond}: {e}", flush=True)
            continue
        cells = pert_data.adata[pert_data.adata.obs["condition"] == cond].X
        if cells.shape[0] == 0:
            continue
        true_expr = np.asarray(cells.mean(0)).ravel()
        pc = np.corrcoef((pred_expr - ctrl_mean)[expr_rank], (true_expr - ctrl_mean)[expr_rank])[0, 1]
        rows.append(dict(condition=cond, gears_pearson_delta=float(pc), gears_err=float(1 - pc), mean_err=1.0))
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"[E1] GEARS end-to-end on Norman: {len(res)} held-out perturbations, mean delta-Pearson "
          f"{res['gears_pearson_delta'].mean():.4f}, mean error {res['gears_err'].mean():.4f}", flush=True)
    print(f"[E1] wrote {args.out} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
