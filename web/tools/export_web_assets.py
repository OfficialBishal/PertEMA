"""Regenerate web/data/ — the static assets the client-side app loads — from the frozen model.

Run from the PertEMA project environment (the one that can import app/backend, i.e. fastapi + the
scientific stack are installed):

    pixi run python web/tools/export_web_assets.py     # or: python web/tools/export_web_assets.py

It imports the tested backend so the display JSONs (transfer heatmap, benchmark, model card) and the
training-set-similarity table come out byte-identical to the Python routes. Reference arrays (embedding,
control baselines, example features) are written as little-endian float32 .bin so the browser reads them
as Float32Array with no re-rounding. The scoring math itself lives in ../js/*.mjs and is a faithful,
parity-verified port of app/backend/{scoring,featurize,ingest}.py.
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BACKEND = os.path.join(REPO, "app", "backend")
MODEL_DIR = os.path.join(REPO, "app", "model", "pertema_model_v0.1.0")
REF = os.path.join(MODEL_DIR, "reference")
OUT = os.path.join(REPO, "web", "data")

for sub in ("model", "ref", "examples", "benchmark"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

sys.path.insert(0, BACKEND)
print("importing backend (loads model + featurizer, fits the similarity kNN)...")
import main as backend  # noqa: E402
from ingest import LONG_TEMPLATE  # noqa: E402

FEAT = backend.FEATURIZER
sizes = {}


def w_json(rel, obj):
    p = os.path.join(OUT, rel)
    with open(p, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    sizes[rel] = os.path.getsize(p)


def w_bin(rel, arr):
    p = os.path.join(OUT, rel)
    arr.astype("<f4").tofile(p)
    sizes[rel] = os.path.getsize(p)


def w_copy(rel, src):
    p = os.path.join(OUT, rel)
    shutil.copyfile(src, p)
    sizes[rel] = os.path.getsize(p)


# ---- model artifacts -------------------------------------------------------
w_copy("model/estimator.json", os.path.join(MODEL_DIR, "estimator.json"))
w_copy("model/provenance.json", os.path.join(MODEL_DIR, "provenance.json"))
w_copy("model/feature_spec.json", os.path.join(MODEL_DIR, "feature_spec.json"))
cal = np.load(os.path.join(MODEL_DIR, "calibration.npz"))
w_json("model/calibration.json", {"iso_x": [float(v) for v in cal["iso_x"]], "iso_y": [float(v) for v in cal["iso_y"]]})

# ---- control baselines -----------------------------------------------------
bz = np.load(os.path.join(REF, "control_baseline.npz"), allow_pickle=True)
genes = [str(g) for g in bz["genes"]]
conditions = [str(c) for c in bz["conditions"]]
stats = np.concatenate([bz["baseline"].ravel(), bz["dropout"].ravel(), bz["donor_var"].ravel()]).astype("<f4")
w_bin("ref/baseline_stats.bin", stats)
w_json("ref/baseline_index.json", {"n_genes": len(genes), "conditions": conditions,
                                   "order": ["baseline", "dropout", "donor_var"], "genes": genes})

# ---- embedding -------------------------------------------------------------
ez = np.load(os.path.join(REF, "gene_embedding.npz"), allow_pickle=True)
gene_ids = [str(g) for g in ez["gene_ids"]]
w_bin("ref/embedding.bin", ez["embedding"].astype("<f4").ravel())
w_json("ref/embedding_index.json", {"n_genes": len(gene_ids), "dim": 50, "gene_ids": gene_ids})

# ---- gene map, similarity, neighbors ---------------------------------------
w_json("ref/gene_map.json", {"sym2ens": FEAT.sym2ens, "ens2sym": FEAT.ens2sym})
w_json("ref/training_set_similarity.json", {k: float(v) for k, v in FEAT._sim.items()})
nn_genes = FEAT._nn_genes
K = 8
_, idx = FEAT._nn.kneighbors(np.array([FEAT.emb[g] for g in nn_genes]), n_neighbors=min(K + 1, len(nn_genes)))
top = {}
for i, g in enumerate(nn_genes):
    ens = str(g); out = []
    for j in idx[i]:
        gj = str(nn_genes[j])
        if gj == ens:
            continue
        out.append(FEAT.ens2sym.get(gj, gj))
        if len(out) >= K:
            break
    top[ens] = out
w_json("ref/gene_top5_neighbors.json", top)

# ---- display JSONs (byte-identical to the Python routes) -------------------
w_json("ref/transfer_heatmap.json", backend.transfer_heatmap())
w_json("model-card.json", backend.model_card_json())
try:
    w_json("benchmark/benchmark.json", backend.benchmark_json())
except Exception as e:
    print(f"  benchmark_json skipped: {e}")

# ---- bundled example -------------------------------------------------------
ex = np.load(os.path.join(REPO, "app", "examples", "example_gladstone.npz"), allow_pickle=True)
w_bin("examples/example_features.bin", ex["features"].astype("<f4").ravel())
w_json("examples/example_meta.json", {"n": int(ex["features"].shape[0]),
                                      "genes": [str(g) for g in ex["genes"]],
                                      "src": [str(s) for s in ex["src"]], "dst": [str(s) for s in ex["dst"]],
                                      "true_error": [float(v) for v in ex["true_error"]]})
w_copy("examples/example_predictions.csv", os.path.join(REPO, "app", "examples", "example_predictions.csv"))
with open(os.path.join(OUT, "examples", "template.csv"), "w") as f:
    f.write(LONG_TEMPLATE)

total = sum(sizes.values())
print(f"\nexported {len(sizes)} assets, {total/1024/1024:.2f} MB total, to {OUT}")
