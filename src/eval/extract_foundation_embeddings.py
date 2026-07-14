"""D1 foundation-model roster: extract PRETRAINED gene embeddings from transcriptomic foundation models and
save them as plain {gene_ids, embedding} npz files in the Ensembl namespace of the parity harness. These are
then consumed by run_parity_foundation.py (in the default env) as frozen-adapted predictors (a light ridge or
kNN head over the frozen embedding), exactly parallel to the co-expression ridge_embed / knn predictors.

Runs in the ISOLATED deep pixi environment (pixi run -e deep python ...), so the reproducible scientific env
is never touched. Only pretrained weights are read, nothing is trained here.

Provenance: foundation models are pretrained on large public single-cell corpora that may overlap the
benchmark cell types, so every foundation predictor carries the UNKNOWN-OVERLAP flag downstream (invariant 2).

Usage:
    pixi run -e deep python src/eval/extract_foundation_embeddings.py geneformer
    pixi run -e deep python src/eval/extract_foundation_embeddings.py scgpt
    pixi run -e deep python src/eval/extract_foundation_embeddings.py gene2vec
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

OUT = "results/features"
FOLDS = "results/splits/gene_folds.csv"        # gene (Ensembl) <-> gene_name (symbol) map lives here


def _sym2ens():
    f = pd.read_csv(FOLDS, dtype=str)
    return dict(zip(f["gene_name"], f["gene"]))


def _save(name, gene_ids, emb):
    emb = np.asarray(emb, np.float32)
    path = os.path.join(OUT, f"foundation_{name}.npz")
    np.savez(path, gene_ids=np.array([str(g) for g in gene_ids]), embedding=emb)
    print(f"wrote {path}: {emb.shape[0]} genes x {emb.shape[1]} dims (namespace Ensembl)")


def extract_geneformer():
    """Geneformer V1-10M (Theodoris 2023): a masked-LM foundation model over Ensembl gene tokens. The token
    embedding table is the pretrained per-gene representation. Ensembl-native, maps straight to the harness."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    repo = "ctheodoris/Geneformer"
    st = hf_hub_download(repo, "Geneformer-V1-10M/model.safetensors")
    tok = hf_hub_download(repo, "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
    token_dict = pickle.load(open(tok, "rb"))          # {ensembl_id or <special>: token_index}
    with safe_open(st, framework="np") as f:
        keys = list(f.keys())
        wkey = next((k for k in keys if k.endswith("embeddings.word_embeddings.weight")), None)
        assert wkey, f"no word_embeddings key in {keys[:8]}"
        W = f.get_tensor(wkey)                          # (vocab, hidden)
    gene_ids, rows = [], []
    for g, idx in token_dict.items():
        if str(g).startswith("<") or not str(g).startswith("ENSG"):
            continue                                    # skip <pad>/<mask>/<cls> and non-Ensembl tokens
        if 0 <= idx < W.shape[0]:
            gene_ids.append(str(g)); rows.append(W[idx])
    _save("geneformer", gene_ids, np.array(rows))


def extract_scgpt():
    """scGPT whole-human (Cui 2024): a generative foundation model. Its gene-token embedding table is the
    pretrained per-gene representation. scGPT keys genes by SYMBOL, mapped to Ensembl via gene_name."""
    from huggingface_hub import hf_hub_download
    import json
    import torch
    # the whole-human checkpoint mirrored on the Hub
    candidates = [("wanglab/scGPT-human", "best_model.pt", "vocab.json"),
                  ("VirtualCell2025/scGPT_human", "best_model.pt", "vocab.json"),
                  ("MohamedMabrouk/scGPT", "best_model.pt", "vocab.json")]
    ckpt = vocabf = None
    for repo, mf, vf in candidates:
        try:
            ckpt = hf_hub_download(repo, mf); vocabf = hf_hub_download(repo, vf); break
        except Exception as e:
            print(f"  scgpt repo {repo} not available: {type(e).__name__}")
    if ckpt is None:
        print("scGPT checkpoint not reachable on the Hub; skipping (documented fallback)."); return
    vocab = json.load(open(vocabf))                     # {gene_symbol: token_index}
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    wkey = next((k for k in sd if k.endswith("encoder.embedding.weight") or k.endswith("gene_encoder.embedding.weight")), None)
    assert wkey, f"no gene embedding key in {list(sd)[:8]}"
    W = sd[wkey].float().numpy()
    s2e = _sym2ens()
    gene_ids, rows = [], []
    for sym, idx in vocab.items():
        ens = s2e.get(sym)
        if ens and 0 <= idx < W.shape[0]:
            gene_ids.append(ens); rows.append(W[idx])
    _save("scgpt", gene_ids, np.array(rows))


def extract_gene2vec():
    """gene2vec (Du 2019): the pretrained gene co-expression embedding GEARS uses as node features. Distinct
    from the transformer foundation models. Labeled honestly as gene2vec, not GEARS-the-GNN. Symbol-keyed."""
    import urllib.request
    url = "https://raw.githubusercontent.com/jingcheng-du/Gene2vec/master/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
    dst = os.path.join(OUT, "gene2vec_raw.txt")
    if not os.path.exists(dst):
        print("downloading gene2vec ...")
        urllib.request.urlretrieve(url, dst)
    s2e = _sym2ens()
    gene_ids, rows = [], []
    with open(dst) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 10:
                continue
            sym = parts[0]; ens = s2e.get(sym)
            if ens:
                gene_ids.append(ens); rows.append([float(x) for x in parts[1:]])
    _save("gene2vec", gene_ids, np.array(rows))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "geneformer"
    {"geneformer": extract_geneformer, "scgpt": extract_scgpt, "gene2vec": extract_gene2vec}[which]()
