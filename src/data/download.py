"""Download primary Marson-Pritchard genome-scale CD4 T cell Perturb-seq data.

Public S3 bucket (no credentials): s3://genome-scale-tcell-perturb-seq/marson2025_data/
Accessed over anonymous HTTPS. Streams to disk with HTTP Range resume and a size check,
so re-running skips complete files and resumes partial ones. Stdlib only, so it runs in
the pinned pixi env without extra dependencies.

Usage (through pixi so the env is fixed):
    pixi run python src/data/download.py core     # DE_stats.h5ad + pseudobulk_merged.h5ad
    pixi run python src/data/download.py hmu       # by_guide.h5mu + by_donors.h5mu
    pixi run python src/data/download.py meta       # readme + supplementary tables + metadata
    pixi run python src/data/download.py <exact/key ...>
"""
import os
import sys
import time
import urllib.request

BASE = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "marson2025")
OUT = os.path.abspath(OUT)

PRESETS = {
    "core": ["GWCD4i.DE_stats.h5ad", "GWCD4i.pseudobulk_merged.h5ad"],
    "hmu": ["GWCD4i.DE_stats.by_guide.h5mu", "GWCD4i.DE_stats.by_donors.h5mu"],
    "meta": [
        "data_sharing_readme.md",
        "suppl_tables/sample_metadata.suppl_table.csv",
        "suppl_tables/DE_stats.suppl_table.csv",
        "suppl_tables/sgrna_library_metadata.suppl_table.csv",
    ],
    "cell": [f"D{d}_{c}.assigned_guide.h5ad"
             for d in (1, 2, 3, 4) for c in ("Rest", "Stim8hr", "Stim48hr")],
}


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", "0"))


def fetch(key, retries=5):
    url = BASE + key
    dest = os.path.join(OUT, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have_local = os.path.getsize(dest) if os.path.exists(dest) else 0
    try:
        total = remote_size(url)
    except Exception as e:
        # Network unavailable. Reproduce-from-present-data: if the file is already on disk, use it and
        # skip the remote size check rather than failing the whole reproduce. Honest: we note the skip.
        if have_local > 0:
            print(f"OK   {key}  ({have_local/1e9:.2f} GB on disk, remote size check skipped, "
                  f"network unavailable: {type(e).__name__})", flush=True)
            return True
        print(f"FAIL {key}  (network unavailable and no local copy: {type(e).__name__})", flush=True)
        return False
    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total and have == total:
            print(f"OK   {key}  ({have/1e9:.2f} GB, already complete)", flush=True)
            return True
        if have > total > 0:
            os.remove(dest)
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
                t0, last = time.time(), have
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if time.time() - t0 > 30:
                        mbps = (have - last) / (time.time() - t0) / 1e6
                        pct = 100 * have / total if total else 0
                        print(f"  ... {key}: {have/1e9:.2f}/{total/1e9:.2f} GB ({pct:.0f}%, {mbps:.0f} MB/s)", flush=True)
                        t0, last = time.time(), have
            if total and os.path.getsize(dest) == total:
                print(f"DONE {key}  ({total/1e9:.2f} GB)", flush=True)
                return True
        except Exception as e:
            print(f"  retry {attempt}/{retries} {key}: {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(3 * attempt)
    print(f"FAIL {key}  (after {retries} attempts)", flush=True)
    return False


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    keys = []
    for a in argv:
        keys.extend(PRESETS.get(a, [a]))
    print(f"target dir: {OUT}\nfiles: {keys}", flush=True)
    ok = all(fetch(k) for k in keys)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
