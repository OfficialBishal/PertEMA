"""Minimal PertEMA API client (standard library only, no dependencies to install). Copy this file and call
the reliability service from your own code. Set PERTEMA_API_TOKEN in the environment if the server is
token-protected. The server never retrains on your ground truth (inviolable invariant), so evaluate() only
measures realized quality.

Example:
    from pertema_client import PertEMA
    api = PertEMA("http://localhost:8000")
    out = api.ingest_csv(open("my_predictions.csv").read())
    for gene, r in zip(out["genes"], out["results"]):
        print(gene, r["reliability"], r["band"])
"""
import json
import os
import urllib.request

CSV_HEADER = "perturbed_gene,gene,predicted_lfc,src_context,dst_context"


class PertEMA:
    def __init__(self, base_url="http://localhost:8000", token=None, timeout=120):
        self.base = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("PERTEMA_API_TOKEN", "")
        self.timeout = timeout

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read().decode() if path == "/report" else json.loads(r.read())

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def health(self):
        return self._get("/health")

    def version(self):
        return self._get("/version")

    def score(self, features):
        """features: list of (n, 64) prediction-time feature rows. Batches in one call."""
        return self._post("/score", {"features": features})

    def explain(self, features):
        """Per-perturbation grouped SHAP attribution toward predicted error (F6)."""
        return self._post("/explain", {"features": features})

    def evaluate(self, features, true_error):
        """Measure realized reliability quality on YOUR out-of-sample ground truth. Never retrains the model."""
        return self._post("/evaluate", {"features": features, "true_error": true_error})

    def ingest_csv(self, csv_text, src_default="Rest", dst_default="Stim48hr"):
        """Score a long-format predictions CSV end to end (parse, featurize, score, explain)."""
        return self._post("/ingest_score", {"csv": csv_text, "src_default": src_default, "dst_default": dst_default})

    def report_html(self, csv_text, src_default="Rest", dst_default="Stim48hr"):
        """Get a self-contained, provenance-stamped HTML reliability report for a predictions CSV."""
        return self._post("/report", {"csv": csv_text, "src_default": src_default, "dst_default": dst_default})


if __name__ == "__main__":
    api = PertEMA(os.environ.get("PERTEMA_BASE_URL", "http://localhost:8000"))
    print("health:", api.health())
    csv = CSV_HEADER + "\nIL2,IL2RA,0.83,Rest,Stim48hr\nIL2,IFNG,-0.21,Rest,Stim48hr\nSTAT5A,MYC,0.30,Rest,Stim48hr\n"
    out = api.ingest_csv(csv)
    for gene, r in zip(out["genes"], out["results"]):
        print(f"{gene}: reliability {r['reliability']:.3f} band {r['band']}")
