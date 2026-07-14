"""API endpoint tests (F10 / N9) using FastAPI's TestClient, no running server needed.

Run: pixi run python app/backend/test_api.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fastapi.testclient import TestClient   # noqa: E402
import main                                 # noqa: E402

client = TestClient(main.app)


def main_test():
    # health
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok", "health"
    # version / provenance
    r = client.get("/version"); prov = r.json()
    assert r.status_code == 200 and prov["provenance_flag"] == "CLEAN", "version"
    assert "measured_accuracy_ceiling_1minus_pearson" in prov, "ceiling in provenance"
    # example scoring with the honest in-sample warning
    r = client.get("/example"); ex = r.json()
    assert r.status_code == 200 and len(ex["results"]) == 400, "example n"
    assert "in_sample_warning" in ex, "example carries the in-sample warning"
    assert set(ex["results"][0]) == {"reliability", "calibrated_error", "band", "conformal_interval"}, "result schema"
    assert abs(ex["calibration_coverage"] - 0.9) < 0.01, "coverage surfaced near target"
    # score a feature matrix
    feats = np.load(os.path.join(HERE, "..", "examples", "example_gladstone.npz"))["features"][:5].tolist()
    r = client.post("/score", json={"features": feats})
    assert r.status_code == 200 and len(r.json()["results"]) == 5, "score"
    # malformed input rejected cleanly (N3), not a crash
    r = client.post("/score", json={"features": [[1.0, 2.0]]})
    assert r.status_code == 422, "bad feature dim rejected with 422"
    # N5 performance: batch scoring is fast (frozen GBT + isotonic + conformal, CPU). Generous ceiling so the
    # assertion is stable across machines, while the measured throughput is documented on /api.
    import time as _t
    big = np.repeat(np.load(os.path.join(HERE, "..", "examples", "example_gladstone.npz"))["features"],
                    3, axis=0)[:1000].tolist()
    _t0 = _t.perf_counter(); rp = client.post("/score", json={"features": big}); dt = _t.perf_counter() - _t0
    assert rp.status_code == 200 and len(rp.json()["results"]) == 1000, "batch score 1000"
    assert dt < 2.0, f"scoring 1000 perturbations should be well under 2s (took {dt*1000:.0f} ms incl. HTTP)"
    # evaluate mode: post the bundled example features + true_error, get realized quality + honest warnings
    ex_npz = np.load(os.path.join(HERE, "..", "examples", "example_gladstone.npz"))
    ev_feats = ex_npz["features"].tolist()
    ev_true = ex_npz["true_error"].tolist()
    r = client.post("/evaluate", json={"features": ev_feats, "true_error": ev_true})
    assert r.status_code == 200, "evaluate ok"
    ev = r.json()
    assert np.isfinite(ev["spearman_reliability_vs_accuracy"]), "evaluate spearman finite"
    assert np.isfinite(ev["risk_coverage_auc"]), "evaluate risk-coverage auc finite"
    assert len(ev["calibration"]) > 0, "evaluate calibration bins present"
    assert any("LEAKAGE" in w for w in ev["warnings"]) and any("SPLIT" in w for w in ev["warnings"]), \
        "evaluate carries explicit leakage and split warnings"
    # mismatched lengths rejected cleanly (N3), not a crash
    r = client.post("/evaluate", json={"features": ev_feats[:5], "true_error": ev_true[:4]})
    assert r.status_code == 422, "evaluate length mismatch rejected with 422"
    # F1: template + ingest a user's long-format predictions CSV, featurize, score
    r = client.get("/template"); assert r.status_code == 200 and "perturbed_gene" in r.text, "template"
    user_csv = ("perturbed_gene,gene,predicted_lfc,src_context,dst_context\n"
                "IL2,IL2RA,0.83,Rest,Stim48hr\nIL2,IFNG,-0.21,Rest,Stim48hr\nIL2,FOXP3,0.11,Rest,Stim48hr\n"
                "STAT5A,IL2RA,0.44,Rest,Stim48hr\nSTAT5A,MYC,0.30,Rest,Stim48hr\n"
                "BATF,JUN,0.5,Rest,Stim48hr\nBATF,FOS,0.4,Rest,Stim48hr\n")
    r = client.post("/ingest_score", json={"csv": user_csv}); assert r.status_code == 200, f"ingest_score {r.text[:200]}"
    ig = r.json()
    assert len(ig["results"]) == 3 and ig["featurize_report"]["mapped_fraction"] == 1.0, "ingest 3 mapped perturbations"
    assert ig["ingestion_report"]["n_perturbations"] == 3, "ingestion report"
    # F6 explainability: ingest carries per-perturbation grouped attributions + a co-expression neighborhood
    assert len(ig["explanations"]) == 3 and ig["explanations"][0]["top_drivers"], "ingest carries explanations"
    grp = ig["explanations"][0]["group_contributions_toward_error"]
    assert "predicted_magnitude" in grp and "coexpression_embedding" in grp, "grouped attribution keys"
    assert isinstance(ig["neighbors"], dict) and len(ig["neighbors"]) == 3, "ingest carries gene neighborhoods"
    # dedicated /explain endpoint over a feature matrix
    r = client.post("/explain", json={"features": feats})
    ex_out = r.json()
    assert r.status_code == 200 and len(ex_out["explanations"]) == 5, "explain endpoint"
    assert ex_out["explanations"][0]["top_drivers"][0]["direction"] in ("more reliable", "less reliable"), "driver direction"
    # F8: self-contained HTML report with the provenance stamp
    r = client.post("/report", json={"csv": user_csv})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), "report is html"
    rep = r.text
    assert "<!doctype html>" in rep.lower() and "PertEMA reliability report" in rep, "report structure"
    assert "provenance flag" in rep.lower() and "build commit" in rep.lower(), "report carries provenance stamp"
    assert "IL2" in rep and "http://" not in rep and "https://" not in rep, "report self-contained (no remote refs)"
    # N1: build commit surfaced on /version
    assert "build_commit" in client.get("/version").json(), "version carries build commit (N1)"
    # bad context is rejected/reported, unmapped gene handled
    bad_ctx = client.post("/ingest_score", json={"csv": "perturbed_gene,gene,predicted_lfc,src_context,dst_context\nIL2,IL2RA,0.5,Rest,BOGUS\n"})
    assert bad_ctx.status_code == 422, "all-bad-context ingest rejected"
    # missing required column -> precise 422
    r = client.post("/ingest_score", json={"csv": "foo,bar\n1,2\n"}); assert r.status_code == 422, "bad schema 422"
    # benchmark: rendered HTML page at /benchmark, raw JSON resource moved to /benchmark.json (P0-4)
    r = client.get("/benchmark.json"); rows = r.json()
    assert r.status_code == 200 and len(rows) > 50 and "provenance" in rows[0], "benchmark.json D8 resource"
    rp = client.get("/benchmark")
    assert rp.status_code == 200 and "<!doctype html>" in rp.text.lower() and "benchmark" in rp.text.lower(), \
        "benchmark serves a rendered HTML page"
    # F4 context-transfer heatmap: 3x3 difficulty matrix, diagonal (within-context) easier than off-diagonal
    r = client.get("/transfer_heatmap"); hm = r.json()
    assert r.status_code == 200 and len(hm["matrix"]) == 3 and len(hm["matrix"][0]) == 3, "transfer heatmap 3x3"
    assert hm["conditions"] == ["Rest", "Stim8hr", "Stim48hr"], "transfer heatmap conditions"
    diag = np.mean([hm["matrix"][i][i] for i in range(3)])
    offdiag = np.mean([hm["matrix"][i][j] for i in range(3) for j in range(3) if i != j])
    assert diag < offdiag, f"within-context transfer easier than cross-context ({diag:.3f} < {offdiag:.3f})"
    # OpenAPI docs auto-generated
    assert client.get("/openapi.json").status_code == 200, "openapi"
    # frontend page served, the working-app-with-the-example flow
    r = client.get("/"); idx = r.text
    assert r.status_code == 200 and "PertEMA" in idx and "Run the bundled example" in idx, "index page"
    # N6 accessibility: keyboard focus ring, ARIA on charts/controls, and a documented colorblind-safe palette
    assert "focus-visible" in idx and 'role="img"' in idx and "aria-label" in idx, "index has focus + ARIA"
    # F2: built-in examples (the /example built-in predictor flow + a one-click load-built-in-example control)
    assert "Run the bundled example" in idx and "Load the built-in example" in idx, "built-in example controls"
    assert "colorblind-safe" in client.get("/faq").text.lower() or "okabe" in client.get("/faq").text.lower(), \
        "faq documents the colorblind-safe palette"
    # methods page carries the honesty content and citations (N2)
    r = client.get("/methods"); assert r.status_code == 200, "methods page"
    mtext = r.text
    assert "Ahlmann-Eltze" in mtext and "conformal" in mtext.lower(), "methods cites references"
    assert "do NOT claim" in mtext or "not recover" in mtext.lower(), "methods carries the honest negatives"
    # interpret-your-results, FAQ, and API docs pages render (F11 / F10)
    for pg in ["/interpret", "/faq", "/api"]:
        r = client.get(pg); assert r.status_code == 200 and len(r.text) > 300, f"{pg} renders"
    # F10: the API docs page shows curl + the Python client and links the interactive schema
    ap_text = client.get("/api").text
    assert "curl" in ap_text and "pertema_client" in ap_text and "/openapi.json" in ap_text, "api page documents usage"
    assert client.get("/redoc").status_code == 200, "redoc served"
    print(f"PASS: / (frontend) /health /version /example ({len(ex['results'])} scored) /score /evaluate /benchmark ({len(rows)} rows) /openapi")
    print(f"  coverage {ex['calibration_coverage']}, model v{ex['model_version']}, malformed input -> 422 (no crash)")
    print(f"  evaluate: spearman {ev['spearman_reliability_vs_accuracy']:.3f}, risk-coverage AUC {ev['risk_coverage_auc']:.3f}, {len(ev['calibration'])} calibration bins")


if __name__ == "__main__":
    main_test()
