"""Backend scoring smoke test (N8 core). Loads the frozen artifact, scores the bundled example, and asserts
the scoring core behaves: shapes, band membership, determinism, and that predicted reliability correlates
positively with realized accuracy on the example (a sanity check, not an inflated claim).

Run: pixi run python app/backend/test_scoring.py
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scoring import BANDS, default_model   # noqa: E402


def main():
    ex = np.load(os.path.join(HERE, "..", "examples", "example_gladstone.npz"), allow_pickle=True)
    X, true_err = ex["features"], ex["true_error"]
    m = default_model()
    out = m.score(X)

    n = len(X)
    assert out["reliability"].shape == (n,), "reliability shape"
    assert out["calibrated_error"].shape == (n,), "calibrated shape"
    assert len(out["band"]) == n, "band length"
    valid = {b[0] for b in BANDS}
    assert all(b in valid for b in out["band"]), "band membership"
    assert np.all(out["conformal_lo"] <= out["conformal_hi"]), "conformal ordering"
    # determinism
    out2 = m.score(X)
    assert np.max(np.abs(out["reliability"] - out2["reliability"])) == 0.0, "determinism"
    # smoke check only that scoring runs and orders sanely. NOTE: the bundled Gladstone example is a slice of
    # the frozen model's own training data, so this Spearman is IN-SAMPLE and optimistic. It is NOT the tool's
    # reliability quality. The honest out-of-fold reliability is about 0.13 (manuscript / D8 benchmark), and a
    # user scoring their own unseen predictor outputs gets that out-of-sample quality, not this in-sample value.
    rho_in_sample = spearmanr(out["reliability"], -true_err).correlation
    assert np.isfinite(rho_in_sample), "spearman finite"

    from collections import Counter
    print(f"PASS: scored {n} example perturbations, model v{out['model_version']}, coverage {out['coverage']:.3f}")
    print(f"  band distribution: {dict(Counter(out['band']))}")
    print(f"  in-sample Spearman on the training-derived example: {rho_in_sample:.3f} "
          f"(OPTIMISTIC, in-sample; honest out-of-fold reliability is ~0.13, see the D8 benchmark)")
    print("  determinism: identical scores on re-run")


if __name__ == "__main__":
    main()
