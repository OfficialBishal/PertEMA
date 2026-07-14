"""Load DE_stats and sanity-check it. Run: pixi run python src/data/verify_de.py"""
import numpy as np

from load_de import CONDITIONS, load_de_stats, ontarget_index


def main():
    de = load_de_stats(layers=("log_fc", "zscore"))
    lfc = de.layers["log_fc"]
    print(f"obs={de.n_obs}  genes={de.n_genes}  log_fc {lfc.shape} {lfc.dtype} "
          f"~{lfc.nbytes/1e9:.2f} GB in mem")
    assert lfc.shape == (33983, 10282)

    print("\nculture_condition counts:")
    print(de.obs["culture_condition"].value_counts().to_string())

    # On-target knockdown check: perturbing gene G should drive G's own log_fc negative (CRISPRi).
    oti = ontarget_index(de)
    has = oti >= 0
    rows = np.where(has)[0]
    self_lfc = lfc[rows, oti[has]]
    sig = de.obs["ontarget_significant"].to_numpy()[rows].astype(bool)
    print(f"\non-target: {has.sum()} rows have their perturbed gene measured")
    print(f"  self log_fc  mean={np.nanmean(self_lfc):.3f}  median={np.nanmedian(self_lfc):.3f}  "
          f"frac<0={np.mean(self_lfc[~np.isnan(self_lfc)]<0):.3f}")
    print(f"  ontarget_significant rows: self log_fc median={np.nanmedian(self_lfc[sig]):.3f} "
          f"(n={sig.sum()})   non-sig median={np.nanmedian(self_lfc[~sig]):.3f}")
    assert np.nanmedian(self_lfc[sig]) < 0, "on-target knockdown should be negative for significant rows"

    # Reproducibility (reliability ground truth) availability
    for c in ["guide_correlation_all", "donor_correlation_all_mean", "donor_correlation_hits_mean"]:
        v = de.obs[c].to_numpy()
        print(f"\n{c}: nonnull={np.isfinite(v).sum()}  "
              f"mean={np.nanmean(v):.3f}  median={np.nanmedian(v):.3f}")

    # Effect magnitude sanity
    mag = np.nanmean(np.abs(lfc), axis=1)
    print(f"\nper-perturbation mean|log_fc|: median={np.median(mag):.4f} max={np.max(mag):.3f}")

    import os
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/verify_de_summary.md", "w") as f:
        f.write("# DE_stats read-correctness (src/data/verify_de.py)\n\n")
        f.write(f"- on-target self log_fc: fraction negative {float(np.mean(self_lfc[~np.isnan(self_lfc)]<0)):.4f}, ")
        f.write(f"median {float(np.nanmedian(self_lfc)):.3f} (n rows with measured target = {int(has.sum())})\n")
        f.write(f"- significant rows self log_fc median {float(np.nanmedian(self_lfc[sig])):.3f}\n")
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
