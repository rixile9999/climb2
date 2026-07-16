#!/usr/bin/env python3
"""Phase B5 — does the label-free recipe hold across backbones?

For each backbone (base ESM2-650M, Hie, ESM2cov), fit b0,b1 label-free against
frequency and against Bloom fitness (CLIB offset frozen), and compare to
equal-weight, CLIB-only, and DMS-supervised (in-sample reference) on the two
leakage-free honest tests: functional DMS escape and temporal emergence.

Question: is "label-free ~ supervised" robust to backbone strength, especially
for the weak base ESM2 where CLIB matters most?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data/observed"
S3_COL = "S3_logclib_z_t0.01"
BACKBONES = {"base_ESM2": "tidy_esm2_650m_l1.parquet",
             "Hie": "tidy_hie.parquet",
             "ESM2cov": "tidy_esm2cov_l1.parquet"}
CUTOFFS = [("spike_le_2021-06.tsv", "spike_2021-09_2022-03.tsv"),
           ("spike_le_2021-12.tsv", "spike_ge_2022-06.tsv"),
           ("spike_le_2022-06.tsv", "spike_ge_2023-01.tsv")]


def fit_b(S1, S2, y, off):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + off) - np.logaddexp(0, X @ w + off))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + off))))
    return minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x[1:]


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = int(y.sum()); n = len(y) - p
    return np.nan if p == 0 or n == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def wmap(df, path):
    w = pd.read_csv(path, sep="\t")
    return df["code"].map(dict(zip(w["wt"] + w["pos"].astype(str) + w["mut"], w["proportion"]))).fillna(0.0).to_numpy()


def main():
    # backbone-independent signals
    ref = pd.read_parquet(ROOT / "outputs/phase0" / BACKBONES["ESM2cov"])
    ref["code"] = ref["original_aa"] + ref["pos"].astype(str) + ref["mutated_aa"]
    allprop = wmap(ref, OBS / "spike_obs_freq.tsv")
    bl = pd.read_csv(OBS / "bloom_spike_fitness.tsv", sep="\t")
    bl["code"] = bl["clade_founder_aa"] + (bl["aa_site"] - 1).astype(str) + bl["mutant_aa"]
    bloom_hi = ref["code"].map(dict(zip(bl["code"], bl["delta_fitness"])))
    y_freq = (allprop >= 0.001).astype(int)
    y_bloom = (bloom_hi >= 2.0).fillna(False).astype(int).to_numpy()
    emg = []
    for e, l in CUTOFFS:
        ep = wmap(ref, OBS / "windows" / e); lp = wmap(ref, OBS / "windows" / l)
        emg.append(((ep < 0.05) & (lp >= 0.25)).astype(int))

    rows = []
    for bname, fn in BACKBONES.items():
        df = pd.read_parquet(ROOT / "outputs/phase0" / fn)
        S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()
        yd = df["esc__DMS"].astype(int).to_numpy()
        methods = {
            "equal(1,1)": (1.0, 1.0),
            "CLIB_only": (0.0, 0.0),
            "freq_fit": tuple(fit_b(S1, S2, y_freq, 1.0 * S3)),
            "bloom_fit": tuple(fit_b(S1, S2, y_bloom, 1.0 * S3)),
            "supervised_DMS": tuple(fit_b(S1, S2, yd, 1.0 * S3)),
        }
        for m, (b0, b1) in methods.items():
            sc = b0 * S1 + b1 * S2 + 1.0 * S3
            rows.append(dict(backbone=bname, method=m, b0=round(b0, 3), b1=round(b1, 3),
                             DMS_AUROC=round(auroc(sc, yd), 4),
                             emergent_AUROC=round(np.nanmean([auroc(sc, y) for y in emg]), 4)))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs/phase0/phaseB5_multibackbone.csv", index=False)
    pd.set_option("display.width", 200)
    for bname in BACKBONES:
        print(f"\n=== {bname} ===")
        print(res[res.backbone == bname].drop(columns="backbone").to_string(index=False))


if __name__ == "__main__":
    main()
