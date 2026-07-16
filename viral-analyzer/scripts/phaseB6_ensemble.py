#!/usr/bin/env python3
"""Phase B6 — does an ENSEMBLE label-free target beat the best single signal?

Composite target = rank-sum of three standardized label-free signals
(log frequency, Bloom fitness, early frequency); positives = top-K. Fit b0,b1
(CLIB frozen) on it, then compare to the best single-signal fit and to supervised
on the honest tests (DMS, per-cutoff emergence). ESM2cov backbone.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data/observed"
S3_COL = "S3_logclib_z_t0.01"
CUTOFFS = [("C1", "spike_le_2021-06.tsv", "spike_2021-09_2022-03.tsv"),
           ("C2", "spike_le_2021-12.tsv", "spike_ge_2022-06.tsv"),
           ("C3", "spike_le_2022-06.tsv", "spike_ge_2023-01.tsv")]


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
    df = pd.read_parquet(ROOT / "outputs/phase0/tidy_esm2cov_l1.parquet")
    df["code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()

    allprop = wmap(df, OBS / "spike_obs_freq.tsv")
    early = wmap(df, OBS / "windows/spike_le_2021-12.tsv")
    bl = pd.read_csv(OBS / "bloom_spike_fitness.tsv", sep="\t")
    bl["code"] = bl["clade_founder_aa"] + (bl["aa_site"] - 1).astype(str) + bl["mutant_aa"]
    bloom = df["code"].map(dict(zip(bl["code"], bl["delta_fitness"]))).fillna(bl["delta_fitness"].min()).to_numpy()

    # composite rank-sum target
    comp = rankdata(np.log(allprop + 1e-9)) + rankdata(bloom) + rankdata(np.log(early + 1e-9))
    y_ens = (comp >= np.percentile(comp, 100 * (1 - 210 / len(comp)))).astype(int)   # ~210 positives

    yd = df["esc__DMS"].astype(int).to_numpy()
    methods = {
        "equal(1,1)": (1.0, 1.0),
        "freq_fit": tuple(fit_b(S1, S2, (allprop >= 0.001).astype(int), 1.0 * S3)),
        "bloom_fit": tuple(fit_b(S1, S2, (bloom >= 2.0).astype(int), 1.0 * S3)),
        "ensemble_fit": tuple(fit_b(S1, S2, y_ens, 1.0 * S3)),
        "supervised_DMS": tuple(fit_b(S1, S2, yd, 1.0 * S3)),
    }
    emg = {}
    for name, e, l in CUTOFFS:
        ep = wmap(df, OBS / "windows" / e); lp = wmap(df, OBS / "windows" / l)
        emg[name] = ((ep < 0.05) & (lp >= 0.25)).astype(int)

    rows = []
    for m, (b0, b1) in methods.items():
        sc = b0 * S1 + b1 * S2 + 1.0 * S3
        row = dict(method=m, b0=round(b0, 3), b1=round(b1, 3), DMS=round(auroc(sc, yd), 4))
        for name in emg:
            row[name] = round(auroc(sc, emg[name]), 4)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs/phase0/phaseB6_ensemble.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== Ensemble label-free target vs single-signal (ESM2cov) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
