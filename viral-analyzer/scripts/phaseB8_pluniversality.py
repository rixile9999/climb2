#!/usr/bin/env python3
"""Phase B8 — universality across GENERAL protein LMs (ESM2 family).

Correction to B5: Hie is Hie et al.'s per-virus LSTM, NOT a general protein LM,
so it is not a valid datapoint for backbone-universality. Here we test the recipe
across four GENERAL PLMs of increasing size/adaptation on SARS-CoV-2 spike:
    ESM2-150M, ESM2-650M, ESM2-3B, ESM2cov(650M, domain-adapted).

For each: fit b0,b1 label-free (Bloom fitness) and supervised(DMS, in-sample ref),
vs equal-weight and CLIB-only, on the honest tests (DMS, temporal emergence).
Question: within general PLMs, does label-free ~ supervised hold, and does the
CLIB offset consistently help? (No Hie here — that is a separate baseline.)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data/observed"
S3_COL = "S3_logclib_z_t0.01"
PLMS = {"ESM2-150M": "tidy_esm2_150m_l1.parquet",
        "ESM2-650M": "tidy_esm2_650m_l1.parquet",
        "ESM2-3B": "tidy_esm2_3b_l1.parquet",
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
    ref = pd.read_parquet(ROOT / "outputs/phase0" / PLMS["ESM2cov"])
    ref["code"] = ref["original_aa"] + ref["pos"].astype(str) + ref["mutated_aa"]
    allprop = wmap(ref, OBS / "spike_obs_freq.tsv")
    bl = pd.read_csv(OBS / "bloom_spike_fitness.tsv", sep="\t")
    bl["code"] = bl["clade_founder_aa"] + (bl["aa_site"] - 1).astype(str) + bl["mutant_aa"]
    bloom = ref["code"].map(dict(zip(bl["code"], bl["delta_fitness"])))
    y_bloom = (bloom >= 2.0).fillna(False).astype(int).to_numpy()
    emg = []
    for e, l in CUTOFFS:
        ep = wmap(ref, OBS / "windows" / e); lp = wmap(ref, OBS / "windows" / l)
        emg.append(((ep < 0.05) & (lp >= 0.25)).astype(int))

    rows = []
    for name, fn in PLMS.items():
        df = pd.read_parquet(ROOT / "outputs/phase0" / fn)
        S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()
        yd = df["esc__DMS"].astype(int).to_numpy()
        methods = {"equal": (1., 1.), "CLIB_only": (0., 0.),
                   "bloom_fit": tuple(fit_b(S1, S2, y_bloom, 1. * S3)),
                   "supervised": tuple(fit_b(S1, S2, yd, 1. * S3))}
        for m, (b0, b1) in methods.items():
            sc = b0 * S1 + b1 * S2 + 1. * S3
            # also protein-only (no CLIB) for equal, to show CLIB-offset contribution
            rows.append(dict(PLM=name, method=m, b0=round(b0, 3), b1=round(b1, 3),
                             DMS=round(auroc(sc, yd), 4),
                             emergent=round(np.nanmean([auroc(sc, y) for y in emg]), 4)))
        # CLIB-offset contribution: equal protein-only vs equal+offset
        po = auroc(1. * S1 + 1. * S2, yd)
        rows.append(dict(PLM=name, method="protein_only(eq)", b0=1., b1=1.,
                         DMS=round(po, 4), emergent=round(np.nanmean([auroc(S1 + S2, y) for y in emg]), 4)))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs/phase0/phaseB8_pluniversality.csv", index=False)
    pd.set_option("display.width", 200)
    for name in PLMS:
        print(f"\n=== {name} ===")
        print(res[res.PLM == name].drop(columns="PLM").to_string(index=False))
    # universality summary
    lf = res[res.method == "bloom_fit"].set_index("PLM")
    su = res[res.method == "supervised"].set_index("PLM")
    print("\n=== label-free (bloom) vs supervised, DMS AUROC, across general PLMs ===")
    for p in PLMS:
        print(f"  {p:10s} bloom_fit={lf.loc[p,'DMS']}  supervised={su.loc[p,'DMS']}  gap={lf.loc[p,'DMS']-su.loc[p,'DMS']:+.4f}")


if __name__ == "__main__":
    main()
