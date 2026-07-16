#!/usr/bin/env python3
"""Phase B4 — cross-signal generalization of the label-free weights.

Fit b0,b1 (CLIB offset frozen) against THREE independent label-free statistical
signals, then cross-evaluate every fitted weight on every escape / fitness target.

  fitting signals (no escape labels):
    freq   : all-time observed frequency  >= 0.001
    bloom  : Bloom phylogenetic fitness   >= 2.0   (missing -> negative)
    temp   : pre-2022 (early) frequency   >= 0.001

  evaluation targets (AUROC, higher=better):
    DMS               : functional escape (leakage-free)
    variant_avg       : mean over Alpha/Beta/Gamma/Delta/Omicron escape
    emergent_avg      : mean over 3 temporal cutoffs (future risers)
    bloom_hi          : Bloom high-fitness set

Question: do the three signals give the SAME weight direction (a universal
label-free operating point), and do they generalize across targets?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "outputs/phase0/tidy_esm2cov_l1.parquet"
OBS = ROOT / "data/observed"
S3_COL = "S3_logclib_z_t0.01"
VARIANTS = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]
CUTOFFS = [("spike_le_2021-06.tsv", "spike_2021-09_2022-03.tsv"),
           ("spike_le_2021-12.tsv", "spike_ge_2022-06.tsv"),
           ("spike_le_2022-06.tsv", "spike_ge_2023-01.tsv")]


def fit_b(S1, S2, y, offset):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + offset) - np.logaddexp(0, X @ w + offset))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + offset))))
    return minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x[1:]


def auroc(score, y):
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    p = int(y.sum()); n = len(y) - p
    return np.nan if p == 0 or n == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def wcode(df, path):
    w = pd.read_csv(path, sep="\t")
    return df["code"].map(dict(zip(w["wt"] + w["pos"].astype(str) + w["mut"], w["proportion"]))).fillna(0.0).to_numpy()


def main():
    df = pd.read_parquet(TIDY)
    df["code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()

    allprop = wcode(df, OBS / "spike_obs_freq.tsv")
    early = wcode(df, OBS / "windows/spike_le_2021-12.tsv")
    bl = pd.read_csv(OBS / "bloom_spike_fitness.tsv", sep="\t")
    bl["code"] = bl["clade_founder_aa"] + (bl["aa_site"] - 1).astype(str) + bl["mutant_aa"]
    bloomv = df["code"].map(dict(zip(bl["code"], bl["delta_fitness"])))

    # --- fitting signals ---
    signals = {
        "freq":  (allprop >= 0.001).astype(int),
        "bloom": (bloomv >= 2.0).fillna(False).astype(int).to_numpy(),
        "temp":  (early >= 0.001).astype(int),
    }
    weights = {}
    for name, y in signals.items():
        b0, b1 = fit_b(S1, S2, y, 1.0 * S3)
        weights[name] = (b0, b1)
    # reference operating points
    weights["equal(1,1)"] = (1.0, 1.0)
    weights["CLIB_only"] = (0.0, 0.0)

    # --- evaluation targets ---
    def emergent_targets():
        out = []
        for e, l in CUTOFFS:
            ep = wcode(df, OBS / "windows" / e); lp = wcode(df, OBS / "windows" / l)
            out.append(((ep < 0.05) & (lp >= 0.25)).astype(int))
        return out
    emg = emergent_targets()
    bloom_hi = (bloomv >= 2.0).fillna(False).astype(int).to_numpy()

    def eval_target(score):
        dms = auroc(score, df["esc__DMS"].astype(int).to_numpy())
        var = np.nanmean([auroc(score, df[f"esc__{v}"].astype(int).to_numpy()) for v in VARIANTS])
        emm = np.nanmean([auroc(score, y) for y in emg])
        bh = auroc(score, bloom_hi)
        return dms, var, emm, bh

    rows = []
    for name, (b0, b1) in weights.items():
        sc = b0 * S1 + b1 * S2 + 1.0 * S3
        dms, var, emm, bh = eval_target(sc)
        rows.append(dict(fit_signal=name, b0=round(b0, 3), b1=round(b1, 3),
                         DMS=round(dms, 4), variant_avg=round(var, 4),
                         emergent_avg=round(emm, 4), bloom_hi=round(bh, 4)))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs/phase0/phaseB4_crosssignal.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== Cross-signal label-free weights & generalization (ESM2cov·L1) ===")
    print(res.to_string(index=False))
    # weight convergence summary
    lf = res[res["fit_signal"].isin(["freq", "bloom", "temp"])]
    print(f"\nlabel-free weight spread:  b0 {lf.b0.min():.3f}-{lf.b0.max():.3f}   b1 {lf.b1.min():.3f}-{lf.b1.max():.3f}")


if __name__ == "__main__":
    main()
