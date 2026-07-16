#!/usr/bin/env python3
"""Phase B3 — de-leaked and higher-power DMS-free evaluation.

(2) Leave-one-variant-out (LOVO) de-leaking.
    The frequency proxy is circular for a variant task because that variant's
    defining mutations ARE the high-frequency ones. Fix: when evaluating variant
    V, fit b0,b1 on the frequency proxy with V's escape mutations REMOVED from the
    fitting data, so the weights never see V's answers. Then evaluate on V.

(3) Multi-cutoff temporal emergence.
    Three date cutoffs, each with a pre-cutoff (early) and post-cutoff (late)
    frequency window. Emergent positives = rare early, common late. proxy_early is
    fit on the early window ONLY (temporally honest). Report AUROC per cutoff and
    a bootstrap CI on the proxy_early - equal-weight AUROC gap.

Writes: outputs/phase0/phaseB3_lovo.csv, phaseB3_temporal.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "outputs/phase0/tidy_esm2cov_l1.parquet"
S3_COL = "S3_logclib_z_t0.01"
AAS = set("ACDEFGHIKLMNPQRSTVWY")
VARIANTS = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]

WIN = ROOT / "data/observed/windows"
CUTOFFS = [   # (name, early_tsv, late_tsv)  -- see scripts/fetch_observed_freq.py
    ("C1 2021-06 -> 21Q4/22Q1", WIN / "spike_le_2021-06.tsv", WIN / "spike_2021-09_2022-03.tsv"),
    ("C2 2021-12 -> 22H2+",     WIN / "spike_le_2021-12.tsv", WIN / "spike_ge_2022-06.tsv"),
    ("C3 2022-06 -> 2023+",     WIN / "spike_le_2022-06.tsv", WIN / "spike_ge_2023-01.tsv"),
]


def load_window(path):
    w = pd.read_csv(path, sep="\t")
    code = w["wt"] + w["pos"].astype(str) + w["mut"]
    return dict(zip(code, w["proportion"]))


def fit_b(S1, S2, y, offset):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + offset) - np.logaddexp(0, X @ w + offset))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + offset))))
    w = minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x
    return w[1], w[2]


def mean_rank(score, y):
    r = pd.Series(-score).rank(method="min").to_numpy()
    return float(r[y == 1].mean())


def auroc(score, y):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return np.nan if npos == 0 or nneg == 0 else (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auroc_ci(sc_a, sc_b, y, B=2000, seed=0):
    """95% CI for AUROC(a) - AUROC(b), bootstrapping positives (neg fixed)."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(B):
        p = rng.choice(pos, len(pos), replace=True)
        idx = np.concatenate([p, neg]); yy = np.concatenate([np.ones(len(p)), np.zeros(len(neg))])
        ds.append(auroc(sc_a[idx], yy) - auroc(sc_b[idx], yy))
    return np.percentile(ds, [2.5, 97.5])


def main():
    df = pd.read_parquet(TIDY)
    df["code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()

    fr = pd.read_csv(ROOT / "data/observed/spike_obs_freq.tsv", sep="\t")
    fr["code"] = fr["wt"] + fr["pos"].astype(str) + fr["mut"]
    allprop = df["code"].map(dict(zip(fr["code"], fr["proportion"]))).fillna(0.0).to_numpy()

    def score(b0, b1):
        return b0 * S1 + b1 * S2 + 1.0 * S3
    eq = score(1.0, 1.0)

    # ---- (2) LOVO -----------------------------------------------------------
    lovo_rows = []
    b0f, b1f = fit_b(S1, S2, (allprop >= 0.001).astype(int), 1.0 * S3)   # full proxy (leaky)
    for V in VARIANTS:
        y = df[f"esc__{V}"].astype(int).to_numpy()
        if y.sum() == 0:
            continue
        keep = y == 0                                    # drop V's escape muts from the fit
        yprox = (allprop[keep] >= 0.001).astype(int)
        b0, b1 = fit_b(S1[keep], S2[keep], yprox, 1.0 * S3[keep])
        sc_lovo = score(b0, b1)
        lovo_rows.append(dict(
            variant=V, n_pos=int(y.sum()),
            equal_AUROC=round(auroc(eq, y), 4),
            proxy_full_AUROC=round(auroc(score(b0f, b1f), y), 4),
            proxy_LOVO_AUROC=round(auroc(sc_lovo, y), 4),
            equal_meanrank=round(mean_rank(eq, y), 1),
            proxy_LOVO_meanrank=round(mean_rank(sc_lovo, y), 1),
            b0_lovo=round(b0, 3), b1_lovo=round(b1, 3),
            dAUROC_LOVO_vs_equal=round(auroc(sc_lovo, y) - auroc(eq, y), 4),
            CI=str([round(x, 4) for x in auroc_ci(sc_lovo, eq, y)]),
        ))
    lovo = pd.DataFrame(lovo_rows)
    lovo.to_csv(ROOT / "outputs/phase0/phaseB3_lovo.csv", index=False)

    # ---- (3) multi-cutoff temporal ------------------------------------------
    temp_rows = []
    for name, ejson, ljson in CUTOFFS:
        early = df["code"].map(load_window(ejson)).fillna(0.0).to_numpy()
        late = df["code"].map(load_window(ljson)).fillna(0.0).to_numpy()
        emergent = ((early < 0.05) & (late >= 0.25)).astype(int)
        if emergent.sum() < 3:
            temp_rows.append(dict(cutoff=name, n_emergent=int(emergent.sum()), note="too few"))
            continue
        b0e, b1e = fit_b(S1, S2, (early >= 0.001).astype(int), 1.0 * S3)   # honest: early only
        sc_pe = score(b0e, b1e)
        ci = auroc_ci(sc_pe, eq, emergent)
        temp_rows.append(dict(
            cutoff=name, n_emergent=int(emergent.sum()),
            equal_AUROC=round(auroc(eq, emergent), 4),
            CLIBonly_AUROC=round(auroc(score(0, 0), emergent), 4),
            proxy_early_AUROC=round(auroc(sc_pe, emergent), 4),
            dAUROC_proxyEarly_vs_equal=round(auroc(sc_pe, emergent) - auroc(eq, emergent), 4),
            CI=str([round(x, 4) for x in ci]),
            b0_early=round(b0e, 3), b1_early=round(b1e, 3),
        ))
    temp = pd.DataFrame(temp_rows)
    temp.to_csv(ROOT / "outputs/phase0/phaseB3_temporal.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("=== (2) Leave-one-variant-out de-leaked (variant escape tasks) ===")
    print(lovo.to_string(index=False))
    print("\n=== (3) Multi-cutoff temporal emergence (proxy_early = honest) ===")
    print(temp.to_string(index=False))


if __name__ == "__main__":
    main()
