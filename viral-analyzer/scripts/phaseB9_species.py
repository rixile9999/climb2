#!/usr/bin/env python3
"""Phase B9 — cross-SPECIES universality of the CLIB mechanistic prior.

Tests whether the CLIB nucleotide-accessibility offset generalizes beyond
SARS-CoV-2 to Influenza HA (Doud 2018) and HIV Env (Dingens 2019), using a base
ESM2-650M backbone. For each species, 5-fold CV over the escape positives:
    protein_only    : b0*S1 + b1*S2
    protein+offset  : b0*S1 + b1*S2 + 1*CLIB
    CLIB_only       : CLIB (no training)
    equal(1,1)      : S1 + S2 + CLIB

If CLIB-offset lifts a base backbone on flu/HIV the way it does on SARS, the prior
is species-universal. (CLIB codons here are modal reverse-translations — an
approximation; exact strain CDS would refine, not change, the accessibility signal.)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
import analyzer as an

SPECIES = {
    "Influenza-HA (H1)": ("Influenza-HA-WSN-ESM2-650M-L1", "Influenza-HA-WSN", "data/species/flu_ha_escape.tsv"),
    "HIV-Env (BG505)": ("HIV-Env-BG505-ESM2-650M-L1", "HIV-Env-BG505", "data/species/hiv_env_escape.tsv"),
}
T = 0.01


def build(strain, key):
    rd = an.RESULT_DATA[key]
    df = an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns"))
    df = an.prepare_df(df, strain)
    Q = an.get_Q(strain)
    out = pd.DataFrame({
        "code": df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"],
        "S1": df["semantic_z"].to_numpy(), "S2": df["log_grammar_z"].to_numpy(),
        "S3": an.get_log_clib_z(df, Q, T),
    })
    return out


def fit(S1, S2, y, off):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + off) - np.logaddexp(0, X @ w + off))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + off))))
    return minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x[1:]


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = int(y.sum()); n = len(y) - p
    return np.nan if p == 0 or n == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def oof(S1, S2, S3, y, use_clib, seeds=5):
    preds = np.zeros(len(y))
    for sd in range(seeds):
        p = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(S1, y):
            off = 1.0 * S3[tr] if use_clib else np.zeros(len(tr))
            b = fit(S1[tr], S2[tr], y[tr], off)
            p[te] = b[0] * S1[te] + b[1] * S2[te] + (1.0 * S3[te] if use_clib else 0.0)
        preds += p / seeds
    return auroc(preds, y)


def main():
    rows = []
    for sp, (key, strain, escf) in SPECIES.items():
        d = build(strain, key)
        esc = pd.read_csv(escf, sep="\t")
        escset = set(esc["wt"] + esc["pos"].astype(str) + esc["mut"])
        y = d["code"].isin(escset).astype(int).to_numpy()
        S1, S2, S3 = d.S1.to_numpy(), d.S2.to_numpy(), d.S3.to_numpy()
        rows.append(dict(
            species=sp, n_cand=len(y), n_esc=int(y.sum()),
            equal=round(auroc(S1 + S2 + S3, y), 4),
            protein_only=round(oof(S1, S2, S3, y, False), 4),
            protein_offset=round(oof(S1, S2, S3, y, True), 4),
            CLIB_only=round(auroc(S3, y), 4),
        ))
    res = pd.DataFrame(rows)
    res.to_csv("outputs/phase0/phaseB9_species.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== Cross-species CLIB universality (base ESM2-650M, 5-fold CV) ===")
    print(res.to_string(index=False))
    print("\nSARS ref (base ESM2, phaseB8): protein_only~0.50, +offset~0.88, CLIB_only 0.883")


if __name__ == "__main__":
    main()
