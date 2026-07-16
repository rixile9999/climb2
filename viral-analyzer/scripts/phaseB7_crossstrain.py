#!/usr/bin/env python3
"""Phase B7 — strong-backbone cross-strain transfer (the flagship open question).

The offset research (Phase 6) could only test cross-strain transfer with the WEAK
base ESM2, because ESM2cov existed only for WildType. Here ESM2cov single-mutant
features have been generated for all five variant strains, so we can test whether
the STRONG domain-adapted backbone transfers across strains.

Leave-one-strain-out over {WildType, Alpha, Beta, Gamma, Delta, Omicron}: fit on
the pooled other strains' mutations (with their >DMS escape labels), predict the
held-out strain. Three models, mirroring Phase 6:
    protein_only     : b0*S1 + b1*S2                  (no CLIB)
    protein+offset   : b0*S1 + b1*S2 + 1*CLIB         (CLIB frozen)
    CLIB_only        : CLIB                            (no training)

Compare mean AUROC to Phase 6's base-ESM baseline (protein-only 0.60,
protein+offset 0.86, CLIB-only 0.864).
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
import analyzer as an

STRAINS = ["WildType", "Alpha", "Beta", "Gamma", "Delta", "Omicron"]
BACKBONE = {  # strain -> registered ESM2cov result key
    "WildType": "SARS-CoV-2-WildType-ESM2cov-L1",
    "Alpha": "SARS-CoV-2-Alpha-ESM2cov-L1", "Beta": "SARS-CoV-2-Beta-ESM2cov-L1",
    "Gamma": "SARS-CoV-2-Gamma-ESM2cov-L1", "Delta": "SARS-CoV-2-Delta-ESM2cov-L1",
    "Omicron": "SARS-CoV-2-Omicron-ESM2cov-L1",
}
ESC = "DMS"          # per-strain escape task: SARS-CoV-2-<S>>DMS
T = 0.01


def build_strain(strain):
    key = BACKBONE[strain]; sname = f"SARS-CoV-2-{strain}"
    rd = an.RESULT_DATA[key]
    df = an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns"))
    df = an.prepare_df(df, sname)
    Q = an.get_Q(sname)
    out = pd.DataFrame({
        "S1": df["semantic_z"].to_numpy(),
        "S2": df["log_grammar_z"].to_numpy(),
        "S3": an.get_log_clib_z(df, Q, T),
    })
    muts = set(an.ESCAPE_DATA[f"{sname}>{ESC}"]["mutations"])
    code = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    out["y"] = code.isin(muts).astype(int).to_numpy()
    out["strain"] = strain
    return out


def fit(S1, S2, y, offset):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + offset) - np.logaddexp(0, X @ w + offset))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + offset))))
    return minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x[1:]


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = int(y.sum()); n = len(y) - p
    return np.nan if p == 0 or n == 0 else (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def main():
    data = {s: build_strain(s) for s in STRAINS}
    for s in STRAINS:
        print(f"{s:9s} n_pos={int(data[s]['y'].sum())}")

    rows = []
    for held in STRAINS:
        tr = pd.concat([data[s] for s in STRAINS if s != held], ignore_index=True)
        te = data[held]
        S1t, S2t, S3t, yt = tr.S1.values, tr.S2.values, tr.S3.values, tr.y.values
        S1e, S2e, S3e, ye = te.S1.values, te.S2.values, te.S3.values, te.y.values
        # protein-only
        b = fit(S1t, S2t, yt, np.zeros(len(yt)))
        au_p = auroc(b[0] * S1e + b[1] * S2e, ye)
        # protein + offset
        b = fit(S1t, S2t, yt, 1.0 * S3t)
        au_po = auroc(b[0] * S1e + b[1] * S2e + 1.0 * S3e, ye)
        # CLIB only
        au_c = auroc(S3e, ye)
        rows.append(dict(held_out=held, n_pos=int(ye.sum()),
                         protein_only=round(au_p, 4), protein_offset=round(au_po, 4),
                         CLIB_only=round(au_c, 4)))
    res = pd.DataFrame(rows)
    mean = res[["protein_only", "protein_offset", "CLIB_only"]].mean().round(4)
    res.to_csv(ROOT + "/outputs/phase0/phaseB7_crossstrain.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Strong-backbone (ESM2cov) leave-one-strain-out transfer ===")
    print(res.to_string(index=False))
    print(f"\nMEAN  protein_only={mean.protein_only}  protein+offset={mean.protein_offset}  CLIB_only={mean.CLIB_only}")
    print("Phase 6 (base ESM2) reference: protein_only~0.60, protein+offset~0.86, CLIB_only~0.864")


if __name__ == "__main__":
    main()
