"""
Phase 6 — cross-WT generalization (leave-one-strain-out).

The real scientific goal is predicting escape on strains NOT used for fitting. We test whether
the offset model transfers across evolutionary backgrounds using base ESM2-650M, the only
backbone available for all six strains (WildType + Alpha/Beta/Gamma/Delta/Omicron), each with
its own >DMS escape set (~18-19 positives), its own sequence features, and its own CLIB offset.

Leave-one-strain-out: for held-out strain S, fit weights on the OTHER five strains, predict S.
    A  protein-only (transferred)   score = b0*S1 + b1*S2
    B  protein + offset (transf.)   score = b0*S1 + b1*S2 + CLIB(t=0.01)
    C  CLIB-only (no training)      score = CLIB(t=0.01)
Per held-out strain: mean-rank / AUROC of that strain's positives; delta B-A with bootstrap CI.

Count assertion is run first for every strain's >DMS list (variant lists may carry the same
coordinate issues found in WildType).

Run:  conda run -n vanalyzer python scripts/phase6_crosswt.py
"""

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import analyzer as an

warnings.filterwarnings("ignore")

T_FIX = 0.01
N_BOOT = 2000
STRAINS = {
    "WildType": ("SARS-CoV-2-WildType", "SARS-CoV-2-WildType-ESM-650M-L1"),
    "Alpha":    ("SARS-CoV-2-Alpha",    "SARS-CoV-2-Alpha-ESM-650M"),
    "Beta":     ("SARS-CoV-2-Beta",     "SARS-CoV-2-Beta-ESM-650M"),
    "Gamma":    ("SARS-CoV-2-Gamma",    "SARS-CoV-2-Gamma-ESM-650M"),
    "Delta":    ("SARS-CoV-2-Delta",    "SARS-CoV-2-Delta-ESM-650M"),
    "Omicron":  ("SARS-CoV-2-Omicron",  "SARS-CoV-2-Omicron-ESM-650M"),
}


def build(strain_key):
    strain_full, result = STRAINS[strain_key]
    rd = an.RESULT_DATA[result]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns")), strain_full)
    Q = an.get_Q(strain_full)
    S1 = df["semantic_z"].to_numpy(); S2 = df["log_grammar_z"].to_numpy()
    clib = np.asarray(an.get_log_clib_z(df, Q, T_FIX), float)
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    muts = set(an.ESCAPE_DATA[f"{strain_full}>DMS"]["mutations"])
    y = np.isin(code, list(muts)).astype(int)
    return dict(S1=S1, S2=S2, off=clib, y=y, n_meta=len(muts), n_match=int(y.sum()))


def fit_point(X, y, offset):
    n, k = X.shape; Xa = np.c_[np.ones(n), X]

    def nll(w):
        eta = Xa @ w + offset
        return -np.sum(y * eta - np.logaddexp(0, eta))

    def grad(w):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w + offset)))
        return -Xa.T @ (y - p)

    return minimize(nll, np.zeros(k + 1), jac=grad, method="L-BFGS-B").x[1:]


def ranks_desc(s):
    return pd.Series(-s).rank(method="min").to_numpy()


def mean_rank(r, y):
    return float(r[y == 1].mean())


def boot(rB, rA, y, seed=0):
    rng = np.random.RandomState(seed); pos = np.where(y == 1)[0]
    a, b = rB[pos], rA[pos]; d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.randint(0, len(pos), len(pos)); d[i] = a[j].mean() - b[j].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    D = {k: build(k) for k in STRAINS}

    print("=== count assertion (>DMS per strain) ===")
    ok = True
    for k, d in D.items():
        flag = "OK" if d["n_match"] == d["n_meta"] else "MISMATCH"
        if d["n_match"] != d["n_meta"]:
            ok = False
        print(f"  {k:9s} matched {d['n_match']:2d} / {d['n_meta']:2d}  {flag}")
    print(f"all strains clean: {ok}\n")

    rows = []
    for held in STRAINS:
        tr = [k for k in STRAINS if k != held]
        S1tr = np.concatenate([D[k]["S1"] for k in tr])
        S2tr = np.concatenate([D[k]["S2"] for k in tr])
        offtr = np.concatenate([D[k]["off"] for k in tr])
        ytr = np.concatenate([D[k]["y"] for k in tr])
        Xtr = np.c_[S1tr, S2tr]
        wA = fit_point(Xtr, ytr, np.zeros_like(offtr))
        wB = fit_point(Xtr, ytr, offtr)

        h = D[held]; Xh = np.c_[h["S1"], h["S2"]]; yh = h["y"]
        sA = Xh @ wA
        sB = Xh @ wB + h["off"]
        sC = h["off"]
        rA, rB, rC = ranks_desc(sA), ranks_desc(sB), ranks_desc(sC)
        dBA = boot(rB, rA, yh)
        rows.append({
            "held_strain": held, "n_pos": int(yh.sum()),
            "mr_A(prot)": mean_rank(rA, yh), "mr_B(prot+off)": mean_rank(rB, yh), "mr_C(CLIBonly)": mean_rank(rC, yh),
            "AUROC_A": roc_auc_score(yh, sA), "AUROC_B": roc_auc_score(yh, sB), "AUROC_C": roc_auc_score(yh, sC),
            "d_B-A": dBA[0], "d_CI": f"[{dBA[1]:.0f},{dBA[2]:.0f}]", "offset_helps": bool(dBA[2] < 0),
        })
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase6_crosswt.csv")
    res.to_csv(out, index=False)
    fmt = {c: (lambda v: f"{v:.4g}") for c in
           ["mr_A(prot)", "mr_B(prot+off)", "mr_C(CLIBonly)", "AUROC_A", "AUROC_B", "AUROC_C", "d_B-A"]}
    print("=== leave-one-strain-out transfer (base ESM2-650M) ===")
    print(res.to_string(index=False, formatters=fmt))
    # pooled summary
    print("\n--- means across held-out strains ---")
    print(f"  mean-rank  A(prot)={res['mr_A(prot)'].mean():.0f}  "
          f"B(prot+off)={res['mr_B(prot+off)'].mean():.0f}  C(CLIBonly)={res['mr_C(CLIBonly)'].mean():.0f}")
    print(f"  AUROC      A={res['AUROC_A'].mean():.3f}  B={res['AUROC_B'].mean():.3f}  C={res['AUROC_C'].mean():.3f}")
    print(f"  strains where offset helps (CI<0): {res['offset_helps'].sum()}/{len(res)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
