"""
Phase 7 (final confirmation) — does enriching the STRONG backbone's own semantic representation help?

Every complexity added so far has failed to beat the simple 2-feature offset (discretize/stack/loss).
The one remaining plausible lever is richer semantics from the strong ESM2cov backbone itself:
its TSV carries two embedding-distance metrics (change_l1, change_l2). We test whether adding the
L2 view (or swapping L1->L2) beats the baseline {S1=L1, S2=grammar} + CLIB offset.

    base     : L1 + grammar            + CLIB offset      (current model)
    +L2      : L1 + L2 + grammar       + CLIB offset      (ridge, lambda inner-CV)
    L2only   : L2 + grammar            + CLIB offset

ESM2cov, DMS + Omicron, 5 seeds x 5-fold, mean-rank delta vs base with bootstrap CI.

Run:  conda run -n vanalyzer python scripts/phase7_metrics.py
"""

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import analyzer as an

warnings.filterwarnings("ignore")

STRAIN = "SARS-CoV-2-WildType"
T_FIX = 0.01
SEEDS = [0, 1, 2, 3, 4]
OUTER_K, INNER_K, N_BOOT = 5, 3, 2000
LAMBDAS = [0.0, 1.0, 10.0, 100.0]


def std(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-15)


def load():
    rd = an.RESULT_DATA["SARS-CoV-2-WildType-ESM2cov-L1"]
    raw = pd.read_csv(rd["path"], sep="\t")
    df = an.prepare_df(an.rename_df(raw, columns=rd["columns"]), STRAIN)
    # align raw extra metric (change_l2) to prepared df via index (prepare_df filters non-standard AAs)
    raw2 = an.rename_df(raw, columns=rd["columns"])
    raw2 = raw2[raw2["mutated_aa"].isin(set("ACDEFGHIKLMNPQRSTVWY"))].reset_index(drop=True)
    L1 = std(df["semantic_z"].to_numpy())            # already L1 semantic (change_l1), re-std
    L2 = std(raw2["change_l2"].to_numpy())
    gram = std(df["log_grammar_z"].to_numpy())
    Q = an.get_Q(STRAIN)
    clib = np.asarray(an.get_log_clib_z(df, Q, T_FIX), float)
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    return dict(L1=L1, L2=L2, gram=gram, off=clib, code=code)


FEATSETS = {"base": ["L1", "gram"], "+L2": ["L1", "L2", "gram"], "L2only": ["L2", "gram"]}


def fit(X, y, offset, lam):
    n, k = X.shape; Xa = np.c_[np.ones(n), X]

    def nll(w):
        return -np.sum(y * (Xa @ w + offset) - np.logaddexp(0, Xa @ w + offset)) + lam * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(Xa @ w + offset)))
        g = -Xa.T @ (y - p); g[1:] += 2 * lam * w[1:]; return g

    return minimize(nll, np.zeros(k + 1), jac=grad, method="L-BFGS-B").x[1:]


def ranks(s):
    return pd.Series(-s).rank(method="min").to_numpy()


def mr(r, y):
    return float(r[y == 1].mean())


def pick_lam(X, y, off, idx, seed):
    best, bmr = 0.0, np.inf
    for lam in LAMBDAS:
        yy = y[idx]; oof = np.full(len(idx), np.nan)
        for itr, ite in StratifiedKFold(INNER_K, shuffle=True, random_state=seed).split(np.zeros(len(idx)), yy):
            tr, te = idx[itr], idx[ite]
            w = fit(X[tr], y[tr], off[tr], lam); oof[ite] = X[te] @ w + off[te]
        m = mr(ranks(oof), yy)
        if m < bmr:
            bmr, best = m, lam
    return best


def oof(X, y, off, seed):
    n = len(y); o = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        lam = pick_lam(X, y, off, tr, seed) if X.shape[1] > 2 else 0.0
        w = fit(X[tr], y[tr], off[tr], lam); o[te] = X[te] @ w + off[te]
    return o


def boot(rX, rB, y):
    rng = np.random.RandomState(0); pos = np.where(y == 1)[0]; a, b = rX[pos], rB[pos]; d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.randint(0, len(pos), len(pos)); d[i] = a[j].mean() - b[j].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    F = load(); off = 1.0 * F["off"]; rows = []
    for task in ["DMS", "Omicron"]:
        muts = set(an.ESCAPE_DATA[f"{STRAIN}>{task}"]["mutations"])
        y = np.isin(F["code"], list(muts)).astype(int)
        rk, au = {}, {}
        for name, cols in FEATSETS.items():
            X = np.column_stack([F[c] for c in cols]); rs, a = [], []
            for sd in SEEDS:
                o = oof(X, y, off, sd); rs.append(ranks(o)); a.append(roc_auc_score(y, o))
            rk[name] = np.mean(rs, 0); au[name] = np.mean(a)
        for name in FEATSETS:
            d = boot(rk[name], rk["base"], y)
            rows.append({"task": task, "n_pos": int(y.sum()), "featset": name,
                         "mean_rank": mr(rk[name], y), "AUROC": au[name],
                         "d_vs_base": d[0], "CI": f"[{d[1]:.0f},{d[2]:.0f}]", "beats_base": bool(d[2] < 0)})
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase7_metrics.csv"); res.to_csv(out, index=False)
    fmt = {c: (lambda v: f"{v:.4g}") for c in ["mean_rank", "AUROC", "d_vs_base"]}
    print("\n===== PHASE 7 — richer semantics within the strong ESM2cov backbone =====")
    print(res.to_string(index=False, formatters=fmt))
    print(f"\nwrote {out}\nbeats_base = mean-rank delta vs {{L1,grammar}}+offset has 95% CI < 0.")


if __name__ == "__main__":
    main()
