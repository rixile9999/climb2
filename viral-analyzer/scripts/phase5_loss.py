"""
Phase 5 — does a ranking-aligned loss beat pointwise logistic?

So far (b0,b1) are fit by pointwise logistic (GLM). The reported metric is a RANKING metric
(mean-rank of positives), so a pairwise learning-to-rank loss (RankNet-style) is, in principle,
better aligned. We A/B the two losses on the fixed-offset model, same nested CV.

    POINT : logistic NLL over labels,          score = b0*S1 + b1*S2 + CLIB(t=0.01)
    PAIR  : Sigma_{i in pos, j in neg} log(1+exp(-(eta_i - eta_j))),  eta = same linear score
            (offset differences enter each pair; ranking is what this directly optimizes)

Backbone ESM2cov; tasks DMS, Omicron; 5 seeds x 5-fold OOF; report OOS mean-rank / AUROC /
AUPRC and the paired bootstrap delta (PAIR - POINT). Winner adopted if its mean-rank CI < 0
(or kept as POINT if indistinguishable, favouring the simpler convex fit).

Run:  conda run -n vanalyzer python scripts/phase5_loss.py
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
from sklearn.metrics import roc_auc_score, average_precision_score
import analyzer as an

warnings.filterwarnings("ignore")

STRAIN = "SARS-CoV-2-WildType"
RESULT = "SARS-CoV-2-WildType-ESM2cov-L1"
T_FIX = 0.01
SEEDS = [0, 1, 2, 3, 4]
OUTER_K, N_BOOT = 5, 2000
MAX_NEG_PAIRS = 4000   # subsample negatives per fit for the pairwise loss (speed; ranking is stable)


def load():
    rd = an.RESULT_DATA[RESULT]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd["columns"]), STRAIN)
    Q = an.get_Q(STRAIN)
    S1 = df["semantic_z"].to_numpy(); S2 = df["log_grammar_z"].to_numpy()
    clib = np.asarray(an.get_log_clib_z(df, Q, T_FIX), float)
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    return S1, S2, clib, code


def fit_point(X, y, offset):
    n, k = X.shape
    Xa = np.c_[np.ones(n), X]

    def nll(w):
        eta = Xa @ w + offset
        return -np.sum(y * eta - np.logaddexp(0, eta))

    def grad(w):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w + offset)))
        return -Xa.T @ (y - p)

    return minimize(nll, np.zeros(k + 1), jac=grad, method="L-BFGS-B").x[1:]


def fit_pair(X, y, offset, seed):
    """RankNet-style pairwise logistic; intercept cancels in pairs so only feature weights."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    if len(neg) > MAX_NEG_PAIRS:
        neg = rng.choice(neg, MAX_NEG_PAIRS, replace=False)
    # difference matrix over all pos x neg pairs
    dX = X[pos][:, None, :] - X[neg][None, :, :]           # (P, N, k)
    dOff = offset[pos][:, None] - offset[neg][None, :]      # (P, N)
    dX = dX.reshape(-1, X.shape[1]); dOff = dOff.reshape(-1)

    def nll(w):
        m = dX @ w + dOff
        return np.sum(np.logaddexp(0, -m))

    def grad(w):
        m = dX @ w + dOff
        s = -1.0 / (1.0 + np.exp(m))     # d/dm log(1+exp(-m)) = -sigmoid(-m)
        return dX.T @ s

    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def ranks_desc(s):
    return pd.Series(-s).rank(method="min").to_numpy()


def mean_rank(r, y):
    return float(r[y == 1].mean())


def oof(X, y, offset, seed, mode):
    n = len(y); o = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        if mode == "point":
            w = fit_point(X[tr], y[tr], offset[tr])
        else:
            w = fit_pair(X[tr], y[tr], offset[tr], seed)
        o[te] = X[te] @ w + offset[te]
    return o


def boot_delta(rP, rB, y, seed=0):
    rng = np.random.RandomState(seed); pos = np.where(y == 1)[0]
    a, b = rP[pos], rB[pos]; d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.randint(0, len(pos), len(pos)); d[i] = a[j].mean() - b[j].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    S1, S2, clib, code = load()
    X = np.column_stack([S1, S2]); offset = 1.0 * clib
    rows = []
    for task in ["DMS", "Omicron"]:
        muts = set(an.ESCAPE_DATA[f"{STRAIN}>{task}"]["mutations"])
        y = np.isin(code, list(muts)).astype(int); npos = int(y.sum())
        rk, au, ap = {}, {}, {}
        for mode in ["point", "pair"]:
            rs, aus, aps = [], [], []
            for sd in SEEDS:
                o = oof(X, y, offset, sd, mode)
                rs.append(ranks_desc(o)); aus.append(roc_auc_score(y, o)); aps.append(average_precision_score(y, o))
            rk[mode] = np.mean(rs, 0); au[mode] = np.mean(aus); ap[mode] = np.mean(aps)
        d = boot_delta(rk["pair"], rk["point"], y)
        rows.append({
            "task": task, "n_pos": npos,
            "mr_point": mean_rank(rk["point"], y), "mr_pair": mean_rank(rk["pair"], y),
            "AUROC_point": au["point"], "AUROC_pair": au["pair"],
            "AUPRC_point": ap["point"], "AUPRC_pair": ap["pair"],
            "d_pair-point": d[0], "d_CI": f"[{d[1]:.0f},{d[2]:.0f}]",
            "pair_wins": bool(d[2] < 0),
        })
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase5_loss.csv")
    res.to_csv(out, index=False)
    fmt = {c: (lambda v: f"{v:.4g}") for c in
           ["mr_point", "mr_pair", "AUROC_point", "AUROC_pair", "AUPRC_point", "AUPRC_pair", "d_pair-point"]}
    print("\n============= PHASE 5 — pairwise ranking loss vs pointwise logistic (ESM2cov) =============")
    print(res.to_string(index=False, formatters=fmt))
    print(f"\nwrote {out}")
    print("\npair_wins = pairwise mean-rank delta vs pointwise has 95% CI < 0.")


if __name__ == "__main__":
    main()
