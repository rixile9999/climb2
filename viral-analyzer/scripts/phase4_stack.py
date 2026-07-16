"""
Phase 4 (fork) — does STACKING multiple protein backbones beat the best single backbone?

This tests the user's original "meta-model" vision honestly: combine the semantic/grammar
features of several protein LMs (+ the shared CLIB offset) and see if out-of-sample DMS/Omicron
ranking improves over the best single backbone. Key facts:
  * CLIB depends only on the WT codon + substitution model, NOT on the protein backbone,
    so there is ONE shared offset (t=0.01, kappa=1) across all models.
  * Labels are scarce (DMS n=19, Omicron n=30) -> more features overfit; L2 (ridge) with an
    inner-CV-selected strength is mandatory, and the honest expectation is that stacking may
    NOT beat the single strong backbone.

Models (all = L2-logistic over the listed z-features + CLIB offset, kappa=1):
    M1  cov            : ESM2cov S1,S2                       (current best baseline)
    M2  cov+base       : + base ESM2 S1,S2
    M3  cov+base+hie   : + Hie S1,S2                         (full stack)

Nested CV: 5 seeds x 5-fold outer; inner 3-fold selects the ridge strength lambda.
Verdict: stacking wins iff its mean-rank delta vs M1 (bootstrap over positives) CI is < 0.

Run:  conda run -n vanalyzer python scripts/phase4_stack.py
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
T_FIX = 0.01
SEEDS = [0, 1, 2, 3, 4]
OUTER_K, INNER_K, N_BOOT = 5, 3, 2000
LAMBDAS = [0.0, 1.0, 10.0, 100.0]
BACKBONES = {
    "cov":  "SARS-CoV-2-WildType-ESM2cov-L1",
    "base": "SARS-CoV-2-WildType-ESM-650M-L1",
    "hie":  "SARS-CoV-2-WildType-Hie",
}


def std(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-15)


def load(result_name):
    rd = an.RESULT_DATA[result_name]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns")), STRAIN)
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    return df, code


def build_aligned():
    """Align all backbones on a common mutation_code; return feature dict + shared CLIB + code."""
    dfc, code_c = load(BACKBONES["cov"])
    dfb, code_b = load(BACKBONES["base"])
    dfh, code_h = load(BACKBONES["hie"])
    # reference universe = ESM2cov; base must match 1:1, Hie joined by code
    assert np.array_equal(code_c, code_b), "cov/base candidate universes differ"
    idx_h = {c: i for i, c in enumerate(code_h)}
    keep = np.array([c in idx_h for c in code_c])
    hsel = np.array([idx_h[c] for c in code_c[keep]])
    feats = {
        "cov_S1": std(dfc["semantic_z"].to_numpy()[keep]),
        "cov_S2": std(dfc["log_grammar_z"].to_numpy()[keep]),
        "base_S1": std(dfb["semantic_z"].to_numpy()[keep]),
        "base_S2": std(dfb["log_grammar_z"].to_numpy()[keep]),
        "hie_S1": std(dfh["semantic_z"].to_numpy()[hsel]),
        "hie_S2": std(dfh["log_grammar_z"].to_numpy()[hsel]),
    }
    # shared CLIB offset (backbone-independent), computed on cov df, aligned to kept rows
    Q = an.get_Q(STRAIN)
    clib = np.asarray(an.get_log_clib_z(dfc, Q, T_FIX), float)[keep]
    code = code_c[keep]
    print(f"aligned universe: {keep.sum()} / {len(code_c)} (Hie-joined)")
    return feats, clib, code


FEATSETS = {
    "M1_cov":          ["cov_S1", "cov_S2"],
    "M2_cov+base":     ["cov_S1", "cov_S2", "base_S1", "base_S2"],
    "M3_cov+base+hie": ["cov_S1", "cov_S2", "base_S1", "base_S2", "hie_S1", "hie_S2"],
}


def fit_ridge_logit(X, y, offset, lam):
    """L2-logistic (intercept unpenalized) with a fixed additive offset; returns feature weights."""
    n, k = X.shape
    Xa = np.c_[np.ones(n), X]

    def nll(w):
        eta = Xa @ w + offset
        return -np.sum(y * eta - np.logaddexp(0, eta)) + lam * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w + offset)))
        g = -Xa.T @ (y - p)
        g[1:] += 2 * lam * w[1:]
        return g

    res = minimize(nll, np.zeros(k + 1), jac=grad, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x[1:]  # drop intercept (irrelevant for ranking)


def ranks_desc(s):
    return pd.Series(-s).rank(method="min").to_numpy()


def mean_rank(r, y):
    return float(r[y == 1].mean())


def pick_lambda(X, y, offset, idx, seed):
    best, best_mr = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        yy = y[idx]; oof = np.full(len(idx), np.nan)
        for itr, ite in StratifiedKFold(INNER_K, shuffle=True, random_state=seed).split(np.zeros(len(idx)), yy):
            tr, te = idx[itr], idx[ite]
            w = fit_ridge_logit(X[tr], y[tr], offset[tr], lam)
            oof[ite] = X[te] @ w + offset[te]
        mr = mean_rank(ranks_desc(oof), yy)
        if mr < best_mr:
            best_mr, best = mr, lam
    return best


def oof_score(X, y, offset, seed):
    n = len(y); oof = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        lam = pick_lambda(X, y, offset, tr, seed) if X.shape[1] > 2 else 0.0
        w = fit_ridge_logit(X[tr], y[tr], offset[tr], lam)
        oof[te] = X[te] @ w + offset[te]
    return oof


def boot_delta(rX, rBase, y, seed=0):
    rng = np.random.RandomState(seed); pos = np.where(y == 1)[0]
    a, b = rX[pos], rBase[pos]; d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.randint(0, len(pos), len(pos)); d[i] = a[j].mean() - b[j].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    feats, clib, code = build_aligned()
    offset = 1.0 * clib
    rows = []
    for task in ["DMS", "Omicron"]:
        muts = set(an.ESCAPE_DATA[f"{STRAIN}>{task}"]["mutations"])
        y = np.isin(code, list(muts)).astype(int)
        npos = int(y.sum())
        ranks = {}
        aurocs = {}
        for mname, cols in FEATSETS.items():
            X = np.column_stack([feats[c] for c in cols])
            rs, au = [], []
            for sd in SEEDS:
                o = oof_score(X, y, offset, sd)
                rs.append(ranks_desc(o)); au.append(roc_auc_score(y, o))
            ranks[mname] = np.mean(rs, 0); aurocs[mname] = np.mean(au)
        base_rank = ranks["M1_cov"]
        for mname in FEATSETS:
            d = boot_delta(ranks[mname], base_rank, y)
            rows.append({
                "task": task, "n_pos": npos, "model": mname,
                "mean_rank": mean_rank(ranks[mname], y), "AUROC": aurocs[mname],
                "d_vs_M1": d[0], "d_CI": f"[{d[1]:.0f},{d[2]:.0f}]",
                "beats_M1": bool(d[2] < 0),
            })
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase4_stack.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    fmt = {c: (lambda v: f"{v:.4g}") for c in ["mean_rank", "AUROC", "d_vs_M1"]}
    print("\n============= PHASE 4 (fork) — multi-backbone stacking =============")
    print(res.to_string(index=False, formatters=fmt))
    print(f"\nwrote {out}")
    print("\nbeats_M1 = stacked model's mean-rank delta vs single ESM2cov (M1) has 95% CI < 0.")


if __name__ == "__main__":
    main()
