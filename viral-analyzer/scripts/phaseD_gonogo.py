"""
Section D — go/no-go: does the CLIB term S3 carry UNIQUE, label-relevant escape signal
beyond the protein features S1 (semantic_z) and S2 (log_grammar_z), under HONEST out-of-sample
evaluation?  If not, the whole offset plan is dropped and we ship kappa=0 (two-feature z-CSCS).

Three models, all on the same candidate universe (~24k single mutants of WildType):
    A  (kappa=0)      : score = b0*S1 + b1*S2                    # no CLIB
    B  (offset kappa=1): score = b0*S1 + b1*S2 + 1.0*S3          # CLIB frozen as offset  <-- the proposal
    C  (free b3)      : score = b0*S1 + b1*S2 + b3*S3            # CLIB coefficient FIT   <-- control

Fits use statsmodels GLM(Binomial); B uses statsmodels' native `offset=`.
Evaluation: 5-fold stratified out-of-fold (OOF) predictions -> rank the full universe ->
per-task mean-rank of the held-out positives (headline, lower=better) + AUROC + AUPRC.
CI: percentile bootstrap over the positive set (2000 resamples) on the mean-rank DELTA.
Screen: in-sample likelihood-ratio test (A vs C) for any independent S3 association.

Clean escape tasks only (count-assertion verified): Omicron (30), DMS (19).

Run:
    conda run -n vanalyzer python scripts/phaseD_gonogo.py
"""

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(0)

PARQUET = os.path.join(ROOT, "outputs", "phase0", "tidy_esm2cov_l1.parquet")
TASKS = ["Omicron", "DMS"]          # count-assertion clean, most positives first
T_LIST = [0.1, 1.0]
N_SPLITS = 5
N_BOOT = 2000


def fit_glm(y, X, offset=None):
    """statsmodels GLM Binomial with intercept; returns fitted result."""
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, Xc, family=sm.families.Binomial(),
                   offset=offset if offset is not None else None)
    return model.fit()


def linpred_no_const(res, X, offset=None):
    """b0*S1 + b1*S2 (+ offset) WITHOUT the intercept (irrelevant for ranking)."""
    coefs = res.params
    # params order: const, then columns of X
    s = np.zeros(len(X))
    for j, col in enumerate(X.columns):
        s = s + coefs[col] * X[col].to_numpy()
    if offset is not None:
        s = s + np.asarray(offset)
    return s


def oof_scores(y, S1, S2, S3):
    """5-fold stratified out-of-fold ranking scores for models A, B(kappa=1), C(free b3)."""
    n = len(y)
    scoreA = np.full(n, np.nan)
    scoreB = np.full(n, np.nan)
    scoreC = np.full(n, np.nan)
    b3s = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    X_all = pd.DataFrame({"S1": S1, "S2": S2, "S3": S3})
    for tr, te in skf.split(X_all, y):
        Xtr2, Xte2 = X_all.iloc[tr][["S1", "S2"]], X_all.iloc[te][["S1", "S2"]]
        Xtr3, Xte3 = X_all.iloc[tr][["S1", "S2", "S3"]], X_all.iloc[te][["S1", "S2", "S3"]]
        ytr = y[tr]
        # A: no CLIB
        rA = fit_glm(ytr, Xtr2)
        scoreA[te] = linpred_no_const(rA, Xte2)
        # B: offset kappa=1
        rB = fit_glm(ytr, Xtr2, offset=1.0 * X_all.iloc[tr]["S3"].to_numpy())
        scoreB[te] = linpred_no_const(rB, Xte2, offset=1.0 * X_all.iloc[te]["S3"].to_numpy())
        # C: free b3
        rC = fit_glm(ytr, Xtr3)
        scoreC[te] = linpred_no_const(rC, Xte3)
        b3s.append(rC.params["S3"])
    return scoreA, scoreB, scoreC, float(np.mean(b3s))


def ranks_desc(score):
    """min-rank, descending (rank 1 = highest score = most escape-like)."""
    return pd.Series(-score).rank(method="min").to_numpy()


def mean_rank(rank, y):
    return float(rank[y == 1].mean())


def boot_delta_meanrank(rankA, rankB, y):
    """bootstrap over positives: distribution of mean-rank(B) - mean-rank(A)."""
    pos = np.where(y == 1)[0]
    rA, rB = rankA[pos], rankB[pos]
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.randint(0, len(pos), len(pos))
        deltas[b] = rB[idx].mean() - rA[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def lrt_S3(y, S1, S2, S3):
    """in-sample LRT: A (S1+S2) vs C (S1+S2+S3). p<0.05 => S3 independently associated."""
    rA = fit_glm(y, pd.DataFrame({"S1": S1, "S2": S2}))
    rC = fit_glm(y, pd.DataFrame({"S1": S1, "S2": S2, "S3": S3}))
    stat = 2.0 * (rC.llf - rA.llf)
    p = stats.chi2.sf(stat, df=1)
    return float(stat), float(p), float(rC.params["S3"])


def main():
    df = pd.read_parquet(PARQUET)
    rows = []
    for task in TASKS:
        y = df[f"esc__{task}"].to_numpy().astype(int)
        n_pos = int(y.sum())
        S1 = df["S1_semantic_z"].to_numpy()
        S2 = df["S2_log_grammar_z"].to_numpy()
        for t in T_LIST:
            S3 = df[f"S3_logclib_z_t{t}"].to_numpy()
            lrt_stat, lrt_p, b3_full = lrt_S3(y, S1, S2, S3)
            sA, sB, sC, b3_cv = oof_scores(y, S1, S2, S3)
            rA, rB, rC = ranks_desc(sA), ranks_desc(sB), ranks_desc(sC)
            mrA, mrB, mrC = mean_rank(rA, y), mean_rank(rB, y), mean_rank(rC, y)
            dBA, loBA, hiBA = boot_delta_meanrank(rA, rB, y)   # offset vs no-CLIB
            dCA, loCA, hiCA = boot_delta_meanrank(rA, rC, y)   # free-b3 vs no-CLIB
            rows.append({
                "task": task, "n_pos": n_pos, "t": t,
                "LRT_p(S3)": lrt_p, "b3_free": b3_cv,
                "mrank_A(noCLIB)": mrA, "mrank_B(offset)": mrB, "mrank_C(freeb3)": mrC,
                "d_B-A": dBA, "d_B-A_CI": f"[{loBA:.0f},{hiBA:.0f}]",
                "d_C-A": dCA, "d_C-A_CI": f"[{loCA:.0f},{hiCA:.0f}]",
                "AUPRC_A": average_precision_score(y, sA),
                "AUPRC_B": average_precision_score(y, sB),
                "AUROC_A": roc_auc_score(y, sA),
                "AUROC_B": roc_auc_score(y, sB),
                "offset_helps": (hiBA < 0),   # mean-rank delta CI fully below 0 (lower=better)
            })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    show = ["task", "n_pos", "t", "LRT_p(S3)", "b3_free",
            "mrank_A(noCLIB)", "mrank_B(offset)", "mrank_C(freeb3)",
            "d_B-A", "d_B-A_CI", "AUPRC_A", "AUPRC_B", "AUROC_A", "AUROC_B", "offset_helps"]
    print("\n===================== SECTION D — go/no-go (ESM2cov-L1, WildType) =====================")
    print(res[show].to_string(index=False,
          formatters={c: (lambda v: f"{v:.3g}") for c in
                      ["LRT_p(S3)", "b3_free", "mrank_A(noCLIB)", "mrank_B(offset)",
                       "mrank_C(freeb3)", "d_B-A", "AUPRC_A", "AUPRC_B", "AUROC_A", "AUROC_B"]}))
    out = os.path.join(ROOT, "outputs", "phase0", "phaseD_gonogo.csv")
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nVERDICT rule: offset (B) beats no-CLIB (A) iff mean-rank delta 95% CI is fully < 0.")
    verdicts = res.groupby("task")["offset_helps"].any().to_dict()
    print("per-task any-t offset_helps:", verdicts)


if __name__ == "__main__":
    main()
