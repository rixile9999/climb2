"""
Phase 3b — can we DROP the evolutionary-time hyperparameter t?

Phase 3 showed the CLIB offset signal lives at very small t (t* -> grid floor), with the gain
saturating below ~0.01, which suggests the effective signal is just the DISCRETE nucleotide
accessibility "how many base changes to reach this amino acid (1/2/3)". We test two
hyperparameter-light replacements for the full inner-CV-selected CLIB offset:

    B_full   offset kappa=1, CLIB S3(t*) with t* chosen by INNER CV        (the current proposal)
    B_fix    offset kappa=1, CLIB S3 at a FIXED t=0.01 (no selection)
    B_min    offset kappa=1, S3 = z(-min_base_changes)  (NO t at all; codon-distance only)
    A        no CLIB                                                        (reference)

"Retains performance" = the simplified model's OOS mean-rank is not significantly worse than
B_full (paired bootstrap delta CI includes 0 / is negligible).

Run:  conda run -n vanalyzer python scripts/phase3b_simplify.py
"""

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import analyzer as an

warnings.filterwarnings("ignore")

STRAIN = "SARS-CoV-2-WildType"
BACKBONES = {
    "ESM2cov":   "SARS-CoV-2-WildType-ESM2cov-L1",
    "ESM2-650M": "SARS-CoV-2-WildType-ESM-650M-L1",
    "Hie":       "SARS-CoV-2-WildType-Hie",
}
TASKS = ["DMS", "Omicron"]
T_LIST = [0.001, 0.003, 0.01, 0.033, 0.1, 0.33, 1.0]
T_FIX = 0.01
SEEDS = [0, 1, 2, 3, 4]
OUTER_K, INNER_K, N_BOOT = 5, 3, 2000


def standardize(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-15)


def min_base_changes(wt_codon, target_aa):
    """minimum Hamming distance (in bases) from the WT codon to ANY codon of target_aa (1..3)."""
    best = 3
    for triple in an.LETTER_TO_CODONS[target_aa]:
        tc = "".join(an.BASE[i] for i in triple)
        h = sum(a != b for a, b in zip(wt_codon, tc))
        best = min(best, h)
    return best


def build_backbone(result_name):
    rd = an.RESULT_DATA[result_name]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns")), STRAIN)
    Q = an.get_Q(STRAIN)
    S1 = df["semantic_z"].to_numpy()
    S2 = df["log_grammar_z"].to_numpy()
    S3 = {t: np.asarray(an.get_log_clib_z(df, Q, t), float) for t in T_LIST}
    # hyperparameter-free codon-distance accessibility: fewer base changes -> more accessible -> higher
    mbc = np.array([min_base_changes(c, m) for c, m in zip(df["codon"], df["mutated_aa"])], float)
    S3_min = standardize(-mbc)
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    print(f"  [{result_name.split('-')[-1]}] min_base_changes dist: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(mbc, return_counts=True))} }")
    return S1, S2, S3, S3_min, code


def fit_coefs(y, cols, offset=None):
    X = sm.add_constant(cols, has_constant="add")
    res = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset if offset is not None else None).fit()
    return np.asarray(res.params[1:], float)


def ranks_desc(score):
    return pd.Series(-score).rank(method="min").to_numpy()


def mean_rank(rank, y):
    return float(rank[y == 1].mean())


def offset_oof_score(S1, S2, S3col, y, seed):
    """OOF ranking score for offset kappa=1 with a GIVEN S3 column (no inner selection)."""
    n = len(y); oof = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]], offset=1.0 * S3col[tr])
        oof[te] = b[0] * S1[te] + b[1] * S2[te] + 1.0 * S3col[te]
    return oof


def inner_pick_t(S1, S2, S3, y, idx, seed):
    def mr_for(t):
        yy = y[idx]; oof = np.full(len(idx), np.nan)
        for itr, ite in StratifiedKFold(INNER_K, shuffle=True, random_state=seed).split(np.zeros(len(idx)), yy):
            tr, te = idx[itr], idx[ite]
            b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]], offset=1.0 * S3[t][tr])
            oof[ite] = b[0] * S1[te] + b[1] * S2[te] + 1.0 * S3[t][te]
        return mean_rank(ranks_desc(oof), yy)
    return min(T_LIST, key=mr_for)


def bfull_oof_score(S1, S2, S3, y, seed):
    n = len(y); oof = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        t = inner_pick_t(S1, S2, S3, y, tr, seed)
        b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]], offset=1.0 * S3[t][tr])
        oof[te] = b[0] * S1[te] + b[1] * S2[te] + 1.0 * S3[t][te]
    return oof


def a_oof_score(S1, S2, y, seed):
    n = len(y); oof = np.full(n, np.nan)
    for tr, te in StratifiedKFold(OUTER_K, shuffle=True, random_state=seed).split(np.zeros(n), y):
        b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]])
        oof[te] = b[0] * S1[te] + b[1] * S2[te]
    return oof


def boot_delta(rankX, rankFull, y, seed=0):
    """paired bootstrap over positives: mean-rank(X) - mean-rank(B_full)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    rX, rF = rankX[pos], rankFull[pos]
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.randint(0, len(pos), len(pos))
        d[b] = rX[i].mean() - rF[i].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rows = []
    for bname, rname in BACKBONES.items():
        S1, S2, S3, S3min, code = build_backbone(rname)
        for task in TASKS:
            muts = set(an.ESCAPE_DATA[f"{STRAIN}>{task}"]["mutations"])
            y = np.isin(code, list(muts)).astype(int)
            npos = int(y.sum())
            if npos < 2:
                continue
            rA, rBf, rFx, rMn = [], [], [], []
            auA, auBf, auFx, auMn = [], [], [], []
            for sd in SEEDS:
                oA = a_oof_score(S1, S2, y, sd)
                oBf = bfull_oof_score(S1, S2, S3, y, sd)
                oFx = offset_oof_score(S1, S2, S3[T_FIX], y, sd)
                oMn = offset_oof_score(S1, S2, S3min, y, sd)
                rA.append(ranks_desc(oA)); rBf.append(ranks_desc(oBf))
                rFx.append(ranks_desc(oFx)); rMn.append(ranks_desc(oMn))
                auA.append(roc_auc_score(y, oA)); auBf.append(roc_auc_score(y, oBf))
                auFx.append(roc_auc_score(y, oFx)); auMn.append(roc_auc_score(y, oMn))
            rankA = np.mean(rA, 0); rankBf = np.mean(rBf, 0); rankFx = np.mean(rFx, 0); rankMn = np.mean(rMn, 0)
            dFx = boot_delta(rankFx, rankBf, y)
            dMn = boot_delta(rankMn, rankBf, y)
            rows.append({
                "backbone": bname, "task": task, "n_pos": npos,
                "mr_A": mean_rank(rankA, y), "mr_Bfull": mean_rank(rankBf, y),
                "mr_Bfix(t=.01)": mean_rank(rankFx, y), "mr_Bmin(codon)": mean_rank(rankMn, y),
                "AUROC_Bfull": np.mean(auBf), "AUROC_Bfix": np.mean(auFx), "AUROC_Bmin": np.mean(auMn),
                "d_fix-full": dFx[0], "d_fix_CI": f"[{dFx[1]:.0f},{dFx[2]:.0f}]",
                "d_min-full": dMn[0], "d_min_CI": f"[{dMn[1]:.0f},{dMn[2]:.0f}]",
                "fix_retains": bool(dFx[1] <= 0 <= dFx[2] or dFx[2] < 0),
                "min_retains": bool(dMn[1] <= 0 <= dMn[2] or dMn[2] < 0),
            })
    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase3b_simplify.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    fmt = {c: (lambda v: f"{v:.3g}") for c in
           ["mr_A", "mr_Bfull", "mr_Bfix(t=.01)", "mr_Bmin(codon)",
            "AUROC_Bfull", "AUROC_Bfix", "AUROC_Bmin", "d_fix-full", "d_min-full"]}
    cols = ["backbone", "task", "n_pos", "mr_Bfull", "mr_Bfix(t=.01)", "mr_Bmin(codon)",
            "AUROC_Bfull", "AUROC_Bfix", "AUROC_Bmin",
            "d_fix-full", "d_fix_CI", "fix_retains", "d_min-full", "d_min_CI", "min_retains"]
    print("\n============= PHASE 3b — dropping the t hyperparameter (WildType) =============")
    print(res[cols].to_string(index=False, formatters=fmt))
    print(f"\nwrote {out}")
    print("\nretains = simplified model's mean-rank NOT significantly worse than B_full "
          "(paired bootstrap delta CI includes 0, or < 0).")


if __name__ == "__main__":
    main()
