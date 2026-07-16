"""
Phase 3 — multi-seed nested cross-validation to confirm (or refuse) the CLIB-offset signal
found in the go/no-go, and to test the hypothesis "the weaker the protein backbone, the more
the nucleotide offset helps".

Backbones (S1=semantic_z, S2=log_grammar_z):  ESM2cov (domain-adapted), ESM2-650M (base), Hie.
Tasks (count-assertion clean):                 DMS (signal), Omicron (near-ceiling control).

Models, all evaluated OUT-OF-SAMPLE (ranking is scale-invariant, so the intercept is dropped):
    A  kappa=0            score = b0*S1 + b1*S2                       (no CLIB, honest baseline)
    B  offset kappa=1     score = b0*S1 + b1*S2 + 1.0*S3(t*)          (t* chosen by INNER CV)   <- the proposal
    C  free b3            score = b0*S1 + b1*S2 + b3*S3(t*)           (t* chosen by INNER CV)   <- control
    L  legacy barycentric score = a*S1 + b*S2 + g*S3(t*), (a,b,g) on simplex, t* + weights fit on TRAIN
       (also reported IN-SAMPLE = the optimistic number the old pipeline prints)

Nested CV: OUTER = repeated (5 seeds) stratified 5-fold -> pooled out-of-fold ranking per seed.
           INNER = stratified 3-fold on the outer-train to pick t (and, for L, the weights).
Headline metric = mean-rank of the held-out positives (lower is better); also AUROC, AUPRC.
Verdict: offset (B) beats no-CLIB (A) iff the paired mean-rank delta 95% CI (bootstrap over
positives, on seed-averaged OOF ranks) is fully < 0.

Run:  conda run -n vanalyzer python scripts/phase3_nestedcv.py
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
T_LIST = [0.001, 0.003, 0.01, 0.033, 0.1, 0.33, 1.0]   # extended 3 decades below the previous 0.033 floor
SEEDS = [0, 1, 2, 3, 4]
OUTER_K, INNER_K, N_BOOT = 5, 3, 2000


# ----------------------------------------------------------------- features
def build_backbone(result_name):
    rd = an.RESULT_DATA[result_name]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd.get("columns")), STRAIN)
    Q = an.get_Q(STRAIN)
    S1 = df["semantic_z"].to_numpy()
    S2 = df["log_grammar_z"].to_numpy()
    S3 = {t: np.asarray(an.get_log_clib_z(df, Q, t), float) for t in T_LIST}
    code = (df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]).to_numpy()
    # degeneracy guard: at tiny t the log(clib+1e-15) floor can collapse many rows to one value
    deg = {t: round(float((np.abs(v - v.min()) < 1e-6).mean()), 3) for t, v in S3.items()}
    print(f"  [{result_name.split('-')[-1]}] S3 fraction-at-floor per t: {deg}")
    return S1, S2, S3, code


# ----------------------------------------------------------------- glm helpers
def fit_coefs(y, cols, offset=None):
    """GLM Binomial with intercept; return feature coefficients (intercept dropped)."""
    X = sm.add_constant(cols, has_constant="add")
    res = sm.GLM(y, X, family=sm.families.Binomial(),
                 offset=offset if offset is not None else None).fit()
    return np.asarray(res.params[1:], float)


def ranks_desc(score):
    return pd.Series(-score).rank(method="min").to_numpy()


def mean_rank(rank, y):
    return float(rank[y == 1].mean())


# ----------------------------------------------------------------- inner selection
def inner_meanrank(S1, S2, S3t, y, idx, seed, mode, kappa=1.0):
    """OOF mean-rank on the outer-train subset `idx` for a candidate S3 column (S3t)."""
    yy = y[idx]
    oof = np.full(len(idx), np.nan)
    skf = StratifiedKFold(INNER_K, shuffle=True, random_state=seed)
    for itr, ite in skf.split(np.zeros(len(idx)), yy):
        tr, te = idx[itr], idx[ite]
        if mode == "offset":
            b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]], offset=kappa * S3t[tr])
            oof[ite] = b[0] * S1[te] + b[1] * S2[te] + kappa * S3t[te]
        else:  # free b3
            b = fit_coefs(y[tr], np.c_[S1[tr], S2[tr], S3t[tr]])
            oof[ite] = b[0] * S1[te] + b[1] * S2[te] + b[2] * S3t[te]
    return mean_rank(ranks_desc(oof), yy)


def pick_t(S1, S2, S3, y, idx, seed, mode):
    scores = {t: inner_meanrank(S1, S2, S3[t], y, idx, seed, mode) for t in T_LIST}
    return min(scores, key=scores.get)


# ----------------------------------------------------------------- one seed of outer CV
def run_seed(S1, S2, S3, y, seed):
    n = len(y)
    oofA, oofB, oofC = (np.full(n, np.nan) for _ in range(3))
    tB_used, tC_used = [], []
    skf = StratifiedKFold(OUTER_K, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(n), y):
        # A
        bA = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]])
        oofA[te] = bA[0] * S1[te] + bA[1] * S2[te]
        # B: offset kappa=1, inner-select t
        tB = pick_t(S1, S2, S3, y, tr, seed, "offset"); tB_used.append(tB)
        bB = fit_coefs(y[tr], np.c_[S1[tr], S2[tr]], offset=1.0 * S3[tB][tr])
        oofB[te] = bB[0] * S1[te] + bB[1] * S2[te] + 1.0 * S3[tB][te]
        # C: free b3, inner-select t
        tC = pick_t(S1, S2, S3, y, tr, seed, "free"); tC_used.append(tC)
        bC = fit_coefs(y[tr], np.c_[S1[tr], S2[tr], S3[tC][tr]])
        oofC[te] = bC[0] * S1[te] + bC[1] * S2[te] + bC[2] * S3[tC][te]
    return oofA, oofB, oofC, tB_used, tC_used


# ----------------------------------------------------------------- legacy barycentric
def legacy_scores(S1, S2, S3, y, seed=0):
    """In-sample (optimistic) and OOS mean-rank for the old 3-weight barycentric grid search."""
    def obj_factory(s1, s2, s3, yy):
        def f(b):
            raw = b[0] * s1 + b[1] * s2 + b[2] * s3
            return mean_rank(ranks_desc(raw), yy)
        return f

    # in-sample: sweep t, minimize weights on full data, take best
    best_ins = np.inf
    for t in T_LIST:
        v, _ = an.minimize_bary_f(obj_factory(S1, S2, S3[t], y))
        raw = v[0] * S1 + v[1] * S2 + v[2] * S3[t]
        best_ins = min(best_ins, mean_rank(ranks_desc(raw), y))

    # OOS: 5-fold, fit t+weights on train, score test, pool
    n = len(y); oof = np.full(n, np.nan)
    skf = StratifiedKFold(OUTER_K, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(n), y):
        best_t, best_v, best_train_mr = None, None, np.inf
        for t in T_LIST:
            v, fv = an.minimize_bary_f(obj_factory(S1[tr], S2[tr], S3[t][tr], y[tr]))
            if fv < best_train_mr:
                best_train_mr, best_t, best_v = fv, t, v
        oof[te] = best_v[0] * S1[te] + best_v[1] * S2[te] + best_v[2] * S3[best_t][te]
    return best_ins, mean_rank(ranks_desc(oof), y), oof


# ----------------------------------------------------------------- bootstrap delta CI
def boot_delta(rankA, rankB, y, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    rA, rB = rankA[pos], rankB[pos]
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.randint(0, len(pos), len(pos))
        d[b] = rB[i].mean() - rA[i].mean()
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ----------------------------------------------------------------- main
def main():
    rows = []
    for bname, rname in BACKBONES.items():
        S1, S2, S3, code = build_backbone(rname)
        for task in TASKS:
            muts = set(an.ESCAPE_DATA[f"{STRAIN}>{task}"]["mutations"])
            y = np.isin(code, list(muts)).astype(int)
            npos = int(y.sum())
            if npos < 2:
                continue

            # multi-seed A/B/C -> average OOF ranks across seeds
            rankA_s, rankB_s, rankC_s = [], [], []
            aurA, aurB, aprA, aprB = [], [], [], []
            tB_all, tC_all = [], []
            for sd in SEEDS:
                oA, oB, oC, tB, tC = run_seed(S1, S2, S3, y, sd)
                rankA_s.append(ranks_desc(oA)); rankB_s.append(ranks_desc(oB)); rankC_s.append(ranks_desc(oC))
                aurA.append(roc_auc_score(y, oA)); aurB.append(roc_auc_score(y, oB))
                aprA.append(average_precision_score(y, oA)); aprB.append(average_precision_score(y, oB))
                tB_all += tB; tC_all += tC
            rankA = np.mean(rankA_s, axis=0); rankB = np.mean(rankB_s, axis=0); rankC = np.mean(rankC_s, axis=0)
            mrA, mrB, mrC = mean_rank(rankA, y), mean_rank(rankB, y), mean_rank(rankC, y)
            dBA, loBA, hiBA = boot_delta(rankA, rankB, y)

            leg_ins, leg_oos, _ = legacy_scores(S1, S2, S3, y)

            from collections import Counter
            rows.append({
                "backbone": bname, "task": task, "n_pos": npos, "N": len(y),
                "mr_A(noCLIB)": mrA, "mr_B(offset)": mrB, "mr_C(freeb3)": mrC,
                "mr_L_oos(legacy)": leg_oos, "mr_L_insample": leg_ins,
                "dBA": dBA, "dBA_CI": f"[{loBA:.0f},{hiBA:.0f}]",
                "AUROC_A": np.mean(aurA), "AUROC_B": np.mean(aurB),
                "AUPRC_A": np.mean(aprA), "AUPRC_B": np.mean(aprB),
                "t*_B(mode)": Counter(tB_all).most_common(1)[0][0],
                "offset_helps": bool(hiBA < 0),
            })

    res = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "phase0", "phase3_nestedcv.csv")
    res.to_csv(out, index=False)

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    fmt = {c: (lambda v: f"{v:.3g}") for c in
           ["mr_A(noCLIB)", "mr_B(offset)", "mr_C(freeb3)", "mr_L_oos(legacy)", "mr_L_insample",
            "dBA", "AUROC_A", "AUROC_B", "AUPRC_A", "AUPRC_B"]}
    print("\n================= PHASE 3 — nested CV (5 seeds x 5-fold), WildType =================")
    cols = ["backbone", "task", "n_pos", "mr_A(noCLIB)", "mr_B(offset)", "mr_C(freeb3)",
            "mr_L_oos(legacy)", "mr_L_insample", "dBA", "dBA_CI",
            "AUROC_A", "AUROC_B", "t*_B(mode)", "offset_helps"]
    print(res[cols].to_string(index=False, formatters=fmt))
    print(f"\nwrote {out}")
    print("\nlegend: mr = mean-rank of held-out positives (lower better). "
          "dBA = mean-rank(B) - mean-rank(A); offset_helps if its 95% CI < 0.")


if __name__ == "__main__":
    main()
