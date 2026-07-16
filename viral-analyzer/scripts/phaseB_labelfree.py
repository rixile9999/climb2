#!/usr/bin/env python3
"""Phase B — label-free (unsupervised-of-escape) fitting of b0, b1.

Idea: choose the two protein-signal weights b0, b1 WITHOUT ever touching the
escape answer key. Instead we fit them against an escape-independent *statistical*
target — the observed frequency of each single amino-acid substitution in the
circulating SARS-CoV-2 population (CoV-Spectrum / LAPIS open data). The CLIB term
stays frozen as an offset (kappa = 1, t = 0.01), exactly as in the offset research.

  fit  :  logistic( y_proxy ~ b0*S1 + b1*S2 ,  offset = 1*CLIB )      (NO escape labels)
          y_proxy = 1[ observed proportion >= threshold ]
  eval :  apply the FIXED (b0, b1) to each escape task; mean-rank + AUROC.

Because the label used to fit (frequency) is disjoint from the label used to
evaluate (curated escape), the escape numbers are honest out-of-sample w.r.t.
escape. We also report the frequency<->escape overlap so the (expected) weak
leakage via natural selection is transparent.

Baselines: (A) equal weight b0=b1=1 · CLIB-only b0=b1=0 · supervised OOF upper bound.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "outputs/phase0/tidy_esm2cov_l1.parquet"
FREQ = ROOT / "data/observed/spike_obs_freq.tsv"
OUT = ROOT / "outputs/phase0/phaseB_labelfree.csv"
S3_COL = "S3_logclib_z_t0.01"
ESC_TASKS = ["DMS", "Omicron", "Alpha", "Beta", "Gamma", "Delta"]
THRESHOLDS = [0.01, 0.001, 0.0001]      # "established / seen / rare-but-real"
SEEDS = range(5)
K = 5


# ---- logistic with fixed offset (kappa=1) --------------------------------------
def fit_b(S1, S2, y, offset):
    X = np.c_[np.ones(len(y)), S1, S2]

    def nll(w):
        eta = X @ w + offset
        return -np.sum(y * eta - np.logaddexp(0, eta))

    def grad(w):
        p = 1 / (1 + np.exp(-(X @ w + offset)))
        return -X.T @ (y - p)

    w = minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x
    return w[1], w[2]      # b0, b1  (intercept irrelevant to ranking)


def mean_rank(score, y):
    r = pd.Series(-score).rank(method="min").to_numpy()
    return float(r[y == 1].mean())


def auroc(score, y):
    # rank-based AUROC (Mann-Whitney), robust without sklearn metric import
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def score_of(df, b0, b1):
    return b0 * df["S1_semantic_z"].to_numpy() + b1 * df["S2_log_grammar_z"].to_numpy() \
        + 1.0 * df[S3_COL].to_numpy()


def boot_ci(delta_fn, y, seed=0, B=2000):
    """95% CI for a per-positive mean-rank delta, bootstrapping the positives."""
    pos = np.where(y == 1)[0]
    rng = np.random.RandomState(seed)
    ds = [delta_fn(rng.choice(pos, len(pos), replace=True)) for _ in range(B)]
    return np.percentile(ds, [2.5, 97.5])


def main():
    df = pd.read_parquet(TIDY)
    df["mutation_code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]

    freq = pd.read_csv(FREQ, sep="\t")
    freq["mutation_code"] = freq["wt"] + freq["pos"].astype(str) + freq["mut"]
    fmap = dict(zip(freq["mutation_code"], freq["proportion"]))
    df["obs_prop"] = df["mutation_code"].map(fmap).fillna(0.0)

    S1 = df["S1_semantic_z"].to_numpy()
    S2 = df["S2_log_grammar_z"].to_numpy()
    S3 = df[S3_COL].to_numpy()
    print(f"n candidates = {len(df)}, observed (>0) = {(df['obs_prop']>0).sum()}")

    # ---- fit b0,b1 label-free at each frequency threshold ----------------------
    fits = {}
    for thr in THRESHOLDS:
        yprox = (df["obs_prop"] >= thr).astype(int).to_numpy()
        b0, b1 = fit_b(S1, S2, yprox, 1.0 * S3)
        fits[thr] = (b0, b1, int(yprox.sum()))
        print(f"[proxy thr={thr:<7}] positives={yprox.sum():5d}  ->  b0(sem)={b0:+.3f}  b1(gram)={b1:+.3f}")

    rows = []
    for task in ESC_TASKS:
        col = f"esc__{task}"
        if col not in df:
            continue
        y = df[col].astype(int).to_numpy()
        if y.sum() == 0:
            continue

        # frequency<->escape overlap (leakage transparency)
        ov = {thr: int(((df["obs_prop"] >= thr).to_numpy() & (y == 1)).sum()) for thr in THRESHOLDS}

        # baselines
        eq = score_of(df, 1.0, 1.0)          # (A) equal weight
        cl = score_of(df, 0.0, 0.0)          # CLIB-only
        base_mr = mean_rank(eq, y); base_au = auroc(eq, y)
        rows.append(dict(task=task, n_pos=int(y.sum()), method="A_equal_w(1,1)",
                         b0=1.0, b1=1.0, mean_rank=base_mr, AUROC=base_au, d_vs_equal=0.0, CI="[0,0]"))
        rows.append(dict(task=task, n_pos=int(y.sum()), method="CLIB_only(0,0)",
                         b0=0.0, b1=0.0, mean_rank=mean_rank(cl, y), AUROC=auroc(cl, y),
                         d_vs_equal=mean_rank(cl, y) - base_mr, CI=""))

        # label-free proxy fits
        for thr, (b0, b1, npos) in fits.items():
            sc = score_of(df, b0, b1)
            mr = mean_rank(sc, y)
            rEq = pd.Series(-eq).rank(method="min").to_numpy()
            rSc = pd.Series(-sc).rank(method="min").to_numpy()
            ci = boot_ci(lambda idx: rSc[idx].mean() - rEq[idx].mean(), y)
            rows.append(dict(task=task, n_pos=int(y.sum()),
                             method=f"proxy_freq>={thr}", b0=round(b0, 3), b1=round(b1, 3),
                             mean_rank=mr, AUROC=auroc(sc, y),
                             d_vs_equal=mr - base_mr, CI=f"[{ci[0]:.0f},{ci[1]:.0f}]",
                             overlap_pos=ov[thr]))

        # supervised OOF upper bound (uses escape labels; for reference only)
        oof = np.full(len(y), np.nan)
        for sd in SEEDS:
            for tr, te in StratifiedKFold(K, shuffle=True, random_state=sd).split(S1, y):
                b0s, b1s = fit_b(S1[tr], S2[tr], y[tr], 1.0 * S3[tr])
                oof[te] = b0s * S1[te] + b1s * S2[te] + S3[te]
        # average over seeds is approximated by last-seed OOF ranks; report mean-rank
        rows.append(dict(task=task, n_pos=int(y.sum()), method="supervised_OOF(ref)",
                         b0=np.nan, b1=np.nan, mean_rank=mean_rank(oof, y), AUROC=auroc(oof, y),
                         d_vs_equal=mean_rank(oof, y) - base_mr, CI=""))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + res.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
