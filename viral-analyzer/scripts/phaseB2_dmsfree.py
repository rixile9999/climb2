#!/usr/bin/env python3
"""Phase B2 — evaluate the model WITHOUT the DMS answer key.

Two DMS-independent ground truths:

  A) Prospective emergence (temporal).  Split observed frequency into an EARLY
     window (<= 2021-12-31) and a LATE window (>= 2022-06-01). A mutation is a
     forward-positive if it was rare early and rose to substantial frequency
     late (a novel riser). The model must rank these high using only pre-cutoff
     information. Protein/CLIB features are frequency-independent, so equal-weight
     and CLIB-only are fully honest forward predictors; the proxy fit uses EARLY
     frequency only.

  B) Phylogeny-derived fitness (convergent evolution).  Bloom et al. estimate a
     per-mutation delta_fitness from observed-vs-expected independent occurrences
     across ~7M sequences on the UShER tree. We measure how well the model score
     agrees with this (Spearman rho, plus AUROC for the high-fitness set). This is
     independent of DMS; equal-weight/CLIB-only are also independent of frequency.

Model variants: A_equal(1,1) · CLIB_only(0,0) · proxy_full · proxy_early ·
                supervised_on_DMS (reference).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TIDY = ROOT / "outputs/phase0/tidy_esm2cov_l1.parquet"
S3_COL = "S3_logclib_z_t0.01"
OUT = ROOT / "outputs/phase0/phaseB2_dmsfree.csv"
AAS = set("ACDEFGHIKLMNPQRSTVWY")

# ground-truth thresholds
EARLY_MAX = 0.05      # "rare early"
LATE_MIN = 0.25       # "common late"
BLOOM_HI = 2.0        # delta_fitness >= this -> high-fitness positive


def load_window(path):
    w = pd.read_csv(path, sep="\t")
    return dict(zip(w["wt"] + w["pos"].astype(str) + w["mut"], w["proportion"]))


def fit_b(S1, S2, y, offset):
    X = np.c_[np.ones(len(y)), S1, S2]
    nll = lambda w: -np.sum(y * (X @ w + offset) - np.logaddexp(0, X @ w + offset))
    grad = lambda w: -X.T @ (y - 1 / (1 + np.exp(-(X @ w + offset))))
    w = minimize(nll, np.zeros(3), jac=grad, method="L-BFGS-B").x
    return w[1], w[2]


def mean_rank(score, y):
    r = pd.Series(-score).rank(method="min").to_numpy()
    return float(r[y == 1].mean())


def auroc(score, y):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return np.nan if npos == 0 or nneg == 0 else (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    df = pd.read_parquet(TIDY)
    df["code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    S1 = df["S1_semantic_z"].to_numpy(); S2 = df["S2_log_grammar_z"].to_numpy(); S3 = df[S3_COL].to_numpy()

    # ---- A: temporal windows -------------------------------------------------
    W = ROOT / "data/observed/windows"
    early = load_window(W / "spike_le_2021-12.tsv"); late = load_window(W / "spike_ge_2022-06.tsv")
    df["early"] = df["code"].map(early).fillna(0.0)
    df["late"] = df["code"].map(late).fillna(0.0)
    emergent = ((df["early"] < EARLY_MAX) & (df["late"] >= LATE_MIN)).astype(int).to_numpy()
    print(f"A) emergent risers (early<{EARLY_MAX}, late>={LATE_MIN}): {emergent.sum()} positives")

    # ---- B: Bloom phylogenetic fitness --------------------------------------
    bl = pd.read_csv(ROOT / "data/observed/bloom_spike_fitness.tsv", sep="\t")
    bl["code"] = bl["clade_founder_aa"] + (bl["aa_site"] - 1).astype(str) + bl["mutant_aa"]
    fmap = dict(zip(bl["code"], bl["delta_fitness"]))
    df["bloom"] = df["code"].map(fmap)
    has_bloom = df["bloom"].notna().to_numpy()
    bloom_hi = ((df["bloom"] >= BLOOM_HI).fillna(False)).astype(int).to_numpy()
    print(f"B) Bloom fitness available for {has_bloom.sum()} candidates; high-fitness(>= {BLOOM_HI}) = {bloom_hi.sum()}")

    # ---- fit the proxy weights ----------------------------------------------
    # proxy_full: fit on all-time observed frequency (from Phase B)
    from pathlib import Path as _P
    fr = pd.read_csv(ROOT / "data/observed/spike_obs_freq.tsv", sep="\t")
    fr["code"] = fr["wt"] + fr["pos"].astype(str) + fr["mut"]
    allprop = df["code"].map(dict(zip(fr["code"], fr["proportion"]))).fillna(0.0).to_numpy()
    b0_full, b1_full = fit_b(S1, S2, (allprop >= 0.001).astype(int), 1.0 * S3)
    # proxy_early: fit on EARLY-window frequency only (temporally honest for A)
    b0_early, b1_early = fit_b(S1, S2, (df["early"].to_numpy() >= 0.001).astype(int), 1.0 * S3)
    # supervised on DMS (reference; uses the answer key)
    yd = df["esc__DMS"].astype(int).to_numpy()
    b0_dms, b1_dms = fit_b(S1, S2, yd, 1.0 * S3)

    variants = {
        "A_equal(1,1)": (1.0, 1.0),
        "CLIB_only(0,0)": (0.0, 0.0),
        "proxy_full(freq)": (b0_full, b1_full),
        "proxy_early(freq<=2021)": (b0_early, b1_early),
        "supervised_on_DMS(ref)": (b0_dms, b1_dms),
    }
    print("\nweights:")
    for k, (a, b) in variants.items():
        print(f"  {k:26s} b0={a:+.3f} b1={b:+.3f}")

    rows = []
    for name, (b0, b1) in variants.items():
        sc = b0 * S1 + b1 * S2 + 1.0 * S3
        # A metrics
        a_mr = mean_rank(sc, emergent); a_au = auroc(sc, emergent)
        # B metrics (restrict to candidates with Bloom values)
        rho = spearmanr(sc[has_bloom], df["bloom"].to_numpy()[has_bloom]).statistic
        b_au = auroc(sc, bloom_hi)
        rows.append(dict(variant=name, b0=round(b0, 3), b1=round(b1, 3),
                         A_emergent_meanrank=round(a_mr, 1), A_emergent_AUROC=round(a_au, 4),
                         B_bloom_spearman=round(rho, 4), B_bloom_hi_AUROC=round(b_au, 4)))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== DMS-free evaluation (A: temporal emergence · B: Bloom phylo-fitness) ===")
    print(res.to_string(index=False))
    print(f"\nN candidates={len(df)}; A positives={emergent.sum()}; B Bloom-scored={has_bloom.sum()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
