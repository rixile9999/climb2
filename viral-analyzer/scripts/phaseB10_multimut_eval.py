#!/usr/bin/env python3
"""Phase B10 (Step B) — multi-mutation CAC on Wu2020 flu HA fitness.

Tests the point->multi hypothesis: for an old species, does MULTI-mutation
accessibility (CLIB with a larger evolutionary time t) help predict combinatorial
fitness, where single-nucleotide CLIB (t=0.01) does not?

multi-CLIB(variant, t) = sum over mutated positions of log P(WT_codon -> variant_aa, t)
using the Influenza-A substitution model. Reports Spearman correlation of each
signal (and a combined CAC) with measured fitness (preference), swept over t,
pooled and split by number of mutations.

Codons at the 6 site-B positions are modal reverse-translations (prototype).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import analyzer as an

MUT_POS = [p - 1 + 16 for p in (156, 158, 159, 190, 193, 196)]
WT6 = {'HK68': 'KGSESV', 'Bk79': 'EESENV', 'Bei89': 'EEYENV',
       'Mos99': 'QKYDST', 'Bris07L194': 'HKFDFA', 'NDako16': 'HNSDFA'}
MODAL = {'A': 'GCC', 'R': 'CGC', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC', 'Q': 'CAG',
         'E': 'GAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'L': 'CTG', 'K': 'AAG',
         'M': 'ATG', 'F': 'TTC', 'P': 'CCC', 'S': 'AGC', 'T': 'ACC', 'W': 'TGG',
         'Y': 'TAC', 'V': 'GTG'}
T_SWEEP = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def main():
    df = pd.read_csv("outputs/phase0/wu2020_features.tsv", sep="\t")
    df = df.dropna(subset=["preference"]).reset_index(drop=True)
    Q = an.get_Q("Influenza-HA-WSN")   # Influenza-A substitution model

    # per-t multi-CLIB
    clib_cols = {}
    for t in T_SWEEP:
        tab = an.get_clib_table(Q, t)
        vals = []
        for _, r in df.iterrows():
            wt6 = WT6[r["strain"]]
            s = 0.0
            for wt_aa, var_aa in zip(wt6, r["variant"]):
                if wt_aa != var_aa:
                    s += np.log(tab.get(f"{MODAL[wt_aa]}->{var_aa}", 1e-15) + 1e-15)
            vals.append(s)
        clib_cols[t] = np.array(vals)

    y = df["preference"].to_numpy()
    sem, gram, nmut = df["semantic"].to_numpy(), df["grammar"].to_numpy(), df["n_mut"].to_numpy()

    def sp(a, b):
        return spearmanr(a, b, nan_policy="omit").statistic

    print("=== single-signal Spearman vs fitness (pooled, n=%d) ===" % len(df))
    print(f"  semantic {sp(sem,y):+.3f}   grammar {sp(gram,y):+.3f}")
    print("\n=== multi-CLIB(t) Spearman vs fitness, and CAC(sem+gram+CLIB) ===")
    print(f"{'t':>6} {'CLIB_only':>10} {'gram+CLIB':>10} {'sem+gram+CLIB':>14}")
    rows = []
    for t in T_SWEEP:
        c = clib_cols[t]
        cac_gc = zscore(gram) + zscore(c)
        cac_all = zscore(sem) + zscore(gram) + zscore(c)
        r_c, r_gc, r_all = sp(c, y), sp(cac_gc, y), sp(cac_all, y)
        print(f"{t:>6} {r_c:>+10.3f} {r_gc:>+10.3f} {r_all:>+14.3f}")
        rows.append(dict(t=t, CLIB_only=round(r_c, 4), gram_CLIB=round(r_gc, 4),
                         sem_gram_CLIB=round(r_all, 4)))

    # does CLIB matter MORE for higher-order (multi) mutants?
    print("\n=== CLIB(best t) Spearman by number of mutations ===")
    best_t = max(T_SWEEP, key=lambda t: abs(sp(clib_cols[t], y)))
    for k in sorted(set(nmut)):
        m = nmut == k
        if m.sum() >= 20:
            print(f"  n_mut={k}: n={m.sum():4d}  CLIB(t={best_t}) {sp(clib_cols[best_t][m], y[m]):+.3f}"
                  f"  grammar {sp(gram[m], y[m]):+.3f}")
    pd.DataFrame(rows).to_csv("outputs/phase0/phaseB10_multimut.csv", index=False)
    print(f"\nbest CLIB t = {best_t}   (t=0.01 CLIB {sp(clib_cols[0.01],y):+.3f} -> t={best_t} {sp(clib_cols[best_t],y):+.3f})")


if __name__ == "__main__":
    main()
