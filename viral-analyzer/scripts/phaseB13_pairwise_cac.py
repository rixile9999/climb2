#!/usr/bin/env python3
"""Phase B13 — complete pairwise CAC on antigenic distance (semantic + multi-CLIB).

B12 predicted antigenic distance from SEMANTIC alone. Here we add the pairwise
MULTI-CLIB term to test whether the complete CAC beats semantic-only.

For a strain pair (A,B) aligned over 328 HA1 positions, the mutation set M is the
positions where they differ. The pairwise multi-CLIB at evolutionary time t is

    CLIB(A,B; t) = 1/2 [ Σ_{i∈M} log P(codon(a_i^A) -> a_i^B, t)
                       + Σ_{i∈M} log P(codon(a_i^B) -> a_i^A, t) ]   (symmetrized)

with codon(·) the modal reverse-translation (codon approximation). We then fit a
signed model [semantic, CLIB(t)] to antigenic distance and compare to semantic
alone, under the same temporal controls as B12 (partial | year, within-year band).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))
import analyzer as an
from Bio import SeqIO

MODAL = {'A':'GCC','R':'CGC','N':'AAC','D':'GAC','C':'TGC','Q':'CAG','E':'GAG','G':'GGC','H':'CAC',
         'I':'ATC','L':'CTG','K':'AAG','M':'ATG','F':'TTC','P':'CCC','S':'AGC','T':'ACC','W':'TGG','Y':'TAC','V':'GTG'}
AAS = set(MODAL)
T_SWEEP = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]


def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    r = lambda x, y: np.corrcoef(x, y)[0, 1]
    rab, rac, rbc = r(ra, rb), r(ra, rc), r(rb, rc)
    return (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))


def main():
    d = np.load(ROOT + "/outputs/phase0/h3_antigenic_emb.npz", allow_pickle=True)
    emb, year = d["emb"], d["year"].astype(int)
    coords = np.column_stack([d["ag_x"].astype(float), d["ag_y"].astype(float)])
    ids = d["ids"]
    # ALIGNED 328-position sequences (position-corresponding across strains) for the
    # pairwise mutation set — NOT the ungapped embedding fasta.
    aln = pd.read_csv(ROOT + "/data/species/h3_antigenic_aligned.tsv", sep="\t")
    amap = dict(zip(aln["id"], aln["aligned"]))
    seqs = [amap[i] for i in ids]
    n = len(ids)
    L = min(len(s) for s in seqs)
    S = np.array([list(s[:L]) for s in seqs])                 # n x L aligned residues

    # precompute CLIB tables per t
    Q = an.get_Q("Influenza-HA-WSN")
    tabs = {t: an.get_clib_table(Q, t) for t in T_SWEEP}

    iu, ju = np.triu_indices(n, k=1)
    sem = np.abs(emb[iu] - emb[ju]).sum(1)
    antig = np.sqrt(((coords[iu] - coords[ju])**2).sum(1))
    ydist = np.abs(year[iu] - year[ju])
    # Hamming (aa differences) — control: is pairwise CLIB just a mutation count?
    ham = np.array([np.sum((S[i] != S[j]) & np.isin(S[i], list(AAS)) & np.isin(S[j], list(AAS)))
                    for i, j in zip(iu, ju)])

    # pairwise multi-CLIB (symmetrized) per t
    def clib_pair(a_res, b_res, tab):
        diff = a_res != b_res
        s = 0.0
        for aa, bb in zip(a_res[diff], b_res[diff]):
            if aa in AAS and bb in AAS:
                s += np.log(tab.get(f"{MODAL[aa]}->{bb}", 1e-15) + 1e-15)
                s += np.log(tab.get(f"{MODAL[bb]}->{aa}", 1e-15) + 1e-15)
        return 0.5 * s

    print(f"strains={n} pairs={len(iu)} aligned_len={L}")
    print("\n=== pairwise multi-CLIB(t) vs antigenic, and CAC[sem+CLIB] (signed, temporal-controlled) ===")
    print(f"{'t':>7} {'CLIB~antig':>11} {'CLIB|year':>10} {'CAC~antig':>10} {'CAC|year':>9} {'CAC Δyr=0':>10}")
    from sklearn.linear_model import LinearRegression
    m0 = ydist == 0
    base_raw = spearmanr(sem, antig).statistic
    base_par = partial_spearman(sem, antig, ydist)
    base_wy = spearmanr(sem[m0], antig[m0]).statistic
    rows = []
    for t in T_SWEEP:
        tab = tabs[t]
        clib = np.array([clib_pair(S[i], S[j], tab) for i, j in zip(iu, ju)])
        # signed CAC = supervised linear fit of [sem, clib] -> antigenic (in-sample direction)
        X = np.column_stack([sem, clib])
        cac = LinearRegression().fit(X, antig).predict(X)
        r_c, r_cpar = spearmanr(clib, antig).statistic, partial_spearman(clib, antig, ydist)
        r_cac, r_cacpar = spearmanr(cac, antig).statistic, partial_spearman(cac, antig, ydist)
        r_cacwy = spearmanr(cac[m0], antig[m0]).statistic
        print(f"{t:>7} {r_c:>+11.3f} {r_cpar:>+10.3f} {r_cac:>+10.3f} {r_cacpar:>+9.3f} {r_cacwy:>+10.3f}")
        rows.append(dict(t=t, CLIB_antig=round(r_c,4), CLIB_par=round(r_cpar,4),
                         CAC_antig=round(r_cac,4), CAC_par=round(r_cacpar,4), CAC_wy=round(r_cacwy,4)))
    pd.DataFrame(rows).to_csv(ROOT + "/outputs/phase0/phaseB13_pairwise_cac.csv", index=False)
    print(f"\nsemantic-only baseline: raw {base_raw:+.3f} | partial|year {base_par:+.3f} | within-year(Δyr=0) {base_wy:+.3f}")

    # --- honest control: pairwise CLIB vs plain Hamming (mutation count) ---
    clib10 = np.array([clib_pair(S[i], S[j], tabs[10.0]) for i, j in zip(iu, ju)])
    from sklearn.linear_model import LinearRegression as LR
    cac_h = LR().fit(np.column_stack([sem, ham]), antig).predict(np.column_stack([sem, ham]))
    cac_c = LR().fit(np.column_stack([sem, clib10]), antig).predict(np.column_stack([sem, clib10]))
    print("\n=== is pairwise CLIB just Hamming (mutation count)? ===")
    print(f"  corr(CLIB, Hamming) = {np.corrcoef(clib10, ham)[0,1]:.4f}")
    print(f"  CAC[sem+Hamming] partial|year {partial_spearman(cac_h, antig, ydist):+.3f} | Δyr=0 {spearmanr(cac_h[m0], antig[m0]).statistic:+.3f}")
    print(f"  CAC[sem+CLIB]    partial|year {partial_spearman(cac_c, antig, ydist):+.3f} | Δyr=0 {spearmanr(cac_c[m0], antig[m0]).statistic:+.3f}")
    print("  -> the complete CAC beats semantic-only, but the gain is GENETIC DISTANCE\n"
          "     (mutation count), which CLIB proxies ~perfectly at strain-pair scale;\n"
          "     the mechanistic codon-accessibility structure washes out (t-independent).")


if __name__ == "__main__":
    main()
