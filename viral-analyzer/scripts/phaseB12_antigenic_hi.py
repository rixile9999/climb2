#!/usr/bin/env python3
"""Phase B12 — multi-mutation antigenic distance with temporal-confound control.

Uses the Smith-2004 H3N2 antigenic map (273 strains, HI-titer-derived 2D antigenic
coordinates; Racmacs h3map2004). Antigenic distance = Euclidean distance in the map
(antigenic units ~ 2-fold HI dilution). Tests whether multi-mutation SEMANTIC change
(ESM2 embedding L1) predicts antigenic distance BEYOND temporal drift, via three
controls:
  1) raw Spearman(semantic, antigenic)
  2) partial Spearman(semantic, antigenic | year distance)   -- controls time
  3) within-year-band pairs (|Δyear| <= k): strains close in time only

The earlier cluster test (B11) confounded antigenic with time; this isolates the
antigenic signal.
"""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path("/home/kist/workspace/climb2/viral-analyzer")


def partial_spearman(a, b, c):
    """Spearman partial correlation of a,b controlling for c (rank-Pearson partial)."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    def r(x, y): return np.corrcoef(x, y)[0, 1]
    rab, rac, rbc = r(ra, rb), r(ra, rc), r(rb, rc)
    return (rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2))


def main():
    d = np.load(ROOT / "outputs/phase0/h3_antigenic_emb.npz", allow_pickle=True)
    emb, year = d["emb"], d["year"].astype(int)
    coords = np.column_stack([d["ag_x"].astype(float), d["ag_y"].astype(float)])
    n = len(emb)
    iu, ju = np.triu_indices(n, k=1)
    sem = np.abs(emb[iu] - emb[ju]).sum(1)                       # semantic L1
    antig = np.sqrt(((coords[iu] - coords[ju])**2).sum(1))       # antigenic distance
    ydist = np.abs(year[iu] - year[ju])
    print(f"strains={n}  pairs={len(sem)}  year {year.min()}-{year.max()}")

    print("\n=== raw associations (all pairs) ===")
    print(f"  Spearman(semantic, antigenic)      = {spearmanr(sem, antig).statistic:+.3f}")
    print(f"  Spearman(semantic, year-dist)      = {spearmanr(sem, ydist).statistic:+.3f}")
    print(f"  Spearman(antigenic, year-dist)     = {spearmanr(antig, ydist).statistic:+.3f}")

    print("\n=== temporal control ===")
    ps = partial_spearman(sem, antig, ydist)
    print(f"  PARTIAL Spearman(semantic, antigenic | year) = {ps:+.3f}")

    print("\n  within-year-band (strains close in time):")
    for k in [0, 2, 5, 10]:
        m = ydist <= k
        if m.sum() >= 50:
            print(f"    |Δyear|<={k:2d}: n={m.sum():6d}  Spearman(semantic, antigenic) = {spearmanr(sem[m], antig[m]).statistic:+.3f}")

    # baseline: does raw genetic (year) predict antigenic as well? and does semantic
    # beat year at predicting antigenic?
    print("\n=== semantic vs time as antigenic predictors ===")
    print(f"  Spearman(year-dist -> antigenic)   = {spearmanr(ydist, antig).statistic:+.3f}")
    print(f"  Spearman(semantic  -> antigenic)   = {spearmanr(sem, antig).statistic:+.3f}")
    ps2 = partial_spearman(antig, sem, ydist)
    print(f"  antigenic explained by semantic beyond year (partial) = {ps2:+.3f}")


if __name__ == "__main__":
    main()
