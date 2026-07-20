#!/usr/bin/env python3
"""Phase B11 (Step B) — multi-mutation antigenic escape on H3N2 clusters.

A real MULTI-mutation escape ground truth: 555 natural H3N2 HA strains labeled by
Smith-2004 antigenic cluster (14 clusters, HK68..SW13). Strains in different
clusters are antigenically escaped; each strain differs from others by MANY
mutations. Tests whether multi-mutation semantic change (mean-pooled ESM2
embedding L1 distance) predicts antigenic escape between diverged strains.

Contrast with B9 (single-mutation DMS escape on flu, AUROC ~0.60-0.64): if
multi-mutation antigenic escape is predicted well, it supports the hypothesis
that escape in old species lives at the multi-mutation level.
"""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/kist/workspace/climb2/viral-analyzer")


def auroc(score, y):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    p = int(y.sum()); n = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def main():
    d = np.load(ROOT / "outputs/phase0/h3_panel_emb.npz", allow_pickle=True)
    emb, cl, order, year = d["emb"], d["cluster"], d["order"].astype(int), d["year"].astype(int)
    n = len(emb)
    # all unordered pairs
    iu, ju = np.triu_indices(n, k=1)
    sem = np.abs(emb[iu] - emb[ju]).sum(1)                 # L1 semantic distance
    diff_cluster = (cl[iu] != cl[ju]).astype(int)          # antigenic escape (binary)
    order_dist = np.abs(order[iu] - order[ju])             # antigenic-order distance
    year_dist = np.abs(year[iu] - year[ju])

    print(f"pairs={len(sem)}  diff-cluster={diff_cluster.sum()} ({100*diff_cluster.mean():.0f}%)")
    print("\n=== multi-mutation semantic vs antigenic escape ===")
    print(f"  AUROC(semantic -> different cluster) = {auroc(sem, diff_cluster):.4f}")
    print(f"  Spearman(semantic, antigenic-order distance) = {spearmanr(sem, order_dist).statistic:+.4f}")
    print(f"  Spearman(semantic, year distance)            = {spearmanr(sem, year_dist).statistic:+.4f}")
    win = sem[diff_cluster == 0]; bet = sem[diff_cluster == 1]
    print(f"\n  within-cluster  semantic: mean {win.mean():.2f}")
    print(f"  between-cluster semantic: mean {bet.mean():.2f}   (ratio {bet.mean()/win.mean():.2f}x)")

    # adjacent-cluster escape (hardest): distinguish neighboring antigenic clusters
    adj = order_dist == 1
    print(f"\n  adjacent-cluster pairs (order dist=1, n={adj.sum()}):")
    same_or_adj = sem[order_dist <= 1]
    y_adj = (order_dist[order_dist <= 1] == 1).astype(int)
    print(f"    AUROC(semantic -> adjacent vs same cluster) = {auroc(same_or_adj, y_adj):.4f}")

    # ordinal drift: does semantic increase monotonically with cluster order gap?
    print("\n  mean semantic by antigenic-order gap:")
    for g in range(0, 8):
        m = order_dist == g
        if m.sum() >= 30:
            print(f"    gap={g:2d}: n={m.sum():6d}  mean semantic {sem[m].mean():.2f}")


if __name__ == "__main__":
    main()
