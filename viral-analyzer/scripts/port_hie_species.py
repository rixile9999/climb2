#!/usr/bin/env python3
"""Port Hie et al.'s influenza-HA (Doud 2018) and HIV-Env (Dingens 2019) escape
data into viral-analyzer scaffolding for the cross-species CLIB universality test.

Produces, per species:
  data/species/<sp>_wt.fasta          WT amino-acid sequence (clean)
  data/species/<sp>_escape.tsv        significant escape single-mutants (pos0,wt,mut)
  data/sequences/seq_<sp>.csv         codon table (pos,codon,aa) via modal reverse-
                                      translation (CLIB approximation — real per-site
                                      codons would need the exact strain CDS)
and prints strain-registration snippets (virus, stat-dist).

Escape significance cutoffs match escape.py: flu HA frac_survive>=0.05 (6 antibodies),
HIV Env >=0.11 (10 antibodies).
"""
import csv
import glob
import os
from pathlib import Path
from Bio import SeqIO

VM = Path("/home/kist/workspace/climb2/viral-mutation")
OUT = Path("/home/kist/workspace/climb2/viral-analyzer")

# most-frequent codon per amino acid (standard reverse-translation table)
MODAL_CODON = {
    'A': 'GCC', 'R': 'CGC', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC', 'Q': 'CAG',
    'E': 'GAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'L': 'CTG', 'K': 'AAG',
    'M': 'ATG', 'F': 'TTC', 'P': 'CCC', 'S': 'AGC', 'T': 'ACC', 'W': 'TGG',
    'Y': 'TAC', 'V': 'GTG',
}


def load_flu():
    seq = str(next(SeqIO.parse(VM / "data/influenza/escape_doud2018/WSN1933_H1_HA.fa", "fasta")).seq)
    pos_map = {}
    with open(VM / "data/influenza/escape_doud2018/pos_map.csv") as f:
        f.readline()
        for line in f:
            a, b = line.rstrip().split(",")
            pos_map[b] = int(a) - 1
    esc = set()
    for fn in glob.glob(str(VM / "data/influenza/escape_doud2018/medianfracsurvivefiles/antibody_*_median.csv")):
        with open(fn) as f:
            f.readline()
            for line in f:
                site, wt, mut, frac = line.rstrip().split(",")
                if float(frac) < 0.05:
                    continue
                pos = pos_map[site]
                if seq[pos] != wt:
                    continue
                esc.add((pos, wt, mut))
    return seq, esc


def load_hiv():
    seq = None
    for r in SeqIO.parse(VM / "data/hiv/escape_dingens2019/Env_protalign_manualeditAD.fasta", "fasta"):
        if r.description == "BG505":
            seq = str(r.seq); break
    pos_map = {}
    with open(VM / "data/hiv/escape_dingens2019/BG505_to_HXB2.csv") as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(",")
            pos_map[fields[1]] = int(fields[0]) - 1
    esc = set()
    for fn in glob.glob(str(VM / "data/hiv/escape_dingens2019/FileS4/fracsurviveaboveavg/*.csv")):
        with open(fn) as f:
            f.readline()
            for line in f:
                fields = line.rstrip().split(",")
                site, wt, mut, frac = fields[0], fields[1], fields[2], float(fields[3])
                if frac < 0.11 or site not in pos_map:
                    continue
                pos = pos_map[site]
                if pos >= len(seq) or seq[pos] != wt:
                    continue
                esc.add((pos, wt, mut))
    return seq, esc


def write_species(sp, virus, seq, esc):
    (OUT / "data/species").mkdir(parents=True, exist_ok=True)
    with open(OUT / f"data/species/{sp}_wt.fasta", "w") as f:
        f.write(f">{sp}\n{seq}\n")
    with open(OUT / f"data/species/{sp}_escape.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["pos", "wt", "mut"])
        w.writerows(sorted(esc))
    # codon table (modal reverse-translation)
    with open(OUT / f"data/sequences/seq_{sp}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pos", "codon", "aa"])
        for i, aa in enumerate(seq):
            w.writerow([i, MODAL_CODON.get(aa, "NNN"), aa])
    # stat-dist from the reverse-translated CDS
    cds = "".join(MODAL_CODON.get(a, "") for a in seq)
    dist = [cds.count(b) for b in "ACGT"]
    n_esc_sites = len({p for p, _, _ in esc})
    print(f"[{sp}] len={len(seq)} escape_muts={len(esc)} escape_sites={n_esc_sites} "
          f"virus='{virus}' stat_dist(ACGT)={dist}")
    return dist


def main():
    flu_seq, flu_esc = load_flu()
    hiv_seq, hiv_esc = load_hiv()
    write_species("flu_ha", "Influenza A", flu_seq, flu_esc)
    write_species("hiv_env", "HIV", hiv_seq, hiv_esc)
    print("\nAdd these to embedding/metadata/sequences.json and register strains "
          "(virus, stat-dist, sequence-path=data/sequences/seq_<sp>.csv).")


if __name__ == "__main__":
    main()
