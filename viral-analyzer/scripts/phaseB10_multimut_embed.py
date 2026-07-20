#!/usr/bin/env python3
"""Phase B10 (Step A) — multi-mutation feature extraction on the Wu2020 flu HA
combinatorial landscape (6 site-B positions, 6 H3 strains, ~4000 variants).

Each variant = strain WT with substitutions at 6 positions (0-6 mutations = MULTI).
Computes, per variant:
  n_mut     : number of mutated positions vs strain WT
  semantic  : L1 distance of mean-pooled ESM2 embedding (variant vs strain WT)
  grammar   : sum of masked-marginal log-prob of the variant residues at the
              mutated positions (WT context) -> joint plausibility
Writes outputs/phase0/wu2020_features.tsv (variant, strain, n_mut, semantic,
grammar, preference). CLIB(t) sweep + correlation is done in the eval step.

Base ESM2-650M backbone (protein LM signal is the species-transferable part).
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

DATA = Path("/home/kist/workspace/data/influenza/fitness_wu2020")
OUT = Path("/home/kist/workspace/climb2/viral-analyzer/outputs/phase0/wu2020_features.tsv")
MODEL = "facebook/esm2_t33_650M_UR50D"
MUT_POS = [p - 1 + 16 for p in (156, 158, 159, 190, 193, 196)]  # 0-based
NAMES = ['HK68', 'Bk79', 'Bei89', 'Mos99', 'Bris07L194', 'NDako16']
WT6 = {'HK68': 'KGSESV', 'Bk79': 'EESENV', 'Bei89': 'EEYENV',
       'Mos99': 'QKYDST', 'Bris07L194': 'HKFDFA', 'NDako16': 'HNSDFA'}
AAS = "ACDEFGHIKLMNPQRSTVWY"


def load_wt_seqs():
    from Bio import SeqIO
    wt = {}
    for r in SeqIO.parse(DATA / "wildtypes.fa", "fasta"):
        wt[r.description] = str(r.seq)
    for s in NAMES:
        for aa, pos in zip(WT6[s], MUT_POS):
            assert wt[s][pos] == aa, (s, pos, wt[s][pos], aa)
    return wt


@torch.no_grad()
def embed_batch(seqs, tok, model, device):
    enc = tok(seqs, return_tensors="pt", padding=True).to(device)
    h = model.esm(enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    # mean over residues excluding cls/eos/pad
    m = (h * mask).sum(1)
    lengths = mask.sum(1)
    mean = (m / lengths)  # includes cls/eos but consistent; refine below
    # exclude cls (idx0) and eos: recompute precisely
    out = []
    for i, s in enumerate(seqs):
        L = int(enc["attention_mask"][i].sum().item())
        out.append(h[i, 1:L - 1].mean(0))
    return torch.stack(out)


@torch.no_grad()
def wt_grammar_table(wt_seq, tok, model, device, max_len=1024):
    """6 x 20 table: log P(aa | WT, position masked) for each site-B position."""
    tab = {}
    ids = tok(wt_seq, return_tensors="pt").input_ids.to(device)
    for pos in MUT_POS:
        tid = pos + 1  # +cls
        masked = ids.clone(); masked[0, tid] = tok.mask_token_id
        logits = model(masked).logits[0, tid]
        logp = torch.log_softmax(logits, -1)
        tab[pos] = {a: logp[tok.convert_tokens_to_ids(a)].item() for a in AAS}
    return tab


def main():
    df = pd.read_csv(DATA / "data_all.csv")
    df = df[df.strain.isin(NAMES)].copy()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForMaskedLM.from_pretrained(MODEL).to(device).eval()
    wt_seqs = load_wt_seqs()

    # per-strain WT embedding + grammar table
    wt_emb, gram_tab = {}, {}
    for s in NAMES:
        wt_emb[s] = embed_batch([wt_seqs[s]], tok, model, device)[0]
        gram_tab[s] = wt_grammar_table(wt_seqs[s], tok, model, device)

    rows = []
    for s in NAMES:
        sub = df[df.strain == s]
        seqs, meta = [], []
        for _, r in sub.iterrows():
            code = r["ID"]
            seq = list(wt_seqs[s])
            nmut = 0; gram = 0.0
            for aa, pos in zip(code, MUT_POS):
                if seq[pos] != aa:
                    nmut += 1
                seq[pos] = aa
                gram += gram_tab[s][pos][aa]
            seqs.append("".join(seq))
            meta.append((code, nmut, gram, r.get("Preference", np.nan)))
        # batched embedding
        emb = []
        for i in range(0, len(seqs), 32):
            emb.append(embed_batch(seqs[i:i + 32], tok, model, device))
        emb = torch.cat(emb)
        for (code, nmut, gram, pref), e in zip(meta, emb):
            sem = torch.norm(e - wt_emb[s], p=1).item()
            rows.append(dict(strain=s, variant=code, n_mut=nmut,
                             semantic=round(sem, 4), grammar=round(gram, 4),
                             preference=pref))
        print(f"{s}: {len(seqs)} variants embedded")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
