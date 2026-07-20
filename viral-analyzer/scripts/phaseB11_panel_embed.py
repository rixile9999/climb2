#!/usr/bin/env python3
"""Phase B11 (Step A) — embed the H3N2 antigenic-cluster strain panel.

Each strain is a natural MULTI-mutation variant. Mean-pooled ESM2-650M embedding
per strain -> outputs/phase0/h3_panel_emb.npz for the multi-mutation antigenic
escape analysis (does semantic change / multi-CAC predict antigenic-cluster
difference between diverged strains).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path("/home/kist/workspace/climb2/viral-analyzer")
MODEL = "facebook/esm2_t33_650M_UR50D"


@torch.no_grad()
def embed(seqs, tok, model, device):
    out = []
    for i in range(0, len(seqs), 16):
        batch = seqs[i:i + 16]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        h = model.esm(enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        for j, s in enumerate(batch):
            L = int(enc["attention_mask"][j].sum().item())
            out.append(h[j, 1:L - 1].mean(0).float().cpu().numpy())
    return np.stack(out)


def main():
    meta = pd.read_csv(ROOT / "data/species/h3_panel_meta.tsv", sep="\t")
    from Bio import SeqIO
    seqd = {r.id: str(r.seq) for r in SeqIO.parse(ROOT / "data/species/h3_panel.fasta", "fasta")}
    seqs = [seqd[i] for i in meta["id"]]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForMaskedLM.from_pretrained(MODEL).to(device).eval()
    emb = embed(seqs, tok, model, device)
    np.savez(ROOT / "outputs/phase0/h3_panel_emb.npz",
             ids=meta["id"].to_numpy(), emb=emb,
             cluster=meta["cluster"].to_numpy(), order=meta["cluster_order"].to_numpy(),
             year=meta["year"].to_numpy())
    print(f"embedded {len(seqs)} strains -> h3_panel_emb.npz")


if __name__ == "__main__":
    main()
