#!/usr/bin/env python3
"""Generate tidy per-mutation feature dumps for multiple backbones, reusing the
exact viral-analyzer machinery (an.prepare_df / an.get_Q / an.get_log_clib_z), so
features are identical to what main.py --scores would compute. Extends phase0_dump
(which built only ESM2cov) to base ESM2 and Hie for the multi-backbone study.

  outputs/phase0/tidy_<tag>.parquet   (one row per single mutant)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "modules"))

import pandas as pd
import analyzer as an

STRAIN = "SARS-CoV-2-WildType"
T_LIST = [0.01, 0.033, 0.1, 0.33, 1.0, 3.3, 10.0]
ESCAPE_TASKS = [k for k in an.ESCAPE_DATA if k.startswith(f"{STRAIN}>")]

BACKBONES = {
    "esm2_150m_l1": "SARS-CoV-2-WildType-ESM-150M-L1",
    "esm2_650m_l1": "SARS-CoV-2-WildType-ESM-650M-L1",
    "esm2_3b_l1": "SARS-CoV-2-WildType-ESM-3B-L1",
    "hie": "SARS-CoV-2-WildType-Hie",
}


def build_tidy(backbone_key):
    rd = an.RESULT_DATA[backbone_key]
    assert rd["strain"] == STRAIN
    raw = pd.read_csv(rd["path"], sep="\t", header=0)
    df = an.rename_df(raw, columns=rd.get("columns"))
    df = an.prepare_df(df, STRAIN)
    Q = an.get_Q(STRAIN)
    tidy = pd.DataFrame({
        "pos": df["pos"].to_numpy(),
        "original_aa": df["original_aa"].to_numpy(),
        "mutated_aa": df["mutated_aa"].to_numpy(),
        "S1_semantic_z": df["semantic_z"].to_numpy(),
        "S2_log_grammar_z": df["log_grammar_z"].to_numpy(),
    })
    tidy["mutation_code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
    for t in T_LIST:
        tidy[f"S3_logclib_z_t{t}"] = an.get_log_clib_z(df, Q, t)
    for task in ESCAPE_TASKS:
        muts = set(an.ESCAPE_DATA[task]["mutations"])
        tidy[f"esc__{task.split('>')[1]}"] = tidy["mutation_code"].isin(muts)
    return tidy


def main():
    os.makedirs("outputs/phase0", exist_ok=True)
    for tag, key in BACKBONES.items():
        tidy = build_tidy(key)
        out = f"outputs/phase0/tidy_{tag}.parquet"
        tidy.to_parquet(out, index=False)
        print(f"wrote {out}  ({len(tidy)} rows)  "
              f"S1std={tidy.S1_semantic_z.std():.3f} S2std={tidy.S2_log_grammar_z.std():.3f}")


if __name__ == "__main__":
    main()
