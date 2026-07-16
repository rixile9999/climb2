"""
Phase 0 — tidy-data dump + count assertion for the CLIB-offset go/no-go.

Reuses the exact viral-analyzer machinery (an.prepare_df / an.get_Q / an.get_log_clib_z)
so the dumped features are byte-identical to what `main.py --scores` would compute.

Backbone: ESM2_coronaviridae (ESM2cov), L1 semantic  ->  S1 = semantic_z, S2 = log_grammar_z.
S3 = log_clib_z (the CTMC nucleotide-accessibility prior), one column per evolutionary time t.

Run:
    conda run -n vanalyzer python scripts/phase0_dump.py
(the script chdir's to the viral-analyzer root itself, so CWD does not matter.)

Outputs:
    outputs/phase0/tidy_esm2cov_l1.parquet   # one row per single mutant (24,188)
    prints the count-assertion table (matched vs metadata) for every WildType>* escape task.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)                       # analyzer.py resolves data/ metadata/ relative to CWD
sys.path.insert(0, os.path.join(ROOT, "modules"))

import numpy as np
import pandas as pd
import analyzer as an

# ---------------------------------------------------------------- config
BACKBONE = "SARS-CoV-2-WildType-ESM2cov-L1"   # semantic = change_l1, grammar = prob
STRAIN = "SARS-CoV-2-WildType"
T_LIST = [0.01, 0.033, 0.1, 0.33, 1.0, 3.3, 10.0]
OUT_DIR = os.path.join(ROOT, "outputs", "phase0")
OUT_PARQUET = os.path.join(OUT_DIR, "tidy_esm2cov_l1.parquet")

# every escape task defined on the WildType strain
ESCAPE_TASKS = [k for k in an.ESCAPE_DATA if k.startswith(f"{STRAIN}>")]


def build_base_df():
    result_data = an.RESULT_DATA[BACKBONE]
    assert result_data["strain"] == STRAIN, result_data["strain"]
    raw = pd.read_csv(result_data["path"], sep="\t", header=0)
    df = an.rename_df(raw, columns=result_data.get("columns"))
    df = an.prepare_df(df, STRAIN)           # -> semantic_z, log_grammar_z, mutation_id, codon
    return df


def count_assertion(tidy: pd.DataFrame) -> pd.DataFrame:
    """For every WildType escape task, compare matched positives vs metadata count,
    and diagnose off-by-one by also testing pos-1 / pos+1 framing."""
    codes_0 = tidy["mutation_code"]                                        # wt + pos       + mut  (pipeline)
    codes_m1 = tidy["original_aa"] + (tidy["pos"] - 1).astype(str) + tidy["mutated_aa"]
    codes_p1 = tidy["original_aa"] + (tidy["pos"] + 1).astype(str) + tidy["mutated_aa"]
    rows = []
    for task in ESCAPE_TASKS:
        muts = set(an.ESCAPE_DATA[task]["mutations"])
        n_meta = len(muts)
        rows.append({
            "task": task.split(">")[1],
            "n_meta": n_meta,
            "match_pos":     int(codes_0.isin(muts).sum()),
            "match_pos_m1":  int(codes_m1.isin(muts).sum()),
            "match_pos_p1":  int(codes_p1.isin(muts).sum()),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_base_df()
    Q = an.get_Q(STRAIN)

    tidy = pd.DataFrame({
        "pos": df["pos"].to_numpy(),
        "original_aa": df["original_aa"].to_numpy(),
        "mutated_aa": df["mutated_aa"].to_numpy(),
        "mutation_id": df["mutation_id"].to_numpy(),      # codon->aa  (for CLIB)
        "S1_semantic_z": df["semantic_z"].to_numpy(),
        "S2_log_grammar_z": df["log_grammar_z"].to_numpy(),
    })
    tidy["mutation_code"] = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]

    # S3 = log_clib_z per evolutionary time t
    for t in T_LIST:
        tidy[f"S3_logclib_z_t{t}"] = an.get_log_clib_z(df, Q, t)

    # Hie rank_cscs (reference baseline), identical to main.py L174-181
    tidy["rank_cscs"] = (
        (df["semantic"].rank(ascending=False, method="min").astype(int)
         + df["grammar"].rank(ascending=False, method="min").astype(int))
        .rank(ascending=True, method="min").astype(int).to_numpy()
    )

    # ---- count assertion ----
    ca = count_assertion(tidy)
    pd.set_option("display.width", 160)
    print("\n================ COUNT ASSERTION (matched vs metadata) ================")
    print(ca.to_string(index=False))
    ok = bool((ca["match_pos"] == ca["n_meta"]).all())
    print(f"\npos-framing (as pipeline builds it) matches metadata for ALL tasks: {ok}")
    if not ok:
        best = {
            "pos": int((ca["match_pos"] == ca["n_meta"]).sum()),
            "pos-1": int((ca["match_pos_m1"] == ca["n_meta"]).sum()),
            "pos+1": int((ca["match_pos_p1"] == ca["n_meta"]).sum()),
        }
        print("tasks fully matched under each framing:", best)

    # attach the pipeline is_escape for a couple of headline tasks (Combined / Omicron / DMS)
    for task in ESCAPE_TASKS:
        muts = set(an.ESCAPE_DATA[task]["mutations"])
        tidy[f"esc__{task.split('>')[1]}"] = tidy["mutation_code"].isin(muts)

    tidy.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET}  ({len(tidy)} rows, {tidy.shape[1]} cols)")
    print("feature sanity:",
          f"S1 mean/std={tidy.S1_semantic_z.mean():.3f}/{tidy.S1_semantic_z.std():.3f}",
          f"| S2 mean/std={tidy.S2_log_grammar_z.mean():.3f}/{tidy.S2_log_grammar_z.std():.3f}")


if __name__ == "__main__":
    main()
