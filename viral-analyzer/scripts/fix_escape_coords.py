"""
Fix the escape-list coordinate bug found by the Phase-0 count assertion.

Findings (ESM2cov WildType candidate universe as ground truth):
  * The repo convention is 0-based indexing (TSV pos 0 == biological residue 1, i.e. pos = bio-1).
    DMS / Alpha / Beta / Gamma / Omicron were curated in this 0-based frame and match perfectly.
  * "SARS-CoV-2-WildType>Actual" was mistakenly curated in 1-based (literature) numbering,
    so it matched only 4/32.  Shifting every position by -1 maps 31/32 into the 0-based universe.
  * "Actual" D769H and all "*X" entries (E155X, F156X, R157X, T94X in Delta/Combined) are NOT
    single-residue substitutions present in the model's candidate set (X = deletion/wildcard;
    D769H = residue-identity mismatch vs the reference), so they cannot be ranked and are dropped.

This script edits metadata/escape_mutants/cov-wt.json in place (after writing a .bak backup),
then re-validates every remaining mutation against the candidate universe.

Run:  conda run -n vanalyzer python scripts/fix_escape_coords.py
"""

import os
import re
import sys
import json
import shutil

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))
import pandas as pd
import analyzer as an

ESC_FILE = os.path.join(ROOT, "metadata", "escape_mutants", "cov-wt.json")
STRAIN = "SARS-CoV-2-WildType"
STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z*])$")


def candidate_set():
    rd = an.RESULT_DATA["SARS-CoV-2-WildType-ESM2cov-L1"]
    df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd["columns"]), STRAIN)
    return set(df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"])


def main():
    S0 = candidate_set()
    with open(ESC_FILE, encoding="utf-8") as f:
        data = json.load(f)

    shutil.copyfile(ESC_FILE, ESC_FILE + ".bak")
    print(f"backup -> {ESC_FILE}.bak\n")

    report = []
    for key, entry in data.items():
        if not key.startswith(STRAIN + ">"):
            continue
        task = key.split(">")[1]
        muts = list(entry.get("mutations", []))
        is_actual = (task == "Actual")

        kept, dropped_nonstd, shifted, unmatched = [], [], 0, []
        for m in muts:
            g = MUT_RE.match(m)
            if not g:
                dropped_nonstd.append(m); continue
            wt, pos, mut = g[1], int(g[2]), g[3]
            if mut not in STD_AA:                       # X / deletion / wildcard -> unscoreable
                dropped_nonstd.append(m); continue
            if is_actual:                               # 1-based -> repo 0-based
                pos -= 1; shifted += 1
            code = f"{wt}{pos}{mut}"
            if code in S0:
                kept.append(code)
            else:
                unmatched.append((m, code))             # e.g. Actual D769H -> D768H (residue mismatch)

        # drop still-unmatched (cannot be ranked); log them
        kept_unique = sorted(set(kept))
        entry["mutations"] = kept_unique
        report.append({
            "task": task, "n_before": len(muts), "n_after": len(kept_unique),
            "shifted(-1)": shifted if is_actual else 0,
            "dropped_nonstandard": dropped_nonstd,
            "dropped_unmatched": [f"{a}->{b}" for a, b in unmatched],
        })

    with open(ESC_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("================ FIX REPORT ================")
    for r in report:
        note = []
        if r["shifted(-1)"]:
            note.append(f"shifted {r['shifted(-1)']} positions -1 (1-based->0-based)")
        if r["dropped_nonstandard"]:
            note.append(f"dropped non-standard {r['dropped_nonstandard']}")
        if r["dropped_unmatched"]:
            note.append(f"dropped unmatched {r['dropped_unmatched']}")
        tag = "  |  ".join(note) if note else "unchanged"
        print(f"  {r['task']:9s} {r['n_before']:2d} -> {r['n_after']:2d}   {tag}")
    print(f"\nwrote {ESC_FILE}")


if __name__ == "__main__":
    main()
