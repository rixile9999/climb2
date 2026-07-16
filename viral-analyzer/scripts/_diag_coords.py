import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "modules"))
import pandas as pd, analyzer as an

STRAIN = "SARS-CoV-2-WildType"
rd = an.RESULT_DATA["SARS-CoV-2-WildType-ESM2cov-L1"]
df = an.prepare_df(an.rename_df(pd.read_csv(rd["path"], sep="\t"), columns=rd["columns"]), STRAIN)
code = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
code_p1 = df["original_aa"] + (df["pos"] + 1).astype(str) + df["mutated_aa"]
S0, Sp1 = set(code), set(code_p1)
seqmap = {int(r.pos): r.original_aa for r in df[["pos", "original_aa"]].drop_duplicates().itertuples()}

print("=== ground truth: WT residue at bio position under each framing (Omicron: K417,E484,F486,N501) ===")
for bio in [417, 484, 486, 501]:
    print(f"  bio{bio}: TSVpos{bio-1}={seqmap.get(bio-1)}   TSVpos{bio}={seqmap.get(bio)}")

print("\n=== framing used by each WildType escape list ===")
for task in sorted(k for k in an.ESCAPE_DATA if k.startswith(STRAIN + ">")):
    muts = set(an.ESCAPE_DATA[task]["mutations"]); nm = len(muts)
    print(f"  {task.split('>')[1]:9s} n={nm:2d}  match_0based={len(muts&S0):2d}  match_1based={len(muts&Sp1):2d}  "
          f"unmatched_either={sorted(muts - S0 - Sp1)}")

print("\n=== Actual: raw(1-based?) -> minus1 -> is it in 0-based pipeline set? ===")
for m in an.ESCAPE_DATA[STRAIN + ">Actual"]["mutations"]:
    g = re.match(r"([A-Za-z])(\d+)([A-Za-z*])", m)
    if not g:
        print(f"  {m}: UNPARSEABLE"); continue
    s = f"{g[1]}{int(g[2])-1}{g[3]}"
    print(f"  {m:8s} -> {s:8s} in0={s in S0}")
