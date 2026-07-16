#!/usr/bin/env python3
"""Fetch observed spike amino-acid substitution frequencies from CoV-Spectrum
(LAPIS open API) and write tidy TSVs used by the label-free (Phase B) analyses.

All-time table  -> data/observed/spike_obs_freq.tsv
Time windows    -> data/observed/windows/spike_<name>.tsv   (0-based positions)

Positions are converted to 0-based to match the repository convention.
Re-run to refresh; the committed TSVs are a snapshot (dataVersion printed).
"""
import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://lapis.cov-spectrum.org/open/v2/sample/aminoAcidMutations?minProportion=0.00001"
AAS = set("ACDEFGHIKLMNPQRSTVWY")

QUERIES = {
    "spike_obs_freq": "",                                          # all time -> data/observed/
    "windows/spike_le_2021-06": "&dateTo=2021-06-30",
    "windows/spike_le_2021-12": "&dateTo=2021-12-31",
    "windows/spike_le_2022-06": "&dateTo=2022-06-30",
    "windows/spike_2021-09_2022-03": "&dateFrom=2021-09-01&dateTo=2022-03-31",
    "windows/spike_ge_2022-06": "&dateFrom=2022-06-01",
    "windows/spike_ge_2023-01": "&dateFrom=2023-01-01",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def main():
    (ROOT / "data/observed/windows").mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        payload = fetch(BASE + q)
        rows = sorted(
            (x["position"] - 1, x["mutationFrom"], x["mutationTo"], x["proportion"], x["count"])
            for x in payload["data"]
            if x.get("sequenceName") == "S" and x.get("mutationFrom") in AAS and x.get("mutationTo") in AAS
        )
        out = ROOT / "data/observed" / f"{name}.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["pos", "wt", "mut", "proportion", "count"])
            w.writerows(rows)
        print(f"{name}: {len(rows)} spike substitutions  (dataVersion={payload['info']['dataVersion']})")


if __name__ == "__main__":
    main()
