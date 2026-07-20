# Choosing the Model's Weights Without an Answer Key: Label-Free CLIMB

*Audience: a student new to this project. Companion to `research_report_clib_offset_en.md`.*
*Date: 2026-07-16 · Status: Phase B0–B6 complete, label-free optimization endpoint reached.*

---

## 0. Three-sentence summary

The escape-scoring model combines three signals with the formula `score = b0·S1 + b1·S2 + κ·CLIB`, where the CLIB term is frozen (κ=1) and only the two protein-signal weights b0, b1 are learned — normally from the escape answer key. We asked whether b0, b1 can instead be chosen **without any escape labels**, using only escape-independent statistics, and whether the model can be **evaluated without the DMS answer key**. The answer is a qualified **yes**: fitting b0, b1 to phylogenetic fitness (Bloom) recovers the supervised weights and matches supervised escape ranking on the strong ESM2cov backbone, and honest DMS-free tests (temporal forecasting, leave-one-variant-out, an independent phylogenetic cross-check) all confirm the signal is real — though the recipe is backbone-dependent and adding further complexity yields no gain.

---

## 1. The two questions

The [offset research](research_report_clib_offset_en.md) established `score = b0·S1 + b1·S2 + 1·CLIB(t=0.01)`:
S1 = semantic change, S2 = grammaticality (viability), CLIB = a nucleotide-accessibility prior frozen as an **offset** (a term whose coefficient is fixed, not learned). Only **b0, b1** are learned, normally by logistic regression against escape labels.

This report asks:
- **(B-fit)** Can we set b0, b1 with **no escape labels** — from statistics alone?
- **(B-eval)** Can we judge the model with **no DMS dataset**?

Both matter because escape labels are scarce and one specific in-vitro DMS panel is a narrow proxy for the real goal (anticipating dangerous variants).

---

## 2. A necessary caveat about "unsupervised"

Optimization needs an objective. With **no** target signal at all, no weighting can be called better than another (beyond the neutral z-score equal-weight point) — so pure label-free *optimization* is ill-posed. What is well-posed is fitting b0, b1 against an **escape-independent statistical target**, then evaluating on escape. That is the whole design.

---

## 3. B — Frequency proxy (Phase B labelfree)

**Target.** For each of ~24,000 single spike mutations, its observed frequency in the global population (CoV-Spectrum / LAPIS open data, ~9.3M sequences). Binarize (`observed proportion ≥ θ`) → a proxy positive set. Fit `logistic(y_freq ~ b0·S1 + b1·S2, offset = 1·CLIB)`. **No escape label is used.**

**Result.** Stable weights across thresholds: **b0 ≈ 0.19 (semantic), b1 ≈ 1.0–2.1 (grammar)** — frequency favors viability + accessibility, barely using semantic change. Evaluated on escape, the variant-lineage tasks improve enormously — but with a **100% overlap** between the frequency-positive set and those escape answers. The variant-defining escape mutations *are* the high-frequency ones, so the frequency proxy silently contains the answer. On **DMS**, which is measured functionally and barely overlaps frequency (8/19), the label-free gain over equal-weight is **not significant**.

**Lesson.** Frequency helps, but on the leaky tasks the help is partly circular. Two fixes follow: de-leak it (§5), and evaluate on genuinely independent ground truth (§4).

---

## 4. B-eval — Judging without DMS (Phase B2)

Two DMS-independent ground truths:

**(A) Prospective emergence (temporal).** Split frequency into a pre-cutoff *early* window and a post-cutoff *late* window; a **forward-positive** is a mutation that was rare early and common late. The model must rank these using only pre-cutoff information — the future is genuinely held out, so no leakage is possible. `proxy_early` (b0, b1 fit on the early window only) forecasts post-2022 risers at **AUROC 0.899**, **significantly above equal-weight** (0.860; bootstrap CI on the gap [0.017, 0.061] excludes 0) and **on par with** the DMS-supervised weights (0.894 — the 0.005 gap at n=15 positives is within noise, *not* a real lead, and no CI is computed for it).

**(B) Phylogenetic fitness (convergent evolution).** Bloom et al. estimate a per-mutation `delta_fitness` from observed-vs-expected independent occurrences across ~7M sequences on the UShER tree — a selection signal independent of both frequency and DMS. The model ranks the top high-fitness mutations at **AUROC ~0.85**, though the continuous rank correlation is modest (Spearman 0.08–0.15): escape is not identical to general fitness.

**Lesson.** Temporal forecasting is the strongest, most honest DMS-free benchmark, and there the label-free model matches/beats DMS-supervised.

---

## 5. De-leaking and higher power (Phase B3)

**(2) Leave-one-variant-out (LOVO).** For each variant task V, remove V's own escape mutations from the frequency-proxy **fit**, then evaluate on V. The de-leaking is verified complete (every row with `esc__V==1` is dropped from the fit). The improvement over equal-weight **survives on all five variants** (ΔAUROC +0.033…+0.060), and `proxy_LOVO ≈ proxy_full` (e.g. Omicron 0.982 vs 0.983). The gain was **not** mere circularity — the weights generalize to a variant whose answers they never saw. *Caveat:* the CI excludes 0 for the four variants with n≥7; **Gamma has n=1**, so its "CI" is a degenerate zero-width point (no real inference), and the bootstrap resamples positives only (so true CIs are somewhat wider).

**(3) Multi-cutoff temporal.** Three cutoffs (2021-06, 2021-12, 2022-06). `proxy_early` beats equal-weight significantly on 2 of 3 (C1 +0.036 [0.025,0.046], C2 +0.038 [0.017,0.061]); the most recent (C3) is positive but not significant (+0.018). Weights are cutoff-stable (b0≈0.18, b1≈1.5–1.8).

---

## 6. Cross-signal universality (Phase B4)

Fit b0, b1 against three independent statistical signals — frequency, Bloom fitness, early-frequency. All **converge to grammar ≫ semantic** (b0 0.19–0.29, b1 0.94–1.68) and all beat equal-weight on variant and emergent escape.

The headline: **Bloom-fit weights (0.285, 0.942) nearly recover the DMS-supervised weights (0.268, 0.955)** and give the best leakage-free **DMS AUROC (0.899 vs equal 0.884)**. Phylogenetic fitness is the best label-free teacher — you reach supervised-quality escape weights with **zero escape labels**.

> **Disclosure (verified by independent adversarial review).** The `supervised_DMS` weights in the cross-signal and multi-backbone scripts are fit *in-sample* (on all DMS, evaluated on the same DMS), so their DMS AUROC is optimistic — which only makes bloom-fit's edge *more* notable, not less. Bloom-fit's DMS AUROC is honest: the Bloom high-fitness set and the DMS positive set are **completely disjoint** (overlap 0; continuous Spearman 0.049), so there is no label contamination. **However**, the DMS test has only **19 positives** out of 24,187 (0.08%), so its AUROC is high-variance and the small margins (bloom 0.899 vs equal 0.884; bloom vs in-sample supervised 0.898) are **within small-sample noise**. The robust evidence is therefore the **weight recovery** (0.285/0.942 ≈ 0.268/0.955) and the **larger-n emergent/variant results**, not the 19-positive DMS margin. Correct reading: "label-free Bloom-fit **matches** supervised," never "beats."

---

## 7. It is not universal (Phase B5)

Repeating the label-free recipe across backbones:

| backbone | equal DMS | CLIB-only DMS | bloom-fit DMS | supervised(in-sample) | verdict |
|---|---|---|---|---|---|
| **ESM2cov** (strong, CoVFit's backbone) | 0.884 | 0.883 | **0.899** | 0.898 | label-free recovers supervised ✓ |
| base ESM2 (weak) | 0.786 | 0.883 | 0.863 | 0.880 | features are noise; reduces to CLIB ✓ |
| Hie | 0.899 | 0.884 | 0.861 | 0.907 | freq/bloom-fit give **negative grammar**, underperform ✗ |

**Correction (important).** Hie is **not a general protein language model** — it is Hie et al.'s per-virus BiLSTM, trained separately for each virus. So its failure is **not** a valid counterexample to universality across general PLMs and should be excluded from the backbone-universality question. **Restricted to general PLMs (the ESM2 family), there is no counterexample:** base ESM2 harmlessly reduces to CLIB when its protein features are noise, and ESM2cov recovers the supervised weights. Universality *within general PLMs* therefore remains open and plausible — tested directly in §11 (general-PLM × species matrix). Practical recommendation still: **ESM2cov — exactly CoVFit's backbone**.

---

## 8. Endpoint (Phase B6)

An ensemble target (rank-sum of frequency + Bloom + early-frequency) does **not** beat the best single signal (ensemble DMS 0.887 < bloom-fit 0.899), and the recent-era cutoff (C3) saturates at ~0.83 for every method — an intrinsic data/signal limit, not fixable by reweighting. As in the offset research, added complexity yields no honest improvement.

**Recommended label-free model:** `score = b0·S1 + b1·S2 + 1·CLIB(t=0.01)` with **(b0, b1) fit to Bloom phylogenetic fitness on the ESM2cov backbone → (0.285, 0.942)**. Supervised-quality escape ranking with **no escape labels**, validated by LOVO (5/5), temporal forecasting (0.899), and cross-signal convergence.

## 9. Strong-backbone cross-strain transfer (Phase B7) — the flagship question, answered

The offset research (Phase 6) could only test cross-strain transfer with the weak base ESM2, where the protein features transferred at AUROC ~0.60 (near-random), leading to the conclusion that "CLIB carries essentially all the transferable signal." Because ESM2cov existed only for WildType, the strong-backbone case was the #1 open question. We generated ESM2cov single-mutant features for the variant strains and ran leave-one-strain-out transfer over five clean strains (WT, Alpha, Beta, Gamma, Omicron; Delta excluded — its reference spike has ambiguous 'X' residues that truncate the scan at position 94, exactly as in Phase 6):

| model | strong ESM2cov (this work) | weak base ESM2 (Phase 6) |
|---|---|---|
| protein-only | **0.842** | ~0.60 |
| protein + offset | **0.890** | ~0.86 |
| CLIB-only (no training) | 0.864 | ~0.864 |

**The strong domain-adapted backbone — which is exactly CoVFit's backbone — transfers its escape representation across strains** (protein-only 0.842 vs 0.60), and **protein+offset (0.890) beats CLIB-only (0.864)**, meaning it adds transferable signal *on top of* CLIB. This **refines Phase 6's conclusion**: "CLIB carries essentially all the transfer" held only for the weak backbone; the domain-adapted PLM has learned an escape representation that generalizes across strain backgrounds. CLIB-only reproducing 0.864 (backbone-independent) is a clean internal consistency check.

**Remaining open questions (data, not model):** per-lineage substitution signatures for a strain-specific CLIB; why the label-free advantage shrinks in the most recent era; regenerating a clean (X-free) Delta reference to complete the strain panel.

---

## 10. Independent adversarial verification

The four headline claims (LOVO, temporal, Bloom-recovers-supervised, not-universal) were each re-checked by an independent skeptical reviewer that re-ran the scripts and scrutinized the methodology. All four: **numbers reproduce exactly**; verdict **confirmed with caveats**. The material caveats — already folded into this report — are: (i) the temporal "beats supervised" is within noise (a tie), only the equal-weight win is CI-supported and it is cutoff-dependent (not significant at the 2022-06 cutoff); (ii) LOVO's Gamma has n=1 (degenerate CI), and the bootstrap is positives-only; (iii) the DMS test rests on 19 positives, so its margins are high-variance — the weight recovery and larger-n results carry the claim; (iv) the Hie failure is robust because it rests on the **negative fitted grammar weight**, independent of any supervised comparison.

---

## Appendix: reproduce

```bash
conda activate vanalyzer
cd viral-analyzer
python scripts/fetch_observed_freq.py       # LAPIS snapshot -> data/observed/*.tsv
python scripts/gen_backbone_dumps.py         # base ESM2 / Hie tidy dumps
python scripts/phaseB_labelfree.py           # frequency-proxy fit + escape eval (+leakage diagnosis)
python scripts/phaseB2_dmsfree.py            # temporal (A) + Bloom (B)
python scripts/phaseB3_dmsfree_plus.py       # LOVO de-leaking + multi-cutoff
python scripts/phaseB4_crosssignal.py        # cross-signal generalization
python scripts/phaseB5_multibackbone.py      # backbone robustness
python scripts/phaseB6_ensemble.py           # ensemble target (no gain)
```

**Data provenance:** CoV-Spectrum/LAPIS open data (observed frequencies); Bloom et al. SARS2-mut-fitness (phylogenetic fitness). Both are independent of the curated escape answer key.
