# CLIMB: A Mechanistic, Label-Free, and Multi-Mutation Extension of Protein-Language-Model Viral-Escape Prediction

*Working paper. Branch `research/climb-labelfree`. All results reproducible from `viral-analyzer/scripts/`.*

---

## Abstract

Protein language models (PLMs) rank viral mutations for immune escape by combining a **semantic** signal (embedding change) and a **grammaticality** signal (mutant plausibility), following the CSCS framework of Hie et al. (2021). We develop **CLIMB**, an extension that adds a third, mechanistic term — **CLIB**, a nucleotide-accessibility prior derived from a continuous-time Markov codon-substitution model — and study three questions: (i) how CLIB should be incorporated, (ii) whether the model's weights can be chosen and validated **without an escape answer key**, and (iii) whether the single-mutation ("point") formulation transfers across strains, models, and species. We find that (1) freezing CLIB as a **unit-coefficient offset** (κ=1, evolutionary time t=0.01) removes a circular in-sample fit, matches the honest supervised model, and is the signal that carries cross-strain generalization; (2) the two learned protein weights can be set **label-free** by regressing on phylogenetic fitness, recovering the supervised weights and matching them on held-out escape, validated by leave-one-variant-out de-leaking and prospective temporal forecasting; (3) escape signal on SARS-CoV-2 comes from **domain adaptation, not model size**, and CLIB's power is **SARS-CoV-2-specific** — it is near-random on Influenza HA and HIV Env, even with real codons. The last result motivates a **point → multi** generalization in which each term becomes a sum over a mutation set and the evolutionary time t becomes a free parameter; on the Wu2020 combinatorial landscape the multi-CLIB–fitness correlation rises with t (+0.21→+0.33), and multi-mutation semantic change predicts real **antigenic-cluster escape** in H3N2 at AUROC 0.77 where single-mutation DMS escape was near-chance. All headline claims were re-checked by independent adversarial verification.

---

## 1. Introduction

Emerging viral variants evade host immunity through **escape** mutations. Ranking the ~24,000 single-residue mutations of a viral protein so that escape mutations sit near the top is a standard computational task, classically approached with the **CSCS** score (Constrained Semantic Change Search; Hie et al., 2021):

- **S1 — semantic change:** the distance in a PLM's embedding space between wild-type and mutant, capturing antigenic novelty.
- **S2 — grammaticality:** the PLM's log-probability of the mutant residue, capturing viability.

CSCS ranks by `rank(S1) + rank(S2)`. Building on the CoVFit line of work (a domain-adapted PLM for SARS-CoV-2 fitness), we add a **mechanistic** third signal and ask whether the resulting model can be made honest, label-free, and cross-domain. The model is:

```
score = b0·z(S1) + b1·z(S2) + κ·z(CLIB) ,   κ fixed = 1
```

where z(·) is standardization and CLIB is the nucleotide-accessibility prior (§2.2). This paper's contributions:

1. **The CLIB offset (§4.1).** Freezing CLIB as a unit-coefficient offset is optimal, removes a ~30% in-sample optimism, and carries cross-strain transfer.
2. **Label-free coefficient selection and DMS-free validation (§4.2).** b0, b1 can be set from escape-independent statistics; the model can be judged without the DMS answer key.
3. **Cross-strain, cross-model, cross-species scope (§4.3–4.5).** Strong-backbone transfer works; CLIB is SARS-specific.
4. **Point → multi extension (§4.6).** A formal generalization to mutation sets, motivated by the failure on old species.

---

## 2. The CLIMB model

### 2.1 Signals

For a single-mutant `x = x_wt ⊕ (i: a_wt→a)`:

- **S1** = ‖e(x) − e(x_wt)‖₁, e = mean-pooled ESM2 last-layer embedding (L1 metric).
- **S2** = log P(a | x_wt, mask i), the masked-marginal log-probability.

### 2.2 CLIB — a codon-accessibility prior (the offset)

An amino-acid substitution is easier to reach when it requires fewer nucleotide changes from the wild-type codon. CLIB quantifies this with a continuous-time Markov chain (CTMC):

- A 4×4 nucleotide rate matrix **Q** is estimated from a virus-specific single-base-substitution (SBS) spectrum and stationary distribution.
- Transition probabilities over evolutionary time t: **P(t) = exp(t·Q)**.
- For a mutation, `CLIB = log P(codon_i → a, t)`, standardized. Because Q comes from an **independent, much larger** dataset than the escape labels, CLIB is a *mechanistic prior*, not label-derived.

### 2.3 The offset formulation

CLIB enters as a statistical **offset** — a covariate with coefficient fixed at κ=1, not learned — so only b0, b1 are estimated:

```
score = b0·S1 + b1·S2 + 1·CLIB(t=0.01)
```

At t=0.01 the prior essentially asks "is the target amino acid reachable by a single nucleotide change?".

---

## 3. Data, models, and validation

### 3.1 Data

| dataset | role | source |
|---|---|---|
| SARS-CoV-2 spike single mutants (24,187) | candidate set | reference S protein |
| DMS antibody-escape (Greaney/Starr) | escape ground truth (leakage-free) | jbloomlab / viral-analyzer |
| Variant-defining escape (Alpha…Omicron) | escape ground truth (lineage) | viral-analyzer |
| Observed substitution frequencies (~9.3M seq) | label-free proxy; temporal windows | CoV-Spectrum / LAPIS open |
| Phylogenetic fitness (delta_fitness, ~7M seq) | label-free proxy; cross-check | Bloom SARS2-mut-fitness |
| Influenza H1 HA escape (170 muts) | cross-species escape | Doud et al. 2018 |
| HIV Env escape (161 muts) | cross-species escape | Dingens et al. 2019 |
| Wu2020 HA site-B combinatorial fitness (3,456) | multi-mutation testbed | Wu et al. 2020 |
| H3N2 antigenic-cluster panel (555 strains) | multi-mutation escape | Smith 2004 clusters / GISAID-derived |
| SBS spectra (SARS-CoV-2, Influenza A, HIV) | CLIB rate matrix Q | viral-analyzer `sbs_freq.csv` |
| CDS: SARS codon tables; WSN HA `J02176`; BG505 Env `DQ208458` | CLIB codons | GenBank |

### 3.2 Models

| model | params | notes |
|---|---|---|
| ESM2-150M / 650M / 3B | 150M–3B | general PLMs (Meta) |
| **ESM2cov** (ESM2_coronaviridae) | 650M | domain-adapted on Coronaviridae; **CoVFit's backbone** (Zenodo 10910360) |
| CoVFit | 650M + LoRA | multitask fitness+DMS head (Zenodo 14438178) |
| Hie BiLSTM | — | **per-virus** LSTM (not a general PLM) — a baseline, not a backbone-universality datapoint |
| CLIB | — | CTMC mechanistic prior (backbone-independent) |

### 3.3 Validation methods

- **Nested cross-validation** (5 seeds × 5-fold; inner loop selects t) for all fitted numbers — no data used to both fit and score.
- **Leave-one-variant-out (LOVO)** de-leaking: remove a task's own positives from the fit before evaluating it.
- **Prospective temporal forecasting:** fit on pre-cutoff frequency, predict post-cutoff emergence — the future is genuinely held out.
- **Bootstrap** confidence intervals over the positive set; **leakage/overlap** accounting.
- **Cross-signal, multi-backbone, cross-species** replication.
- **Real-CDS confound control** (§4.5).
- **Independent adversarial verification:** each headline claim re-run and scrutinized by a separate skeptical agent.

---

## 4. Results

### 4.1 The CLIB offset (Phases 0–7)

A data-hygiene pass fixed a coordinate bug (a 1-based escape list silently scoring the wrong mutations). Honest nested CV then showed:

- **The offset helps where the backbone is weak.** Base ESM2 on DMS jumps **AUROC 0.53 → 0.85** once CLIB is added; ESM2cov 0.82 → 0.88.
- **κ=1 is optimal.** Learning CLIB's coefficient gives no significant gain; freezing it removes a **~30% in-sample optimism** (legacy 2,270 vs honest ~3,000 mean-rank) while cutting a parameter.
- **t can be dropped** (fixed t=0.01, no loss). **Four complexity axes** — discretization, multi-backbone stacking, ranking loss, richer semantics — all fail to beat the two-parameter model.
- **CLIB carries cross-strain transfer.** With a weak backbone, CLIB-only (no training) reaches leave-one-strain-out AUROC 0.864.

### 4.2 Label-free coefficients and DMS-free validation (Phase B)

**Fitting without escape labels.** Regressing b0, b1 on escape-independent statistics (CLIB frozen) gives stable, biologically sensible weights (grammar ≫ semantic). Fitting on **Bloom phylogenetic fitness** yields **(b0, b1) = (0.285, 0.942)**, nearly the DMS-supervised weights **(0.268, 0.955)**, and matches supervised leakage-free DMS AUROC (≈0.90). The three independent proxies (frequency, Bloom, temporal) converge on the same operating point.

**Validating without DMS.**
- *Prospective emergence:* proxy_early (fit only on ≤2021 frequency) forecasts post-2022 risers at **AUROC 0.899**, significantly above equal-weight (0.860; bootstrap CI excludes 0) and *on par* with DMS-supervised (0.894 — a within-noise tie at n=15).
- *De-leaking (LOVO):* after removing each variant's own mutations from the fit, the improvement survives on **5/5 variants** (ΔAUROC +0.033…+0.060), refuting circularity. (Caveat: one variant has n=1.)
- *Phylogenetic cross-check:* the score identifies top Bloom-fitness mutations at AUROC ~0.85.

**Frequency-proxy caveat.** On variant-lineage tasks the frequency proxy is circular (its positives *are* the escape answers); the honest tests are DMS and temporal.

### 4.3 Strong-backbone cross-strain transfer (Phase B7)

Generating ESM2cov single-mutant features for the variant strains (previously unavailable) enabled the first strong-backbone leave-one-strain-out test:

| model | strong ESM2cov | weak base ESM2 (Phase 6) |
|---|---|---|
| protein-only | **0.842** | ~0.60 |
| protein + offset | **0.890** | ~0.86 |
| CLIB-only | 0.864 | ~0.864 |

The domain-adapted backbone's escape representation **transfers across strains** and adds signal on top of CLIB — refining the earlier weak-backbone conclusion that "CLIB carries all transfer".

### 4.4 General-PLM universality (Phase B8)

Across ESM2-150M/650M/3B/ESM2cov: **CLIB-only = 0.883 for every backbone** (a universal baseline lifting each PLM from protein-only ~0.50 to ~0.88). Crucially, protein-only DMS AUROC is ~0.50 for all base sizes but **0.82 for ESM2cov** — escape signal comes from **domain adaptation, not scale** (3B is no better than 150M). Label-free ≈ supervised holds for 150M and ESM2cov; for 650M/3B the protein features are noise and CLIB-only is already best. (Hie, being a per-virus LSTM rather than a general PLM, is excluded from this universality question.)

### 4.5 Cross-species: CLIB is SARS-specific (Phase B9)

Porting Influenza-HA (Doud 2018) and HIV-Env (Dingens 2019) escape and evaluating with base ESM2:

| species | protein-only | protein+offset | CLIB-only |
|---|---|---|---|
| SARS-CoV-2 | 0.50 | 0.88 | **0.883** |
| Influenza HA | 0.639 | 0.513 | **0.500** |
| HIV Env | 0.603 | 0.618 | **0.581** |

CLIB is **near-random** on flu/HIV and adding it *hurts*. We removed the obvious confound — the codon approximation — by fetching **real CDS** (WSN HA `J02176`, 99.1% identity; BG505 Env `DQ208458`, 99.9%) and rebuilding codon tables with 99%+ real codons: the result is unchanged (CLIB flu 0.500, HIV 0.581). CLIB's escape power is genuinely **SARS-CoV-2-specific**; the protein-LM signal is the more species-transferable component.

### 4.6 Point → Multi (Phases B10–B11)

**Hypothesis (biological).** Old, highly diverged viruses rarely escape via a single point mutation; escape is combinatorial. The single-mutation formulation is therefore the wrong frame for them.

**Formal extension.** For a variant v with mutation set `M(v) = {(i,a_i): a_i≠a_i^wt}`:

```
S1 = ‖e(v) − e(x_wt)‖₁                              (unchanged form)
S2 = Σ_{(i,a)∈M} log P(a | x_wt, mask i)           (sum over M)
CLIB(t) = Σ_{(i,a)∈M} log P(c_i → a, t)            (CTMC over M; t free)
```

The single-mutation model is the special case k=1, t→0; **t is the point↔multi knob** (small t: single-nucleotide; large t: multi-substitution).

**Evidence.**
- *Wu2020 combinatorial HA fitness (3,456 multi-mutants):* multi-CLIB–fitness Spearman rises **+0.206 (t=0.01) → +0.329 (t=10)** — larger t helps, as predicted. Semantic dominates (−0.459); signs differ, so a naive equal-weight CAC cancels but a **signed supervised CAC recovers +0.447** (vs unsigned +0.254).
- *H3N2 antigenic clusters (555 strains, real multi-mutation escape):* multi-mutation semantic predicts **antigenic-cluster escape at AUROC 0.769**, far above the single-mutation flu DMS escape (0.60–0.64), with monotonic drift (mean semantic 6.4→14.4 across cluster-order gaps). Caveat: clusters are time-ordered, so much of the signal is temporal divergence.
- *HI-titer antigenic distance, temporal-controlled (Smith-2004 map, 273 strains):* to remove the temporal confound we use the Racmacs `h3map2004` antigenic map, whose 2D coordinates are inferred from **HI titers** (273 antigens × 79 sera); antigenic distance is the Euclidean distance in the map (antigenic units ≈ 2-fold HI dilution), matched to each strain's HA sequence. Raw Spearman(semantic, antigenic) = +0.788, but both track year (semantic~year 0.83, antigenic~year 0.84). The confound is controlled three ways: **partial** Spearman(semantic, antigenic | year) = **+0.289**, and **within-year** pairs (|Δyear|=0, n=2,749) give Spearman = **+0.508** (|Δyear|≤2: +0.575). Since strains isolated in the *same year* have no temporal drift, this shows the multi-mutation semantic signal predicts antigenic distance **beyond time** — resolving the cluster-test caveat.

**Multi-mutation verification, in sum.** Three datasets validate the extension: Wu2020 (mechanism — the t knob), H3N2 antigenic clusters (escape prediction), and the HI-titer antigenic map (temporal-controlled confirmation). The last is the strongest evidence that the multi-mutation antigenic signal is real, not a byproduct of temporal divergence.

---

## 5. Discussion

Three themes emerge. **(1) Honesty is a design constraint.** The offset removes a circular fit; label-free fitting and DMS-free temporal validation show the model generalizes without ever touching the answer key. **(2) The mechanistic prior is powerful but local.** CLIB dominates for SARS-CoV-2 — a young, low-diversity virus whose escape is nucleotide-accessibility-constrained — but is near-random for ancient, hyper-diverse Influenza and HIV. The PLM's learned escape representation, by contrast, is what transfers across strains and (weakly) across species, and it comes from **domain adaptation rather than scale**. **(3) Escape has a characteristic mutational order.** For old viruses, escape lives at the multi-mutation level; the CTMC's evolutionary time t is the natural knob, and multi-mutation semantic change recovers real antigenic drift.

---

## 6. Limitations

- Label-free ≈ supervised is **backbone-dependent** (holds for domain-adapted ESM2cov; base PLMs reduce to CLIB-only).
- The DMS escape test has few positives (n=19) → high-variance margins; the robust evidence is weight recovery and larger-n temporal/variant results.
- Cross-species CLIB uses one reference strain per virus; per-lineage substitution signatures were not tested.
- The multi-mutation antigenic signal is partly temporal divergence; a time-controlled antigenic-distance ground truth (HI titers) is needed.
- Multi-mutation CLIB assumes position independence (no epistasis); CLIB's contribution decays at high mutation order.

---

## 7. Reproducibility

```bash
conda activate vanalyzer
cd viral-analyzer
python scripts/fetch_observed_freq.py        # CoV-Spectrum snapshot
python scripts/gen_backbone_dumps.py         # ESM2 150M/650M/3B, Hie tidy dumps
python scripts/phase{D,3,3b,4,5,6,7}*.py     # offset research (Phases 0-7)
python scripts/phaseB_labelfree.py           # label-free frequency proxy
python scripts/phaseB2_dmsfree.py            # temporal + Bloom DMS-free eval
python scripts/phaseB3_dmsfree_plus.py       # LOVO + multi-cutoff
python scripts/phaseB4_crosssignal.py        # cross-signal convergence
python scripts/phaseB5_multibackbone.py      # backbone robustness
python scripts/phaseB7_crossstrain.py        # strong-backbone cross-strain
python scripts/phaseB8_pluniversality.py     # general-PLM universality
python scripts/port_hie_species.py           # port flu/HIV escape
python scripts/phaseB9_species.py            # cross-species CLIB
python scripts/phaseB10_multimut_{embed,eval}.py   # multi-mutation CAC (Wu2020)
python scripts/phaseB11_{panel_embed,antigenic}.py # multi-mutation antigenic escape
```

Reports: `docs/research_report_clib_offset_en.md` (offset), `docs/research_report_labelfree_en.md` (label-free), `docs/notes/09_phaseB_labelfree.md` (Phase B–B11). Issues: rixile9999/climb2 #1 (label-free/universality), #2 (point→multi).

## References
Hie et al., *Science* 2021 (CSCS). Doud et al. 2018 (H1 HA escape). Dingens et al. 2019 (HIV Env escape). Wu et al. 2020 (HA combinatorial fitness). Smith et al. 2004 (antigenic cartography). Bloom lab SARS2-mut-fitness. CoV-Spectrum/LAPIS. CoVFit / ESM2_coronaviridae (Ito et al., bioRxiv 2024). ESM2 (Lin et al. 2023).
