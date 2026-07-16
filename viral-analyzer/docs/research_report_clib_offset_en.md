# Freezing CLIB as an Offset: Optimizing an Immune-Escape Scoring Model

*Audience: a student new to this project. Every technical term is explained the first time it appears.*
*Date: 2026-07-16 · Status: Phases 0–7 complete, optimization endpoint reached.*

---

## 0. Three-sentence summary

This project ranks single-residue mutations of a viral protein by how likely each one is to cause **immune escape** (a change that lets the virus evade antibodies). The original scoring method chose the weights that combine its signals by fitting them directly to the answer key it was then evaluated on, which inflates the reported performance. We asked whether one particular idea — taking the term that comes from **nucleotide (genetic) statistics and freezing it as a fixed "offset" instead of learning its weight** — is actually sound, and we tested it with honest cross-validation across multiple random seeds and multiple protein models. The answer is a qualified **yes**: the frozen term genuinely helps when the protein model is weak or the task is hard (e.g. it lifts a weak backbone's AUROC from 0.53 to 0.85), and it is the signal that carries generalization to unseen viral strains, while adding any further model complexity yields no honest improvement.

---

## 1. Background: the problem this project solves

### 1.1 What we predict
The SARS-CoV-2 (COVID-19) spike protein has about 1,270 amino-acid positions. Changing any one position to a different amino acid (a single mutation) produces roughly 24,000 variants. The goal is to rank these variants so that the **immune-escape mutations** — the ones that let the virus evade vaccines or antibodies — sit near the top. Measuring every mutation experimentally is infeasible, so a computational model puts "the most dangerous-looking ones first."

### 1.2 The existing score: CaCSCS
For each mutation the existing model computes **three signals** and combines them into one number.

- **S1 — semantic (how much the protein "reads" differently):** the distance, in a protein language model's embedding space, between the original and the mutated sequence. A large distance means the protein looks antigenically novel, which correlates with escape.
- **S2 — grammar (viability):** the model's estimated probability that this amino acid belongs at this position (in log units). This captures whether the mutated protein still "makes sense," i.e. whether the virus can survive the change.
- **S3 — CLIB (nucleotide accessibility):** see §1.4. It measures how easy the mutation is to reach at the genetic (codon) level.

> **What is a protein language model (PLM)?** An AI trained on amino-acid sequences instead of text. The best-known one is **ESM2**; this project also uses **ESM2_coronaviridae** (ESM2cov for short), a version further trained on coronavirus sequences. Like a "fill-in-the-blank" model, it learns which amino acids are natural at each position (giving S2) and produces the sequence embedding used for S1.

Each signal is **standardized to a z-score** (rescaled to mean 0, standard deviation 1) so that quantities on different scales can be added fairly. The three are then combined:

```
raw = alpha * S1 + beta * S2 + gamma * S3        (alpha + beta + gamma = 1)
final = the rank of the 24,000 mutations sorted by raw, descending
```

Here `alpha, beta, gamma` are the **weights** on the three signals.

### 1.3 The flaw: post-hoc fitting
How did the original code choose `alpha, beta, gamma`? It searched the weight space for the values that **minimize the average rank of the known escape mutations** — and then reported that same average rank. This is like reading the exam answers, tuning your answers to match, and then boasting about your score. When you pick the weights against the answer key and then measure yourself with the same answer key, the number you get is better than your true performance on unseen data. This is called **in-sample bias** or a **circular fit**: it arises because no data was held out for validation.

### 1.4 CLIB (S3): the nucleotide-statistics term
An amino acid is encoded by three nucleotide bases (a codon, e.g. `AAG`). Some substitutions require changing only one base; others require two or three. In other words, some amino-acid changes are **genetically easier to reach** than others. CLIB quantifies this accessibility:

- The virus's real mutation tendencies (which base tends to change into which) are estimated from a **large, separate dataset** of single-base-substitution counts (192 sequence contexts) to build a 4×4 rate matrix **Q**.
- The transition probability over an evolutionary time **t** is `P = exp(t·Q)` (a continuous-time Markov model).
- For each mutation, CLIB is the probability that the original codon turns into any codon of the target amino acid within time `t`. Its logarithm, standardized, is **S3 = log_clib_z**.

The crucial point: **S3 comes from an entirely different (and much larger) dataset than the escape answer key.** That makes it a "mechanistic prior" — biological background knowledge, not something learned from the labels.

---

## 2. The idea: freeze CLIB as an "offset"

### 2.1 Why freeze it
S3's information is already well estimated from the large nucleotide dataset. Re-fitting its weight against a mere few dozen escape labels would (1) waste that reliable prior and (2) burn scarce labels, inviting overfitting. So the proposal is to **stop learning S3's weight and fix it at a constant κ (default 1)**. In statistics, a term added with a fixed, non-learned coefficient is called an **offset**.

### 2.2 A terminology correction
The idea was first phrased as "use CLIB as a bias term," which is imprecise.

- A **bias term (intercept)** is a single constant added to every example, and it *is* learned.
- An **offset** varies per example (S3 differs for every mutation) but its coefficient is fixed and *not* learned.

Since S3 varies per mutation, it cannot be a bias term; it is exactly an **offset**. The final model is:

```
score = b0 * S1 + b1 * S2 + kappa * S3        (kappa fixed; only b0, b1 are learned)
```

Now **only two parameters (b0, b1) are learned.** The protein model (S1, S2) and the nucleotide term (S3) are both **frozen**, and the scarce labels are spent only on the two weights.

---

## 3. Phase 0 — data hygiene, and a real bug

Before building anything, you must check that the features and the answer key line up. We counted whether each escape mutation actually appears in the 24,000-mutation candidate set (a **count assertion**).

This exposed a genuine bug:

- Most lists (DMS, Alpha, Beta, Gamma, Omicron) matched perfectly under the repository's convention of **0-based indexing** (the first residue is numbered 0).
- The **`Actual` list was written in 1-based (textbook) numbering** — a coordinate-system bug — so only 4 of its 32 entries matched; shifting every position down by 1 fixed 31 of them.
- The unmatched entries in `Delta`/`Combined` were **deletions or wildcards ("X")**, which are not single substitutions and cannot appear in the candidate set at all.

Because the old pipeline never verified counts, it had been **silently scoring the wrong mutations** on these tasks. We normalized `Actual` and dropped the non-standard entries; all count assertions now pass (the original was backed up).

**Lesson:** a simple "do the labels line up with the candidates?" check catches bugs that hide for a long time. This assertion belongs in the pipeline permanently.

---

## 4. Phase 1 (go/no-go) — does CLIB carry independent signal?

Before committing to a multi-day plan, we cheaply tested the riskiest assumption: does S3 (CLIB) carry escape signal beyond S1 and S2?

We used honest **5-fold cross-validation** (fit the weights on 4 folds, score the held-out fold) and compared three models: A (no CLIB), B (offset with κ=1), C (learn S3's weight too). A model "helps" only if the 95% confidence interval of its mean-rank improvement — obtained by bootstrapping the positives — excludes zero.

| task | mean-rank A→B | 95% CI of improvement | AUROC A→B | helps? |
|---|---|---|---|---|
| Omicron | 401 → 379 | [−72, +16] | 0.984 → 0.985 | no |
| **DMS** | 4,680 → 3,250 | **[−2492, −466]** | 0.807 → 0.866 | **yes** |

**Verdict: a qualified GO.** CLIB helps significantly, even out-of-sample, exactly where the protein model is weak (DMS, AUROC 0.807), and adds nothing where the model is already near-perfect (Omicron, 0.984). It also confirms that the old "CAC helps most on DMS at small t" claim was real, not merely an in-sample artifact — though the benefit is narrower than "helps everywhere."

---

## 5. Phase 2 (Phase 3 script) — confirmation across seeds and backbones

We hardened the result with 5 random seeds × 5-fold **nested cross-validation** (an inner loop selects the time `t`, so even that choice is out-of-sample), across three backbones of differing strength, comparing the offset model against the old barycentric method both in-sample and out-of-sample.

| backbone | task | A (no CLIB) | B (offset) | AUROC A→B | helps? |
|---|---|---|---|---|---|
| base ESM2 | DMS | 11,400 | 3,710 | **0.53 → 0.85** | yes |
| Hie | DMS | 4,620 | 2,260 | 0.81 → 0.91 | yes |
| ESM2cov | DMS | 4,330 | 2,870 | 0.82 → 0.88 | yes |
| ESM2cov | Omicron | 408 | 401 | 0.984 → 0.984 | no |

Three conclusions:

1. **Hypothesis confirmed — the weaker the backbone, the more CLIB helps.** The extreme case is base ESM2, which is near-random on DMS (AUROC 0.53) but jumps to 0.85 once the offset is added. The only "no help" cell is the strong ESM2cov on Omicron, where the model is already near-perfect.
2. **The DMS signal is robust** across all backbones and all seeds (every confidence interval excludes zero).
3. **Honest validation quantified the old method's optimism.** The legacy in-sample number for ESM2cov/DMS was ~2,270, versus ~3,000 out-of-sample — about a **30% inflation**. And the 2-parameter offset matched the honest 3-parameter legacy, so freezing CLIB costs nothing while removing the circular fit and cutting a parameter. Learning S3's weight (free-b3) helped only marginally, confirming that **fixing κ=1 is the right trade-off.**

---

## 6. Phase 3 (Phase 3b script) — can we drop the time hyperparameter?

The inner CV kept driving `t` to the smallest value in the grid, and the benefit saturated below ~0.01. So we tested whether `t` can be removed entirely.

- **Fixing t = 0.01 retained full performance in all six cells** (every paired bootstrap interval includes zero). So the evolutionary-time hyperparameter can simply be dropped.
- A fully **discrete** replacement — "minimum number of base changes to reach the target amino acid" — retained performance for strong/medium backbones but significantly degraded the weak base ESM2, so we keep the continuous CLIB.

**Decision:** the final scorer is `score = b0·S1 + b1·S2 + κ·CLIB(t=0.01)` with κ=1 — **two learned parameters, zero hyperparameters.** In the t→0 limit, CLIB essentially reduces to "is this amino acid reachable by a single nucleotide change?", which is the biological core of the signal.

---

## 7. Phase 4 (fork) — does stacking multiple backbones help?

This tested the original "meta-model" vision honestly, on a dedicated git branch: combine several protein models' features (plus the shared CLIB offset — note CLIB depends only on the codon, not the backbone) under an L2-regularized logistic model.

| task | model | mean-rank | AUROC | vs single ESM2cov | beats it? |
|---|---|---|---|---|---|
| DMS | cov+base+hie | 2,525 | 0.896 | −341 [−812, +72] | no (not significant) |
| Omicron | cov+base+hie | 827 | 0.966 | +443 [210, 742] | no (significantly worse) |

**Stacking does not beat the single best backbone.** On DMS the full stack shows a hopeful point estimate but does not reach significance with only 19 positives; on Omicron, mixing in weaker backbones dilutes the strong signal and significantly hurts. This is a valuable **negative result** — direct evidence that label scarcity caps model complexity. The stacking branch was rejected, and only the experiment record was merged back.

---

## 8. Phase 5 — ranking-aligned loss; Phase 7 — richer semantics

Two more attempts to squeeze out gains, both negative:

- **Phase 5 (loss function):** a pairwise learning-to-rank (RankNet) loss is statistically indistinguishable from the plain pointwise logistic (both mean-rank intervals include zero). With only two free weights, the logistic solution is already near rank-optimal, so the loss choice does not matter.
- **Phase 7 (richer semantics):** adding a second embedding-distance metric (L2) to the strong ESM2cov backbone does not help (Omicron gets significantly worse; DMS is unchanged), and swapping L1 for L2 is identical.

Together with Phases 3b and 4, these make **four consecutive no-improvement results** (discretize, stack, loss, metrics).

---

## 9. Phase 6 — cross-strain generalization (the high point)

The real scientific goal is predicting escape on strains not used for fitting. Using base ESM2-650M (the only backbone available for all strains), we ran **leave-one-strain-out** over five clean strains: for each held-out strain, fit on the other four, then predict the held-out one.

| model | mean-rank | AUROC | description |
|---|---|---|---|
| A protein-only (transferred) | 9,631 | 0.601 | base protein features barely transfer (near-random) |
| B protein + offset (transferred) | 3,428 | 0.858 | offset helps on 5/5 strains (CIs exclude 0) |
| **C CLIB-only (no training at all)** | **3,257** | **0.864** | best, with no training and no protein model |

Findings:

1. **The offset transfers robustly** across strains (significant on all five).
2. **With a weak backbone, CLIB carries essentially all of the transferable signal.** The base protein features transfer at AUROC 0.60 (barely above chance), while **CLIB alone — no training, no protein model — reaches AUROC 0.864**, matching or slightly beating the full model. This is the strongest demonstration that CLIB is a genuinely transferable mechanistic prior.

Two caveats to read alongside this:

- **Backbone-limited:** the strong ESM2cov backbone exists only for WildType, so we could not test strong-backbone transfer. When the backbone is strong, the protein features clearly matter in-strain (Phase 2).
- **Possible ascertainment bias:** if the curated escape sets are enriched for nucleotide-accessible mutations, CLIB's apparent power could be overstated. The Baum et al. DMS measures all mutations functionally, so the bias is likely limited, but we cannot fully rule it out without the curation details. This is the **key open question for future validation.**

(Delta was excluded here because its base-ESM feature file is truncated to positions 0–93 — a data-generation gap, not a coordinate bug.)

---

## 10. Final conclusion — why we stopped

**Confirmed model:** `score = b0·S1 + b1·S2 + 1.0·CLIB(t=0.01)` — two learned parameters, no hyperparameters. It removes the circular in-sample fit (eliminating ~30% optimism), matches the honest legacy method, and is simpler.

**Why this is the endpoint:** four independent complexity axes (discretization, multi-backbone stacking, ranking loss, richer semantics) all failed to beat the simple two-parameter model under honest out-of-sample evaluation. Phase 6 then showed the offset is the signal that carries cross-strain generalization, confirming its value at the deepest level. The remaining gains lie in **data, not the model.**

**What more data could unlock (the current ceiling):**
1. Recompute the strong ESM2cov backbone's features for the variant strains, to test strong-backbone cross-strain transfer.
2. More escape labels, which would let stacking be revisited.
3. A direct check of the ascertainment-bias question from Phase 6.
4. Per-lineage nucleotide substitution signatures, to make CLIB strain-specific.

All of these require new data or computation beyond this repository.

---

## Appendix: how to reproduce

```bash
conda activate vanalyzer   # numpy, pandas, scipy, statsmodels, scikit-learn, pyarrow
cd viral-analyzer

python scripts/phase0_dump.py        # tidy dump + count assertion
python scripts/fix_escape_coords.py  # one-time coordinate fix (writes cov-wt.json.bak)
python scripts/phaseD_gonogo.py      # go/no-go
python scripts/phase3_nestedcv.py    # nested CV across backbones
python scripts/phase3b_simplify.py   # drop the t hyperparameter
python scripts/phase4_stack.py       # multi-backbone stacking (rejected)
python scripts/phase5_loss.py        # pairwise vs pointwise loss
python scripts/phase6_crosswt.py     # leave-one-strain-out transfer
python scripts/phase7_metrics.py     # richer within-backbone semantics
```

**Mini-glossary:** z-score (standardized value) · offset (an added term with a fixed, non-learned coefficient) · in-sample / out-of-sample (fit to the answer key vs. evaluated on unseen data) · cross-validation (splitting data to validate on held-out parts) · mean-rank (average position of the positives; lower is better) · AUROC / AUPRC (summaries of ranking quality) · bootstrap (resampling to estimate uncertainty) · LRT (a test of whether adding a term matters) · backbone (the protein model that produces S1 and S2) · CLIB (the nucleotide-accessibility term).
