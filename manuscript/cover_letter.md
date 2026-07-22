# Cover Letter — Nature Communications

Dear Editor,

We are pleased to submit our manuscript, **"The mutation process label-free calibrates protein language models for viral immune-escape prediction,"** for consideration as an Article in *Nature Communications*.

**The problem, and why it is timely.** Protein language models (PLMs) are now widely used to rank amino-acid substitutions for viral immune escape. Yet a 2025 systematic evaluation (Allman et al., *J. R. Soc. Interface*) reported that their zero-shot scores fail to separate escape from non-escape mutations on SARS-CoV-2 Spike, and concluded that reliable prediction requires *supervised* fine-tuning toward specific phenotypes. This has left the field with a stark choice: abandon label-free PLM scoring, or invest in scarce, expensive escape labels. Our work resolves this tension.

**Our central advance.** We show that the missing ingredient is not supervision but the **mutation process**. PLMs score substitutions in protein space, whereas variation is supplied as stochastic nucleotide changes that make some amino-acid substitutions far more genetically accessible than others. We introduce a post-hoc, model-agnostic calibration (CAC) that augments any PLM's semantic-change and grammaticality scores with a codon-transition accessibility prior derived from a continuous-time nucleotide Markov process. Critically, the calibration requires **no escape labels**: freezing the accessibility prior and fixing the remaining protein-signal weights by regression against an *independent* phylogenetic-fitness signal (Bloom & Neher observed-versus-expected substitution counts on the viral tree) recovers the escape-supervised weights and matches held-out deep-mutational-scanning recovery. The calibration is **self-adaptive** — it leans on the mutation process when a backbone's protein features are uninformative and on the PLM when they are — and it **forecasts prospectively**, predicting post-cutoff emergent substitutions from pre-cutoff data alone (AUROC ≈ 0.90). It also exposes an interpretable evolutionary-horizon crossover from mutational accessibility to protein-level constraint.

**Why this is new and distinct.** Unlike EVEscape (Nature 2023), which couples a learned constraint model to a *structural/biophysical* accessibility prior, ours is a genotype-level codon-transition process used to *calibrate* a PLM rather than as a multiplicative factor. Unlike CoVFit (*Nat. Commun.* 2025), which fine-tunes a PLM on surveillance reproduction numbers and DMS escape, our method uses a related evolutionary signal with **no escape labels and no retraining**. And unlike frequency-based fitness-inference methods such as PyR0 (Science 2022), we do not infer fitness — we use the evolutionary signal to fix a PLM's calibration. To our knowledge, this is the first demonstration that an independent evolutionary observable can determine a PLM's escape-scoring weights without any phenotype labels, directly answering the label-free-failure critique.

**Significance and fit for *Nature Communications*.** The result is immediately practical: it provides a portable, plug-in calibration for frozen foundation models, and because it needs no escape labels, it is deployable to newly emerging pathogens for which DMS data do not yet exist. Conceptually, it recasts a reported failure of protein-space scoring as a calibration problem solved by the mutation process itself — a mutation-supply-times-selection decomposition with population-genetic precedent, bridging protein language modeling, molecular evolution, and stochastic-process theory. We believe this interdisciplinary contribution, and its direct engagement with an active debate in the field, are well matched to the broad readership of *Nature Communications*.

**Statements.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. All source code and processed data are publicly available (GitHub and Zenodo; DOI 10.5281/zenodo.15744443), and the label-free-calibration analyses are archived in an accompanying branch.

We suggest that expert reviewers be drawn from three areas: (i) phylogenetic fitness inference and viral molecular evolution; (ii) protein language models for variant-effect / escape prediction; and (iii) codon-substitution and mutation-selection modeling. We would be glad to provide specific names on request, and to exclude any reviewers at the editor's discretion.

Thank you for considering our work. We look forward to your response.

Sincerely,

Chansoo Kim and Cherlhyun Jeong, on behalf of all authors
Korea Institute of Science and Technology (KIST)
che.jeong@kist.re.kr · eau@ust.ac.kr
