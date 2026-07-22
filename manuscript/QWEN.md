# QWEN.md — Project Context

## Project Overview

This repository contains a **Physical Review Letters (PRL) manuscript** and its supplementary material on the topic of **codon-aware calibration of protein language models (PLMs) for genotype–phenotype evolutionary forecasting**.

**Title:** *Mutation-process priors calibrate protein language models for genotype–phenotype evolutionary forecasting*

**Authors:** Hyoseok Jang, Sangchul Lee, David S. Yang, Haneol Cho, Cherlhyun Jeong, Chansoo Kim (KIST / UST / Sejong University)

**Core idea:** Protein language models rank amino-acid substitutions by learned constraints, but natural variation arises through stochastic nucleotide changes. The paper introduces **CAC (codon-aware constrained semantic change)** — a post-hoc calibration layer that augments PLM scores with a codon-transition prior inferred from mutation spectra, without retraining. Evaluated on a retrospective SARS-CoV-2 Spike benchmark.

## Directory Structure

| Path | Description |
|---|---|
| `main_prl_revised.docx` | Revised Word manuscript (latest revision incorporating reviewer feedback) |
| `feedback_review.md` | Revision notes documenting what was changed and why in response to reviewer feedback |
| `_Bio__Viral_Escape_1/` | Main paper LaTeX source and figure assets |
| `_Bio__Viral_Escape_1/main_prl.tex` | PRL-format LaTeX source (revtex4-2) |
| `_Bio__Viral_Escape_1/main_nature.tex` | Earlier Nature-format LaTeX source |
| `_Bio__Viral_Escape_1/sn-bibliography.bib` | BibTeX bibliography |
| `_Bio__Viral_Escape_1/main_prl.pdf` | Compiled PRL PDF |
| `_Bio__Viral_Escape_1/Figure_Nature_*.pdf/.png` | Main figures (1–5); `Figure_Nature_1_revised` is the corrected version |
| `_Bio__Viral_Escape_1__Supplementary_/` | Supplementary material: LaTeX source (`main.tex`), compiled PDF, supplementary figures and tables |

## Key Concepts

- **CAC (Codon-Aware Constrained semantic change):** Combines PLM semantic-change score, grammaticality score, and a codon-accessibility prior into a single ranking score.
- **CLIMB (Codon-LIkelihood Markov-Bias):** The codon-accessibility term derived from a continuous-time Markov process on nucleotides with generator Q estimated from SARS-CoV-2 substitution spectra.
- **ESM-2:** General-purpose protein language model (used at 150M, 650M, and 3B parameter scales).
- **hie (HiE):** A specialized PLM that incorporates evolutionary information.
- **Markov horizon T:** Dimensionless parameter controlling the breadth of the mutational neighborhood (not calendar time).

## Build Commands

### Main Paper (PRL format)
```bash
docker run --rm -v $(pwd)/_Bio__Viral_Escape_1:/work -w /work texlive/texlive sh -c \
  "pdflatex -interaction=nonstopmode main_prl.tex && bibtex main_prl && pdflatex -interaction=nonstopmode main_prl.tex && pdflatex -interaction=nonstopmode main_prl.tex"
```

### Supplementary Material
```bash
docker run --rm -v $(pwd)/_Bio__Viral_Escape_1__Supplementary_:/work -w /work texlive/texlive sh -c \
  "pdflatex -interaction=nonstopmode main.tex && biber main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"
```

## LaTeX Conventions

- Main paper uses `revtex4-2` document class with `aps,prl,reprint,superscriptaddress,nofootinbib` options.
- Supplementary material uses standard `article` class with `geometry` (1-inch margins).
- Supplementary sections, figures, and tables are prefixed with "S" (e.g., Section S1, Figure S1, Table S1).
- Bibliography style: `sn-nature.bst` for the Nature-format source; standard BibTeX for PRL format. Supplementary uses `biblatex` with `biber` backend.
- LaTeX build artifacts (`.aux`, `.bbl`, `.log`, etc.) are gitignored.

## Revision Status

The manuscript has been revised once based on reviewer feedback. Key changes (documented in `feedback_review.md`):
- Reframed from "viral escape prediction" to "retrospective genotype–phenotype evolutionary forecasting"
- Removed language that implied teleological evolution ("evolution proposes")
- Repositioned CAC/CLIMB as a PLM calibration layer within existing mutation-selection framework
- Clarified that Markov horizon T is dimensionless, not calendar time
- Revised Figure 1 to fix "Constrain Score" → "Constrained Score" typo
- Added safety-policy-aware language to avoid operational variant-construction interpretation

## Guidelines for Assistants

- This is a **scientific manuscript**, not a software project. Edits should preserve academic tone and scientific accuracy.
- LaTeX files are the source of truth; the `.docx` is a parallel deliverable.
- When editing `.tex` files, maintain the existing formatting style (indentation, line breaks around equations).
- Figure PDFs are binary assets — do not regenerate them unless explicitly asked.
- The bibliography is in `sn-bibliography.bib`; new references should follow its existing BibTeX key convention (e.g., `AuthorYearKeyword`).
- Korean-language notes in `feedback_review.md` are intentional; do not translate them.
