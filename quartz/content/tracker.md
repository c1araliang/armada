---
title: 2. Changes
description: Important changes and pending actions from pipeline development.
tags:
  - tracking
  - pipeline
---


## Pending Actions

Ongoing or partially addressed processing edges from output inspection:

|Problem|Status & Impact|
|---|---|
|**Wronged Rejects?** (Semantic gate)|Some semantic rejects from the classifier boundary check appear legitimate in context. Borderline outputs (score 0.45-0.55) remain the immediate priority for human review to adjust the PCA+LogReg classifier.|
|**Reduplicated Sentencess**|A sentence matching both `immigrant` and `refugee` gets exported twice to `semantic_filter_results.tsv`. Only 12 instances identified in Dolma sample. It functions essentially like the `Sri Lankan` split constraint. Completely negligible impact at corpus scale.|
|**Outlier 2nd-screening**|Outputs surfacing with `AgI`/`PI`/`SI` = 0 or > 2 are actively exported to `output_review.tsv` for syntactic sanity checks.|
|**Passive Mismatch**|**Partially addressed:** SRL assigns `PI` via `ARG1`; spaCy `nsubjpass`/`pcomp+auxpass` retained as fallback guard. Cross-sentence coreference tracking remains unaddressed systemically.|

## Important Changes

(2026-04-02)

1. **Association layer rewritten.**
`PPMI` was removed from the active collocate-discovery path and replaced with sentence-level, non-adjacent `LLR / LogDice` scoring. Avoid inflate counts of adjacent lexicalized pairs.

2. **Role extraction upgraded from dependency-only to hybrid SRL.**
`AgI / PI / SI` are now extracted primarily through a Hugging Face SRL model.

3. **Demographic Lexicons Massively Expanded.**
`TARGET_TOKENS` expanded to >140 groups (Global South / historically marginalized), and `CONTRAST_TOKENS` expanded to >40 groups (European / Anglosphere / dominants). Broad tokens (`citizen`, `minority`, `native`) no longer trigger the gate independently to prevent noise.

4. **SEAT computation standardized and SEAT-full added.**
SEAT uses MiniLM sentence embeddings. WEAT was switched from spaCy GloVe to the exact same MiniLM encoder to ensure cross-corpus comparability. `SEAT-full` (computed on *all* raw lexical hits bypassing semantic filters) allows computation of `Δ-SEAT` (`SEAT-full` - `SEAT-filtered`), quantifying "associative contamination" from non-human or institutional usages in the raw corpus.

5. **Reporting tightened.**
Per-group summary outputs are now filtered to `N ≥ 50` for statistical stability at full scale.

6. **Rule-based `AttI` replaced with prototype embedding matching.**
`negAttI` and `posAttI` are no longer assigned via fixed rules. A new attitudinal module scores a local group-centered context snippet against negative and positive prototypes via cosine similarity.

7. **Frame inventory (`F⁻` / `F⁺`) refreshed from LLR candidates.**
`WEAT` and `SEAT` no longer anchor exclusively on the static manual frame taxonomy. At each pipeline run, we compare current LLR candidates against seed frame terms and augment `F⁻` / `F⁺` accordingly, writing the result into `candidate_terms.json`. The file is regenerated each run rather than accumulated. Manual seed sets are kept as polarity anchors.
Auto-admission currently overfires on generic discourse words (`apply`, `work`, `community`); the intended resolution is Step 1 (human baseline): annotators grouping high-LLR collocates into confirmed frame types provide exactly the community-validated gate that prevents generic candidates from entering `F⁻` / `F⁺` unrestricted.

8. **Classifiers & Disambiguation Hardened.**
The final sentence filter layer now uses PCA over target embeddings instead of TF-IDF, preventing sparse vector overfitting. The polysemous group resolution (`PROMPT_BANK`) was rebuilt into a universal template registry, ensuring consistent and scalable zero-shot disambiguation across ambiguous tokens like `polish`, `foreign`, or `black`.