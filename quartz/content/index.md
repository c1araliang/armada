---
title: A Framing Framework for Framing Biased Frames in LLM Training Material
description: Central hub for the ARMADA PhD project on detecting systematic framing bias in LLM training data.
tags:
  - methodology
  - pipeline
  - resources
---



**DC:** Wuyue `Clara` Liang

**Latest Update**: 2026-05-13

## Current Situation

* Corpus selection; Small-scale testing.
* Methodology being revised.

## Core Idea

*How are minority/immigrant groups systematically framed in LLM training data?*

Binary numeric indices per demographic group:

|Index|Measures|Operationalized as|
|-|-|-|
|**Subjecthood**|Syntactic subjecthood diagnostic|Proportion of occurrences as grammatical subject; not used as agency by itself|
|**AgI**|Agency — is the group portrayed as *doing* things with control/volition?|Proportion of occurrences as semantic/social agent|
|**PI**|Patienthood — is the group *acted upon*?|Proportion as patient|
|**SI**|Subjectivity — is the group granted *autonomous consciousness*?|Proportion as subject of mental-state verbs|
|**frame_negAttI**|Negative evaluative framing|Share of group sentences with target-bound F⁻ frame terms|
|**frame_posAttI**|Positive evaluative framing|Share of group sentences with target-bound F⁺ frame terms|
|**netAttI**|Net evaluative framing|`frame_negAttI − frame_posAttI`; local prototype AttI is diagnostic only|

These indices, together with WEAT and CEAT association scores, form the complex dimensions of the **Evaluative Framing Index (EFI)** — a per-group framing profile (see [[decisions#EFI Architecture]]).

### Structure overview

```text
Phase 1 - extract.py (offline, per-corpus)
---------------------------------------------------------------------------
Phase 2 - run_pipeline.py (analysis)
---------------------------------------------------------------------------
semantic_filter_results.tsv
    |
    +--> LLR / LogDice -> candidate collocates
    |        |                     |
    |        |                     +-- GTE ModernBERT cos-sim gate vs seed F-/F+
    |        |                             (human review: planned)
    |        v
    |      auto_negative_terms, auto_positive_terms  (accumulated word list for AttI)
    |
    |      seed_negative_terms, seed_positive_terms  (sentences -> neg/pos centroids)
    |        |
    |        +--> WEAT         (GTE ModernBERT, type-level vs centroids)        --> per-group WEAT
    |        +--> CEAT-filtered (GTE ModernBERT, sampled contexts vs centroids) --> per-group CEAT + N/SE
    |                               |
    |                               lexical_all.txt ------> CEAT-full (vs centroids)
    |                                                         |
    |                                                    Δ-CEAT = CEAT-full − CEAT-filtered
    |
    +--> preprocess  -> target-binding layer --> primary group identification
                                                    -> scope/review flags
                                                    -> local attitude diagnostics
                                                    -> SRL + predicate cues
                                                    -> per-group Subjecthood, AgI, PI, SI
                                                    -> target-bound frame-AttI
                                                                       |
                                                                       v
                                                     +----------------------+
                                                     | Group × Dimension    |
                                                     | matrix               |
                                                     | [AgI PI SI netAttI   |
                                                     |  WEAT CEAT-filtered] |
                                                     +----------+-----------+
                                                                |
                                                                v
                                                     +----------------------+
                                                     | PCA -> EFI (PC1)     |
                                                     +----------+-----------+
                                                                |
                                                                v
                                                     +----------------------+
                                                     | group_stats.tsv      |
                                                     | WEAT, CEAT-filtered, |
                                                     | CEAT-full, Δ-CEAT,   |
                                                     | EFI, Subj/AgI/PI/SI/frame-AttI |
                                                     +----------------------+
```

## What I Built and What It Showed

### Pipeline steps (Updated)

1. **Lexicons** — (Roughly) define target groups (*immigrant, refugee, asian...*), contrast groups (*european, citizen, local...*), ~~negative frames (*flood, swarm, tide...*), positive frames (*contribute, welcome, build...*)~~, and mental-state verbs (*believe, fear, decide...*), — no pre-specified frames.

For formal analysis, a `full-pass filter` is required.

The lexicon should thus be maintained and expanded in cross-reference with `NRC lexicon` by Saif M. Mohammad to close false-negative gaps. The original idea was a three-layer regex-spacy-llm per-word screening, which proved time-costly and wasting computing power.

Latest preprocessing flow:

|Layer | Source | Task | Purpose |
|---|---|---|---|
| **1. Lexical Gate** | lexicons.py extract.py | import and combine `TARGET_TOKENS` and `CONTRAST_TOKENS` into a regex pattern (GROUP_RE) to filter documents that mention demographic terms | First-stage filter using keyword matching |
| **2. ~~Syntactic filter~~ Semantic Retrieval** | ~~spaCy dep-parse~~ extract.py | teach pretrained sentence-transformers model `MiniLM` crude `POS_QUERIES` and `NEG_QUERIES` distiction, i.e., what are we (not) looking for | Embedding-based refined similarity scorings |
| **3. ~~Semantic screen~~ Classifier training** | filter_training_samples.txt | local, embedding-based supervised learning with ~~TF-IDF~~ `MiniLM + PCA + LogisticRegression` to compute probability of relevance | Final binary classification (RELEVANT vs IRRELEVANT) avoiding high-dimensional overfitting |

Preliminary results (2026-05-21) from `Dolma_v1.6_sample`, i.e., minimal Dolma, parquet 1/70:

|Metric | Value | Meaning|
|---|---|---|
|total_sentences| 1477953 | sentences extracted and evaluated.|
|lexical_hits | 139316 | sentences containing at least one TARGET or CONTRAST token.|
|semantic_pass | 5927 |sentences passed embedding similiarity test.|
|classifier_pass | 5291 |sentences with high relevant probability (≥ 0.56)|
|borderline_review | 14643 |sentences with medium relevant probability — needs human review, of which the results can be used to optimise the `PCA + LogisticRegression pipeline`.|

Visualized `extract.py`:

```
Parquet document
       ↓
[Lexical gate]  ─────────────→ (Logs all hits to semantic_filter_lexical_all.txt for CEAT-full)
       ↓ hit
Split into sentences (Regex hardened against Mr./Dr./Mrs. and other abbreviations)
       ↓
[Semantic retrieval] 
       ↓ pass
[Embedding PCA classifier]
       ↓
  relevant_prob ≥ 0.56 → KEPT          → results.tsv
  relevant_prob ≥ 0.45 → BORDERLINE    → review.tsv  
  prob < 0.45 → DROPPED
```

Then measure the **sentence-level, non-adjacent highest-Log-likelihood ratio (LLR) > LogDice collocates of each target AND contrast group** to produce an empirically discovered statistical evidence of collocation profile for both sides.

1. **Human Baseline for LM Reference** — inter-annotator linguistics expert agreement on ~~*Lexicon Construction*~~, *Sentence Preclassification*, and ***grouping empirically discovered high-LLR collocates into frame types***, based on selected excerpt, to provide a community-validated layer of legitimacy.

The problematic 1st version measured predefined results, while Sinclair's corpus linguistics method runs the other direction: `observe` → `classify`. Classification of existing data--not predicting what data should look like and composing frames from scratch--is a more natural task for linguists.

1. **Framing** — Develop a composite frame taxonomy (metaphorical: natural disaster, dehumanization, invasion, contribution...; attitudinal: positive-negative, verbal-adjectival...) based on post-hoc classification and loop auto-refresh.
2. **Preprocessing** — Sentence Preclassification → Strip noise (HTML, encoding artifacts) if there's any → spaCy token-level annotations (lemma, POS, dependency relation).
3. **Feature extraction** — For each target token or small target span, the pipeline now separates Subjecthood from AgI, applies affectedness/mental-state predicate cues, and routes negation, correction, quotation, contrast, and ambiguous frame binding into review flags. Reported AttI uses target-bound frame association; local prototype matching remains diagnostic.
4. **Association testing (WEAT + CEAT)** — Using seed sentence prototypes encoded into F⁻ and F⁺ centroids:
    * **WEAT** (static embeddings): type-level — is the vector for *immigrant* closer to the F⁻ centroid or F⁺ centroid?
    * **CEAT** (contextualized sentence embeddings): sampled context distribution — across encoded sentences containing *immigrant*, is the contextual embedding closer to the F⁻ centroid or F⁺ centroid? The pipeline reports the mean plus `N` and `SE`.
5. **EFI via PCA** — Assemble group × dimension matrix [AgI, PI, SI, frame-derived netAttI, WEAT, CEAT]. Run PCA on groups with `N ≥ 50`. Subjecthood is retained as a diagnostic column, not an EFI dimension. At the current stage, PCA should be treated as an exploratory dimensionality-reduction step rather than a finalized definition of "negative framing"; the interpretation of PC1 remains open for later discussion.
6. **Output** — Per-sentence table (targets, indexical counts) + per-group summary (proportionalized indices, WEAT/CEAT scores, EFI, PCA loadings, regression β), with the reported group table filtered to lemmas with `N ≥ 50`.

### Design decisions from early testing

Detailed rationales for pipeline architecture choices, alternatives rejected, and risk evaluations have been moved to the decisions log.
→ See [[decisions]]

## Explorables

### Changes

[[tracker#Implemented Changes]] → [[tracker#Current Status]]

### To-Dos

[[todo#Action Queue]] → [[todo#Broad Plans]]

### Active Logs

[[log]]

## Reading List & Novelty Check

The complete SOTA literature review, categorized by tags (methodology, framing, benchmark) and sorted by relevance, has been moved to [[reading]]

Partial overlaps exist but what makes ARMADA novel is not that each individual technical/conceptual component is unprecedented, but that such components are integrated in a way I have not found in prior work on LLM bias-related studies.

My work merges:

- bottom-up frame discovery from corpus statistics,
- group-level role profiling (w/ redesigned target-aware semantic categories: agency, patienthood, subjectivity),
- association testing anchored in the discovered framing inventory,
- and a composite per-group framing profile.

into a reproducible pipeline, closing the causal chain, i.e. biased training input - bias direction - biased output.

**Methodological note**: Even as model probes, existing studies on pretraining data predominantly confirm pre-theorized stereotype categories (in some cases generated by LMs w/ clear priors), making them blind to emerging, subtle, or culturally situated framing biases. A model trained on text that systematically frames immigrants as patients with strong affectedness (w/o ever using a slur) would score perfectly fine on existing benchmarks, even though the linguistic framing is clearly and subtly skewed.

## Side Notes

Best reference: NRC Emotion Lexicon; Görge et al. (2025) provide LM-generated word list, and Kadan et al. (NLP Journal 2024) provide target terms for affective bias, both are references for more diverse training data; 

VerbNet (ancient, messy categories), WordNet (frozen, awkward syntax), and MECORE predicate database (48 predicates, theory-curated for cross-linguistic representativeness, not corpus coverage) are supplementary references but insufficient as primary sources.
