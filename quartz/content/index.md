---
title: A Framing Framework for Framing Biased Frames in LLM Training Material
description: Central hub for the ARMADA PhD project on detecting systematic framing bias in LLM training data.
tags:
  - methodology
  - pipeline
  - resources
---



**DC:** Wuyue (Clara) Liang

**Latest Update**: 2026-04-10

## Current Situation

* Corpus selection; Small-scale testing.
* Methodology being revised.

## Core Idea

*How are minority/immigrant groups systematically framed in LLM training data?*

Binary numeric indices per demographic group:

|Index|Measures|Operationalized as|
|-|-|-|
|**AgI**|Agency — is the group portrayed as *doing* things?|Proportion of occurrences as grammatical agent|
|**PI**|Patienthood — is the group *acted upon*?|Proportion as grammatical patient|
|**SI**|Subjectivity — is the group granted *autonomous consciousness*?|Proportion as subject of mental-state verbs|
|**negAttI**|Negative attitudinal attribution|Proportion scored as negative by prototype embedding matcher|
|**posAttI**|Positive attitudinal attribution|Proportion scored as positive by prototype embedding matcher|

These indices, together with WEAT and SEAT association scores, form the complex dimensions of the **Evaluative Framing Index (EFI)** — a per-group framing profile (see [[#efi-architecture|EFI Architecture]]).

### Structure overview

```text
Phase 1 — extract.py (offline, per-corpus)
──────────────────────────────────────────────────────────────────
Phase 2 — run_pipeline.py (analysis)
──────────────────────────────────────────────────────────────────
semantic_filter_results.tsv
    │
    ├──► LLR / LogDice → Top collocates → human classification
    │         ↓
    │    F⁻, F⁺  (frame attribute sets)
    │         ↓
    │    WEAT (MiniLM type-level)   SEAT-filtered (MiniLM token-level)
    │         ↓                          ↓
    │    per-group WEAT score       per-group SEAT score ────────┐
    │                               ↓                            │
    │                          F⁻/F⁺ centroids                   │
    │                               ↓                            │
    │                     lexical_all.txt ──► SEAT-full          │
    │                                             ↓              │
    │                                         Δ-SEAT ────────────┤
    │                                                            │
    └──► (spaCy)-SRL → feature extraction                        │
              ↓                                                  │
         per-group AgI, PI, SI, negAttI, posAttI ────────────────┤
                                                                 ↓
                                                    ┌─────────────────────┐
                                                    │ Group × Dimension   │
                                                    │ matrix              │
                                                    └──────────┬──────────┘
                                                               ↓
                                                    ┌─────────────────────┐
                                                    │ PCA → EFI           │
                                                    │  PC1 = axis of max  │
                                                    │  cross-group var.   │
                                                    └──────────┬──────────┘
                                                               ↓
                                                    group_stats.tsv
                                                    (WEAT, SEAT, SEAT-full,
                                                     Δ-SEAT, EFI, AgI/PI/SI…)
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
| **2. ~~Syntactic filter~~ Semantic Retrieval** | ~~spaCy dep-parse~~ extract.py | teach pretrained sentence-transformers model `miniLM` hardcoded `POS_QUERIES` and `NEG_QUERIES`, i.e., what are we (not) looking for | Embedding-based refined similarity scorings |
| **3. ~~Semantic screen~~ Classifier training** | filter_training_samples.txt | local, embedding-based supervised learning with ~~TF-IDF~~ `MiniLM + PCA + LogisticRegression` to compute probability of relevance | Final binary classification (RELEVANT vs IRRELEVANT) avoiding high-dimensional overfitting |

Preliminary results (2026-04-02) from `Dolma_v1.6_sample`, i.e., minial Dolma, parquet 1/70:

|Metric | Value | Meaning|
|---|---|---|
|total_sentences| 1392502 | sentences extracted and evaluated.|
|lexical_hits | 123976 | sentences (8.903%) containing at least one TARGET or CONTRAST token.|
|semantic_pass | 5442 |sentences (0.391%) passed embedding similiarity test.|
|classifier_pass | 2895 |sentences (0.208%) with high relevant probability (≥ 0.56)|
|borderline_review | 1590 |sentences with medium relevant probability (0.45–0.55) — needs human review, of which the results can be used to optimise the `PCA + LogisticRegression pipeline`.|

Visualized `extract.py`:

```
Parquet document
       ↓
[Lexical gate]  ─────────────→ (Logs all hits to semantic_filter_lexical_all.txt for SEAT-full)
       ↓ hit
Split into sentences (Regex hardened against Mr./Dr./Mrs. abbreviations)
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
2. **Preprocessing** — Sentence Preclassification (see also [[#furthering-question-3-complex-contexts|Furthering Question 3]]) → Strip noise (HTML, encoding artifacts) if there's any → (spaCy still kept as a scaffolding codebase) produce token-level annotations (lemma, POS, dependency relation).
3. **Feature extraction** — For each target token or small target span in each sentence, determine AgI/PI/SI primarily via Transformer-based **SRL**. Attitudinal indices (`negAttI` / `posAttI`) are now assigned via prototype-based local embedding matching, reusing the shared MiniLM encoder, rather than exact adjective/psych-verb lookups. Then proportionalize per group across the corpus.
4. **Association testing (WEAT + SEAT)** — Using frame attribute sets (F⁻, F⁺) discovered by **LLR / LogDice** and classified by annotators:
    * **WEAT** (static embeddings): type-level — is *immigrant* closer to F⁻ or F⁺ compared to *citizen*?
    * **SEAT** (contextualized sentence embeddings): token-level — averaging over each *occurrence* of *immigrant* in context, is it closer to F⁻ or F⁺? Now computed with MiniLM sentence embeddings rather than spaCy document vectors.
5. **EFI via PCA** — Assemble group × dimension matrix [AgI, PI, SI, netAttI, WEAT, SEAT]. Run PCA on groups with `N ≥ 10`. At the current stage, PCA should be treated as an exploratory dimensionality-reduction step rather than a finalized definition of "negative framing"; the interpretation of PC1 remains open for later discussion.
6. **Output** — Per-sentence table (targets, indexical counts) + per-group summary (proportionalized indices, WEAT/SEAT scores, EFI, PCA loadings, regression β), with the reported group table filtered to lemmas with `N ≥ 10`.

### Design decisions from early testing

Detailed rationales for pipeline architecture choices, alternatives rejected, and risk evaluations have been moved to the decisions log.
→ See [[decisions]]

## Explorables

### Changes

[[tracker#Important Changes]] → [[tracker#Pending Actions]]

### To-Dos

[[todo#Questions]] → [[todo#Broad Plans]]

### Active Logs

[[log]]

## Novelty Check

What makes ARMADA novel is the integration:

* discover frames bottom-up (LLR on pretraining data) → anchor WEAT/SEAT on those empirical discoveries → extract syntactic/semantic roles per demographic group (AgI/PI/SI via SRL) → combine into a composite index (EFI via PCA) → all on the same corpus, closing the causal chain.

**No existing work does all four steps together, on LLM pretraining data.**

The closest we'd need to cite as "prior work that does not integrate" are:

Most relevant near-miss: **Mendelsohn, Tsvetkov & Jurafsky (2020/2021)** develop a computational framework for dehumanization analysis — measuring negative evaluations, denial of agency, moral disgust, and vermin/animal metaphors using w2v cosine similarity to dehumanization concept clusters — applied as a case study to **LGBTQ discourse in *The New York Times* (1986–2015)**, not immigration. But: (a) they work on news media, not pretraining corpora; (b) their four dehumanization dimensions are predefined from social psychology theory, not empirically discovered per corpus; (c) they don't extract syntactic roles (AgI/PI/SI) at all. The 2025 follow-up (**"When People are Floods," Mendelsohn & Budak, ACL 2025**) applies a new combined word-level + document-level LLM technique to seven metaphor concepts (water, vermin, animal, parasite, physical pressure, commodity, war) in 400K US immigration tweets, studying ideology and engagement effects — still no role extraction, still predefined metaphor categories, still social media not pretraining data.

Partial overlap 1: **Bamman, O'Connor & Smith (2013) — "Learning Latent Personas of Film Characters"** extract agent verb / patient verb / attribute patterns per character from dependency parses (nsubj, dobj, nsubjpass) on 42,306 Wikipedia movie plot summaries — conceptually the ancestor of our AgI/PI. But: (a) encyclopedic narrative text / film domain; (b) Dirichlet latent variable models for persona clustering (not measurement); (c) no bias framing, no embeddings, no demographic group profiling; (d) purely syntactic extraction — we uses transformer-based SRL as the primary mechanism, capturing predicate-argument structure that surface dep-parse misses (e.g., control verbs, nominalized predicates, long-distance dependencies).

Partial overlap 2: ***Connotation Frames (Rashkin et al., 2016)*** provides a crowd-annotated lexicon of ~1,000 high-frequency transitive verbs from the NYT corpus, encoding agent/theme power, agency, and emotional affect per verb — conceptually adjacent to our SI (mental-state verbs) and AgI/PI distinction, and clearly citable.

Partial overlap 3: **BIAS Detection Framework (Kurpicz-Briki et al.) / IssueBench / DecodingTrust / StereoSet etc.** These are distinct in scope but share the same limitation. The BIAS Framework and IssueBench use predefined word/prompt lists to probe embedding-level or generation-level stereotype associations. 

**Methodological gap**: Even as model probes, they can only confirm pre-theorized stereotype categories, making them blind to emerging, subtle, or culturally situated framing biases. A model trained on text that systematically frames immigrants as grammatical patients (without ever using a slur) would score perfectly fine on StereoSet, even though the linguistic framing is clearly and subtly biased.

## Reading List

The complete literature review, categorized by tags (methodology, framing, benchmark) and sorted by relevance, has been moved to [[reading]]

## Side Notes

[[reading#Lexical Sources]]