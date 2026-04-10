---
title: 1. Design Decisions
description: Rationales for pipeline architecture choices, alternatives rejected, and risk evaluations.
tags:
  - design-decision
  - methodology
---


## Prior Knowledge

If our goal is method development—like the current bias-detection framework—then using prior knowledge encoded in a pretrained model to focus on relevant cases (e.g., sentences mentioning minority groups) is more defensible.

If one evaluates directly on raw corpora:

* The distribution of irrelevant senses will dominate.
* Set indices become diluted or skewed.
* The evaluation no longer reflects bias toward actual groups.
Raw inclusion clearly introduces a confound, not just variance.

We're not trying to infer the natural distribution of linguistic patterns, but only to ensure the model sees the cases that matter for the task.

The methodological goal, therefore, is not to eliminate priors, but to ensure they are:

* explicit,
* justified,
* not aligned with the hypothesis being tested.
* not making any descriptive generalization that applies to the complete dataset.

## LLR / LogDice instead of PPMI

“Are these two words systematically associated?”

~~Levy & Goldberg (2014) showed that word2vec skip-gram implicitly factorizes a shifted PMI matrix. WEAT on word embeddings therefore already encodes PMI-derived associations. The earlier `EFI = α·PPMI + (1-α)·WEAT` weighted a raw signal against a smoothed version of itself; the α had no theoretical grounding.~~

## WEAT vs. SEAT

Both are kept — they measure different granularities:

* **WEAT** (Word Embedding Association Test): one static vector per word type, trained on the corpus. Is *immigrant* as a type closer to F⁻ or F⁺ compared to *citizen*? Stable, interpretable, type-level.
* **SEAT** (Sentence Embedding Association Test): one vector per token occurrence (contextualized). Collects all sentences containing a target word, encodes each via a sentence encoder, and averages cosine similarities to F⁻/F⁺ attribute sentences. Sensitive to contextual variation, token-level.

Both produce per-group scores. SEAT captures context (e.g., *"immigrant workers contribute"* vs. *"immigrant workers were detained"* yield different vectors), while WEAT gives a single stable baseline per word.

Given this is a generalizable pipeline to probe how different datasets exhibit bias through a standard lens, using a standardized pre-trained encoder proves beneficial.

* No matter we run Dolma, Wikipedia, or Reddit text through the exact same MiniLM encoder, the encoder's priors are held constant.
Any difference in the resulting SEAT/WEAT scores can be attributed to the differences in the input sentences.
* It is much faster and more practical than training a custom word2vec model from scratch every time a user inputs a new dataset.

## EFI Architecture

EFI (Evaluative Framing Index) is no longer a single formula but a **per-group framing profile** extracting the most relevant information from the initial input. Each group is described by a vector of dimensions:

```
EFI(group) = [ AgI, PI, SI, netAttI, WEAT, SEAT ]
```

The composite scalar is currently derived via **PCA** (Principal Component Analysis) on the group × dimension matrix, but this should be treated as exploratory rather than settled. The **first principal component (PC1)** captures the axis of maximum cross-group variance — the dimension along which groups differ most. Whether that axis can be read straightforwardly as "negative framing" is left open for later discussion.

* **PC1 loadings** = the empirically observed weighting of the current dimensions in this sample.
* **PC1 scores** = a provisional scalar ranking along that empirical axis.
* **Variance explained** = if PC1 > 50%, a single dominant axis accounts for most cross-group variation; if low, framing is multidimensional (also a finding).

No arbitrary weights. The data determines what "evaluative framing" consists of.

The regression `WEAT/SEAT ~ AgI + PI + SI + AttI` remains a complementary analysis: it tests whether syntactic-semantic patterns predict embedding-level association specifically, whereas PCA currently serves as a descriptive diagnostic of covariance structure rather than a final scalar definition.

## SI is independent from AgI

A group can be 100% agent (AgI=1.0) but 0% subjective (SI=0) — portrayed as *doing* things but never *thinking or feeling*. This distinction matters: it separates "active agency" from "full autonomous personhood." Nuances that define "autonomous personhood" will be extensively furthered with embodied semantics.