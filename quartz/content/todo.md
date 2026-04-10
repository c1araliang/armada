---
title: 3. To-Dos
description: Research questions under investigation and broad plans for pipeline development.
tags:
  - methodology
  - pipeline
---

## Questions

~~**1. Does dependency parsing suffice for role extraction, or should I explore dedicated deep-learning SRL models?**~~
~~While spaCy currently suffices for role detection, specialized Transformer-based SRL models (e.g. via Huggingface) can be integrated later if the project requires higher semantic granularity.~~

**2. Can "embodied and enactive cognition" help refine the nuances and categorization AgI/PI/SI indices?**
For example: *"immigrants build"* vs *"immigrants arrive"* — differ clearly on the perpection end despite both being agentive descriptions. Action parsing and verb-type weighting could deeply enrich the analysis.

**3. How can we handle complex contexts and textual reversals?**

* **Contextual Reversals:** *"Communities that had once welcomed migrants began to restrict access..."* (PPMI captures *welcome+migrant* but ignores the pivot to *restrict*).
* **Defended Attacks:** *"People believe immigrants create problems, but the truth is..."* (High negative PPMI from *immigrants+problems*, clear PI and no SI for *immigrants*—yet the sentence defends them).

**Potential solutions:** ~~Instead of sentiment analysis, use spaCy to look for negation modifiers (`neg`) or adversarial conjunctions like *but*, *however*~~; or, employ frozen LLMs like `GPT-4` or `Llama-3` as annotators to see how often our metrics get it "wrong" compared to a context-aware LLM.  ([Furthering Question 3](#furthering-question-3-complex-contexts))

**4. MWEs**

`Korean immigrant`, `undocumented foreign nationals` exhibit same group repetition:
should we pick only 1 essential profilel?

This policy ensured that only `white Europeans` would be allowed to immigrate to the new country, while empowering the state to deport existing `non-white immigrants.`:
should we treat repeated targets as standalone instances, but for example, multiplied by a penalty coefficient, given such repetition reinforces stereotypical racial image.

`Asian American` demonstrates contrastive MWE:
should we pick only the minority profile?

**5. Minority Political Grouping**

`USSR`,`communist`,`conservatist` clearly present in data, encoded with certain opinion, discard or expand?

~~**6. Non-human objects**~~

~~`foreign language`, `foreign country`, also present with research value. But our framework is designed for human groups?~~
When a model is trained on "Chinese products are cheap", the contextualized vector for "Chinese" absorbs negative valence from "cheap" through distributional learning.
While static embeddings (like Word2Vec/WEAT) irreversibly crush all senses into a single vector (permanently fusing the contexts of "black market" and "black people"), contextual embeddings (like MiniLM/SEAT/LLMs) dynamically construct distinct vectors for different contexts. However, the *underlying parameters* (attention layers, feed-forward weights) generating these distinct vectors are still shared. This causes **valence bleed**: the negative distributional associations of non-human contexts (e.g., "black sheep", "white noise") inevitably drag the token's network parameters toward negative framing, structurally contaminating the model's baseline representation of the demographic group.

Question is, **by filtering out nonhuman objects as IRRELEVANT, are we missing a real source of bias in the LLM's representations?**

Tentatively solving via:

* **SEAT-filtered**: average MiniLM embeddings of sentences resolving to human entities (post-pipeline).
* **SEAT-full**: average MiniLM embeddings of *all* sentences matching the token (pre-filtering, raw from the lexical gate).
* **Δ-SEAT** (`SEAT-full` − `SEAT-filtered`): quantifies exactly the magnitude and direction of **associative contamination**.

If `Δ-SEAT` is large for a demographic term, it empirically proves that non-human usages in the broader English language (idioms, objects, abstracts) exert a structural drag that biases the model's representation of those people. First-pass Dolma extractions demonstrate exactly this: `white` and `black` showed extreme `Δ-SEAT` drift (+0.0443 and +0.0252 toward F⁻ respectively) driven by generic semantic contamination, while unambiguous terms like `refugee` showed virtually none (-0.0006).

## Broad Plans

### Scale through training-data levels

LLM training has three canonical stages, each with distinct data — our framework should generalize across all of them as long as the extraction skeleton holds:

| Level | Data type | Example open dataset | What bias looks like here |
|-|-|-|-|
| **1. Pretraining** | Raw web/book text (trillions of tokens) | **Dolma** [(*v1_6-sample, 16.4GB=10 billion tokens*)](https://huggingface.co/datasets/devingulliver/dolma-v1_6-sample) with a toolkit for curating datasets for language modeling | Distributional: co-occurrence patterns, metaphorical framing |
| **2. Instruction Tuning (SFT)** | (instruction, input, output) triples | **Stanford Alpaca 52k** (52k demos generated via self-instruct from text-davinci-003) | Instructional: biased task completions, stereotyped examples |
| **3. RLHF / Preference** | Human-ranked response pairs | Anthropic HH-RLHF, UltraFeedback | Evaluative: which "preferred" answers encode bias |

Start from Level 1 (basic pretraining data) with a manageable subset and scale up.

### Furthering Question 3 (complex contexts)

This expands on [[#pipeline-steps-updated|Preprocessing (Step 3)]]. When surface-level statistics contradicts the actual rhetorical intent of the sentence (contextual reversals, defended attacks), three complementary approaches can help:

| | Heuristic filter | Sentence-level SEAT | Argumentation mining |
|-|-|-|-|
| **Mechanism** | Rule-based: scan for `neg`, `but`, `however` + target-frame co-occurrence | Embedding-based: whole-sentence cosine distance to "negative/defensive framing" prototype | ML classifier: detect claim/attack/support rhetorical structure |
| **What it catches** | Explicit pivot markers and simple negations | Implicit semantic orientation of the full sentence | Rhetorical structure regardless of surface markers |
| **Misses** | Subtle, marker-free pivots | Fine-grained argumentative direction (attack vs. concession) | Anything outside training distribution; requires labeled data |
| **Cost** | Near-zero (extending spaCy pipeline) | Low (one sentence-encoder forward pass; extends WEAT → SEAT already in pipeline) | Higher (specialized BERT/LSTM finetuned on argumentation corpora) |

**Possible approach:** Heuristics + SEAT as a cheap joint **filter** (syntax + semantics); argumentation mining as a suggested **flagger** on sentences that slip through. Run this pre-classification on the corpus *before* the main pipeline to categorize and extract outlier sentences for targeted inspection, reducing noise in aggregate metrics.

* Pre-existing biases at the embedding/vector level (e.g., via WEAT) are worth keeping in mind — even if not our direct focus, they mean subtle biases are structurally inevitable and set a floor for what corpus-level intervention alone can fix.