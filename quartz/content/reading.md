---
title: 4. Reading List
description: Complete literature review, categorized by tags and sorted by relevance.
tags:
  - literature
  - literature/bias-detection
  - literature/methodology
  - literature/benchmark
---

## General Interest

WildChat (Allen AI) + ShareChat (2512.17843) Both are large-scale corpora of real user-LLM conversations. The "realness" of human input doesn't outweigh its structural messiness. Employing incomplete conversational data risks breaking existing extraction heuristics. As Level 2 data, structured Alpaca is significantly more suitable (or syntactically tractable) in terms of data compatibility.

"Don't Erase, Inform!" (2505.24538) AI tool to detect offensive terms in cultural heritage metadata. Their detection is lexical, i.e., flagging known offensive terms from a "multilingual vocabulary co-created with marginalized communities, researchers..." After checking the performance of their online parser and the DE-BIAS vocabulary, it can be cross-referenced but is more orthogonal to our objective, i.e., contemporary metaphorical biased frame detection, given the cultural heritage context.

"Decoding Hate" (2410.00775) Qualitative analysis of how 7 LLMs respond to hate speech inputs, including politically correct hate speech. They study post-training output behaviour, with a focus on safety guardrails and alignment behavior. Not relevant to systematic demographic framing patterns.

"ChatGPT-generated texts show authorship traits" (2508.16385) ChatGPT systematically prefers nouns over verbs across registers, showing a distinct "linguistic backbone" from humans who anchor in tense/aspect/mood. Dentella et al. is a worthwhile (citable) note-to-self that verb-reliant extraction yield may vary across levels, but the effect is attenuated, given we are doing SRL.

"Emergent social conventions and collective bias" (2410.08948) LLM agent populations spontaneously develop shared conventions, and collective biases emerge even when individual agents are tested as unbiased. Yet the bias noted is a property of the interaction process, not reducible to individual-agent testing. Ashery et al.(2025) motivationally strengthen our argument: training-data-level bias detection matters because collective deployment of LLMs can resurrect or amplify biases that were mitigated on an individual level.

"Just Ask for Calibration" (2305.14975) RLHF distorts token-level probability calibration; verbalized confidence is better calibrated. Might be relevant as an implementation detail for a future validation step: if we use LLM-as-judge, its confidence scores should come from verbalized output, not logprobs.

DecodingTrust (NeurIPS 2023) Comprehensive trustworthiness benchmark for GPT-3.5/GPT-4 output across toxicity, stereotype bias, robustness, privacy, and fairness. The stereotype bias dimension, which tests whether models systematically produce stereotyped associations with demographic groups (e.g., linking "Muslim" to "terrorism" in generated text), mirrors our concerns, serving as evidence for the bias-input-propagate-output causal chain.

"Towards an Ontology-Driven Approach to Document Bias" (Russo & Vidal, JAIR 2025) Develops Doc-BiasO, an ontology providing a machine-readable vocabulary for documenting bias throughout the ML lifecycle: formal categories of bias types, metric definitions (indicators vs. measures), and FAIR-aligned terminology reusing PROV-O, SKOS, MLS, DCAT. Their goal is documentation; ours is measurement. Pipeline-irrelevant (we discover framing bottom-up via LLR, not top-down from an ontology), but their vocabulary/categorization is absorbable: Doc-BiasO's taxonomy of bias types and standardized metric naming could be adopted in ARMADA's reporting/output layer to make our EFI dimensions interoperable with the broader Trustworthy AI literature, without changing the extraction pipeline itself.

"IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance" (2502.08395) Introduces 2.49M prompts to measure issue bias (e.g., favoring specific political perspectives) in LLM writing, finding persistent biases across 10 SOTA models. Backs our motivating argument: it provides fresh, large-scale evidence that training-derived biases consistently restrict the diversity of perspectives in LLM outputs, reinforcing the need for dataset-level diagnostics like ARMADA.

"The BIAS Detection Framework" (2407.18689) Kurpicz-Briki et al. implement WEAT, SEAT, LPBS, and CrowS-Pairs for bias detection in word embeddings and language models across multiple European languages. Shares our WEAT+SEAT tooling but differs fundamentally: they test embedding-level stereotype associations in pre-trained models using artificial word lists, whereas we measure corpus-level framing by running SRL to extract syntactic/semantic roles (AgI/PI/SI) and empirically discovering collocates (LLR) directly on the training data, then using those empirical discoveries to anchor our WEAT/SEAT. Methodologically relevant as a benchmark reference for our Python WEAT/SEAT implementation.

"Detecting Bias and Intersectional Bias in Italian Word Embeddings and Language Models" (GeBNLP 2025) Puttick & Kurpicz-Briki extend WEAT/SEAT/LPBS to Italian, introducing GG-FISE for intersectional bias while accounting for grammatical gender. Their intersectionality angle (gender × ethnicity) is driven by Italian morphological gender, which guarantees grammatically intertwined bias. This is conceptually interesting and citable when we extend to handle compound above-noted demographic intersections/repetitions, i.e., ethnicity × nationality like 'Korean American' or 'White American', via span-aware resolution rather than morphological parsing.

"Bias Score: Estimating Gender Bias in Sentence Representations" (Azzalini, Dolci & Tanelli, 2022; 2023) Proposes a sentence-level bias metric: a normalized sum of per-word cosine similarities to a "gender direction" in embedding space, distinguishing neutral gender information from stereotypical bias. Moderately relevant: their sentence-level granularity parallels our SEAT computation, and their word-importance weighting (scoring the whole sentence by weighting individual words vs a target concept) could inform a future refinement of multiple indices, potentially refining the context token weighting in the SEAT computation.

## Lexical Sources

Best source: NRC Emotion Lexicon (~14k entries), filtered to verbs only (cross-check POS via spaCy vocab), then frequency-filtered against Dolma sample. Produces a linguistically grounded, corpus-attested lexicon. VerbNet (ancient, messy categories), WordNet (frozen, awkward syntax), and MECORE predicate database (48 predicates, theory-curated for cross-linguistic representativeness, not corpus coverage) are supplementary references but insufficient as primary sources.

## Existing Studies on Bias in Pretraining Data

| Study | Focus | Method | Corpus |
|---|---|---|---|
| [Feng et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.656/) | Political bias propagation from pretraining into downstream unfairness | Trace political leaning along social/economic axes from corpus → LM → task outputs | Diverse pretraining corpora (Best Paper Award) |
| [Kadan et al. (NLP Journal 2024)](https://arxiv.org/abs/2301.09003) | Affective bias: skewed emotion–demographic associations | Corpus-level affective word distribution analysis + model-level class/intensity evaluation | Large-scale pretraining + fine-tuning corpora |
| [Itzhak et al. (COLM 2025)](https://openreview.net/forum?id=KQhUEoPmJy) | Where cognitive biases originate: pretraining vs. finetuning | Cross-tuning: swap instruction datasets between models with different pretraining histories to isolate source | Spotlight paper; instruction-tuned LLMs |
| [Udagawa et al. (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.2/) | Social bias in Common Crawl via protected-attribute regard | Protected-attribute detection + regard (polarity) classification; simple mitigation by balancing regard | Common Crawl |
| [Görge et al. (arXiv 2512.10734)](https://arxiv.org/abs/2512.10734) | Representation bias + explicit stereotypes across protected groups | LLM-generated word lists → Demographic Representation Score → sociolinguistic stereotype filter → counterfactual data augmentation | Textual pretraining data |
| [Mendelsohn & Budak (ACL 2025)](https://aclanthology.org/2025.acl-long.398/) | Dehumanizing metaphors in immigration discourse | Word-level + document-level LLM technique for seven metaphor concepts (water, vermin, animal, etc.); ideology × engagement analysis | 400K US immigration tweets |
| [Entity Framing (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.17/) | Entity role portrayal in news (protagonist / antagonist / innocent) | LLM zero-shot + fine-tuned XLM-R for 22 fine-grained narrative archetypes in multilingual news | 1,378 news articles, 5 languages |

### Shared Focus

These works collectively establish that bias is present in pretraining data, that it propagates through model training into downstream behavior, and that it can be partially traced or mitigated at the data level. Itzhak et al. in particular make the causal case that pretraining—not alignment or finetuning—is the primary source, directly motivating upstream intervention like F3BF. IssueBench (2502.08395) corroborates this at scale: persistent perspective biases in 10 SOTA models trace back to training-data-level asymmetries.

### Main Gap

Existing work converges on a shared methodological ceiling: it measures *outcomes* of bias (polarity, regard, political leaning, emotion association, metaphor frequency) using **predefined category systems**—attribute word lists, stereotype templates, StereoSet/DecodingTrust probes, annotated metaphor concepts. Even the most granular approaches (Udagawa's regard classification, Mendelsohn's metaphor tagging) assume the frame shapes in advance and then confirm them.

This makes them blind to emerging, subtle, or culturally situated framing that never uses flagged vocabulary. A corpus that systematically casts immigrant groups as grammatical patients—without a single slur, without any of the seven dehumanization metaphors—scores clean on all existing benchmarks. The linguistic framing is nonetheless real and measurable.

No existing study jointly models:
- **Bottom-up frame discovery** from corpus statistics (LLR / LogDice collocate profiling, without pre-specified categories),
- **Syntactic-semantic role extraction** (AgI / PI / SI via transformer-based SRL, capturing control verbs, nominalized predicates, and long-distance dependencies that surface dependency parsing misses),
- **Contextualized association testing anchored to empirically discovered frames** (WEAT at type level + SEAT at token/occurrence level, both using the same empirical F⁻/F⁺ attribute sets derived from collocate classification),
- and **integrated composite group profiling** (EFI via PCA on a group × dimension matrix, letting the data determine what "evaluative framing" structurally consists of) —

all applied to the **same LLM pretraining corpus** as one closed causal chain from surface pattern to representation.

### Why F3BF is Unique?

**F3BF is mechanism-centered, not outcome-centered.** It does not ask *is there bias?* — existing work has answered that. It asks *which linguistic mechanisms carry it, and how do they interact?* Four properties set it apart:

1. **Bottom-up frame discovery.** Frames emerge from LLR / LogDice collocation statistics on the corpus itself, then get classified by expert annotators. No predefined stereotype categories are imposed. This follows Sinclair's corpus linguistics methodology: observe → classify, rather than predict → confirm.

2. **Role granularity via SRL.** AgI, PI, and SI are not equivalent: a group can show high AgI (doing things) yet zero SI (never thinking or feeling), which surface-level sentiment or polarity scores cannot disentangle. Transformer SRL captures predicate-argument structure across control verbs and nominalized predicates that dependency-parse-only approaches miss.

3. **Closed causal chain.** Surface collocation patterns → frame attribute sets → WEAT/SEAT association scores → syntactic-semantic role indices → composite EFI — all measured on the same corpus, with the same encoder held constant across runs. Any cross-corpus difference in scores is attributable to the input data, not the model priors.

4. **Pretraining scope.** All comparison studies either work on social media, news, or model outputs. F3BF targets the upstream training source directly — Dolma v1.6 as a representative large-scale pretraining corpus — which is where the bias enters before any alignment or fine-tuning can address it.
