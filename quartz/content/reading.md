---
title: 4. Reading List
description: Complete literature review, categorized by tags and sorted by relevance.
tags:
  - literature
  - literature/bias-detection
  - literature/methodology
  - literature/benchmark
---


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