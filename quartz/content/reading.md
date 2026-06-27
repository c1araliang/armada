---
title: 4. Reading List
description: Complete literature review, categorized by tags and sorted by relevance.
tags:
  - literature

---


## Scope

This report focuses on **how linguistic mechanisms propagate bias from training data into representations and contextual effects**.

The central question is:

> **How can we detect, measure, and validate the linguistic mechanisms that propagate bias in LLM, from surface patterns to representations to contextual effects?**

RepE is central because it gives the strongest account in this set of **why vector operations on hidden states are meaningful**, while also showing what representation-level intervention cannot explain on its own.

---

## 1. Research Problem

The question is not just whether a model is biased. It is:

- which **linguistic mechanisms** carry that bias,
- how those mechanisms can be **detected and measured**,
- and how they can be **validated across levels**, from corpus patterns to model representations to contextual effects.

The mechanisms I care about are things like:

- collocational patterns,
- discourse patterns (metaphorical, attitudinal, narrative, etc.),
- agency/patienthood/subjectivity,
- distributional semantic association / semantic clustering.

Three levels are relevant:

1. **Surface level**: linguistic patterns evident via corpus analysis 
2. **Representation level**: do those patterns correspond to stable directions or structures in model representations?
3. **Contextual/behavioral level**: do those structures affect how the model interprets or evaluates content in **demographically specific contexts**?

## 2. Holistic SOTA Relevance Report

The literature is organized by **where bias is studied** and **which part of the pipeline is targeted**.

### 2.1 Studies of bias informing LLM-bias

Foundational work on bias in language and pretraining corpus ethics, forming the conceptual basis for analyzing representational framing. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Said (1978)](https://www.penguinrandomhouse.com/books/168494/orientalism-by-edward-w-said/) *Orientalism* | Postcolonial cultural theory | Foundational critique of Western discursive construction of the "Orient"; informs the postcolonial framing of minority/dominant asymmetry in my lexicons | **Theory** | `postcolonial` `lexicon-foundation` |
| 2 | [Dowty (1991)](https://doi.org/10.2307/415037) Thematic Proto-Roles | Formal semantics | Replaces discrete thematic roles with gradient Proto-Agent / Proto-Patient entailment clusters; grounding for AgI and PI dimensions | **Theory** | `role-extraction` `formal-semantics` |
| 3 | [Hall (1992/2018)](https://www.politybooks.com/bookdetail?isbn=9780745609607) "The West and the Rest" | Postcolonial cultural theory | Discourse and power analysis of how "the West" was constructed via binary opposition with "the Rest"; conceptual foundation for the minority/dominant typology in my lexicons | **Theory** | `postcolonial` `lexicon-foundation` |
| 4 | [Quijano (2000)](https://muse.jhu.edu/article/23906) Coloniality of Power | Latin American studies | Coloniality of power / Eurocentrism as a structuring global axis; motivates the postcolonial frame for the target/contrast group partition | **Theory** | `postcolonial` `lexicon-foundation` |
| 5 | [Sinclair (2004)](https://www.routledge.com/Trust-the-Text-Language-Corpus-and-Discourse/Sinclair/p/book/9780415317689) *Trust the Text* | Language corpora | Foundational corpus-driven methodology: meaning emerges from observed textual patterns rather than intuition; basis for bottom-up, minimal-prior frame discovery | **Methodology** | `corpus-linguistic` `bottom-up` |
| 6 | [Bamman, O'Connor & Smith (2013)](https://aclanthology.org/P13-1035/) Latent Personas | 42,306 Wikipedia movie plot summaries | Agent verb / patient verb / attribute patterns from dependency parses (nsubj, dobj, nsubjpass); Dirichlet persona clustering | **Analysis** | `role-extraction` `corpus-linguistic` |
| 7 | [Rashkin, Singh & Choi (2016)](https://doi.org/10.18653/v1/P16-1030) Connotation Frames | *New York Times* corpus | Crowd-annotated lexicon of ~1,000 transitive verbs encoding agent/theme power, agency, and emotional affect per verb | **Analysis** | `role-extraction` `corpus-linguistic` `verb-level` |
| 8 | [Sap et al. (2017)](https://aclanthology.org/D17-1247/) Connotation Frames of Power & Agency | ~2,000 film screenplays | Extends Rashkin's connotation-frame lexicon to power and agency dimensions; crowd-sourced annotations for ~2,000 verbs; applied to gender bias in films | **Analysis** | `role-extraction` `corpus-linguistic` `verb-level` |
| 9 | [Mendelsohn, Tsvetkov & Jurafsky (2020)](https://doi.org/10.3389/frai.2020.00055) Dehumanization Framework | *New York Times* (1986–2015), LGBTQ discourse | w2v cosine similarity to four theory-derived dehumanization concept clusters; denial of agency measured alongside metaphor | **Analysis** | `metaphor-framing` `predefined-categories` `demographic-profiling` |
| 10 | [Bender et al. (2021)](https://doi.org/10.1145/3442188.3445922) Stochastic Parrots | Large language models (general) | Risks of ever-larger LLMs: environmental cost, encoded societal biases from uncurated training data, and the illusion of meaning from statistical pattern matching; LM lacking actual communicative intent, let alone experiencee-based world or pragmatic percetion. | **Analysis** | `llm-risks` `training-data` |
| 11 | [Mendelsohn & Budak (ACL 2025)](https://aclanthology.org/2025.acl-long.398/) "When People are Floods" | 400K US immigration tweets | Word-level + document-level LLM technique for seven predefined metaphor concepts; ideology × engagement analysis | **Analysis** | `metaphor-framing` `predefined-categories` `demographic-profiling` |


- `role-extraction` (#6–8): Rashkin (#7) is the closest overlap with my role indices and attitudinal dimensions; Sap et al. (2017) (#8) extends the same verb lexicon to power and agency entailments, directly motivating the AgI dimension; Bamman (#6) is the syntactic ancestor of AgI/PI but has no bias analysis or embeddings.
- Mendelsohn 2020 (#9) and Mendelsohn & Budak 2025 (#11) form a `metaphor-framing` pair: #9 studies news/LGBTQ discourse via w2v dehumanization clusters; #11 studies immigration tweets using LLMs as measurement tool. Both are corpus-level analyses of human discourse bias; neither studies LLMs as the object.
- `corpus-linguistic` methodology (#2, #5): Sinclair (#5) grounds the bottom-up, corpus-driven approach; Dowty (#2) provides the thematic proto-role theory for AgI/PI. Collocate statistics (Dunning LLR, Rychlý LogDice) and association metrics (WEAT, CEAT) are listed in §2.6 Tools & Methods.
- `postcolonial` (#1, #3, #4): Said (#1), Hall (#3), and Quijano (#4) are the theoretical anchors for the minority/dominant lexicon typology; they are listed here as foundational sources, not bias-detection methods.
- `corpus-ethics` foundation (#10): Bender et al. (2021) (#10) is the seminal critique of training LLMs on massive uncurated corpora, establishing the core ethical motivation for my focus on pretraining data analysis.
- None integrates role extraction with association testing or collocate-driven frame discovery.

### 2.2 Studies of bias in LLM

#### 2.2.1 Model Internal

Bias probed in contextualized embeddings or model-side association structure rather than generated text. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Ethayarajh (2019)](https://aclanthology.org/D19-1006/) How Contextual are Contextualized Representations? | BERT, ELMo, GPT-2 geometry | Shows embeddings occupy a narrow cone (anisotropy); contextual variance in upper layers but not fully separable by sense; motivates why CEAT/WEAT cannot cleanly isolate human-referent from non-human-referent demonym senses | **Analysis** | `representation-mechanism` `polysemy` |
| 2 | [Haber & Poesio (2021)](https://aclanthology.org/2021.findings-emnlp.226/) Patterns of Polysemy | BERT-Large on graded word-sense similarity | Contextual models correlate with human sense-similarity judgments but consistently fail on certain polysemic alternations; empirical grounding for why demonym polysemy (human vs. institution) is not cleanly separated | **Analysis** | `polysemy` `representation-mechanism` |
| 3 | [Buijtelaar & Pezzelle (EACL 2023)](https://aclanthology.org/2023.eacl-main.163/) Psycholinguistic Analysis of BERT's Compounds | BERT, noun compounds | Higher transformer layers best represent compound meaning; moderate alignment with human lexeme-meaning-dominance judgments; constituent meaning does not average uniformly — the head contributes disproportionately | **Analysis** | `representation-mechanism` `compound-semantics` |
| 4 | [Zaitova et al. (Findings NAACL 2025)](https://aclanthology.org/2025.findings-naacl.228/) Attention on MWEs | BERT-based models, six Indo-European languages | Fine-tuning shifts attention over multiword expressions non-uniformly by task (semantic vs. syntactic); reveals that constituent-level attention is idiomaticity-dependent, motivating my explicit compound-resolution rules | **Analysis** | `representation-mechanism` `compound-semantics` |

- WEAT and CEAT association testing methods are moved to §2.6 Tools & Methods.
- Ethayarajh (#1) and Haber & Poesio (#2) are cited in short paper Limitations (§Realist vs Instrumentalist): contextual encoders do not cleanly separate polysemic alternations along a stable axis, which motivates Δ-CEAT as a contamination diagnostic.
- Zaitova (#4) and Buijtelaar & Pezzelle (#3) probe how BERT-based models internally process compound and multiword expressions; the findings ground §Compound/MWE Group Mentions limitation: attention over compound constituents is non-uniform and meaning concentrates in deeper layers, so asymmetric suppression of dominant-side constituents is an approximation without guaranteed encoder-side support.
- This subsection is model-side but not output-side: it captures representational structure rather than generated judgments or behaviors.

#### 2.2.1.1 Animacy Conflation in Geometric Space

Geometric work on how models fail to separate human-referent from non-human-referent senses within a single token. This block deepens the Ethayarajh (#1) and Haber & Poesio (#2) anchors above: those establish that contextual encoders cannot cleanly separate polysemic senses; this block explains *why* and documents the downstream output effect. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Klafka & Ettinger (ACL 2020)](https://aclanthology.org/2020.acl-main.434/) Spying on Your Neighbors | BERT, ELMo, GPT; SVO transitive sentence probing | Suite of probing tasks on transitive sentences; tests recoverability of subject/object animacy, number, gender, and verb features from every token position | **Analysis** | `representation-mechanism` `probing` `animacy` |
| 2 | [Wdowicz (2025)](https://doi.org/10.1007/s43681-025-00898-z) Not a Mirror, a Caricature | GPT models prompted with American / Polish / Japanese identity cues | Self-Construal Scale (30 items); models produce fully separated, near-zero within-culture variance profiles — caricature rather than mirror; anglocentric default in 75% of neutral conditions | **Analysis** | `representation-mechanism` `demographic-profiling` |

- Klafka & Ettinger (#1) show that animacy is recoverable with near-perfect accuracy from verb and object token embeddings — *more* robustly than from the subject token itself (especially in GPT). The finding is distributional spread, not geometric disentanglement: animacy is present and non-locally encoded across all token positions, which is the probing-level pre-condition for why CEAT cannot isolate human-referent from non-human-referent contexts for a demonym appearing in both. Note: the paper frames this as a positive result (features *are* recoverable), not as a failure of disentanglement — the failure framing belongs to Ethayarajh (#1) and Haber & Poesio (#2) in §2.2.1.
- Wdowicz (#2) documents the *output* of animacy conflation: when human identity is underspecified, models collapse to a high-resource dominant-culture prototype, which is the LLM-side mirror of the corpus-level frequency collapse I measures via CEAT-full and Δ-CEAT.

#### 2.2.1.2 Identity Conflation at Intersectionality

Studies on how models fail to maintain distinct representational space for individuals whose identity intersects multiple minority axes. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Ghai, Hoque & Mueller (CHI 2021)](https://doi.org/10.1145/3411763.3451587) WordBias | Static word embeddings | Relative Norm Difference across multiple demographic axes (race × gender × religion); parallel-coordinate visualization of intersectional bias distribution | **Detection** | `intersectional` `visualization` `polysemy` |
| 2 | [Dev et al. (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.150/) Harms of Gender Exclusivity | GloVe, BERT; NER, coreference, MT tasks | Survey of non-binary individuals + quantitative analysis of pronoun frequency disparities; cyclical erasure mechanism: binary-skewed corpora → binary models → perpetuated erasure | **Analysis** | `intersectional` `non-binary` `representation-mechanism` |
| 3 | [Khan et al. (ACL 2025)](https://openreview.net/forum?id=zOw2it5Ni6) WinoIdentity | 5 LLMs, WinoBias extended with 25 demographic markers across 10 attributes | Coreference Confidence Disparity metric; confidence drops up to 40% for doubly-disadvantaged intersectional identities in anti-stereotypical settings; high performance driven by stereotype memorization | **Detection** | `intersectional` `benchmark` `coreference` |
| 4 | [Pawar et al. (EMNLP 2025 Findings)](https://aclanthology.org/2025.findings-emnlp.1207/) Presumed Cultural Identity | Aya, DeepSeek, Llama, Mistral-Nemo, GPT-4o-mini; suggestion-seeking prompts | Name-as-proxy analysis: models adapt responses based on presumed cultural identity from name alone; leads to misattribution and identity flattening toward dominant high-resource cultural defaults | **Analysis** | `intersectional` `demographic-profiling` `representation-mechanism` |
| 5 | [Shieh et al. (2026)](https://doi.org/10.1038/s41467-025-68004-9) Intersectional Biases in Narratives | Open-ended generation across GPT-family models | Omission, subordination, and stereotyping patterns for minoritized intersectional subgroups in implicit (non-identity-prompted) narratives; moves beyond explicit-prompt evaluation paradigm | **Analysis** | `intersectional` `output-behavior` `emergent-bias` |

- Ghai (#1) provides geometric visualization evidence that intersectional bias in static embeddings is non-additive — the distribution is not the sum of each component's bias — which is the embedding-space correlate of CEAT's emergent intersectional bias finding (now listed in §2.6).
- Dev (#2) establishes the cyclical erasure mechanism for non-binary identities: the same binary-forcing logic applies to demonym polysemy (human vs. inanimate use), where the model lacks geometric capacity to separate senses for low-frequency groups.
- Khan (#3) provides a quantitative signature of the problem in coreference: confidence disparity as high as 40% at intersectional identity cells, driven by stereotype memorization rather than reasoning — directly relevant to  finding that CEAT scores for compound demonyms (`Asian American`, `undocumented immigrant`) may reflect memorized collocational patterns rather than stable representational structure.
- Pawar (#4) and Wdowicz (§2.2.1.1 #1) form a complementary pair: Pawar shows name-triggered cultural flattening in prompted tasks; Wdowicz shows it in identity-cued generation. Both converge on the same corpus-level explanation: high-resource dominant-culture training signal overwhelms low-frequency minority signal at the representational level.
- Shieh (#5) is citable as evidence that bias persists even when explicit demographic labels are absent — strengthening the motivation for corpus-level upstream analysis rather than output-level filtering.

#### 2.2.2 Output

Bias probed in generated judgments, benchmark responses, or collective output behavior. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Nangia et al. (2020)](https://aclanthology.org/2020.emnlp-main.154/) CrowS-Pairs | Masked language model outputs | 1,508 minimal-pair sentences across 9 bias categories; measures preference for stereotypical completions | **Detection** | `benchmark` `predefined-categories` |
| 2 | [Nadeem, Bethke & Reddy (2021)](https://aclanthology.org/2021.acl-long.416/) StereoSet | BERT, GPT-2, RoBERTa, XLNet outputs | Context Association Test (CAT) measuring stereotypical bias across gender, profession, race, and religion while tracking language modelling ability via ICAT score | **Detection** | `benchmark` `predefined-categories` |
| 3 | [Ouyang et al. (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) InstructGPT | GPT-3 with RLHF | Foundational RLHF pipeline (SFT → reward model → PPO); establishes the alignment procedure whose covert-bias limits are probed by Hofmann and RepE | **Analysis** | `alignment` `implementation-detail` |
| 4 | [Wang et al. (NeurIPS 2023)](https://decodingtrust.github.io/) DecodingTrust | GPT-3.5 / GPT-4 outputs | Trustworthiness benchmark across toxicity, stereotypes, fairness, privacy, robustness; large-scale audit of output-side bias and safety failures | **Detection** | `benchmark` `predefined-categories` |
| 5 | [Hofmann et al. (2024)](https://doi.org/10.1038/s41586-024-07856-5) Dialect Prejudice | LLM outputs (GPT, etc.) | LLMs covertly assign negative stereotypes (character, employability, criminality) to African American English speakers; RLHF conceals overt bias but intensifies covert bias | **Detection** | `contextual-effects` `audit` `alignment` |
| 6 | [Germani & Spitale (2025)](https://doi.org/10.1126/sciadv.adz2924) Source Framing | LLM evaluative judgments | Source attribution audit: identical content, swapped demographic labels; measures evaluation shift in context | **Detection / analysis** | `contextual-effects` `audit` |
| 7 | [IssueBench (2025)](https://arxiv.org/abs/2502.08395) | 10 SOTA LLM outputs | 2.49M prompts probing perspective bias in writing assistance; large-scale evidence that issue bias persists through alignment | **Detection** | `benchmark` `predefined-categories` |
| 8 | [Ashery et al. (2025)](https://arxiv.org/abs/2410.08948) Emergent Collective Bias | Multi-agent LLM populations | Individually-unbiased agents still produce biased collective behavior; motivates upstream data-level intervention | **Analysis** | `emergent-bias` `multi-agent` |
| 9 | [Dentella et al. (2025)](https://arxiv.org/abs/2508.16385) "ChatGPT-generated texts show authorship traits" | ChatGPT outputs across registers | ChatGPT systematically prefers nouns over verbs, showing a distinct "linguistic backbone" from humans who anchor in tense/aspect/mood; citable caveat that verb-reliant extraction yield may vary, attenuated by SRL | **Analysis** | `linguistic-structure` `generation-pattern` |
| 10 | ["Decoding Hate" (2025)](https://arxiv.org/abs/2410.00775) | 7 LLM outputs on hate speech inputs | Qualitative analysis of LLM responses to hate speech including politically correct hate speech; focus on safety guardrails and alignment behavior, not systematic demographic framing | **Analysis** | `output-behavior` `alignment` |

- Source Framing (#6) shows contextual effects without decomposing the linguistic mechanism.
- InstructGPT (#3) and Hofmann (#5) are cited together in `cache.tex` for the dual alignment findings: while RLHF reduces overt toxicity, covert representational and dialect biases persist or even intensify after alignment. InstructGPT (#3) establishes the RLHF pipeline showing toxicity/safety improvements alongside persistent bias, while Hofmann (#5) provides direct empirical evidence of covert dialect prejudice persisting post-alignment.
- Benchmarks (#1, #2, #4, #7): StereoSet (#2) and CrowS-Pairs (#1) are the predefined-inventory benchmarks whose construct validity is questioned by Blodgett et al. (see §2.2.5); DecodingTrust (#4) and IssueBench (#7) confirm the problem at scale but remain output-side evaluations.
- Ashery (#8) motivates upstream intervention: individually debiased models can still produce collective bias.
- Dentella (#9) is a citable note that ChatGPT's noun-over-verb preference may affect verb-reliant extraction yield, though the effect is attenuated under SRL.
- "Decoding Hate" (#10) studies post-training output behavior, not systematic framing.

#### 2.2.3 Input

Bias analyzed in pretraining data, news, social media, or annotated narrative collections. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Kadan et al. (NLP Journal 2024)](https://arxiv.org/abs/2301.09003) | WikiEn, BookCorpus, WebText-250, C4-Val + SemEval-2018 EI-oc; BERT, GPT-2, XLNet, T5 | Corpus-level affective word distribution + model-level emotion–demographic association evaluation | **Analysis** | `pretraining-data` `predefined-categories` |
| 2 | [Mahmoud et al. (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.17/) Entity Framing | 1,378 news articles, 5 languages | LLM zero-shot + fine-tuned XLM-R for 22 narrative archetypes (protagonist / antagonist / innocent) | **Analysis** | `role-extraction` `multilingual` |
| 3 | [Udagawa et al. (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.2/) | Common Crawl (pretraining) | Protected-attribute detection + regard (polarity) classification; mitigation by balancing regard across groups | **Detection / mitigation** | `pretraining-data` `predefined-categories` `demographic-profiling` |
| 4 | [Görge et al. (2025)](https://arxiv.org/abs/2512.10734) | Small Heap corpus (OpenWebText2, CC-News, Wikipedia) + 3 LLMs (0.6B–8B) | LLM-generated word lists, Demographic Representation Score, stereotype filter, counterfactual augmentation; debiased fine-tuning did *not* consistently improve benchmark scores | **Detection / mitigation** | `pretraining-data` `predefined-categories` `mitigation` |

- Entity Framing (#2) assigns narrative roles (shared `role-extraction` tag with §2.1 #7–8) but uses a taxonomy-first approach and no association testing.
- `pretraining-data` (#1, #3, #4) measure polarity, representation counts, or affect distributions rather than collocate-grounded discourse structures.
- Görge et al. (#4) shows that data debiasing improves dataset-level bias measures, but does not consistently improve model-level bias benchmark performance; i.e., cleaning obvious demographic skew does not automatically make models fairer.
- Kadan et al. (#1) combines corpus-level emotion-word distribution and co-occurrence analysis with model-level emotion evaluation. It spans input *and* model-internal bias, but keeps them **correlational** — parallel measurements without corpus intervention. This is the gap Feng et al. and Itzhak et al. close in §2.2.4 (causal evidence) and that RepE closes at the representation level.

#### 2.2.4 All stages

Bias traced across pipeline stages or causally attributed to a specific stage. RepE leads because it shares the goal of mechanistic analysis and provides the theoretical grounding for vector-space operations. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Zou et al. (2023)](https://arxiv.org/abs/2310.01405) RepE | Internal hidden states + downstream generation | Concept directions via contrastive activation extraction; causal reading, steering, and intervention on representation-level bias; tested on Llama-2-13b-chat | **Analysis / mitigation** | `representation-mechanism` `causal-intervention` |
| 2 | [Feng et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.656/) Trails of PoliBias | Corpus → LM → downstream tasks | Political-compass probing of LMs + controlled continued pretraining on matched partisan corpora + downstream fine-tuning with per-group/source fairness comparison | **Analysis** | `causal-tracing` `pretraining-data` `predefined-categories` |
| 3 | [Itzhak et al. (COLM 2025)](https://openreview.net/forum?id=8sKcAWOf2D) Origins of Cognitive Biases | Pretraining vs. finetuning histories | Cross-tuning: swap instruction datasets between models with different pretraining backbones; models cluster by backbone — pretraining is the primary causal source | **Analysis** | `causal-tracing` `pretraining-data` |

- RepE (#1) proves that bias occupies causally active linear directions in hidden-state space, validating the geometric reasoning behind WEAT/SEAT, prototype matching, and PCA-based EFI. Its limit: it cannot identify which corpus framing patterns installed those directions.
- Feng (#2) and Itzhak (#3) share `causal-tracing` and together establish the causal case for upstream intervention: Feng traces the propagation path; Itzhak isolates the origin via controlled backbone swaps.

#### 2.2.5 Bonus

Relevant work that informs evaluation, reporting, or corpus selection but does not fit cleanly into the representation / output / input / all-stage taxonomy. Entries are sorted by year (ascending).

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Blodgett et al. (2021)](https://aclanthology.org/2021.acl-long.81/) Stereotyping Norwegian Salmon | StereoSet, CrowS-Pairs, WinoBias, Winogender | Measurement-modeling critique of four NLP bias benchmarks; demonstrates that none operationalizes a clearly defined construct of stereotyping; challenges construct validity of predefined-inventory benchmarks | **Critique** | `benchmark` `measurement-validity` |
| 2 | [Tian et al. (2023)](https://arxiv.org/abs/2305.14975) "Just Ask for Calibration" | RLHF-tuned LLM outputs | RLHF distorts token-level probability calibration; verbalized confidence is better calibrated; relevant as implementation detail if LLM-as-judge is used for validation | **Analysis** | `calibration` `implementation-detail` |
| 3 | [Soldaini et al. (ACL 2024)](https://aclanthology.org/2024.acl-long.840/) Dolma | Three-trillion-token English pretraining corpus | Open corpus (ODC-BY) underlying OLMo; Core data used as of June 2026; best-resource award ACL 2024 | **Corpus** | `corpus-resource` `pretraining-data` |
| 4 | [Russo & Vidal (JAIR 2025)](https://doi.org/10.1613/jair.1.19388) Doc-BiasO | ML lifecycle documentation | Ontology for bias types and metric definitions; absorbable for reporting layer, pipeline-irrelevant | **Documentation** | `ontology` `interoperability` |
| 5 | [WildChat (Allen AI) + ShareChat (2025)](https://arxiv.org/abs/2512.17843) | Large-scale real user–LLM conversations | Conversational corpora; structural messiness of incomplete conversational data risks breaking extraction heuristics; structured Alpaca significantly more suitable for syntactic tractability | **Corpus** | `corpus-resource` `assessed-rejected` |
| 6 | ["Don't Erase, Inform!" (2025)](https://arxiv.org/abs/2505.24538) | Cultural heritage metadata | Lexical flagging of offensive terms from a multilingual vocabulary co-created with marginalized communities; cross-referenceable for lexical coverage but orthogonal (cultural heritage context, not contemporary metaphorical framing) | **Detection** | `lexical-detection` `multilingual` |

- Blodgett (#1) critiques StereoSet and CrowS-Pairs (§2.2.2 #1–2) as lacking clearly defined constructs; directly supports moving away from predefined-inventory benchmark design.
- Tian (#2) matters only as an auxiliary evaluation note for any future LLM-as-judge validation step.
- Dolma (#3) is the source corpus; listed here because it is a resource paper, not a bias-detection method.
- Doc-BiasO (#4) is useful for metric naming and interoperability, not for the bias-detection pipeline itself.
- WildChat + ShareChat (#5) were assessed and rejected: the "realness" of user input does not outweigh structural messiness for the extraction pipeline.
- "Don't Erase, Inform!" (#6) is orthogonal in domain and its DE-BIAS vocabulary appears domain-restricted.

### 2.6 Tools & Methods

Implementation tools and statistical methods cited as components of the current pipeline. These are not bias-detection papers; they are listed separately to avoid inflating the substantive literature counts. Entries are sorted by year (ascending).

| # | Citation | Used for |
|:---|:---|:---|
| 1 | [Dunning (1993)](https://aclanthology.org/J93-1003/) Log-Likelihood Ratio — *Computational Linguistics* | LLR statistic for sparse collocation scoring in frame-term discovery |
| 2 | [Rychlý (2008)](https://www.sketchengine.eu/wp-content/uploads/2015/03/Lexicographer-Friendly_2008.pdf) LogDice — Corpus lexicography | LogDice association measure used alongside LLR for frame-term ranking |
| 3 | [Caliskan, Bryson & Narayanan (2017)](https://doi.org/10.1126/science.aal4230) WEAT | Type-level embedding association metric (WEAT) comparing group target embeddings with F⁻/F⁺ frame centroids in GTE ModernBERT space |
| 4 | [Shi & Lin (2019)](https://arxiv.org/abs/1904.05255) Simple BERT SRL | BERT-based SRL; the dannyshao fine-tuned variant used for PI patient-label detection |
| 5 | [Honnibal et al. (2020)](https://doi.org/10.5281/zenodo.1212303) spaCy | Phase 2 dependency parsing, NER, compound canonicalization, inanimate-head guard |
| 6 | [Guo & Caliskan (2021)](https://doi.org/10.1145/3461702.3462536) CEAT | Contextualized embedding association metric (CEAT / Δ-CEAT) evaluated over natural pretraining corpus contexts |
| 7 | [Azzalini, Dolci & Tanelli (2022)](https://www.semanticscholar.org/paper/Bias-Score%3A-Estimating-Gender-Bias-in-Sentence-Azzalini-Dolci/a8caf23b86b050ad217a05db6aac94396e73d37a) Bias Score | Sentence-level bias scoring, providing weighting analogy for context token relevance in embedding evaluation |
| 8 | [Zhang et al. (EMNLP 2024)](https://aclanthology.org/2024.emnlp-industry.103/) mGTE | GTE-ModernBERT encoder for frame-refresh, WEAT, CEAT, CEAT-full, and AttI prototype scoring |

### 2.3 Relevance gap

Across 11 foundational + 11 model-internal (4 core + 2 animacy-conflation + 5 identity-conflation) + 10 output + 4 input + 3 all-stage + 6 bonus + 8 tools works, three converging gaps identified:

1. **Predefined frames dominate bias-on-LLM work, but bottom-up discovery is not absent.** 
Most existing bias measurement assumes the frame shape in advance: fixed target–attribute word lists, polarity/regard/affect schemes, or predefined metaphor and narrative-role typologies. Bottom-up alternatives do exist: Bamman (§2.1 #6) clusters latent personas from dependency-parsed agent/patient verb patterns, and corpus-assisted discourse studies, e.g. Gabrielatos & Baker (2006) on UK refugee coverage, and the RASIM project, use LLR-based collocate analysis to surface frame structure from a corpus rather than impose it. The actual gap is integration: such bottom-up frame discovery is rarely connected to association testing on demographic groups, and almost never run on pretraining-scale corpora driving contemporary LLMs. We close that link: minimal attitudinal polarity seeds anchor LLR/LogDice surface on target/contrast groups in corpus (Dolma); candidate terms enter F⁻/F⁺ only if both empirically differential (via LLR) and semantically grounded (MiNiLM cosine-sim gate), accompanied by human review to ensure frame type alignment (which does require partial reruns concerning association testing and sequential EFI via PCA); the frame inventory updates along the pipeline rather than being fixed in advance. 
2. **No composite group profiling.** No study assembles syntactic-semantic role indices + association scores into a per-group multidimensional framing profile and derives a composite summary of cross-group variation through PCA.
3. Current pipeline produces diagnostic outputs consumable by pre-processing mitigation, and does not currently claim intra- or post-processing coverage. 

---

## RepE

> Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405. https://arxiv.org/abs/2310.01405

### A. Research problem addressed by RepE

RepE (tested primarily on **Llama-2-13b-chat**) asks the following question:

> **Can high-level concepts such as bias be detected and controlled as directions in model representations?**

The paper adopts a **Hopfieldian** view of neural cognition: representations are the fundamental unit of analysis, not individual neurons or circuits (the Sherringtonian view pursued by mechanistic interpretability). This justifies working with **population-level geometric operations** (PCA, cosine similarity) rather than tracing circuits.

RepE distinguishes two extraction targets: **concepts** (declarative knowledge, e.g. truthfulness, bias — elicited via "Consider the amount of [concept] in...") and **functions** (procedural knowledge, e.g. honesty, power-seeking — elicited via contrastive experimental/reference prompts). Bias is treated as a concept in §6.3, with the specific template *"Consider the bias in the following scenario..."*

### B. Sequenced Hierarchy of RepE

```text
RepE — sequenced flow
═══════════════════════════════════════════════════════════

1. READING  (offline, before inference)
   └─ LAT (Linear Artificial Tomography)
      ├─ design contrastive stimuli (concept-present vs absent)
      ├─ collect hidden-state activations per stimulus
      └─ fit linear model (default: PCA on activation
         differences → 1st principal component)
         → reading vector  [static, reusable]

2. CONTROL  (at inference — pick ONE controller path)
   ├─ path A: reading vector  [from step 1, static]
   │     OR
   ├─ path B: contrast vector  [computed on-the-fly from
   │          same input under two contrastive prompts;
   │          input-adaptive, dynamic]
   │
   └─ + choose operator ─┬─ linear combination (h' = h + αv)
                          ├─ piece-wise transform
                          └─ projection (erase direction)
      → explicit intervention on hidden states

3. LoRRA  (optional — modifies weight matrices, not hidden states)
   └─ in this paper: contrast vectors construct the loss function
      → train low-rank adapters on ATTENTION WEIGHTS to
        approximate that target (paper's scope; other weights
        and reading-vector supervision are not ruled out)
      → deploy WITHOUT vectors at inference; NO separate operator step
         (similar performance, negligible compute overhead)
   ⚠ vectors intervene on activations + require explicit operator (h' = h + αv);
     LoRRA bakes the effect into weights: (W + AB)x = Wx + ABx —
     the operator is now implicit; "controllers refer to low-rank
     weight matrices rather than vectors".
```

### C. Expanded Explanation

RepE assumes that many high-level concepts are encoded as approximately linear directions in hidden-state space. The underlying idea is not that "bias" is one point in space, but that it is an **axis of variation**.

**Step 1 — Reading via LAT.** 

Design contrastive stimuli that differ on the target concept (e.g., biased vs unbiased scenarios). Collect hidden-state activations for both conditions. Default linear model: **PCA on the difference** between concept-present and concept-absent activations; the **first principal component** is the **reading vector**. Alternatively, a logistic regression probe. The result is a fixed, reusable direction in representation space.

**Step 2 — Control.** Pick one controller path:

- **Reading vector** (from step 1): static, concept-general, works across inputs.
- **Contrast vector**: computed at inference from the same input under two contrastive prompts. Input-adaptive but requires paired prompting at runtime.

Reading vector and contrast vector are **alternative** ways to obtain a direction $v$ at **inference time** — choose one based on the compute budget; the contrast vector achieves SOTA performance but requires 3× more inference compute (paired prompting per input); the reading vector is static and cheaper.

Then apply an **operator** to intervene on hidden states. In the simplest case, **linear combination**:

$$
h' \leftarrow h + \alpha v
$$

where $h$ is the current hidden representation, $v$ is the controller direction, $\alpha$ is a tunable control-strength hyperparameter, and $h'$ is the edited representation. If $\alpha > 0$, the model is pushed toward the concept; if $\alpha < 0$, against it. Other operators: **piece-wise transform** (non-linear reshaping) and **projection** (erase the direction entirely).

**Step 3 — LoRRA** (optional).

LoRRA is **not** a third inference-time alternative at the same level. It is a **training-time compression** step: contrast vectors construct the loss function (reading vectors are optional input), and low-rank adapters on attention weights are trained to approximate what contrast-vector intervention would produce. 

Unlike vectors, which add to hidden states at inference via an explicit operator ($R' = R \pm v$), LoRRA modifies **weight matrices** directly — specifically attention weights. Zou introduces operators *after* defining all three controllers, meaning the operator framework formally covers LoRRA too, but for LoRRA the linear combination is implicit: $(W + AB)x = Wx + ABx$. The effect is the same in spirit — a directional push in representation space — but it is baked into the weights rather than applied at runtime.

Once trained, LoRRA is merged into the model — no vectors, no operator needed at inference — persistent weight-level adjustment with negligible compute overhead.

### D. Why RepE is convincing

RepE uses a four-step causal hierarchy:

1. **reading predicts**: the direction predicts concept presence.
2. **control changes**: adding or subtracting it changes behavior.
3. **erasure disrupts**: removing it disrupts concept-related performance.
4. **restoration recovers**: restoring it recovers the behavior.

This upgrades vector-space analysis from "useful geometry" to **causal representational evidence**.

### E. Key findings

- **Bias is a structural property of learned representations, not a surface artifact.** 
- **Interpretability and intervention share the same substrate.** Reading vectors that identify a bias direction are directly usable as control signals — which means any mechanism identified at corpus level could in principle become a RepE steering target for downstream validation.
- **A unified bias subspace may exist**. Bias directions extracted from one stereotype domain can transfer to others, suggesting low-dimensional shared structure.
- **RLHF does not necessarily remove bias geometrically**. RLHF is weight-level training that updates the model to produce refusals on bias-adjacent prompts. But the biased *direction* installed by pretraining remains causally active; RLHF works around it, not erase it. Once Jailbreaks technics defeat the gate (RLHF or frontend censorship), the underlying direction flows through unfiltered. 

### F. Comparison across aspects

The relevant comparison is **SOTA in §2 vs RepE vs My Work**. 

| Aspect | SOTA in §2 | RepE | My work |
|:---|:---|:---|:---|
| Main target | Outputs, embeddings, corpora, or pipeline stages | Internal representations and generation | Training material and corpus-derived framing mechanisms |
| Main strength | Broad coverage of the problem space | **Mechanistic and causal representation-level evidence** | **Mechanism-specific linguistic analysis at corpus level** |
| Detects contextual effects? | Partially / unevenly | Yes, indirectly via steering tests | Not by itself |
| Identifies internal mechanism? | Usually not, or only partially | **Yes** | Indirectly, via downstream validation |
| Traces patterns back to training material? | Sometimes, but usually with broad predefined categories | No | **Yes** |
| Separates different framing mechanisms? | Rarely | No | **Yes** |
| Supports mitigation/intervention? | Sometimes, often category-based | **Yes**, but coarse and potentially overcorrective | Yes, through upstream corpus intervention |
| Linguistic relevance | Mixed | Medium | **High** |


### G. Limitations, challenges, and future directions

#### Main limitations

- **Training-data blindness**: RepE can show where bias lives in the model/hyperspace, but not which sources installed it.
- **Linearity may be too simple**: intersectional or context-sensitive bias may not fit one global direction.
- **Weak contextual specificity**: the same intervention may fire across contexts where the concept should be treated differently.
  - **Overcorrection**: legitimate demographic-specific information, e.g. Black Females are de facto reported to be most affected by sarcoidosis, becomes neutralized by the intervention.
- **Hyperparameter dependence**: effect depends on layer choice and coefficient strength.

### H. Implications

RepE is both a validation paper and a boundary marker.

#### Why it validates my direction

RepE strengthens the logic:

- **Current developmental orientation is warranted**: if bias occupies causally active directions in the trained model, then studying where those directions were installed — in the training corpus — is the right level of analysis. RepE confirms the mechanism exists; I target its origin.
- **WEAT / CEAT** become more theoretically credible if social associations really do occupy meaningful linear subspaces.
- **Prototype-based attitudinal matching** becomes easier to justify if concept directions are readable through contrastive geometry.
- **PCA-based EFI construction** becomes more interpretable as a dimensional summary of structured variation, even if it is not identical to a pure causal direction.

#### What my work must add

RepE is best understood as a post-training control layer, not as a replacement for upstream corpus design. It can suppress harmful representational tendencies, but because the control is relatively coarse, it is not a full solution to context-sensitive fairness.

It's insufficient not because it is weak, but because it operates at a different stage, like rectifying adultLMs' behavior by some force, instead of compiling correct textbooks for and properly educating babyLMs.

In the future, one can:

- Use RepE-style steering as a **validation layer** for mechanisms first identified in training data.

- Extend from single vector to context-aware subspace.

The overcorrection issue in §G reflects a representational choice in RepE itself. Its bias stimuli are scenario-level (*"Consider the bias in the following scenario..."*), framing bias as a single *biased vs. unbiased* dichotomy, and the default extraction keeps only the first principal component of the contrast. The resulting vector averages across linguistically distinct mechanisms — attitudinal vs. indexical, factual-demographic vs. evaluative-stereotype, F⁻ vs. F⁺ — so steering along it hits all of them at once, neutralizing legitimate demographic information alongside evaluative bias. Follow-up work already moves towards this direction (e.g. [Gaussian Concept Subspace](https://arxiv.org/abs/2410.00153), ICLR 2025, modeling concepts as distributions over a subspace rather than points). My work fits here not as a design for that subspace but as the source of the mechanism distinctions any such subspace would need to respect.



---

## CEAT

> Guo, W., & Caliskan, A. (2021). *Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases*. In *Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society* (pp. 122-133). https://doi.org/10.1145/3461702.3462536

### What problem it addresses

> If we want to measure bias in contextual embeddings, should we treat bias as one average score, or as a distribution across many natural contexts? 

Tested on **ELMo, BERT, GPT, and GPT-2** using **Reddit Comment Dataset 2014** as the source of natural contexts.

### Core contribution

CEAT improves on WEAT/SEAT-style testing by:

- using **natural contexts** (Reddit sentences) rather than a few synthetic templates,
- treating contextual bias as a **distribution** rather than a single point estimate,
- and making **intersectional** and **emergent intersectional** bias, i.e., those that appear at an intersection (e.g., African American females; Mexican American females) but don't exist for either constituent category alone, visible.

**Statistical innovation**: CEAT aggregates sampled WEAT effect sizes (N=10,000) using a **random-effects meta-analytic model**, producing a Combined Effect Size (CES) that properly accounts for heterogeneity across contexts. This is what makes "bias as distribution" statistically rigorous rather than merely reporting a histogram.

### Key quantitative findings

- Bias magnitude correlates **negatively** with how contextualized the model is: more contextualized models (BERT, GPT-2) show lower average bias than less contextualized ones (ELMo), but bias is still **present in all models**.
- Intersectional biases associated with members of **multiple minority groups** (African American females, Mexican American females) have the **highest magnitude** across all models — stronger than single-category biases.

### Relevance

CEAT clarifies two points:

1. contextualized bias measurement should report **variation**, not just averages;
2. intersectionality should be treated as a real methodological requirement, not a bonus feature.

### How CEAT improves the current WEAT / Δ-SEAT pipeline

The current pipeline produces **one scalar per term**: a WEAT effect size, a SEAT score, and a Δ-SEAT. CEAT's contribution is to replace those with **distributions**, using the actual corpus sentences rather than bleached templates. Three concrete upgrades:

**1. Point estimate → distribution (directly applicable to Δ-SEAT)**

Instead of `Δ-SEAT = SEAT-full − SEAT-filtered` yielding one number per term, CEAT-style sampling produces:

- For each term, draw N=10,000 subsets of its context sentences from Dolma
- Compute SEAT-full and SEAT-filtered for each sample
- Report a distribution of Δ values with CES, mean, and variance

A term with **high mean Δ and low variance** is systematically contaminated across all contexts. A term with **high mean Δ and high variance** is contaminated only in certain discourse domains. That distinction is currently invisible in the pipeline and directly affects how EFI scores should be interpreted.

**2. Centroid collapse → full distribution**

The previously installed SEAT already uses real Dolma sentences (not bleached templates like vanilla SEAT's `"This is WORD."`), so the contextualization is meaningful. But such implementation still **collapses those sentences to a single centroid** — averaging GTE ModernBERT embeddings before computing cosine similarity — producing one scalar per group. CEAT's upgrade is to never collapse: keep the full set of per-sentence embeddings and sample from them, so the effect size is a distribution rather than a point. 

**3. Intersectional testing for compound group terms**

CEAT's Intersectional Bias Detection (IBD) tests whether bias emerges at an intersection that does not exist for either component alone. For compound group terms (`Asian American`, `undocumented immigrant`) in EFI, running IBD-style testing alongside per-component WEAT scores reveals emergent intersectional bias directions that the PCA reduction would otherwise miss if the per-group vectors were computed independently.

---

## 3. What Is Novel in My Positioning?

the field has studied multitude of aspects but individually; **no existing work integrates the linguistic grounding to construct an analysis/detection pipeline on LLM training material**.

## 4. In-Group Synergies and Takeaways

### 4.1 In-group synergies under "Domain Alignment and Bias Identification"

Our research package cluster around different levels of the **grounding stage of LLM development**. T2.3 centers around **uncertainty, confidence, and output reliability**; they help determine when an answer is unstable, miscalibrated, or inconsistent. T2.1 works more on the **conceptual grounding** layer by clarifying what grounding should mean in NLP plus **reasoning support and grounding verification**, asking whether outputs are supported by structured knowledge or retrieved evidence.

My work complements these lines of work by targeting a different failure mode: an answer can be confident, consistent, or grounded and still encode biased group portrayal inherited from training material. It's relevant to also **explainability, grounding, uncertainty, and robustness**, but from the upstream side of the pipeline.

The main takeaways are:

- A **grounding or verification WP** could use **identified discourse patterns** to distinguish factual grounding problems from bias in group portrayal.
- An **interpretability / representation WP** could use RepE-style steering to test whether **discovered mechanisms** correspond to identifiable concept directions.
- An **uncertainty / confidence WP** could compare calibration and confidence signals with bias-sensitive context patterns; high confidence does not rule out bias.
- A **robustness / evaluation WP** could test whether corpus or representation interventions improve contextual robustness.

---

## 5. Detailed References

see the latest version of [bibliography](https://raw.githubusercontent.com/c1araliang/armada/main/My%20Library.bib) 
