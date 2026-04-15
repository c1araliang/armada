---
title: 4. Reading List
description: Complete literature review, categorized by tags and sorted by relevance.
tags:
  - literature

---

WP2 Domain Alignment and Bias Identification: T2.2 Bias Analysis/Detection (F3BF)

## Scope

This report focuses on **how linguistic mechanisms propagate bias from training data into representations and contextual effects**.

The central question is:

> **How can we detect, measure, and validate the linguistic mechanisms that propagate bias, from surface patterns to representations to contextual effects?**

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

Foundational work on bias in language, predating or independent of LLM architectures. Entries are ordered by relevance to F3BF, with role/discourse overlap first and association-testing overlap second.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Rashkin, Singh & Choi (2016)](https://doi.org/10.18653/v1/P16-1030) Connotation Frames | *New York Times* corpus | Crowd-annotated lexicon of ~1,000 transitive verbs encoding agent/theme power, agency, and emotional affect per verb | **Analysis** | `role-extraction` `corpus-linguistic` `verb-level` |
| 2 | [Bamman, O'Connor & Smith (2013)](https://aclanthology.org/P13-1035/) Latent Personas | 42,306 Wikipedia movie plot summaries | Agent verb / patient verb / attribute patterns from dependency parses (nsubj, dobj, nsubjpass); Dirichlet persona clustering | **Analysis** | `role-extraction` `corpus-linguistic` |
| 3 | [Mendelsohn, Tsvetkov & Jurafsky (2020)](https://doi.org/10.3389/frai.2020.00055) Dehumanization Framework | *New York Times* (1986–2015), LGBTQ discourse | w2v cosine similarity to four theory-derived dehumanization concept clusters; denial of agency measured alongside metaphor | **Analysis** | `metaphor-framing` `predefined-categories` `demographic-profiling` |
| 4 | [Caliskan, Bryson & Narayanan (2017)](https://doi.org/10.1126/science.aal4230) WEAT | Static word embeddings (GloVe) | IAT analogue on word vectors; foundational proof that corpus-derived semantics encode human-like biases | **Detection** | `WEAT/SEAT-family` `predefined-categories` |
| 5 | [Azzalini, Dolci & Tanelli (2022)](https://www.semanticscholar.org/paper/Bias-Score%3A-Estimating-Gender-Bias-in-Sentence-Azzalini-Dolci/a8caf23b86b050ad217a05db6aac94396e73d37a) Bias Score | Sentence representations | Normalized per-word cosine similarity to a gender direction; word-importance weighting parallels SEAT context token weighting | **Detection** | `WEAT/SEAT-family` `sentence-level` |
| 6 | ["Don't Erase, Inform!" (2025)](https://arxiv.org/abs/2505.24538) | Cultural heritage metadata | Lexical flagging of offensive terms from a multilingual vocabulary co-created with marginalized communities; cross-referenceable for lexical coverage but orthogonal (cultural heritage context, not contemporary metaphorical framing) | **Detection** | `lexical-detection` `multilingual` |

- `role-extraction` (#1–2) are the most direct conceptual ancestors of AgI/PI/SI. Rashkin is the closest overlap with F3BF's role indices and attitudinal dimensions; Bamman is the syntactic ancestor of AgI/PI but has no bias analysis or embeddings.
- Mendelsohn 2020 (#3) is the nearest miss for discourse-level bias analysis: it measures denial of agency alongside metaphor, but the dimensions are theory-derived and the corpus is news.
- `WEAT/SEAT-family` (#4–5): Caliskan is the foundational association-testing method F3BF inherits; Azzalini adds sentence-level weighting adjacent to SEAT context token weighting. 
- "Don't Erase, Inform!" (#6) is orthogonal in domain and its DE-BIAS vocabulary appears domain-restricted. 
- None integrates role extraction with association testing or collocate-driven frame discovery.

### 2.2 Studies of bias in LLM

#### 2.2.1 Representation

Bias probed in contextualized embeddings or model-side association structure rather than generated text. Entries are ordered by method overlap first, then multilingual extension.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Guo & Caliskan (2021)](https://doi.org/10.1145/3461702.3462536) CEAT | Contextualized embeddings (ELMo, BERT) | Extend WEAT to natural contexts; treat bias as a *distribution* over occurrences; intersectional and emergent intersectional bias | **Detection** | `WEAT/SEAT-family` `intersectional` `distributional` |
| 2 | [Kurpicz-Briki et al. (2024)](https://arxiv.org/abs/2407.18689) BIAS Framework | Word embeddings + LMs, European languages | WEAT, SEAT, LPBS, CrowS-Pairs with predefined word lists; tests *model* associations, not *corpus* framing | **Detection** | `WEAT/SEAT-family` `predefined-categories` `multilingual` |
| 3 | [Puttick & Kurpicz-Briki (GeBNLP 2025)](https://aclanthology.org/2025.gebnlp-1.3/) | Italian word embeddings + LMs | WEAT/SEAT/LPBS + GG-FISE for intersectional bias; grammatical gender as an additional bias axis | **Detection** | `WEAT/SEAT-family` `intersectional` `multilingual` |

- `WEAT/SEAT-family` (#1–3) share the core measurement method with F3BF. CEAT is closest: it treats bias as a distribution over natural occurrences and handles intersectionality. BIAS Framework and Puttick apply the same tooling across European languages. All three test against **predefined attribute sets**; none derives those sets from corpus collocate statistics.
- This subsection is model-side but not output-side: it captures representational association structure rather than generated judgments or behaviors.

#### 2.2.2 Output

Bias probed in generated judgments, benchmark responses, or collective output behavior. Entries are ordered by contextual depth first, then benchmarks, then emergent or stylistic effects.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Germani & Spitale (2025)](https://doi.org/10.1126/sciadv.adz2924) Source Framing | LLM evaluative judgments | Source attribution audit: identical content, swapped demographic labels; measures evaluation shift in context | **Detection / analysis** | `contextual-effects` `audit` |
| 2 | [Wang et al. (NeurIPS 2023)](https://decodingtrust.github.io/) DecodingTrust | GPT-3.5 / GPT-4 outputs | Trustworthiness benchmark across toxicity, stereotypes, fairness, privacy, robustness; large-scale audit of output-side bias and safety failures | **Detection** | `benchmark` `predefined-categories` |
| 3 | [IssueBench (2025)](https://arxiv.org/abs/2502.08395) | 10 SOTA LLM outputs | 2.49M prompts probing perspective bias in writing assistance; large-scale evidence that issue bias persists through alignment | **Detection** | `benchmark` `predefined-categories` |
| 4 | [Ashery et al. (2025)](https://arxiv.org/abs/2410.08948) Emergent Collective Bias | Multi-agent LLM populations | Individually-unbiased agents still produce biased collective behavior; motivates upstream data-level intervention | **Analysis** | `emergent-bias` `multi-agent` |
| 5 | [Dentella et al. (2025)](https://arxiv.org/abs/2508.16385) "ChatGPT-generated texts show authorship traits" | ChatGPT outputs across registers | ChatGPT systematically prefers nouns over verbs, showing a distinct "linguistic backbone" from humans who anchor in tense/aspect/mood; citable caveat that verb-reliant extraction yield may vary, attenuated by SRL | **Analysis** | `linguistic-structure` `generation-pattern` |
| 6 | ["Decoding Hate" (2025)](https://arxiv.org/abs/2410.00775) | 7 LLM outputs on hate speech inputs | Qualitative analysis of LLM responses to hate speech including politically correct hate speech; focus on safety guardrails and alignment behavior, not systematic demographic framing | **Analysis** | `output-behavior` `alignment` |

- Source Framing (#1) shows contextual effects without decomposing the linguistic mechanism.
- Benchmarks (#2–3) confirm the problem at scale but remain output-side evaluations rather than causal tracing from training data.
- Ashery (#4) motivates upstream intervention: individually debiased models can still produce collective bias.
- Dentella (#5) is a citable note that ChatGPT's noun-over-verb preference may affect verb-reliant extraction yield, though the effect is attenuated under SRL.
- "Decoding Hate" (#6) studies post-training output behavior, not systematic framing.

#### 2.2.3 Input

Bias analyzed in pretraining data, news, social media, or annotated narrative collections. Entries are ordered by linguistic depth first, then pretraining-data scope.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Mendelsohn & Budak (ACL 2025)](https://aclanthology.org/2025.acl-long.398/) "When People are Floods" | 400K US immigration tweets | Word-level + document-level LLM technique for seven predefined metaphor concepts; ideology × engagement analysis | **Analysis** | `metaphor-framing` `predefined-categories` `demographic-profiling` |
| 2 | [Mahmoud et al. (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.17/) Entity Framing | 1,378 news articles, 5 languages | LLM zero-shot + fine-tuned XLM-R for 22 narrative archetypes (protagonist / antagonist / innocent) | **Analysis** | `role-extraction` `multilingual` |
| 3 | [Udagawa et al. (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.2/) | Common Crawl (pretraining) | Protected-attribute detection + regard (polarity) classification; mitigation by balancing regard across groups | **Detection / mitigation** | `pretraining-data` `predefined-categories` `demographic-profiling` |
| 4 | [Görge et al. (2025)](https://arxiv.org/abs/2512.10734) | Textual pretraining data | LLM-generated word lists, Demographic Representation Score, stereotype filter, counterfactual augmentation; debiased fine-tuning did *not* consistently improve benchmark scores | **Detection / mitigation** | `pretraining-data` `predefined-categories` `mitigation` |
| 5 | [Kadan et al. (NLP Journal 2024)](https://arxiv.org/abs/2301.09003) | Pretraining + fine-tuning corpora | Corpus-level affective word distribution + model-level emotion–demographic association evaluation | **Analysis** | `pretraining-data` `predefined-categories` |

- Mendelsohn & Budak (#1) is the nearest miss for F3BF's immigration focus: it still relies on predefined metaphor categories, uses social media rather than pretraining data, and has no role extraction.
- Entity Framing (#2) assigns narrative roles (shared `role-extraction` tag with §2.1 #1–2) but uses a taxonomy-first approach and no association testing.
- `pretraining-data` (#3–5) share F3BF's data scope but not its linguistic granularity: they measure polarity, representation counts, or affect distributions rather than collocate-grounded discourse structures. Görge (#4) is particularly instructive because counterfactual debiasing did not consistently improve benchmark scores.
- Kadan (#5) bridges corpus affect distributions and model-side association evaluation, but its primary analytical starting point is still the training-data distribution.

#### 2.2.4 All stages

Bias traced across pipeline stages or causally attributed to a specific stage. RepE leads because it shares the goal of mechanistic analysis and provides the theoretical grounding for F3BF's vector-space operations.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Zou et al. (2023)](https://arxiv.org/abs/2310.01405) RepE | Internal hidden states + downstream generation | Concept directions via contrastive activation extraction; causal reading, steering, and intervention on representation-level bias | **Analysis / mitigation** | `representation-mechanism` `causal-intervention` |
| 2 | [Feng et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.656/) Best Paper | Corpus → LM → downstream tasks | Political leaning traced from pretraining through LM into hate speech and misinformation predictions | **Analysis** | `causal-tracing` `pretraining-data` `predefined-categories` |
| 3 | [Itzhak et al. (COLM 2025)](https://openreview.net/forum?id=KQhUEoPmJy) Spotlight | Pretraining vs. finetuning histories | Cross-tuning: swap instruction datasets between models with different pretraining backbones; models cluster by backbone — pretraining is the primary causal source | **Analysis** | `causal-tracing` `pretraining-data` |

- RepE (#1) proves that bias occupies causally active linear directions in hidden-state space, validating the geometric reasoning behind WEAT/SEAT, prototype matching, and PCA-based EFI. Its limit: it cannot identify which corpus framing patterns installed those directions. 
- Feng (#2) and Itzhak (#3) share `causal-tracing` and together establish the causal case for upstream intervention: Feng traces the propagation path; Itzhak isolates the origin via controlled backbone swaps.

#### 2.2.5 Bonus

Relevant work that informs evaluation, reporting, or corpus selection but does not fit cleanly into the representation / output / input / all-stage taxonomy.

| # | Study | Target | Key method | Goal | Tags |
|:---|:---|:---|:---|:---|:---|
| 1 | [Tian et al. (2023)](https://arxiv.org/abs/2305.14975) "Just Ask for Calibration" | RLHF-tuned LLM outputs | RLHF distorts token-level probability calibration; verbalized confidence is better calibrated; relevant as implementation detail if LLM-as-judge is used for validation | **Analysis** | `calibration` `implementation-detail` |
| 2 | [Russo & Vidal (JAIR 2025)](https://doi.org/10.1613/jair.1.15195) Doc-BiasO | ML lifecycle documentation | Ontology for bias types and metric definitions; absorbable for F3BF's reporting layer, pipeline-irrelevant | **Documentation** | `ontology` `interoperability` |
| 3 | [WildChat (Allen AI) + ShareChat (2025)](https://arxiv.org/abs/2512.17843) | Large-scale real user–LLM conversations | Conversational corpora; structural messiness of incomplete conversational data risks breaking extraction heuristics; structured Alpaca significantly more suitable for syntactic tractability | **Corpus** | `corpus-resource` `assessed-rejected` |

- Tian (#1) matters only as an auxiliary evaluation note for any future LLM-as-judge validation step.
- Doc-BiasO (#2) is useful for metric naming and interoperability, not for the bias-detection pipeline itself.
- WildChat + ShareChat (#3) were assessed and rejected: the "realness" of user input does not outweigh structural messiness for the extraction pipeline.

### 2.3 Relevance gap for F3BF

Across 6 foundational + 3 representation + 6 output + 5 input + 3 all-stage + 3 bonus works, four converging gaps define F3BF's niche:

1. **Predefined categories throughout.** Whether the attribute is polarity, regard, affect, political leaning, or metaphor type, all existing bias measurement assumes the frame shape in advance. No work discovers frame sets bottom-up from corpus collocate statistics (LLR / LogDice).
2. **Role extraction isolated from bias measurement.** Rashkin (§2.1 #1), Bamman (§2.1 #2), and Entity Framing (§2.2.3 #2) each extract role-level patterns, but none combines them with semantic association testing or demographic group profiling.
3. **Association testing disconnected from empirical frame discovery.** `WEAT/SEAT-family` studies (§2.1 #4–5, §2.2.1 #1–3) test associations against predefined attribute sets. No study anchors WEAT/SEAT stimuli to frame sets discovered from the corpus being analyzed.
4. **No composite group profiling.** No study assembles role indices + association scores + attitudinal dimensions into a per-group framing profile and reduces the structure through data-driven PCA.

---

## RepE 

> Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405. https://arxiv.org/abs/2310.01405

### A. Research problem addressed by RepE

RepE asks the following mechanistic question:

> **Can high-level concepts such as bias be detected and controlled as directions in model representations?**

If social bias is encoded in roughly linear directions, then the vector-space methods used in F3BF, including **WEAT, SEAT, prototype matching, and PCA-based summary dimensions**, gain a stronger theoretical foundation.

### B. Taxonomy of methods inside RepE

RepE itself contains a useful mini-taxonomy:

| Method class | What it does | Why it matters |
|:---|:---|:---|
| **Representation reading** | Extracts concept directions from hidden states using contrastive examples | Lets us identify whether a concept is present in the model's internal geometry |
| **Representation control** | Adds or subtracts those directions during inference | Lets us test whether the concept is causally active |
| **LoRRA / weight-level integration** | Bakes a concept direction into low-rank adapters | Shows how representation control can become a practical deployment intervention |

### C. High-level explanation of the representative methods

#### 1. Representation reading

RepE assumes that many high-level concepts are encoded as approximately linear directions in hidden-state space. The basic workflow is:

1. Construct contrastive examples, such as biased versus anti-biased or honest versus dishonest completions.
2. Extract hidden activations.
3. Compute a concept direction from the difference structure of those activations.
4. Use that direction to classify whether the concept is present.

The underlying idea is not that "bias" is one point in space, but that it is an **axis of variation**.

#### 2. Representation control

Once a direction is extracted, RepE modifies generation by adding or subtracting it from hidden states at selected layers:

$$h' \leftarrow h + \alpha v$$

This turns interpretability into intervention. If subtracting a bias direction reduces biased outputs, then the direction is not merely correlated with bias; it is part of the causal mechanism of generation.

#### 3. Three representative intervention variants

| Variant | Mechanism | Strength | Weakness |
|:---|:---|:---|:---|
| **Reading Vector** | Add/subtract one fixed concept direction | Cheap and simple | Can be too coarse |
| **Contrast Vector** | Build a direction relative to the current stimulus | More adaptive, often stronger | More expensive and input-dependent |
| **LoRRA** | Train low-rank adapters around the target direction | No extra inference-time steering cost | Less transparent once merged into deployment workflow |

### D. Why RepE is convincing: causal evaluation, not just probing

RepE's strongest contribution is methodological: it does not stop at correlation and instead uses a four-step causal hierarchy:

1. **Correlation**: the direction predicts concept presence.
2. **Manipulation**: adding or subtracting it changes behavior.
3. **Termination**: removing it disrupts concept-related performance.
4. **Recovery**: restoring it recovers the behavior.

This upgrades vector-space analysis from "useful geometry" to **causal representational evidence**.

### E. Key findings most relevant to my research problem

The RepE results I would emphasize are:

- **Bias is causally active in representation space**. This supports treating bias not as a purely surface-level prompt artifact, but as a property of learned internal structure.
- **The same representational machinery can be used for reading and control**. So interpretability and intervention are tightly linked.
- **A unified bias subspace may exist**. Bias directions extracted from one stereotype domain can transfer to others, suggesting low-dimensional shared structure.
- **RLHF does not necessarily remove bias geometrically**. It may suppress or route around it behaviorally while leaving the representational tendency intact.
- **Steering can overcorrect**. The sarcoidosis case shows that subtracting a "bias" direction can also suppress true demographic signal.

### F. Comparison across the aspects most relevant to my topic

The relevant comparison is **SOTA in §2 vs RepE vs F3BF**. CEAT functions as a **complement to F3BF's workflow**, especially for contextual variation and intersectionality.

| Aspect | SOTA in §2 | RepE | F3BF |
|:---|:---|:---|:---|
| Main target | Outputs, embeddings, corpora, or pipeline stages | Internal representations and generation | Training material and corpus-derived framing mechanisms |
| Main strength | Broad coverage of the problem space | **Mechanistic and causal representation-level evidence** | **Mechanism-specific linguistic analysis at corpus level** |
| Detects contextual effects? | Sometimes | Yes, indirectly via steering tests | Not by itself |
| Identifies internal mechanism? | Usually not, or only partially | **Yes** | Indirectly, via downstream validation |
| Traces patterns back to training material? | Sometimes, but usually with broad predefined categories | No | **Yes** |
| Separates different framing mechanisms? | Rarely | No | **Yes** |
| Supports mitigation/intervention? | Sometimes, often category-based | **Yes**, but coarse and potentially overcorrective | Yes, through upstream corpus intervention |
| Linguistic relevance | Mixed | Medium | **High** |

**RepE is strongest on internal mechanism and causal manipulation, while F3BF is strongest on linguistic decomposition and source tracing**. **CEAT complements F3BF by improving contextual association measurement, especially through distributional variation and intersectionality**.

### G. Limitations, challenges, and future directions

#### Main limitations

- **Training-data blindness**: RepE can show where bias lives in the model, but not which corpus patterns installed it.
- **No linguistic decomposition**: one bias direction does not tell me whether the issue is dehumanization, patient framing, denied agency, or negative attitudinal attribution.
- **Linearity may be too simple**: intersectional or context-sensitive bias may not fit one global direction.
- **Layer dependence**: steering success depends on where the intervention is applied, and the theory of layer choice is still weak.
- **Overcorrection**: removing the direction can also remove legitimate demographic information.

#### Open research challenges

- How can we distinguish **subtle yet harmful framing** from **accurate group-conditioned facts**?
- How should we represent **intersectional bias**, which may not be linearly separable?
- Can concept directions be mapped back to **specific discourse patterns** in the pretraining corpus?
- Can we move from one opaque "bias axis" to **mechanism-specific axes** such as threat framing, victim framing, or agency denial?

#### Future directions

- Use RepE-style steering as a **validation layer** for mechanisms first identified by F3BF in training data.
- Extend F3BF from single-group terms to **intersectional and multi-word group expressions**.
- Combine framing diagnostics with **uncertainty and grounding checks** so we can better separate unsupported, unstable, and systematically biased answers.

### H. Implications for F3BF

RepE is both a validation paper and a boundary marker.

#### Why it validates my direction

RepE strengthens the logic behind several components of **F3BF**:

- **WEAT / SEAT** become more theoretically credible if social associations really do occupy meaningful linear subspaces.
- **Prototype-based attitudinal matching** becomes easier to justify if concept directions are readable through contrastive geometry.
- **PCA-based EFI construction** becomes more interpretable as a dimensional summary of structured variation, even if it is not identical to a pure causal direction.

#### Why it also shows what my work must add

RepE cannot answer the question that most directly motivates my project:

> **Which linguistic mechanisms in training material made the model learn those directions, and how can we validate them from surface pattern to representation to contextual effect?**

It's insufficient not because it is weak, but because it solves a **different layer** of the problem.

---

## CEAT

> Guo, W., & Caliskan, A. (2021). *Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases*. In *Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society* (pp. 122-133). https://doi.org/10.1145/3461702.3462536

### What problem it addresses

If we want to measure bias in contextual embeddings, should we treat bias as one average score, or as a distribution across many natural contexts?

### Core contribution

CEAT improves on WEAT/SEAT-style testing by:

- using **natural contexts** rather than a few synthetic templates,
- treating contextual bias as a **distribution** rather than a single point estimate,
- and making **intersectional** and **emergent intersectional** bias visible.

### Relevance to F3BF

CEAT clarifies two points for F3BF:

1. contextualized bias measurement should report **variation**, not just averages;
2. intersectionality should be treated as a real methodological requirement, not a bonus feature.

### Limit relative to F3BF

CEAT improves contextual bias measurement, but like RepE, it does not identify **which linguistic mechanisms in the corpus created the effect**.

---

## 3. What Is Novel in My Positioning?

The four gaps identified in §2.3 converge on a single claim: the field has studied framing, bias, roles, and associations individually, but **no existing work integrates all four on LLM training material in one analysis pipeline**.

F3BF within ARMADA addresses this by closing the chain: bottom-up frame discovery (LLR/LogDice → human classification → F⁻/F⁺) → role-based analysis (AgI, PI, SI via transformer SRL) → empirically anchored association testing (WEAT, SEAT, Δ-SEAT, attitudinal matching) → composite group profiling (EFI via PCA) — all on a single pretraining corpus (Dolma v1.6).

## 4. In-Group Synergies and Takeaways

### 4.1 In-group synergies under "Domain Alignment and Bias Identification"

Our research package cluster around different levels of the **grounding stage of LLM development**. T2.3 centers around **uncertainty, confidence, and output reliability**; they help determine when an answer is unstable, miscalibrated, or inconsistent. T2.1 works more on the **conceptual grounding** layer by clarifying what grounding should mean in NLP plus **reasoning support and grounding verification**, asking whether outputs are supported by structured knowledge or retrieved evidence.

F3BF complements these lines of work by targeting a different failure mode: an answer can be confident, consistent, or grounded and still encode biased group portrayal inherited from training material. This makes F3BF relevant to also **explainability, grounding, uncertainty, and robustness**, but from the upstream side of the pipeline.

The main takeaways are:

- A **grounding or verification WP** could use **F3BF's discourse patterns** to distinguish factual grounding problems from bias in group portrayal.
- An **interpretability / representation WP** could use RepE-style steering to test whether **F3BF-discovered mechanisms** correspond to identifiable concept directions.
- An **uncertainty / confidence WP** could compare calibration and confidence signals with bias-sensitive context patterns; high confidence does not rule out bias.
- A **robustness / evaluation WP** could test whether corpus or representation interventions improve contextual robustness.
- A **benchmarking / statistics WP** could extend **F3BF's SEAT analysis** with CEAT-style heterogeneity estimates and intersectional testing.

---

## 5. Detailed References

**§2.1 Studies of bias informing LLM-bias**

Azzalini, F., Dolci, T., & Tanelli, M. (2022). *Bias Score: Estimating Gender Bias in Sentence Representations*. *Sistemi Evoluti per Basi di Dati*.

Bamman, D., O'Connor, B., & Smith, N. A. (2013). *Learning Latent Personas of Film Characters*. In *Proceedings of ACL 2013* (pp. 352–361). https://aclanthology.org/P13-1035/

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). *Semantics derived automatically from language corpora contain human-like biases*. *Science*, 356(6334), 183–186. https://doi.org/10.1126/science.aal4230

*Don't Erase, Inform!* (2025). arXiv:2505.24538. https://arxiv.org/abs/2505.24538

Mendelsohn, J., Tsvetkov, Y., & Jurafsky, D. (2020). *A Framework for the Computational Linguistic Analysis of Dehumanization*. *Frontiers in Artificial Intelligence*, 3, 55. https://doi.org/10.3389/frai.2020.00055

Rashkin, H., Singh, S., & Choi, Y. (2016). *Connotation Frames: A Data-Driven Investigation*. In *Proceedings of ACL 2016* (pp. 311–321). https://doi.org/10.18653/v1/P16-1030

**§2.2.1 Representation**

Guo, W., & Caliskan, A. (2021). *Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases*. In *Proceedings of AIES 2021* (pp. 122–133). https://doi.org/10.1145/3461702.3462536

Kurpicz-Briki, M., et al. (2024). *The BIAS Detection Framework*. arXiv:2407.18689. https://arxiv.org/abs/2407.18689

Puttick, S., & Kurpicz-Briki, M. (2025). *Detecting Bias and Intersectional Bias in Italian Word Embeddings and Language Models*. In *Proceedings of GeBNLP 2025* (pp. 33–51). https://aclanthology.org/2025.gebnlp-1.3/

**§2.2.2 Output**

Ashery, et al. (2025). *Emergent social conventions and collective bias in LLM populations*. arXiv:2410.08948. https://arxiv.org/abs/2410.08948

*Decoding Hate* (2025). arXiv:2410.00775. https://arxiv.org/abs/2410.00775

Dentella, V., et al. (2025). *ChatGPT-generated texts show authorship traits*. arXiv:2508.16385. https://arxiv.org/abs/2508.16385

Germani, F., & Spitale, G. (2025). *Source framing triggers systematic bias in large language models*. *Science Advances*, 11(45), eadz2924. https://doi.org/10.1126/sciadv.adz2924

*IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance* (2025). arXiv:2502.08395. https://arxiv.org/abs/2502.08395

Wang, B., et al. (2023). *DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models*. In *Proceedings of NeurIPS 2023*. https://decodingtrust.github.io/

**§2.2.3 Input**

Görge, et al. (2025). *Textual Data Bias Detection and Mitigation: An Extensible Pipeline with Experimental Evaluation*. arXiv:2512.10734. https://arxiv.org/abs/2512.10734

Kadan, A., Deepak, P., Bhadra, S., Gangan, M. P., & Lajish, V. L. (2024). *Understanding Latent Affective Bias in Large Pre-trained Neural Language Models*. *Natural Language Processing Journal*, 7. https://arxiv.org/abs/2301.09003

Mahmoud, et al. (2025). *Entity Framing and Role Portrayal in the News*. In *Findings of ACL 2025* (pp. 302–326). https://aclanthology.org/2025.findings-acl.17/

Mendelsohn, J., & Budak, C. (2025). *When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models*. In *Proceedings of ACL 2025* (pp. 8079–8103). https://aclanthology.org/2025.acl-long.398/

Udagawa, T., Zhao, Y., Kanayama, H., & Bhattacharjee, B. (2025). *Bias Analysis and Mitigation through Protected Attribute Detection and Regard Classification*. In *Findings of EMNLP 2025* (pp. 16–25). https://aclanthology.org/2025.findings-emnlp.2/

**§2.2.4 All stages**

Feng, S., Park, C. Y., Liu, Y., & Tsvetkov, Y. (2023). *From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models*. In *Proceedings of ACL 2023* (Best Paper). https://aclanthology.org/2023.acl-long.656/

Itzhak, I., Belinkov, Y., & Stanovsky, G. (2025). *Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs*. In *Proceedings of COLM 2025* (Spotlight). https://openreview.net/forum?id=KQhUEoPmJy

Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405. https://arxiv.org/abs/2310.01405

**§2.2.5 Bonus**

Russo, A., & Vidal, M.-E. (2025). *Towards an Ontology-Driven Approach to Document Bias*. *Journal of Artificial Intelligence Research*. https://doi.org/10.1613/jair.1.15195

Tian, K., et al. (2023). *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback*. arXiv:2305.14975. https://arxiv.org/abs/2305.14975

*WildChat + ShareChat* (2025). arXiv:2512.17843. https://arxiv.org/abs/2512.17843