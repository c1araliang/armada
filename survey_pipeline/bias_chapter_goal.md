# Bias Chapter — Cycle Goal File

**Parent survey:** `../survey.md` (do not duplicate; always read from there)
**Primary literature source:** `quartz/content/reading.md` (read this first before any literature search)
**Skill:** survey-production
**Scope:** Bias sections only (survey.md lines 213–318)
**Cycle start:** 2026-06-27
**Human author:** owns Introduction, Knowledge Grounding, Training-time Grounding,
Synergies, Conclusion. **Pipeline must not modify these sections under any circumstances.**

---

## Working Abstract (Bias Chapter)

This chapter addresses bias as a distinct evaluation dimension interacting with LLM
grounding. LLMs inherit not only lexical associations but also recurrent discourse frames
that place social groups into events, roles, and epistemic positions [@bender2021stochastic;
@feng2023pretraining; @itzhak2025planted]. Fine-tuning techniques such as RLHF reduce
overt toxicity, yet covert representational biases persist or intensify after alignment
[@hofmann2024ai; @NEURIPS2022_b1efde53]. Existing benchmarks that rely on predefined
target–attribute pairings have limited construct validity [@blodgett-etal-2021-stereotyping].
We organize the literature by **where bias is introduced and measured** across the LLM
development pipeline: foundational/pre-LLM conceptual groundings, model-internal
representation structure (embeddings, animacy conflation, intersectionality), model output
behavior (benchmarks, contextual effects, emergent collective bias), pretraining input
material (corpus audits, emotion distributions, narrative framing), and cross-stage causal
analyses. We position the ARMADA pipeline as closing the gap between bottom-up corpus
framing discovery and association testing on pretraining-scale data.

---

## Chapter Structure (Bias Sections — authoritative)

```
§ Bias as an Evaluation Dimension          [intro: framing, motivation, definition]
  §.1  Bias and Representation             [scope statement + prior work gap]
  §.2  Foundational Works                  [pre-LLM: thematic roles, dehumanization,
                                            postcolonial theory, corpus-driven methodology]
  §.3  Model Internal                      [contextualized embeddings, polysemy/animacy
                                            conflation, intersectionality in geometry]
  §.4  Model Output                        [predefined-inventory benchmarks, contextual
                                            effects, emergent collective bias]
  §.5  Input Material                      [pretraining corpus audits, affective distributions,
                                            narrative/entity framing]
  §.6  All Stages                          [RepE causal tracing, cross-backbone experiments,
                                            corpus→LM→task causal chain]
  §.7  Relevant Works                      [evaluation tools, corpus resources, ontologies,
                                            rejected corpora]
  §.8  Bias as a Representation Problem    [synthesis: ARMADA positioning + RepE as
                                            validation layer + open challenges]
```

---

## Literature Map (from reading.md — use as primary citation source)

> **Instruction to pipeline:** reading.md is the primary literature for this chapter.
> Do NOT run broad literature searches that duplicate entries already present here.
> Run targeted gap-filling searches only for items marked ⚠ MISSING below.

### §.2 Foundational Works

Already catalogued in reading.md §2.1 (sorted by year):

| Key | Citation | Role in chapter | Depth |
|-----|----------|-----------------|-------|
| said1977Orientalism | Said (1978) *Orientalism* | Postcolonial framing of minority/dominant asymmetry | B |
| dowty1991thematic | Dowty (1991) Thematic Proto-Roles | Formal semantics grounding for AgI/PI dimensions | B |
| hall2018west | Hall (1992/2018) "The West and the Rest" | Binary opposition construction; lexicon typology motivation | B |
| quijano2000Coloniality | Quijano (2000) Coloniality of Power | Eurocentrism as structuring global axis | B |
| sinclair2004trust | Sinclair (2004) *Trust the Text* | Corpus-driven methodology; bottom-up frame discovery | B |
| bamman2013latent | Bamman, O'Connor & Smith (2013) Latent Personas | Agent/patient verb patterns from dependency parses | A |
| rashkin2016connotation | Rashkin, Singh & Choi (2016) Connotation Frames | ~1,000 verb lexicon encoding power/agency/affect | A |
| sap2017connotation | Sap et al. (2017) Connotation Frames of Power & Agency | Extension to power/agency; gender bias in film | A |
| mendelsohn2020dehumanization | Mendelsohn, Tsvetkov & Jurafsky (2020) | w2v dehumanization clusters; metaphor framing | B |
| bender2021stochastic | Bender et al. (2021) Stochastic Parrots | Training data ethics; corpus-level bias motivation | A |
| mendelsohn2025floods | Mendelsohn & Budak (ACL 2025) "When People are Floods" | Metaphor framing in immigration tweets; LLM as tool | B |

⚠ MISSING in reading.md for §.2:
- Sociolinguistics references for pragmatic context / reappropriation (flagged in survey.md line 235)
- Gabrielatos & Baker (2006) on corpus-assisted discourse (mentioned in reading.md §2.3 gap #1 but not catalogued)

### §.3 Model Internal

Already catalogued in reading.md §2.2.1 + §2.2.1.1 + §2.2.1.2:

| Key | Citation | Sub-section | Depth |
|-----|----------|-------------|-------|
| ethayarajh-2019-contextual | Ethayarajh (2019) How Contextual | Anisotropy; polysemy non-separability | A |
| haber-poesio-2021-patterns-polysemy | Haber & Poesio (2021) Patterns of Polysemy | Contextual failure on polysemic alternations | A |
| buijtelaar2023compounds | Buijtelaar & Pezzelle (EACL 2023) | Compound head dominance in BERT | B |
| zaitova2025attention | Zaitova et al. (NAACL Findings 2025) | Attention shifts on MWEs; idiomaticity-dependent | B |
| klafka2020animacy | Klafka & Ettinger (ACL 2020) Spying on Your Neighbors | Animacy recoverable across positions; non-local encoding | A |
| wdowicz2025caricature | Wdowicz (2025) Not a Mirror, a Caricature | Anglocentric default; identity underspecification → prototype collapse | B |
| ghai2021wordbias | Ghai, Hoque & Mueller (CHI 2021) WordBias | Intersectional bias visualization; non-additive structure | B |
| dev2021gender | Dev et al. (EMNLP 2021) Harms of Gender Exclusivity | Cyclical erasure of non-binary identities | B |
| khan2025winoidentity | Khan et al. (ACL 2025) WinoIdentity | Confidence disparity at intersectional identity cells | A |
| pawar2025cultural | Pawar et al. (EMNLP Findings 2025) Presumed Cultural Identity | Name-triggered cultural flattening | B |
| shieh2026intersectional | Shieh et al. (2026) Intersectional Biases in Narratives | Omission/subordination in implicit narrative generation | A |

### §.4 Model Output

Already catalogued in reading.md §2.2.2:

| Key | Citation | Sub-section | Depth |
|-----|----------|-------------|-------|
| nangia-etal-2020-crows | Nangia et al. (2020) CrowS-Pairs | Minimal-pair benchmark; 9 categories | B |
| nadeem-etal-2021-stereoset | Nadeem, Bethke & Reddy (2021) StereoSet | CAT benchmark; ICAT score | B |
| NEURIPS2022_b1efde53 | Ouyang et al. (NeurIPS 2022) InstructGPT | RLHF pipeline; covert bias persists | A |
| wang2023decodingtrust | Wang et al. (NeurIPS 2023) DecodingTrust | Large-scale output audit | B |
| hofmann2024ai | Hofmann et al. (2024) Dialect Prejudice | RLHF conceals overt but intensifies covert bias | A |
| germani2025framing | Germani & Spitale (2025) Source Framing | Demographic label swap; evaluation shift | B |
| issuebench2025 | IssueBench (2025) | 2.49M prompts; perspective bias through alignment | B |
| ashery2025collective | Ashery et al. (2025) Emergent Collective Bias | Individually-unbiased agents → collective bias | A |
| dentella2025chatgpt | Dentella et al. (2025) | ChatGPT noun-over-verb preference; verb-extraction caveat | C |
| decodinghate2025 | "Decoding Hate" (2025) | Safety guardrails; post-training output behavior | C |

### §.5 Input Material

Already catalogued in reading.md §2.2.3:

| Key | Citation | Sub-section | Depth |
|-----|----------|-------------|-------|
| kadan2024corpus | Kadan et al. (NLP Journal 2024) | Emotion–demographic correlation; corpus + model-level | B |
| mahmoud2025framing | Mahmoud et al. (ACL Findings 2025) Entity Framing | 22 narrative archetypes; 5 languages | B |
| udagawa2025crawl | Udagawa et al. (EMNLP 2025) | Protected-attribute detection in Common Crawl | A |
| gorge2025debiasing | Görge et al. (2025) | Data debiasing ≠ benchmark improvement | A |

### §.6 All Stages (Causal)

Already catalogued in reading.md §2.2.4:

| Key | Citation | Role | Depth |
|-----|----------|------|-------|
| zou2023repe | Zou et al. (2023) RepE | Causal bias directions in hidden states; validation layer for corpus-level mechanisms | A |
| feng2023pretraining | Feng et al. (ACL 2023) Trails of PoliBias | Corpus→LM→task causal chain; controlled pretraining | A |
| itzhak2025planted | Itzhak et al. (COLM 2025) Origins of Cognitive Biases | Pretraining is primary causal source (cross-tuning) | A |

### §.7 Relevant Works (Tools / Resources / Adjacent)

Already catalogued in reading.md §2.2.5 + §2.6:

| Key | Citation | Type | Notes |
|-----|----------|------|-------|
| blodgett-etal-2021-stereotyping | Blodgett et al. (2021) Stereotyping Norwegian Salmon | Critique | Benchmark construct validity challenge |
| tian2023calibration | Tian et al. (2023) "Just Ask for Calibration" | Tool note | RLHF calibration distortion; if LLM-as-judge used |
| soldaini2024dolma | Soldaini et al. (ACL 2024) Dolma | Corpus | Source corpus for ARMADA |
| russo2025docbiaso | Russo & Vidal (JAIR 2025) Doc-BiasO | Documentation | Bias ontology; reporting layer |
| dunning1993llr | Dunning (1993) LLR | Method | Sparse collocation scoring |
| rychly2008logdice | Rychlý (2008) LogDice | Method | Frame-term ranking |
| caliskan2017semantics | Caliskan, Bryson & Narayanan (2017) WEAT | Method | Type-level embedding association |
| shi2019srl | Shi & Lin (2019) Simple BERT SRL | Method | Patient-label detection |
| honnibal2020spacy | Honnibal et al. (2020) spaCy | Method | Dependency parsing / NER |
| guo2021ceat | Guo & Caliskan (2021) CEAT | Method | Contextualized embedding association; emergent intersectional bias |
| zhang2024gte | Zhang et al. (EMNLP 2024) mGTE | Method | GTE-ModernBERT encoder |

---

## Open Items (pipeline work queue — process in order)

- [ ] **§.2 Foundational Works:** Write prose covering rows above. Add missing sociolinguistics refs for pragmatic context / reappropriation (survey.md line 235). Search for Gabrielatos & Baker (2006) corpus-assisted discourse paper.
- [ ] **§.3 Model Internal:** Write prose for §2.2.1 (core polysemy/geometry) → §2.2.1.1 (animacy conflation) → §2.2.1.2 (intersectionality). Emphasize how Ethayarajh + Haber/Poesio motivate Δ-CEAT as contamination diagnostic.
- [ ] **§.4 Model Output:** Write prose. Pair InstructGPT (#3) + Hofmann (#5) for the RLHF dual finding. Group predefined-inventory benchmarks (#1, #2, #4, #7) under a construct-validity critique via Blodgett. Highlight Ashery (#8) as motivation for upstream intervention.
- [ ] **§.5 Input Material:** Write prose. Lead with Görge (#4) finding: data debiasing ≠ benchmark improvement. Connect to ARMADA's upstream corpus-level framing analysis.
- [ ] **§.6 All Stages:** Write prose. Lead with RepE as mechanistic validation. Pair Feng + Itzhak to establish pretraining as primary causal source.
- [ ] **§.7 Relevant Works:** Write brief subsection. Flag Tian (#2) and Doc-BiasO (#4) as implementation-detail references only.
- [ ] **§.8 Bias as a Representation Problem:** Write synthesis. Connect: bottom-up discovery gap (reading.md §2.3 gap #1) → composite group profiling gap (gap #2) → ARMADA as closing both. RepE as future validation layer per reading.md §H. Current pipeline: diagnostic outputs for pre-processing mitigation only (gap #3).
- [ ] **§.1 Bias and Representation:** Update once §.2–§.8 are drafted; revise scope statement to accurately reflect section content.
- [ ] After all sections drafted: verify citation keys against `My Library.bib`; check every 20 citations per Gate 1 protocol.

---

## Taxonomy Keywords (for gap-filling searches only)

> Most literature is already in reading.md. Run new searches only for:

- sociolinguistics reappropriation solidarity language bias NLP
- Gabrielatos Baker corpus-assisted discourse refugees CDA
- pragmatic context group identity language model
- ingroup language speaker bias LLM

---

## Seed Papers (already in BibTeX — verify keys in `My Library.bib`)

From reading.md — all 10 original seeds plus key additions:
- caliskan2017semantics, bender2021stochastic, blodgett-etal-2021-stereotyping
- nadeem-etal-2021-stereoset, nangia-etal-2020-crows, hofmann2024ai
- feng2023pretraining, itzhak2025planted, ethayarajh-2019-contextual
- haber-poesio-2021-patterns-polysemy, zou2023repe, guo2021ceat
- soldaini2024dolma, NEURIPS2022_b1efde53

---

## Scope / Angle / Audience

- **Scope:** Bias in LLMs from corpus-level input to model output, including foundational
  NLP/sociolinguistics bias work and cross-stage causal analyses. Excludes fairness in
  systems beyond NLP. Does **not** cover Knowledge Grounding or Training-time Grounding.
- **Angle:** Pipeline-stage taxonomy (where bias enters and is measured). Emphasis on
  pretraining corpus as upstream origin — connects to ARMADA's Phase 1 corpus framing work.
  Key differentiator: bottom-up collocate-driven frame discovery + association testing
  (no work in reading.md does both at pretraining scale).
- **Audience:** NLP researchers familiar with LLMs; ARMADA WP2 reviewers; grounding survey
  readers needing bias as a third axis.

---

## Cycle Log
<!-- Orchestrator appends one line per phase boundary. -->
