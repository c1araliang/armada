---
title: cache
description: Temporary page for storing drafts to be reviewed and submitted 
tags:
 - cache
---

## [May 16] AILC Draft - 433/500 words

**Title**

A Computational Pipeline for Reproducible Social Bias Analysis of LLM Training Data

**Abstract**

This poster presents ongoing work from the EU-funded ARMADA project, on the task "analysis of bias in training materials for chatbots/LLM via linguistic corpora analysis." The project builds on converging evidence that social bias can propagate from training distributions into model representations and contextual behaviour. Such bias is not limited to explicit stereotypes and toxic expression. It may also emerge through underlying recurrent semantic associations, stance patterns, evaluative framing, and asymmetrical depitctions of social groups, including inter-group power relations, and the socially loaded construal of human agency, in situated discourse across contexts.

Existing LLM-bias research has produced strong results, but its main approaches often remain separated across different stages of the bias-production causal chain. Corpus-specific discourse studies are difficult to generalize; pretraining-data audits often rely on sentiment-oriented measures or predefined wordlists with strong priors; representation-level methods can identify or steer bias-related directions, but may offer limited interpretability and risk overcorrection; and output benchmarks usually measure bias only after its installation. The resulting gap is one of integration: even bottom-up analyses of biased framing are rarely connected to demographic association testing, and are almost never applied to the pretraining-scale corpora that shape contemporary LLMs.

To address this problem, the current prototype combines large-scale corpus filtering, bottom-up collocate discovery, target/contrast-group comparison, and qualitative validation. Rather than beginning from extensively predefined stereotype or framing categories, **the goal of the design is to rely on minimal referential input and mitigate prior assumption by first uncovering salient collocational evidence through LLR and LogDice measures.** These automatically discovered lexical and constructional patterns are then validated through post hoc human-led annotation and interpreted by fine-grained, small-scale corpus-assisted linguistic analysis. The aim is not one-off moral labelling, but discovery-first and reproducible bias measurement, with evaluation introduced at a later stage.

Still at the early development stage, test runs have highlighted limitations in the current **semantic feature construction**. Dependency parsing via spaCy and predicate-argument semantic role labelling do not reliably capture target-specific semantic profiles, especially when the designed metrics–agency, patienthood, and subjectivity–aim to encode multidimensional social frames that cannot be reduced to simple contrastive detection. Likewise, prototype-based attitudinal recognition proved brittle when evaluative language cluster around multiple demographic groups in **a both syntactically and semantically complex construction.** 

Future revisions will therefore introduce explicit target binding to improve attitude attribution, and tighter compositional handling for a wider range of discursive constructions (including negation, quotation, contrast, and reported speech). By reporting the conceptual architecture, preliminary diagnostic findings and failure cases, and planned validation strategy, this work shows how linguistic analysis can make computational bias detection more sensitive to discourse-level meaning and more interpetable for fairer conversational AI.

## [Jun 4] Venice Draft - 300 words


## [Jun 8] AACL SRW Short Paper - 4 pages

**Toward Target-Aware Bias Detection: A Linguistically Grounded Pipeline (for Group Profiling/Validation?)**

Anonymous ACL submission

**Abstract**

Bias measurement in NLP often begins with a fixed target--attribute relation: predefined demographic terms, predefined affective categories, predefined stereotypes, or predefined metaphor inventories. Such designs are useful for controlled benchmarking, but they risk missing recurrent linguistic mechanisms that are not already encoded in the test design. This paper presents a work-in-progress pipeline for detecting candidate framing bias in large-scale language-model training material. The proposed framework treats bias not as a single sentiment score, but as a recurrent pattern of target/contrast-group construal across distributional association, semantic-role attribution, contextual embedding association, and attitudinal framing. Its central output is the Evaluative Framing Index (EFI), a per-group multidimensional framing profile consisting of agency, patienthood, subjectivity, attitudinal association, WEAT/SEAT-style association scores, and exploratory PCA-based summaries of cross-group variation. At the current stage, automatic outputs are treated as diagnostic candidates rather than final bias measurements. Preliminary testing on a small Dolma sample shows that a lexical gate, MiniLM-based semantic retrieval, and an embedding-based relevance classifier can reduce a large raw corpus into a manageable set of demographic-context sentences for later human validation. The paper contributes a methodological argument and prototype architecture: bottom-up frame discovery should be connected to target/contrast-group profiling before bias is interpreted as a stable property of model representations or outputs.

**1 Introduction**

Large language models inherit not only lexical associations from their training data, but also recurrent ways of placing social groups into events, roles, affects, and discourse frames. A model may avoid explicit slurs while still reproducing a skewed pattern in which one group is repeatedly represented as threatening, passive, irrational, dependent, or culturally alien. Such patterns are not always visible to benchmark designs that test only fixed target--attribute pairs or output-level refusals. The problem is therefore not only whether a model is biased, but which linguistic mechanisms carry the bias, how these mechanisms can be detected in training material, and how they may later propagate into model representations and contextual behavior.

This paper proposes a validation-aware pipeline for measuring such mechanisms. The motivating question is:

> How are minority and immigrant groups systematically and linguistically framed in large-scale language-model training data?

The proposed answer is not a single scalar. The framework instead constructs a per-group framing profile: a vector of distributional, semantic-role, attitudinal, and embedding-association dimensions. These dimensions are then inspected component-wise and, only exploratorily, summarized through principal component analysis (PCA). The aim is to avoid the common shortcut in which bias is first pre-shaped by the researcher and then rediscovered by the metric.

The current system remains incomplete. In particular, target/contrast-group identification and semantic attribution require redesign, and the frame-semantic layer is not yet validated by human annotation. This incompleteness is treated as part of the paper's scope rather than hidden as a weakness. The present contribution is methodological: it specifies how bottom-up corpus statistics, target-aware semantic attribution, and contextual association testing can be assembled into one reproducible diagnostic pipeline. Preliminary outputs are used only to show feasibility and failure modes.

The paper makes three contributions:

1. It argues for bias detection as recurrent framing analysis rather than fixed target--attribute matching alone.
2. It proposes the Evaluative Framing Index (EFI) as a multidimensional per-group profile rather than a pre-weighted bias score.
3. It presents a prototype corpus-filtering and association pipeline that converts large-scale pretraining material into candidate framing evidence for later human validation.

**2 Related Work and Gap**

Existing bias research covers several levels: corpus input, model representation, and output behavior. Word Embedding Association Test (WEAT) and its contextual descendants show that distributional representations encode human-like social associations. Contextualized variants further show that bias should be treated as a distribution across naturally occurring contexts rather than as one averaged point estimate. Output-side benchmarks and audits demonstrate that aligned systems can still show demographic sensitivity in judgments, generation, or evaluation. Representation-level work, including representation engineering, suggests that high-level concepts such as bias can correspond to manipulable directions or subspaces in model activations.

These lines of work establish that bias exists and can be measured at different stages. However, they often leave the linguistic carrier underspecified. Fixed word lists and polarity schemes test whether a hypothesized association is present. They are less suited to discovering how a corpus repeatedly constructs a group through role assignment, metaphor, collocation, and contextual construal. A sentence can be demographically relevant and evaluatively biased without matching an obvious stereotype word list. For example, repeated patienthood or affectedness can frame a group as acted upon rather than acting, even if no overt negative adjective appears.

Corpus-assisted discourse studies offer a complementary route. Log-likelihood ratio (LLR), collocation analysis, and post-hoc classification can surface recurrent discourse patterns from the corpus itself rather than imposing a frame inventory in advance. Work on latent personas and connotation frames also shows that agent/patient patterns and verb-level role relations can reveal social construal. Yet these bottom-up linguistic approaches are rarely integrated with association-testing methods such as WEAT/SEAT, and even more rarely applied to pretraining-scale corpora that shape contemporary LLM behavior.

The gap is therefore not that no component exists. Rather, the gap is integration. Existing work has studied embedding association, discourse framing, role extraction, affect, benchmarks, and representation steering, but mostly as separate measurement traditions. The present framework attempts to connect them: bottom-up frame discovery from corpus statistics, target/contrast-aware semantic attribution, contextual embedding association, and composite group profiling.

**3 Framework: Bias as Recurrent Framing**

The central assumption is that bias in training material is not exhausted by isolated lexical negativity. It can appear as a recurrent alignment between a demographic group and a set of linguistic positions: who is agentive, who is acted upon, who is described as thinking or feeling, who co-occurs with threat or dependency, and whose contexts are pulled toward negative or positive attribute regions in embedding space.

For each demographic group *g*, the framework constructs an Evaluative Framing Index:

```text
EFI(g) = [AgI, PI, SI, AttI, WEAT, SEAT-filtered, SEAT-full, Δ-SEAT]
```

The dimensions are defined as follows.

- Agency Index (AgI). The proportion of relevant occurrences in which the group is construed as an agent or initiator of an event. The target is not merely grammatical subjecthood, but event-level agency.

 - Patienthood Index (PI). The proportion of relevant occurrences in which the group is acted upon, affected, detained, displaced, excluded, attacked, governed, or otherwise placed in a patient-like role.

 - Subjectivity Index (SI). The proportion of relevant occurrences in which the group is granted autonomous consciousness, belief, intention, feeling, evaluation, or other subject-like mental perspective. This is kept separate from agency: a group can be active without being represented as internally subjective.

 - Attitudinal Index (AttI). A target-aware estimate of positive or negative evaluative attribution in the local context. The current version is under redesign: early prototype matching against polarity seeds is not sufficient for the final system.

 - WEAT. A type-level association score asking whether a group term is closer to discovered negative or positive frame terms than a contrast-group term is.

 - SEAT-filtered. A contextual association score computed over sentences retained as demographically relevant by the filtering pipeline. [CEAT>SEATs??]

  - SEAT-full. A contextual association score computed over all lexical hits before semantic filtering.
  - Δ-SEAT. The difference between SEAT-full and SEAT-filtered. This is intended as a diagnostic for valence bleed: association induced by broad lexical environments that may not actually be about the demographic group as a social target.



EFI is therefore not a sentiment score. It is also not a final scalar ranking. It is a group-by-dimension profile. PCA is used only as an exploratory summary of covariance among dimensions. If the first principal component explains a large share of variance, it may indicate a dominant axis along which groups differ in the current sample. If it does not, the result is still informative: framing is multidimensional and should not be collapsed prematurely.

**4 Pipeline**

The prototype pipeline has two major branches: a distributional branch and a semantic-attribution branch. The distributional branch is currently more implemented; the semantic branch is under revision.

***4.1 Corpus Filtering***

The initial implementation targets a small sample of Dolma v1.6. The filtering stage is designed to avoid both raw-corpus dilution and excessive hypothesis injection. Directly evaluating all sentences in a large corpus would allow irrelevant senses to dominate: *foreign key*, *black tea*, *white hat*, or place names may trigger demographic lexicons without being about social groups. The first methodological task is therefore candidate extraction, not final interpretation.

The current filtering pipeline proceeds in three steps.

First, a cheap lexical gate selects documents or sentences containing target or contrast-group terms. Target terms include minority, immigrant, refugee, racial, ethnic, and related demographic expressions. Contrast terms include citizen, local, western, national, and related majority or reference-group expressions. This gate is intentionally broad.

Second, MiniLM-based semantic retrieval compares candidate sentences against positive and negative query sets. Positive queries describe sentences about immigrants, refugees, racial or ethnic minority communities, foreign workers, asylum seekers, displaced families, and demographic group treatment. Negative queries describe irrelevant lexical confounds such as weather, geography, software, food, and color-object senses. Sentences pass if their positive semantic score and positive--negative margin exceed fixed thresholds.

Third, an embedding-based classifier estimates relevance. Training examples are manually labelled as relevant or irrelevant, encoded with MiniLM, reduced through PCA, and classified with logistic regression. High-probability sentences are kept; borderline cases are written to a review file for later human annotation and threshold revision.

In a preliminary run over one minimal Dolma sample, the system processed 1,392,502 extracted sentences. The lexical gate retained 123,976 sentences (8.90%). The semantic filter retained 5,442 sentences (0.39%). The classifier retained 2,895 high-probability relevant sentences (0.21%), with 1,590 additional borderline sentences marked for review. These numbers should not be read as corpus-level findings. They are pipeline diagnostics: the current extractor can sharply reduce the search space while preserving a reviewable borderline zone.

***4.2 Bottom-Up Frame Discovery***

From the retained sentences, the distributional branch computes non-adjacent collocational association for target and contrast groups. LLR is used to identify terms that are statistically over-associated with a group, and LogDice is retained as a complementary measure less dominated by raw corpus size. PPMI is not used as the primary collocate score because it can inflate rare events and adjacent multiword expressions; it also overlaps theoretically with embedding association measures that already encode PMI-like distributional structure.

Candidate frame terms are not admitted simply because they are associated. They must satisfy two conditions:

1. They are empirically differential: stronger around a target group than around the relevant contrast group, or vice versa.
2. They are semantically grounded: close enough to minimal positive or negative polarity anchors, or later grouped by human annotators into frame categories.

This preserves a bottom-up direction of analysis. The frame inventory is discovered and revised along the pipeline rather than fixed before the corpus is observed.

***4.3 Association Testing***

After candidate frame terms have been discovered and classified, the framework computes association scores. WEAT asks a type-level question: is a demographic term, considered as a word type, closer to discovered negative frames or positive frames relative to a contrast term? SEAT asks the contextual version: across actual corpus occurrences, are the sentences containing the target pulled toward one frame region more strongly than the contrast-group contexts?

The distinction matters. WEAT provides a stable type-level baseline, but it cannot model local usage. SEAT-filtered uses only demographically relevant sentences and should better approximate the framing contexts of interest. SEAT-full uses the larger lexical-hit set and is useful precisely because it may be contaminated. Δ-SEAT measures the difference between the two, making valence bleed visible rather than silently averaging it into the score.

***4.4 Semantic Attribution Under Revision***

The semantic branch is the least stable part of the current pipeline. Earlier versions relied on comparatively simple subject/object or predicate/argument proxies. That is insufficient. Agency, patienthood, and subjectivity are not hard-coded syntactic labels. A group can be grammatically subject but semantically patient-like, as in passive or affected constructions; conversely, a group can be syntactically embedded but still be the central intentional actor.

The planned redesign therefore moves toward target-level semantic attribution. Candidate sentences will first be grouped by structural clarity. Sentences with straightforward predicate--argument structure can be processed with SRL-like tools. More complex contexts require a frame-semantic or transformer-assisted layer, with fallback to manual validation. The goal is not to force all examples into one parser, but to assign role indices only where the system has enough target-specific evidence.

This redesign is central for the final system, but it is not required to make the current short paper coherent. The present paper treats the semantic branch as a planned validation-critical component, while using the implemented distributional and filtering branches as evidence that the framework is technically viable.

**5 Preliminary Diagnostics**

The preliminary system is not yet a validated bias detector. Its outputs are diagnostic in two senses: they show where the pipeline can reduce the corpus successfully, and they expose which error classes must be controlled before substantive claims are made.

First, the filtering pipeline shows that raw lexical inclusion is too noisy. A demographic lexicon alone retrieves many irrelevant cases, including color terms, nationality adjectives in food or geography contexts, and technical terms such as *foreign key*. This supports the decision to keep semantic retrieval and relevance classification before downstream measurement.

Second, collocational measures surface both potentially meaningful frames and predictable artefacts. Adjacent multiword expressions, demographic descriptors, punctuation/tokenisation issues, and institutional phrases can receive high association values without functioning as framing evidence. The paper therefore treats LLR/LogDice output as candidate evidence, not as an automatic frame inventory.

Third, preliminary PCA over the current group-by-dimension matrix suggests that embedding-association dimensions may dominate the first latent axis more strongly than symbolic role dimensions. This is only a hypothesis. At the current stage, the semantic-role dimensions are not reliable enough to support an empirical conclusion. The safer interpretation is methodological: component-level inspection is necessary before any PCA-derived scalar is treated as an EFI score.

Fourth, Δ-SEAT is promising as an error diagnostic. If SEAT-full and SEAT-filtered diverge strongly, then the broader lexical environment is adding association not present in demographically relevant contexts. This helps separate actual target framing from lexical valence bleed.

These diagnostics motivate the next iteration: human validation, target-aware semantic attribution, more robust target/contrast grouping, and component-level ablation before any final index is reported.

**6 Validation Plan**

The next stage requires human input at three points.

 - Sentence relevance. Borderline sentences from the relevance classifier should be manually reviewed. This will calibrate the semantic filter and prevent the pipeline from silently excluding difficult but important cases.

 - Frame classification. High-LLR and high-LogDice collocates should be grouped into frame categories by annotators after observation, not before. This preserves the corpus-linguistic direction of analysis: observe, classify, then measure. Minimal polarity anchors can seed the process, but they should not determine the final frame inventory.

 - Target-level semantic attribution. AgI, PI, and SI require validation because role assignment is linguistically subtle. A small manually annotated sample should check whether automatic role labels correctly identify the target group, the predicate, affectedness, agency, and subjectivity. A second annotator or expert adjudication would be needed for any claim beyond proof of concept.

The validation design should also include ablations. The pipeline should be rerun with alternative seed lists, without the semantic retrieval stage, with and without contrast terms, and with separate reporting of WEAT, SEAT-filtered, SEAT-full, Δ-SEAT, and role dimensions. The point is not to defend one composite score, but to identify which component contributes which type of evidence.

**7 Limitations** 

The current framework has four main limitations.

First, it depends on initial target and contrast lexicons. These priors are necessary for candidate extraction, but they must be explicit and not aligned too closely with the hypothesis being tested. The target lexicon should locate demographic contexts; it should not encode the expected frame.

Second, the semantic-attribution branch is unfinished. Until target-level role assignment is validated, AgI, PI, and SI should be treated as planned dimensions, not reliable measurements.

Third, MiniLM supplies useful retrieval and association geometry, but it also brings its own pretrained priors. Holding the encoder constant across corpora can support comparison, but it does not remove model bias from the measurement instrument.

Fourth, PCA is descriptive, not causal. PC1 may summarize cross-group variance, but it is not automatically a bias direction. Any interpretation of a dominant axis must be checked against component loadings, annotation, and alternative dimensional summaries.

These limitations are not peripheral. They define the validation agenda. A bias-detection pipeline that hides its priors, parser errors, and embedding assumptions risks turning technical convenience into a false empirical claim.

**8 Conclusion**

This paper presented a work-in-progress pipeline for detecting candidate framing bias in large-scale LLM training material. The proposed framework treats bias as recurrent target/contrast-group construal across distributional association, semantic roles, attitudinal framing, and contextual embedding association. Its output, EFI, is defined as a multidimensional per-group profile rather than a final sentiment-like scalar.

The current prototype demonstrates feasibility at the corpus-filtering and distributional-diagnostic stages, but it does not yet provide validated empirical findings about bias. That distinction is deliberate. The goal of the short paper is to establish a defensible research design: bottom-up frame discovery should be linked to target-aware semantic profiling and embedding association before one claims to measure bias. The next stage is human validation, semantic-attribution redesign, and ablation-based evaluation of each pipeline component.

**Mentor Questions**

1. Is the contribution sufficiently clear as a work-in-progress short paper, or should it be reframed as a thesis proposal?
2. Which part should be foregrounded for a 4-page version: corpus filtering, bottom-up frame discovery, EFI architecture, or validation design?
3. Is the preliminary Dolma filtering result enough as a feasibility diagnostic, given that semantic-role extraction is still under revision?
4. Should PCA-based EFI be retained in the paper, or moved to future work until the semantic dimensions are validated?
5. How much human annotation is minimally necessary before the final SRW submission can make a safe methodological claim?

**References to Add in Final ACL Format**

