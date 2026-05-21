---
title: cache
description: Temporary page for storing drafts to be reviewed and submitted 
tags:
 - cache
---

## [May 16] AILC Draft - Submitted

**Title**

*Unbias* Is What We Need: Reproducible Social Bias Profiling of LLM Training Data

**Abstract**

This poster presents ongoing work from the EU-funded ARMADA project, on the task
“analysis of bias in training materials for chatbots/LLMs via linguistic corpora analysis.” It
builds on evidence that bias propagates from training data distributions into model
representations and output behaviour. Such bias is not limited to stereotypes and overt
discrimination; it may also emerge explicitly through recurrent semantic associations, stance
patterns, evaluative framing, as well as implicitly through nuanced asymmetrical depictions of
social groups—particularly how minority and immigrant communities are systematically
framed relative to dominant groups—in situated discourse across contexts.

Existing LLM-bias research has produced strong results, but its main approaches often remain
separated across distinct stages of the bias-production chain. The resulting gap is one of
integration: even bottom-up biased framing analyses are rarely connected to demographic
association testing and are almost never applied to the pretraining-scale corpora that shape
contemporary LLMs. This proposed pipeline addresses that by combining large-scale data
filtering, bottom-up collocate discovery, inter- and per-group profiling, and qualitative
validation. Rather than imposing extensively predefined stereotype or framing categories, it
seeks to limit evaluative priors by first uncovering salient collocational evidence through LLR
and LogDice measures. These candidate patterns then feed a computational profiling layer that
positions target and contrast groups across semantic structures, evaluative frames, and
embedding association dimensions. The resulting profiles are to be validated through post-hoc
small-scale human-led annotation and interpreted via fine-grained corpus analysis. This
approach prioritises a discovery-first, reproducible architecture for bias measurement over one-
off moral labelling.

Early test runs have highlighted limitations of standard semantic feature construction.
Dependency parsing and predicate-argument role labelling often underspecify target-specific
semantic attribution, especially when they are used to encode multidimensional social frames
like *Agency*, *Patienthood*, and *Subjectivity*. Likewise, local attitudinal diagnostics reveal
significant attribution ambiguity in contextually complex constructions where evaluative
language co-occurs with multiple demographic groups. In response, current revisions are
introducing explicit target binding alongside tighter compositional handling for structural
phenomena such as double negation, embedded clauses, anaphora. By reporting the conceptual
architecture, preliminary diagnostic findings, and the proposed validation strategy, this work
demonstrates how linguistic analysis can make computational bias detection more sensitive to
discourse-level meaning and more interpretable for fairer conversational AI.

## [Jun 4] Venice Draft - 300 words


## [Jun 8] AACL SRW Short Paper - max-5 pages

**Toward Target-Aware Framing Bias Detection: A Linguistically Grounded Pipeline for Group Profiling**

Anonymous AACL submission

**Abstract**

This paper presents ARMADA, a pipeline for detecting candidate framing bias in large-scale language-model pretraining data. The framework treats bias not as a single sentiment score but as a recurrent pattern of target and contrast-group construal across distributional association, semantic-role attribution, target-bound evaluative framing, and contextual embedding association (Caliskan et al., 2017; Rashkin et al., 2016; Lucy et al., 2022). For each demographic group the pipeline constructs a multidimensional profile: *Subjecthood*, *Agency Index* (AgI), *Patienthood Index* (PI), *Subjectivity Index* (SI), target-bound *frame-derived AttI*, *WEAT*, *CEAT*, *Δ-CEAT*, and an exploratory PCA-based *Evaluative Framing Index* (EFI). A preliminary single-shard run on the Dolma v1.6 sample (Soldaini et al., 2024) suggests that the principal axis of cross-group variation is dominated by embedding-association rather than syntactic-role evidence. The contribution is methodological: the paper specifies the architecture, surfaces design forks, and defines validation requirements rather than reporting final group-level scores.

**1 Introduction**

Large language models inherit not only lexical associations from their training data but also recurrent ways of placing social groups into events, roles, and discourse frames (Caliskan et al., 2017; Bender et al., 2021; Blodgett et al., 2020). A model may avoid explicit slurs while still reproducing a skewed pattern in which a group is repeatedly represented as threatening, passive, or dependent. Such patterns are not always visible to benchmark designs that test only fixed target–attribute pairs (Nadeem et al., 2021; Nangia et al., 2020).

This paper proposes a pipeline for detecting such mechanisms in pretraining data. The motivating question is: how are minority and immigrant groups linguistically framed relative to dominant groups? The answer is not a single scalar. The framework constructs a per-group profile that separates grammatical prominence, semantic agency, affectedness, subjectivity, evaluative frame association, and contextual embedding association. The aim is to avoid the shortcut in which bias is first pre-shaped by the researcher and then rediscovered by the metric (Blodgett et al., 2020; Goldfarb-Tarrant et al., 2021).

The paper makes three claims:

1. Framing bias detection benefits from being treated as recurrent multidimensional construal rather than fixed target–attribute matching.
2. A two-phase pipeline can connect pretraining-data filtering, target-aware semantic attribution, target-bound frame association, and embedding association under a shared target set.
3. EFI is best treated as a descriptive group-profile summary; component loadings rather than the composite carry the diagnostic weight.

**2 Related Work and Gap**

Prior work spans data input, model representation, and output behaviour. WEAT and its contextual descendants show that distributional representations encode human-like social associations (Caliskan et al., 2017; May et al., 2019; Guo & Caliskan, 2021). Output-side benchmarks demonstrate that aligned systems still show demographic sensitivity (Nadeem et al., 2021; Nangia et al., 2020). Representation-level work suggests that abstract concepts may correspond to manipulable directions in model activations (Zou et al., 2023). Surveys document that these traditions are typically pursued separately and that the linguistic carrier is often left underspecified (Blodgett et al., 2020; Stanczak & Augenstein, 2021).

Fixed word lists test whether a hypothesised association is present but are less suited to discovering how a dataset repeatedly constructs a group through role assignment and collocation. Repeated patienthood, for instance, can frame a group as acted upon even when no overt negative adjective appears (Sap et al., 2017). Work on connotation frames (Rashkin et al., 2016), agent–patient asymmetries (Sap et al., 2017), and contextual semantic axes (Lucy et al., 2022) shows that role patterns can reveal social construal. Statistical association measures such as log-likelihood ratio (Dunning, 1993) and LogDice (Rychlý, 2008) provide complementary discovery tools.

The gap is integration: embedding association, role extraction, and frame discovery are usually treated as separate traditions. ARMADA connects them in one pipeline anchored to the same target and contrast lexicons.

**3 Framework**

For each demographic group *g*, the pipeline reports:

- *Subjecthood*: syntactic subjecthood, reported separately from agency (Dowty, 1991).
- *AgI*: proportion of mentions in which *g* is construed as a semantic agent.
- *PI*: proportion of mentions in which *g* is acted upon or affected.
- *SI*: proportion of mentions linked to mental-state predicates.
- *Frame-derived AttI*: target-bound evaluative frame association (*netAttI*).
- *WEAT*: type-level embedding association with F⁻/F⁺ seed centroids.
- *CEAT* (filtered) and *Δ-CEAT*: contextual association over filtered vs. all lexical hits; the difference diagnoses associative contamination.
- *EFI_PC1*: exploratory PCA over [AgI, PI, SI, netAttI, WEAT, CEAT], oriented by an empirical loading anchor.

**4 Pipeline**

Phase 1 produces a filtered sentence-level subset from pretraining data; Phase 2 computes the per-group profile.

***4.1 Pretraining-Data Filtering***

Phase 1 streams Dolma v1.6 (Soldaini et al., 2024) parquet shards.[^1] The preliminary run uses a single shard (187,078 documents; 1,477,953 sentences). A three-stage gate applies: (1) a lexical gate selects sentences containing target or contrast-group tokens, excluding highly polysemous lemmas as standalone triggers; (2) MiniLM semantic retrieval (Wang et al., 2020) compares candidates against positive and negative query sets, with rescue lanes for strong-margin and demonym + human-head patterns; (3) a MiniLM-embedding PCA + logistic-regression classifier retains high-probability rows and routes borderline rows to a review file with diagnostic flags. A separate file retains all lexical hits for CEAT-full. On the preliminary shard: 139,316 lexical hits (9.4% of sentences), 5,291 retained, 14,643 routed to review. Full target/contrast lexicons are reported in Appendix A.

[^1]: Dolma v1.6 sample, 16.4 GB / ~10 billion tokens, Hugging Face, ODC-BY licence.

***4.2 Target Resolution and Semantic Attribution***

Phase 2 preprocesses retained sentences with spaCy (Honnibal et al., 2020), then resolves group mentions through a shared lexicon and a GTE ModernBERT (Zhang et al., 2024) semantic backoff layer for genuinely ambiguous tokens (*black*, *polish*, *native*, *asylum*).

Role attribution combines dependency parsing with a BERT-based SRL model (Devlin et al., 2019; Shi & Lin, 2019) as structural evidence, but the primary gate for AgI, PI, and SI is target-conditioned prototype similarity: the Phase 2 encoder scores an annotated context window against dimensional prototype sentences. A dimension is assigned only when its similarity exceeds a floor and beats competing dimensions by a relative margin. This replaces verb-class membership checks with a continuous, target-aware mechanism. Negation of the governing predicate blocks all role assignment and routes the mention to review.

The role-theoretic profile is restricted to human-referent mentions; non-human-head modifications (*Iranian cuisine*, *foreign key*) are suppressed. This scope choice is motivated by construct validity (see §6).

***4.3 Frame Discovery and Target Binding***

Non-adjacent LLR (Dunning, 1993) and LogDice (Rychlý, 2008) scores surface candidate frame terms. Candidates are compared against sentence-level seed F⁻/F⁺ examples via the Phase 2 encoder; auto-admitted terms accumulate in a JSON inventory. Auto-admitted word-level terms are used only for frame-AttI binding, not as WEAT/CEAT anchors (which use seed-sentence centroids directly).

Frame terms are bound to the nearest group anchor via dependency, shared predicate, or bounded proximity. Scope flags (negation, correction, quotation, contrast, multi-group) block or route flagged attachments to review. Reported *netAttI* is therefore target-bound: a frame term attaching to a different group does not contribute to *g*'s score.

***4.4 WEAT, CEAT, Δ-CEAT, EFI***

WEAT encodes group lemmas against seed centroids, yielding a type-level association difference (Caliskan et al., 2017). CEAT encodes sampled filtered contexts and scores each against the same centroids, reporting mean, N, and SE per group (Guo & Caliskan, 2021). CEAT-full reuses the same centroids on all lexical hits; Δ-CEAT quantifies contamination from non-demographic environments. EFI is PCA over [AgI, PI, SI, netAttI, WEAT, CEAT] for groups with N ≥ 50, with PC1 oriented by an empirical loading anchor.

**5 Preliminary Observations**

The preliminary run is reported as quantified pipeline behaviour rather than group-level scores.

**Filtering residue.** Of 139,316 lexical hits, 5,291 pass the strict gate (0.36% of total sentences). 14,643 borderline rows are routed to review by flag type: ~50% low semantic margin, ~25% high semantic / low classifier, remainder rescue or reference-noise. This constitutes a quantified review residue rather than a discarded set.

**Multidimensional divergence.** Group profiles do not move uniformly: a group can show high Subjecthood with low AgI, or high WEAT with near-zero netAttI. This non-redundancy is the property the profile is designed to expose.

**Loading-pattern hypothesis.** Under the current configuration, PC1 appears dominated by embedding-association dimensions (WEAT, CEAT) rather than syntactic-role dimensions (AgI, PI, SI). If replicated at full scale, this would suggest that the principal axis of cross-group variation sits in distributional geometry rather than symbolic role structure. This observation is hypothesis-generating; PC1 stability with a small group count requires replication.

**6 Design Discussion**

**Complex constructions.** The pipeline detects negation, quotation, correction, contrast, and multi-group sentences through lexical scope cues. Flagged frame attachments are excluded from reported counts; flagged role attributions are routed to review. Full structural parsing of negation scope is brittle at scale (Morante & Sporleder, 2012); the current design is precision-oriented. Defended attacks remain a known blind spot.

**Subjecthood in EFI.** Subjecthood is under evaluation as an additional EFI input. Because it is explicitly separated from AgI, their joint presence in the PCA allows the data to reveal whether syntactic prominence and semantic agency co-vary or diverge. A symmetric Objecthood dimension is not included because PI already encompasses patient-as-object and affectedness cases.

**Frame auto-admission.** The LLR + centroid-similarity gate admits generic collocates more readily than narrow frame terms. The auto-admitted inventory is treated as candidate evidence requiring human review; WEAT and CEAT remain anchored to seed centroids to limit noise propagation.

**Human-referent scope.** Role-theoretic dimensions are computed only on human-referent mentions. There is no strong evidence that the encoder cleanly separates human and non-human senses of group adjectives; Δ-CEAT quantifies the gap. Preliminary Δ-CEAT does not show categorical divergence, supporting future extension toward holistic group-signifier profiling.

**Encoder and lexicon priors.** The pipeline is parameterised by pretrained encoders that may carry their own biases. Encoders are held constant across groups; encoder-substitution is a planned ablation. Target/contrast lexicons shape what is extracted and are reported in full (Appendix A). Comparison against alternative demographic lexicons (Dixon et al., 2018; Czarnowska et al., 2021) is planned.

**7 Next Stages**

Validation proceeds along four tracks: (1) sentence relevance — borderline and rescue rows reviewed against manual judgement; (2) group resolution — ambiguous tokens checked for inanimate-head suppression and semantic-backoff decisions; (3) frame binding — target-bound links checked in multi-group, negated, and contrastive sentences; (4) semantic roles — annotated sample tests prototype-similarity confirmation, negation-scope blocking, and SRL-as-auxiliary evidence.

**8 Limitations**

The framework depends on initial lexicons (reported in Appendix A). Encoder priors are unavoidable at this scale. The frame inventory is partly auto-refreshed and requires human review. Δ-CEAT diagnoses contamination but does not resolve it. EFI is descriptive; PC1 stability with the current group count is itself a limitation.

**9 Conclusion**

The pipeline treats framing bias as recurrent target/contrast-group construal across distributional association, semantic-role attribution, target-bound frame association, and contextual embedding association. The preliminary run indicates that the principal axis of cross-group variation is dominated by embedding-association evidence. The next stage is to validate the per-group profile against human judgement and to extend the human-referent scope toward holistic group-signifier profiling.

**References**

- Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? *FAccT 2021*, 610–623.
- Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *ACL 2020*, 5454–5476.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.
- Czarnowska, P., Vyas, Y., & Shah, K. (2021). Quantifying social biases in NLP: A generalization and empirical comparison of extrinsic fairness metrics. *TACL*, 9, 1249–1267.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*, 4171–4186.
- Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. *AAAI/ACM AIES 2018*, 67–73.
- Dowty, D. (1991). Thematic proto-roles and argument selection. *Language*, 67(3), 547–619.
- Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. *Computational Linguistics*, 19(1), 61–74.
- Goldfarb-Tarrant, S., Marchant, R., Sánchez, R. M., Pandya, M., & Lopez, A. (2021). Intrinsic bias metrics do not correlate with application bias. *ACL-IJCNLP 2021*, 1926–1940.
- Guo, W., & Caliskan, A. (2021). Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases. *AAAI/ACM AIES 2021*, 122–133.
- Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength natural language processing in Python.
- Lucy, L., Tadimeti, D., & Bamman, D. (2022). Discovering differences in the representation of people using contextualized semantic axes. *EMNLP 2022*, 3477–3494.
- May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). On measuring social biases in sentence encoders. *NAACL 2019*, 622–628.
- Morante, R., & Sporleder, C. (2012). Modality and negation: An introduction to the special issue. *Computational Linguistics*, 38(2), 223–260.
- Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models. *ACL-IJCNLP 2021*, 5356–5371.
- Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-Pairs: A challenge dataset for measuring social biases in masked language models. *EMNLP 2020*, 1953–1967.
- Rashkin, H., Singh, S., & Choi, Y. (2016). Connotation frames: A data-driven investigation. *ACL 2016*, 311–321.
- Rychlý, P. (2008). A lexicographer-friendly association score. *RASLAN 2008*, 6–9.
- Sap, M., Prasettio, M. C., Holtzman, A., Rashkin, H., & Choi, Y. (2017). Connotation frames of power and agency in modern films. *EMNLP 2017*, 2329–2334.
- Shi, P., & Lin, J. (2019). Simple BERT models for relation extraction and semantic role labeling. *arXiv:1904.05255*.
- Soldaini, L., Kinney, R., Bhagia, A., et al. (2024). Dolma: An open corpus of three trillion tokens for language model pretraining research. *ACL 2024*, 15725–15788.
- Stanczak, K., & Augenstein, I. (2021). A survey on gender bias in natural language processing. *arXiv:2112.14168*.
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS 2020*.
- Zhang, X., Zhang, Y., Long, D., et al. (2024). mGTE: Generalized long-context text representation and reranking models for multilingual text retrieval. *arXiv:2407.19669*.
- Zou, A., Phan, L., Chen, S., et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.
