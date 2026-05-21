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
* not aligned with the hypothesis being tested,
* not making any descriptive generalization that applies to the complete dataset.

## Encoder Split

The pipeline now separates the extraction encoder from the analysis encoder.

Reason:

* Phase 1 extraction is a high-throughput filtering step. It scans many lexical hits and only decides which sentences enter the reviewable filtered corpus. Using GTE ModernBERT here is substantially slower on a local MacBook Air.
* MiniLM is acceptable for Phase 1 if it is treated as a recall-oriented gate, not as a reported embedding metric.
* Phase 2 analysis still needs one stronger, fixed encoder for semantic disambiguation, frame refresh, WEAT, CEAT, and CEAT-full so reported scores share one embedding geometry.

Implementation decision:

* `extract.py` defaults to `EXTRACTION_EMBEDDING_PRESET = "minilm"` for semantic retrieval and the PCA+LogReg relevance classifier.
* The extraction preset is intentionally overrideable with `ARMADA_EXTRACTION_PRESET=gte_modernbert_base` for A/B calibration runs.
* `X/run_pipeline.py`, `X/semantic_group_resolver.py`, frame refresh, WEAT, CEAT, and CEAT-full default to `ANALYSIS_EMBEDDING_PRESET = "gte_modernbert_base"`.
* The analysis preset is overrideable with `ARMADA_ANALYSIS_PRESET`, but reported runs should keep it fixed unless explicitly comparing encoders.
* `X/embedding_config.py` centralizes both presets and the model catalog.

Consequences:

* Phase 1 semantic scores and classifier probabilities are extraction provenance only. They should not be compared directly to Phase 2 WEAT/CEAT/frame similarity scores.
* Phase 1 thresholds are MiniLM-specific; Phase 2 thresholds are GTE-specific.
* Re-running Phase 1 under a different extraction encoder changes corpus composition and must be reported.
* GTE ModernBERT may improve Phase 1 recall on complex long sentences; the reason not to default to it locally is throughput, not a methodological claim that it cannot improve extraction.
* Phase 2 results remain comparable across corpora only when the analysis encoder stays fixed.
* The catalog entries in `embedding_config.py` are options, not a claim that every listed model is downloaded or active.

## Extraction Gate Design

Extraction failure is not only an encoder problem. The correct Phase 1 review artifact is `dolma/semantic_filter_review.tsv`, not `X/output_review.tsv`. `semantic_filter_report.txt` is useful only when its timestamp matches the review/results files from the same extraction run; otherwise regenerate or ignore it for calibration.

The current review rows mostly passed the semantic gate and then landed near the classifier boundary (`relevant_probability` between `BORDERLINE_PROB_MIN` and `CLASSIFIER_THRESHOLD`). That points first to classifier calibration, training examples, and reference/index-like corpus noise, not to an encoder-only fix.

The extraction gate therefore uses two lanes:

```text
STRICT:          semantic_pos >= SEMANTIC_MIN
                 and semantic_margin >= SEMANTIC_MARGIN_MIN

SEMANTIC_RESCUE: semantic_pos >= SEMANTIC_RESCUE_MIN
                 and semantic_margin >= SEMANTIC_RESCUE_MARGIN_MIN

LEXICAL_HUMAN_RESCUE: lexical group token is structurally tied to
                      a human head noun or person suffix
```

`STRICT` rows enter the final filtered corpus at `CLASSIFIER_THRESHOLD`. `SEMANTIC_RESCUE` rows use a stricter `RESCUE_CLASSIFIER_THRESHOLD`; low-confidence rescue rows still go to review. Reference/index-like rows are blocked from final output and routed to review even when the classifier is high.

Review files include `review_flags` so calibration can separate:

* `low_semantic_margin`: semantic similarity itself is uncertain;
* `high_semantic_low_classifier`: retrieval thinks the sentence is relevant but the classifier is conservative;
* `reference_noise_like:*`: index, URL, markup, or bibliographic/citation-like text;
* `semantic_rescue`: a likely semantic false reject surfaced by positive-vs-negative margin;
* `lexical_human_rescue`: a likely semantic false reject surfaced by lexical-human structure, e.g. `American lady` or `Peruvian boys`.

This threshold split follows the 2026-05-13 extraction report: many rescue rows had high classifier probabilities and clearly relevant group framing, while only a small share of high-probability rescue rows carried reference-noise flags. The conservative response is not to lower the main semantic floor globally, but to let high-confidence rescue rows enter final output behind a higher classifier threshold and to route lexical-human false rejects to review at a lower review-only floor.

This matters because simply switching from MiniLM to GTE may not solve extraction by itself:

* query wording still defines the semantic target;
* absolute cosine floors are model-specific;
* a strong positive margin can be more informative than an absolute score when the query set underfits older, literary, or syntactically unusual sentences;
* Phase 1 review rows need to be judged by their bucket/flags and classifier probabilities, while `X/output_review.tsv` reflects Phase 2 target/role/binding issues.

## LLR / LogDice instead of PPMI

“Are these two words systematically associated?”

~~Levy & Goldberg (2014) showed that word2vec skip-gram implicitly factorizes a shifted PMI matrix. WEAT on word embeddings therefore already encodes PMI-derived associations. The earlier `EFI = α·PPMI + (1-α)·WEAT` weighted a raw signal against a smoothed version of itself; the α had no theoretical grounding.~~

## WEAT vs. CEAT

Both are kept — they measure different granularities:

* **WEAT** (Word Embedding Association Test): one type-level vector per group/frame term from the frozen encoder. Is *immigrant* as a type closer to F⁻ or F⁺ compared to *citizen*? Stable, interpretable, type-level.
* **CEAT-style contextual association** (Contextualized Embedding Association Test logic): one score distribution per group, built from sampled sentence contexts. Each context is scored as `cos(context, F⁻ centroid) - cos(context, F⁺ centroid)`, then reported as a group mean with `N` and `SE`.

Both produce per-group scores. CEAT captures context (e.g., *"immigrant workers contribute"* vs. *"immigrant workers were detained"* yield different vectors), while WEAT gives a single stable baseline per word.

CEAT replaces the old SEAT-style centroid averaging because ARMADA needs a distributional contextual metric, not just another single centroid score:

* it keeps the context-sensitive association signal that motivated the old full-vs-filtered comparison in the first place;
* it exposes uncertainty through `CEAT_N` and `CEAT_SE`;
* it is compute-bounded by deterministic sampling, so local GTE ModernBERT runs do not need to encode every lexical hit;
* it fits the target-bound design better: AgI/PI/SI/frame-AttI operate on mention-level evidence, while CEAT summarizes the broader contextual association distribution around reported groups.

Given this is a generalizable pipeline to probe how different datasets exhibit bias through a standard lens, using a standardized pre-trained encoder proves beneficial.

* No matter we run Dolma, Wikipedia, or Reddit text through the exact same GTE ModernBERT encoder, the encoder's priors are held constant.
Any difference in the resulting CEAT/WEAT scores can be attributed to the differences in the input sentences.
* It is much faster and more practical than training a custom word2vec model from scratch every time a user inputs a new dataset.

Operationally, CEAT-full has two legitimate run scopes:

* `ARMADA_CEAT_FULL_MODE=reported` (default): compute CEAT-full only for groups that will be written to `group_stats.tsv`. This preserves the reported output surface and avoids spending GTE ModernBERT time on groups below the reporting threshold.
* `ARMADA_CEAT_FULL_MODE=all`: compute CEAT-full for every lexical-hit group. Use this only for diagnostic audits of unreported groups or final exhaustive checks.

`ARMADA_CEAT_FULL_MODE=skip` is a development shortcut only; it intentionally omits `CEAT_full` and `delta_CEAT` and should not be used for reported results. `ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP` and `ARMADA_CEAT_MAX_FRAME_CONTEXTS` bound embedding work; the defaults are meant for local iteration, not a final power analysis.

This is an ARMADA operationalization of CEAT's core insight from Guo & Caliskan: contextualized embeddings should be treated as a distribution of association effects rather than a single static vector ([arXiv:2006.03955](https://arxiv.org/abs/2006.03955)). The current implementation reports mean and SE; it does not yet claim the full random-effects meta-analysis used in the original CEAT paper.

## EFI Architecture

EFI (Evaluative Framing Index) is no longer a single formula but a **per-group framing profile** extracting the most relevant information from the initial input. Each group is described by a vector of dimensions:

```
EFI(group) = [ AgI, PI, SI, frame_netAttI, WEAT, CEAT ]
```

The composite scalar is currently derived via **PCA** (Principal Component Analysis) on the group × dimension matrix, but this should be treated as exploratory rather than settled. The **first principal component (PC1)** captures the axis of maximum cross-group variance — the dimension along which groups differ most. Whether that axis can be read straightforwardly as "negative framing" is left open for later discussion.

* **PC1 loadings** = the empirically observed weighting of the current dimensions in this sample.
* **PC1 scores** = a provisional scalar ranking along that empirical axis.
* **Variance explained** = if PC1 > 50%, a single dominant axis accounts for most cross-group variation; if low, framing is multidimensional (also a finding).

No arbitrary weights. The data determines what "evaluative framing" consists of.

The regression `WEAT/CEAT ~ AgI + PI + SI + frame_netAttI` remains a complementary analysis: it tests whether syntactic-semantic patterns and frame-level evaluative association predict embedding-level association specifically, whereas PCA serves as a descriptive diagnostic of covariance structure rather than a final scalar definition.

## Target-Conditioned Semantic Dimensions

### Why these four dimensions

The four dimensions (Subjecthood, AgI, PI, SI) come from the observation that social-semantic construal of groups is multidimensional and each dimension carries different theoretical weight:

- **Subjecthood** — the baseline syntactic fact: is the group grammatically placed as the sentence subject? A group can be a grammatical subject while being entirely acted upon ("The refugees were deported"). Separating this from AgI prevents syntax from becoming ideology.
- **AgI (Agency Index)** — does the text attribute *volitional control or intentional efficacy* to the group? "Immigrants organized a protest" vs. "Immigrants arrived". Agency is a social construct under active dispute in immigration discourse; it is not simply ARG0.
- **PI (Patienthood Index)** — does the text attribute *affectedness or being acted upon*? A group can be PI as syntactic object, as passive subject, or as the subject of an affectedness predicate ("The refugees suffered"). This maps to constructions of vulnerability, passivity, and victimhood.
- **SI (Subjectivity Index)** — does the text attribute *mindedness: feelings, beliefs, perceptions, or inner experience*? A group that is represented as thinking or feeling is represented as a full social agent; a group portrayed only as feared or mistrusted by others is construed through others' subjectivity, not its own.

These four map exactly onto the distinct failure modes identified in the extraction report and build.md: syntactic subject ≠ agency; objecthood ≠ patienthood; mental-state predicate target ≠ mental-state predicate subject.

### Why the hybrid score, not pure cosine

Pure prototype similarity (cosine of sentence embedding against dimension-defining prototypes) was considered and rejected as the sole evidence source. The problem is disambiguation of target role:

```text
"They feared deportation."
Target = they / migrants → SI: high (target is the experiencer)
                           negAttI: not necessarily high

"People feared them."
Target = them / migrants → SI: low (target is NOT the experiencer)
                           negAttI: high (target is construed as fear-inducing)
```

Sentence-level embedding similarity will conflate these because both sentences have high cosine similarity to mental-state prototype sentences. Only explicit target-role information makes them separable. This is why the design is a hybrid:

```text
score(dimension, target) =
      semantic_similarity(focus_text, prototype)   ← target-bound context embedding
    + syntactic_evidence(dep, SRL)                 ← who is arg0/arg1/nsubj
    - scope_penalty(negation, correction)          ← "not a threat", "falsely accused"
    - non_target_penalty(binding_distance)         ← frame term is about someone else
```

The `focus_text` is the annotated context window `[GROUP:token][PRED:head_verb] ...sentence...`, already implemented in `step3_attitudinal_prototypes._build_focus_text()`. This shifts the question from *"is this verb similar to 'think'?"* to *"does this context, centered on the group and its governing predicate, resemble a context where [GROUP] is thinking?"*.

For **AttI**, prototype/definition similarity is appropriate because evaluative stance is expressed at the clause/sentence level and the question is straightforwardly about the local context.

For **AgI/PI**, prototype similarity alone is too weak — syntactic and voice evidence (passive, `nsubjpass`, affectedness predicates) is more reliable and cheaper. Prototype similarity is used as auxiliary confirmation, not as the primary gate.

For **SI**, prototype similarity is appropriate *only if the target's role is explicit* in the focus text — which the `[GROUP:token][PRED:verb]` formatting ensures.

The redesign is not "one classifier decides every metric." The core unit is:

```text
(target group mention, local predicate/frame evidence, scope flags)
```

Each reported semantic dimension answers a target-conditioned question:

|Dimension|Question|Evidence|
|---|---|---|
|`Subjecthood`|Is the group grammatically placed as subject?|dependency/SRL syntax only; diagnostic, not evaluative|
|`AgI`|Is the group attributed control, volition, or intentional efficacy?|target-bound predicate evidence + SRL/dependency support + low-agency suppression|
|`PI`|Is the group attributed affectedness, vulnerability, or being acted upon?|patient labels, passive/object evidence, affectedness predicates such as `suffer`, `flee`, `detain`, `deport`|
|`SI`|Is the group attributed mindedness, feeling, belief, or interior state?|target-bound mental-state predicates such as `fear`, `believe`, `hope`, `worry`|

So "target-conditioned" means the score is not assigned because a sentence contains a relevant word somewhere. It is assigned only when the relevant predicate/frame can be connected to a resolved group mention.

The evidence stack is deliberately hybrid:

* **definition/prototype layer**: each dimension has a semantic definition (agency = volitional control; patienthood = affectedness; subjectivity = mindedness);
* **syntactic/SRL layer**: dependency and SRL provide candidate links between group mentions and predicates;
* **predicate-cue layer**: verb classes correct obvious construct mismatches, e.g. `arrive` gives subjecthood but not AgI; `suffer` gives PI; `fear` gives SI without automatic AgI;
* **scope/review layer**: negation, correction, quotation, reported speech, contrast, and ambiguous target binding do not disappear into the score; they surface as review flags.

This is why generic SRL is demoted rather than removed. SRL can suggest `ARG0` / `ARG1`, but ARMADA's constructs are social-semantic attributions, not ordinary PropBank roles.

## Q & A

 Major failures in feature construction: target binding, local attitude scope, subjecthood/agency collapse, and SRL overreach.

|Problem|Current answer|
|---|---|
|Nearby evaluative language may not target the group: "Migrants were falsely accused of being dangerous."|Frame terms are bound to group mentions before reported AttI is counted; correction/denial blocks the frame from the reported numerator and routes it to review.|
|Positive and negative language can target different entities: "The minister praised volunteers while blaming refugees for the crisis."|`bound_frames` records which frame term is bound to which group; unrelated local positivity no longer becomes positive AttI for the wrong group.|
|Compositional reversal: "not a threat", "falsely accused", "it is false that..."|Scope flags block those frame terms from reported frame-AttI and expose them through `frame_binding_flags` / `frameReview`.|
|Syntactic subject does not equal agency: "The refugees arrived."|`Subjecthood` is now separate from `AgI`; low-control predicates suppress automatic agency.|
|Patienthood can occur without objecthood: "The refugees suffered."|Affectedness predicates can add `PI` even when the group is syntactic subject.|
|Subjectivity can occur without strong agency: "The refugees feared deportation."|Mental-state predicates add `SI`; non-volitional mental states do not automatically add `AgI`.|
|Generic SRL is not the right construct for social framing.|SRL is auxiliary evidence; final AgI/PI/SI depends on target binding, predicate semantics, and review flags.|
|MWE / modifier-head group phrases can double count anchors.|Primary `GroupMention` anchors drive association and frame-AttI; same-head MWE children are suppressed there.|
|Political labels can contaminate demographic claims.|Political labels are reported separately as `political` and excluded from demographic frame-candidate discovery.|

This is still not a full discourse parser. The claim is narrower: the pipeline now makes the main failure modes visible and prevents the most obvious false positives from entering the reported group-level metrics.

## AttI as Frame Association

Local prototype similarity is not removed; it is demoted. It can still answer:

```text
Does the local snippet around this group resemble a positive/negative attitude prototype?
```

But that is not the same as:

```text
Is the corpus systematically framing this group through F⁻ or F⁺ evaluative frames?
```

The first question is a sentence-local diagnostic. It is brittle under target ambiguity, negation, quotation, and defended attacks. The second question is the reported AttI claim, and it belongs at the frame-association layer.

Reported AttI is now computed at group level:

```
frame_negAttI = share of group sentences with target-bound F- frame terms
frame_posAttI = share of group sentences with target-bound F+ frame terms
netAttI       = frame_negAttI - frame_posAttI
```

This makes the claim weaker and more defensible:

* local prototype AttI says: nearby context resembles a positive/negative attitude prototype;
* target-bound frame AttI says: across the corpus, this group is statistically linked to F⁻ or F⁺ evaluative frames, after obvious scope-blocked cases are excluded from the reported numerator.

The second claim fits ARMADA's corpus-framing goal better and is structurally closer to `Δ-CEAT`: both compare an association signal after target/scope filtering against broader distributional pressure.

## Complex Sentences

The pipeline should not claim to fully resolve every long, embedded, quoted, or anaphoric sentence. The operational rule is:

* count AgI/PI/SI when target binding is clear enough;
* compute evaluative framing at corpus level through target-bound frame association and WEAT/CEAT;
* keep local prototype AttI as a review/debug signal;
* route negation, quotation, defended attacks, and unclear anaphora into review flags where possible.

This keeps automatic metrics interpretable without pretending to solve all discourse-pragmatic structure.

## Political Group Scope

Political or ideological labels such as `communist`, `soviet`, and `conservatist` remain exploratory social-group terms for now, but they should not be used to overstate a purely demographic claim.

Operationally:

* keep them in extraction as corpus signals;
* report them as `political`;
* exclude them from demographic frame-candidate discovery;
* do not use them as core evidence for minority/immigrant demographic framing unless separately justified.

If the project narrows to strict demographic categories, these terms should move to a separate political-group inventory rather than silently remaining mixed with ethnic, racial, or migration-status terms.

## SI is independent from AgI

A group can be 100% agent (AgI=1.0) but 0% subjective (SI=0) — portrayed as *doing* things but never *thinking or feeling*. This distinction matters: it separates "active agency" from "full autonomous personhood." Nuances that define "autonomous personhood" will be extensively furthered with embodied semantics.
