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

The current extraction gate has two lanes (no classifier):

```text
STRICT:        semantic_pos >= SEMANTIC_MIN (0.34)
               AND semantic_margin >= SEMANTIC_MARGIN_MIN (0.03)

STRONG_MARGIN: semantic_margin >= SEMANTIC_STRONG_MARGIN (0.10)
```

A row entering either lane is kept unless its `review_flags` contain a `reference_noise_like:*` entry (index/URL/bibliographic patterns), in which case it is blocked from `semantic_filter_results.tsv` and routed to review. Rows that pass neither lane but still have `semantic_margin >= REVIEW_MARGIN_MIN (0.05)` go to review for human inspection.

Review files include `review_flags` so calibration can separate:

* `strong_margin`: row entered the STRONG_MARGIN lane (high pos-vs-neg differential, low absolute pos);
* `low_semantic_margin`: semantic similarity itself is uncertain;
* `high_semantic_low_margin`: retrieval pos is high but margin is below the STRICT floor;
* `reference_noise_like:*`: index, URL, markup, or bibliographic/citation-like text;
* `semantic_borderline`: catch-all when no other flag fires.

The two lanes are complementary, not redundant. STRICT captures sentences whose POS query is well-matched in absolute terms (typical Western news framing). STRONG_MARGIN captures sentences whose absolute pos is low because the sentence is short, colloquial, or stylistically distant from the queries, but where the gate is still confident the sentence belongs to the demographic-framing class because pos clearly exceeds neg.

### Classifier removal (2026-05)

Earlier extractions used a third decision layer: a PCA + logistic-regression classifier trained on a hand-labelled `filter_training_samples.txt` (≈258 rows). The classifier was meant to filter polysemous and lexical false positives that semantic margin alone could not catch (e.g. `asylum for the insane`, demonym-as-institution names).

A controlled ablation was run on a 10,000-sentence sample of `semantic_filter_lexical_all.txt`, restricted to candidates with `semantic_margin >= 0.10`:

* with classifier (`prob >= 0.45`): **287 kept** out of 458 candidates (37% drop rate);
* without classifier: **458 kept**;
* the 171 sentences the classifier dropped, sorted by margin, were inspected manually.

Almost all of the classifier's drops at this margin tier were true RELEVANT sentences whose surface form did not match any pattern in the training set — refugee/asylum sentences in policy or organizational context, demonym + animate human structures with non-immigration framing, statistical demographic descriptions. The training set had no realistic way to cover all such surface variations, because:

* the corpus has 184 distinct demonyms in `GATE_TOKENS`, only 30+ of which receive enough training examples to anchor a PCA principal direction;
* most demonyms appear in the corpus across topics far broader than the immigration/refugee framing that dominates the labelled training set (sample frequency analysis on `semantic_filter_lexical_all.txt`: e.g. `sudanese` 25% demonym+human / 27% demonym+inanimate, `syrian` 27% / 23%, `afghan` 16% / 19%);
* a single-shard PCA on 258 rows projects the 384-dim MiniLM space onto ≈85 components fit to whatever clusters happened to be over-represented (refugee/migrant/immigrant in the labelled set), so unfamiliar surface forms collapse into the IRRELEVANT half-space regardless of meaning.

The reverse ablation (sample of 100 sentences kept by margin-only filter, manually labelled) showed a false-positive rate of ≈14%, vs. ≈7-8% with the classifier. So the classifier reduced FP by half but at the cost of a 37% recall loss on the same boundary band. The trade was not favourable for our target population (low-frequency demonyms in non-immigration framing), and Phase 2 `resolve_group_token()` and `INANIMATE_NOUNS` already filter most of the surviving FP at mention-resolution time.

The classifier, the training file consumed by it, and the LEXICAL_HUMAN / SEMANTIC_RESCUE rescue lanes that existed only to compensate for classifier conservatism, were removed together. `semantic_filter_results.tsv` no longer carries a `relevant_probability` column. The replacement gate is the two-lane design above.

This decision is reversible: re-introducing a classifier is acceptable if a substantially larger, surface-balanced training set becomes available. The current view is that growing the training set further by hand mostly redistributes the same systematic bias rather than removing it.

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

EFI (Evaluative Framing Index) is a **per-group framing profile** rather than a single formula. Each group is described by a six-dimension vector:

```
EFI(group) = [ AgI, PI, SI, frame_netAttI, WEAT, CEAT ]
```

PCA on the group × dimension matrix yields principal components. Earlier versions tried to flip PC1's sign so "negative framing" was always positive (`orientation_anchor = +PI + netAttI + WEAT + CEAT − AgI − SI`); that flip silently encoded a prior assumption that the six dims covary toward a single negative-framing axis. The 2026-05-24 calibration on the live corpus showed this assumption is empirically wrong on this corpus: AgI / PI / SI cluster opposite frame-netAttI / CEAT (correlations `PI ↔ CEAT = −0.30`, `SI ↔ CEAT = −0.43`, `AGI ↔ netAttI = −0.42`). The flip was removed.

The current report is two-component:

* **PC1** captures the axis of largest cross-group variance — whatever that axis happens to be in the sample.
* **PC2** is the largest variance orthogonal to PC1.
* **Sign Calibration**: PCA component signs are mathematically arbitrary ($\pm v$). To ensure deterministic orientations across different SVD solvers and data runs, `X/run_pipeline.py` implements post-hoc SVD sign calibration in `_compute_efi`: PC2 is locked deterministically so that WEAT loading is positive ($+0.637$) and SI loading is negative ($-0.626$). Under this orientation, positive PC2 indicates higher static embedding association / lower subjectivity (e.g., *indigenous* $+2.233$), and negative PC2 indicates higher subjectivity / lower static embedding association (e.g., *american* $-2.264$).
* **Loadings** for both components are reported; substantive interpretation comes from which dims load on which axis for the current run, not from a fixed sign convention.
* **Per-group scores** for both PCs are written to `group_stats.tsv` (`EFI_PC1`, `EFI_PC2`).

This means EFI no longer claims to give a single scalar of "negative framing" per group. What it gives is the two largest axes of how groups differ in this corpus along the six framing dimensions. The corresponding research finding from the current corpus: bias surfaces along **two distinct modalities** — *contextual salience* (PC1: all dimensions load positively, explaining ~40.1% of variance, capturing groups embedded in high-intensity crisis/conflict narratives vs. diffuse contexts) and the tradeoff between *static structural evaluation* in the embedding space vs. *dynamic compositional subjectivity* in context (PC2: WEAT loading $+0.637$ vs. SI loading $-0.626$, explaining ~24.2% of variance). Both are legitimate readings of bias, and aggregating them into one scalar would erase the distinction.


The regression `WEAT/CEAT ~ AgI + PI + SI + frame_netAttI` remains a complementary analysis: it tests whether attribution dimensions and frame-level evaluative association predict type-level / context-level embedding association, whereas PCA describes covariance structure across groups. The two analyses answer different questions.

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

### Why prototype surface variation is controlled

Prototype lists were originally written by hand in a single dominant tense/aspect/number per dimension: PI was uniformly past-passive plural, AttI was uniformly past-copular, AGI/SI were mostly past simple. With a relative-margin gate (`DIM_MARGIN=0.04`), a tense-driven cosine shift on the order of 0.01–0.03 was enough to flip the winning dimension on present-progressive sentences such as *"Asians are being attacked"*. The asymmetry was not uniform across dimensions, so PI was systematically under-attributed relative to AGI/SI on present-tense data.

The fix is structural rather than threshold-based, but it differs by construct.

For **AttI**, polarity lives in the complement, so the seeds are stored as canonical complements (`unwanted and unwelcome`, `welcomed and respected`, etc.) and surface-varied at module load across a tense × number grid using simple `be`-form rules (no external inflection library required).

For **AGI / PI / SI**, an earlier verb-grid approach was tried — bare verb lemmas expanded across tense/aspect/number with `lemminflect` — and removed. Two problems: a verb list re-introduces the same evaluative-prior problem the project explicitly avoided when it eliminated `SUBJECTIVE_VERBS` etc. from `lexicons.py`; and stative verbs (`know`, `believe`, `understand`) generated ungrammatical progressive cells. The current design is six hand-written paraphrase sentences per dimension. Index 0 is the ruling definition (AGI ↔ PI structurally parallel: "They are agents bringing about effects, physically or mentally, through their own decisions." / "They are patients being affected, physically or emotionally, by others' actions or attitudes."). Indices 1–5 cover present/past/perfect tense, simple/progressive aspect, active/passive/mixed/embedded-passive voice, he/she/none gender, and plural `they` / singular / collective `the group` number. Each list deliberately includes both pleasant and unpleasant outcomes ("for better or worse", "praised by some and blamed by others", "loves, hates, hopes, and believes") so dimensional similarity does not collapse into evaluative polarity (PI ≠ negAttI; AGI ≠ posAttI). PI seeds also include both direct-action affectedness and indirect attitude-as-affectedness ("how others felt about her", "what others believe about them"), reflecting that patienthood in framing analyses is not exhausted by physical action.

Reflexive constructions ("they organized themselves") remain a known limitation that whole-sentence cosine similarity at this layer cannot fully separate from non-reflexive agency ("they organized a movement"). The pipeline does not try to resolve this through paraphrase coverage; target-bound SRL evidence and review flags downstream handle the boundary cases. Surface variation is applied only on the prototype side; `_build_focus_text` is unchanged because the `[GROUP:token][PRED:verb]` markup is still the cleanest way to expose syntactic role to the encoder.

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

### Two-tier frame admission

The frame inventory F⁻ / F⁺ is auto-refreshed each run from LLR candidates against the seed prototypes. The admission gate is two-tier, not single-cosine.

The first tier scores `cos(candidate_word, full_seed_sentence)` and selects the larger of the NEG / POS sides. This is the "is this candidate used in the same kind of corpus context as known framing language?" question. It uses the encoder's contextual capacity, but it is permissive: long humanitarian-topic seed sentences encode into a broadly positive-discourse vector that any topical word (`right`, `journey`, `story`, `organization`, `high`) lands near regardless of its own polarity. The mirror problem on the negative side admits topic markers like `asylum`, `racial`, `wall`, `humanitarian`, `water`, `protest`.

The second tier scores `cos(candidate_word, sentiment_anchor_word)` against four abstract anchors stored in `candidate_terms.json`: `bad` / `negative` for F⁻ and `good` / `positive` for F⁺. This is a word-level polarity sanity check. Anchors are deliberately abstract and minimal — not corpus-attested exemplars and not domain-specific — because their job is only to verify that the candidate's underlying sentiment direction matches what the sentence-level context suggested. A topical word like `asylum` can have high cosine to a long humanitarian-context sentence while having near-zero anchor sentiment margin; the second tier catches the cases where it has the *wrong* sign, not just a small one.

A candidate is admitted only when the sentence tier clears its margin (`|neg_sim - pos_sim| >= 0.06`) and the anchor direction agrees in sign with the sentence direction (`anchor_diff * sentence_diff > 0`). Anchor magnitude is **not** thresholded: after the seed cleanups in entry 28+, anchor `|diff|` clusters in the 0.01–0.05 range and any absolute floor would block almost everything. The magnitude check sits on the sentence tier where the signal is more reliable; the anchor tier is a direction-only veto.

Two design constraints kept this simple:

* Anchors never enter centroid geometry. WEAT, CEAT, and CEAT-full all read centroids from `seed_*_terms` only. Reported scores stay 100% corpus-derived; the abstract anchors only gate which words enter the inventory.
* Anchors are four words, not a curated lexicon. A larger anchor list would re-introduce the wordlist-prior problem the project explicitly avoided when it removed `SUBJECTIVE_VERBS` and `CLASSIFIED_FRAMES`. Sentiment polarity is the only thing the anchor tier needs to test, so four anchor words are sufficient.

Known limitations the two-tier rule does not solve: evaluative adjectives that are not framing terms (`right`, `high`) still pass anchor sentiment polarity because they lexically *are* sentiment-bearing; and metaphorical framing (e.g., `wave`, `flood`, `surge` for migration) is not detectable by either tier because metaphor source domains share little distributional similarity with abstract evaluative anchors. Metaphor framing is left to a future separate diagnostic layer rather than forced through the frame-admission gate.

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

This independence is enforced operationally: each dimension has its own absolute prototype floor and a single mention can pass any combination of AGI / PI / SI. The earlier `_dim_wins(score, others)` rule treated the dimensions as exclusive (each mention wins at most one dim by clearing a relative `DIM_MARGIN` against the others), which contradicted the design and silently suppressed multi-role mentions like `Refugees protested the deportation orders that affected them` (AgI on `protested`, PI on `affected`). The current `_dim_passes(score, floor)` rule uses per-dim floors calibrated to the 70th percentile of each dim's empirical cosine distribution — `AGI_FLOOR=0.626`, `PI_FLOOR=0.637`, `SI_FLOOR=0.597`. PI's mean cosine sits ~0.04 above SI's against the paraphrase prototypes, so a uniform floor would systematically suppress SI and pin AGI/SI near zero by accident of cosine geometry rather than by data. Per-dim calibration removes that artifact.

Three corollaries follow. First, syntactic evidence and prototype evidence are no longer symmetric across dimensions. PI fires on SRL `PATIENT_LABELS` / `dobj` / `nsubjpass` / `pcomp+auxpass` without requiring prototype confirmation, because the PI prototype geometry sits closer to AGI prototypes on the most prototypical English patienthood pattern (`B verbs A`, A in object position), which means a prototype gate would actively suppress the syntactic signal it is meant to confirm. Disagreement is recorded as a review flag, not a veto. Second, SRL ARG0 requires AGI prototype confirmation (`agi_passes`) to filter unaccusative subjects (`refugees suffered` lands below `AGI_FLOOR`); a sentiment-anchor veto was tried and removed — anchor margin tracks sentence-level polarity, not verb argument structure, so it was an unreliable unaccusativity proxy. Third, SI has a minimum target-as-experiencer guard — the target must be `nsubj` / `nsubjpass` or carry an SRL ARG0 / ARG1 label — so SI no longer false-fires on bare modifiers inside prepositional phrases.

## MWE / Compound Group Mention Resolution

### The problem

spaCy tokenizes multi-word demographic expressions as separate tokens. Without explicit handling:

- `asylum seeker(s)` — two tokens; neither `asylum` nor `seeker` resolves on its own under the token-level inventory check.
- `people of color` — four tokens; `people` is not in the demographic inventory, `color` is not a group token.
- `African American` — two tokens; both `african` (minority) and `american` (dominant/ambiguous) are in the inventory, producing a double-count.
- `native African` — `native` is in both `TARGET_TOKENS` and `CONTRAST_TOKENS`; inventory order is not a principled tiebreaker.
- `Spanish speaking immigrants` — `spanish` is a demographic token, `immigrants` is a demographic token; both would fire independently, double-counting one mention.

The old composite route addressed the `African American` case by emitting a hyphenated lemma (`african-american`) when a dominant-side modifier modified a minority-side head. This was fragile: it only fired when the head was in the dominant-side ambiguous inventory, so `native African` (minority head) produced two mentions instead of one. It also produced a third lemma class (`african-american`) that was not in either inventory, complicating aggregation.

### The idiomaticity principle

The new design follows the idiomaticity principle from MWE research: when a compound demographic expression has a dominant constituent and a minority constituent, the minority constituent carries the framing signal. This is justified by the hypothesis that contextual encoders attribute framing exposure to the semantically dominant constituent — `African American` is distributionally closer to `Black` than to `White` in embedding space, so suppressing `American` and keeping `African` is the more defensible choice. `Mexican citizen` is not a compound demonym; `citizen` is a civic-majority framing word that should not inherit the minority framing of `Mexican`. The head is suppressed; the modifier carries the mention.

For genuinely ambiguous dual-group tokens (`native`, `spanish`, `citizen`) the inventory order is not a principled tiebreaker. The new rule consults the syntactic context: if the dual-group token modifies a head that is itself demographic, it defers to the head's side. If it is a head with a demographic modifier child, it defers to the modifier. If neither is available, it falls back to the GTE semantic resolver (for tokens in `SEMANTIC_DISAMBIGUATION_TOKENS`) or returns `None`.

### Implementation

Three new mechanisms in `resolve_group_token()`:

1. **`COMPOUND_TARGET_HEADS`** — a small dict keyed by head lemma that describes either a compound-modifier child (`asylum` → `seeker`) or a `prep + pobj` chain (`of` → `color` under `people`). The head emits the canonical compound lemma (`asylum-seeker`, `people-of-color`); the modifier child is suppressed on its own pass through the resolver. spaCy `PhraseMatcher` was avoided to keep the resolver token-by-token.

2. **`DUAL_GROUP_TOKENS`** — the intersection `TARGET_TOKENS & CONTRAST_TOKENS`. When a dual-group token is a modifier, it defers to the head's group side (or to a sibling modifier's side). When it is a head with a demographic modifier child, it defers to the modifier and suppresses itself. When neither is available, it falls back to the semantic resolver or returns `None` rather than letting inventory order pick a side arbitrarily.

3. **Sibling-modifier suppression** — when two modifiers of the same head both resolve to demographic groups, the dominant-side modifier is suppressed and the minority-side modifier carries the mention. This handles `African American voters` (both `African` and `American` are amod of `voters`; `American` is dominant-side and is suppressed).

The `X speaking Y` pattern (`Spanish speaking immigrants`) is handled by a separate rule: if a demographic token modifies a `speak/speaking` head that itself modifies a demographic head, the language-modifier token is suppressed so only the demographic head fires.

### What was removed

The old composite-route hyphenation (`f"{lemma}-{head_group}"`) and the conjoined-modifier hyphenation (`Asian and Latino Americans` → `asian-latino`) were deleted. Conjoined modifiers now each emit as standalone mentions via their own token paths. This means `Asian and Latino Americans` yields two mentions (`asian`, `latino`) rather than one composite lemma. The composite lemma was a third class that complicated aggregation and was not grounded in any lexical resource.

### Known limits

- `asylum` as a single-token target is occasionally over-broad (`applying for asylum` where the noun denotes the institution rather than a person). Future work: add `asylum` to `SEMANTIC_DISAMBIGUATION_TOKENS` so the GTE resolver decides per-context.
- Dual-group tokens without a resolvable syntactic context (no demographic head, no demographic sibling modifier) return `None` rather than guessing. This is precision-oriented; some valid mentions are missed.
- The idiomaticity principle is a hypothesis, not an empirically validated claim for this corpus. Δ-CEAT partially quantifies the gap between human-referent scope and broader signifier-level distribution; a direct test of compound-constituent attribution would require targeted annotation.

## Keyword-Extraction Resolution (2026-06-07)

### Problem

`resolve_group_token()` had grown to ~305 lines with 14+ branches: `AMBIGUOUS_NOUNS`, `DUAL_GROUP_TOKENS`, `STRONG_TARGET_TOKENS`, `AMBIGUOUS_TARGET_MODIFIERS`, `AMBIGUOUS_CONTRAST_MODIFIERS`, three semantic-resolver call sites, and a modifier guard requiring a confirmed human head. This complexity produced systematic false negatives:

- `African` as `acomp` ("I am African inside") → suppressed because `inside` is not a human head.
- `German` modifying `anthropologists` → suppressed because `anthropologists` is not in `HUMAN_NOUNS`.
- `American` in "Friends, countrymen, Americans" → suppressed because `nounish` detection failed on the conjunct.

Polysemous civic tokens (`citizen`, `native`, `local`, `national`, `domestic`, `majority`, `native-born`, `nationalist`) in the token sets caused a different class of false positives: "national park", "local government", "native speaker" all resolved as demographic mentions despite being civic/institutional uses. These tokens consistently appeared in the frequency-ranked top-8 and entered the active extraction set, consuming slots that could have gone to genuinely demographic tokens.

### Decision

1. **Remove civic tokens from TARGET_TOKENS and CONTRAST_TOKENS entirely.** These are not demographic tokens; they describe civic/institutional status. `citizen`, `native`, `majority` remain in `HUMAN_NOUNS` (they are valid human-referent head nouns for rescue/inanimate checks). `national` is excluded from `HUMAN_NOUNS` because it overwhelmingly denotes institutions ("national policy", "national security").

2. **Replace the 14-branch resolver with keyword extraction + inanimate suppression.** The simplified `resolve_group_token()` is ~100 lines:
   - Compound resolution (asylum-seeker, people-of-color, native-american)
   - Compound-child suppression
   - `non-` prefix unconditional suppression (`non-white` ≠ `white`)
   - `anti-`/`pro-` prefix modifier+head guard (stance prefix, referent is real)
   - `X speaking Y` suppression
   - Minority-child dominant-head suppression (African American → african)
   - Sibling-modifier suppression
   - Inanimate head → suppress
   - Otherwise → emit

   No human-head requirement. No `AMBIGUOUS_*` / `STRONG_*` / `DUAL_GROUP_*` branching. No semantic resolver.

3. **Remove `SemanticGroupResolver` entirely.** The GTE ModernBERT model is still loaded in `run_pipeline.py` for WEAT/CEAT/frame-refresh, but it is loaded directly via `SentenceTransformer` instead of through the resolver wrapper. The resolver's disambiguation role is now handled by inanimate suppression: if the head noun is in `INANIMATE_NOUNS` or has an inanimate entity type, the modifier token is suppressed. `INANIMATE_NOUNS` was expanded to cover gaps that the resolver previously handled (dog breeds, statistics terms, `accent`, `parliament`, `engineering`, etc.).

4. **Add `native-american` to `COMPOUND_TARGET_HEADS`.** Without this, "Native Americans" emitted `american` (dominant) with `native` suppressed. The compound spec maps `american`/`americans` with modifier-child `native` → `native-american` (minority).

5. **Add Phase 1 inanimate-adjacency pre-filter.** Before MiniLM scoring, check if every gate token in a sentence is adjacent (±2 whitespace tokens) to an inanimate noun. If so, skip MiniLM scoring. This catches "German law", "black hole", "American government" before any embedding work. The all-lexical file (`semantic_filter_lexical_all.txt`) still includes these sentences for CEAT-full.

### Consequence

- Phase 1 extraction must be re-run: the gate regex is compiled from the changed `TARGET_TOKENS ∪ CONTRAST_TOKENS − GATE_EXCLUDE_TOKENS`, so the extraction corpus changes.
- `active_labels.json` will contain different top-8 sets after re-extraction (civic tokens no longer compete for slots).
- The `DUAL_GROUP_TOKENS` mechanism is dead (empty intersection). The `AMBIGUOUS_NOUNS` branch and `TARGET_NOUN_MODIFIER_MAP` / `CONTRAST_NOUN_MODIFIER_MAP` are removed. `SEMANTIC_DISAMBIGUATION_TOKENS`, `STRONG_TARGET_TOKENS`, `STRONG_CONTRAST_TOKENS`, `AMBIGUOUS_TARGET_MODIFIERS`, `AMBIGUOUS_CONTRAST_MODIFIERS` are removed.
- False negatives from the human-head modifier guard are eliminated.
- Some false positives from polysemous modifiers (e.g., `black sheep`, `white noise` if head is not in `INANIMATE_NOUNS`) may increase slightly. These are mitigated by MiniLM semantic scoring in Phase 1 and by inanimate suppression in Phase 2, but the coverage of `INANIMATE_NOUNS` is now the critical determinant.
- The `semantic_group_resolver.py` source file was already deleted; only a `.pyc` cache remained. This change makes the deletion explicit — no code depends on the resolver class any longer.

### 2026-06-07 Refinement of Active Label Gating and Compound Counting

1. **Constituent-Aware Active Label Gating**: The active label gate was previously enforced strictly on the final resolved canonical. This caused genuinely demographic compound canonicals like `native-american` to be suppressed, because they were not in the top-8 demographic list (`active_labels.json`). Enforcing gating on the constituent parts of the compound (e.g. allowing `native-american` since its constituent `american` is in the active list) resolves this without leaking unrelated non-active demographic groups.
2. **Compound Counting and Negation Discounting in Pre-Scan**: In Phase 1's pre-scan frequency counting, compound tokens (like `native-american` and `asylum-seeker`) and negated terms (like `non-white` or `non-Chinese`) were previously either split or counted towards the positive standalone tokens. By matching them explicitly with regex and discounting their constituent counts from the raw word frequencies, we obtain clean, non-duplicated frequencies for all target and contrast groups in `demographic_word_counts.tsv`.
3. **SRL Cache Invalidation**: Since `resolve_group_token` is called during SRL role extraction to filter predicate hints, any changes to target resolution rules mean that cached SRL annotations will lack the newly resolved target contexts. Deleting `srl_cache.pkl` is required to force a fresh annotation pass.

### 2026-06-07 Correction: Compound Part-Match Bypass Removed; `foreign` Removed from TARGET_TOKENS

Two issues surfaced from reviewing output.md against active_labels.json:

**1. Compound part-match bypass was methodologically wrong.**
The constituent-aware gating introduced in entry 57 allowed `native-american` to resolve whenever `american` was in the active set, on the assumption that constituent attention allocation makes the compound "active enough." This is not defensible: `native-american` has its own frequency count (495 on the current shard), which is well below the top-8 threshold. Compounds have distinct distributional profiles from their constituents — `native american` sentences are about indigenous history and land rights, not about the same population as `american` alone. Letting the compound ride the constituent's active status means reporting results for a group that was not selected by the frequency-based reproducibility criterion. The fix is to revert to exact-canonical matching: `canonical not in _ACTIVE_EXTRACTION_TOKENS → return None`, no part-splitting. `native-american` will return to reported output only if and when it enters the top-8 target list on a shard where indigenous group mentions are frequent enough.

**2. `foreign` is too polysemous as a bare adjective and was removed from TARGET_TOKENS.**
`foreign` was admitted as a target token to catch "foreign worker/student/national" usage. In the Dolma shard, however, 79 of 411 `foreign` sentences pair it with inanimate heads (`foreign policy`, `foreign land`, `foreign minister`, `foreign direct investment`). Inanimate suppression blocks the modifier-of-inanimate cases, but `foreign` as a predicative or standalone use (`foreign incidents`, `foreign objects`, `foreign gods`) still leaks through. The person-referent signal of `foreign` is already covered by `foreigner`/`foreigners`, which are unambiguously nominal and person-referent. Removing `foreign` from `TARGET_TOKENS` closes the leak without losing coverage of the intended population. The freed active-label slot goes to `jewish` (2799 occurrences, next in frequency rank after color tokens are excluded).

### 2026-06-23 Phase 1 Recall Audit & Calibration Decoupling

A standalone recall audit (`ablation_recall_audit.py`) was run on a random sample of 300 sentences from `semantic_filter_lexical_all.txt` to evaluate the recall loss of the MiniLM cascade filter relative to GTE-ModernBERT.

1. **Embedding Scale Mismatch**: Direct transfer of MiniLM's absolute thresholds (`SEMANTIC_MIN = 0.34`, `SEMANTIC_MARGIN_MIN = 0.03`, `RESCUE_POS_MIN = 0.25`) to GTE-ModernBERT resulted in an artificially high False Negative Rate of **49.1%** (130/265). This was caused by ModernBERT's higher baseline similarity scores, which made the low `pos >= 0.34` threshold trivial to pass, thus admitting inanimate phrases (e.g. "DTI white matter segmentation", "Spanish style homes", "American Folk-Blues Festival").
2. **Calibrated Audit**: Once GTE-ModernBERT's thresholds were adjusted to match its embedding scale (`SEMANTIC_MIN = 0.62`, `SEMANTIC_MARGIN_MIN = 0.06`, `SEMANTIC_STRONG_MARGIN = 0.12`, `RESCUE_POS_MIN = 0.60`, `RESCUE_MARGIN_MIN = 0.06`), the True False Negative Rate dropped to **6.8%** (18/265). *(Note: These high calibrated thresholds evaluate GTE-ModernBERT as a standalone primary filter without inanimate pre-filtering. In `extract.py`'s Step 5 ModernBERT fine-screening pass on pre-filtered review candidates, ModernBERT re-evaluates review rows against standard POS/NEG and RESCUE query sets using pipeline constants with `rp >= 0.53` margin bypass).*
3. **Engineering Tradeoff Validation**: This audit confirms that the MiniLM cascade captures **93.2%** of relevant demographic sentences that a calibrated GTE-ModernBERT model would admit, while enabling a massive throughput speedup (MiniLM runs thousands of sentences/sec, whereas ModernBERT is constrained to ~50 sentences/sec on Mac MPS).

### Group count (K) expansion and positive review category (2026-07-07)

#### Problem
PC2 (dehumanization / subjectivity axis) showed severe instability under Leave-One-Out (LOO) checks. Removing `spanish` caused PC2 loading cosine similarity to drop to 0.15, and removing `white` dropped it to 0.55. Furthermore, the Bartlett sphericity test was not significant ($p = 0.17 > 0.05$), meaning the group profile matrix correlations were insufficient to support factor extraction. At the same time, the frame auto-refresh pipeline admitted 87 terms, introducing significant noise on the positive side (topical FPs like `insurance`, `coverage`, `rise`).

#### Decision
1. **Expand Group Count K**: The PC2 instability is a small-N observation problem (16 groups × 6 dimensions). To achieve a stable principal component structure, the group limit in `extract.py` was increased from `top 8` to `top 14` targets and contrasts (yielding 27 active groups in the pipeline). This directly improves the observation-to-variable ratio from 2.67 to 4.50.
2. **Positive Review Category**: To handle the structural asymmetry of positive evaluative framing (which tends to co-occur as diffuse functional/institutional collocates rather than polar adjectives), we introduced a manual `review_positive_terms` list in `candidate_terms.json`. This quarantines topic-collocate FPs (e.g. `insurance`, `coverage`, `holiday`) so they are carried forward in the JSON without polluting AttI frame binding.
3. **Batch Interactive Terminal Review**: To eliminate manual JSON edits while keeping the human-in-the-loop audit practical, the frame refresh is now a batch interactive process. The pipeline displays all proposed auto-admissions at once and queries if the user wants to override any of them. If the user selects a term, it prints the full candidate entry metadata (including `found_with`, LLR/LogDice, GTE cosines, and anchors) and lets the user choose: Admit [a], Quarantine/Review [r], or Skip/Blacklist [s]. The choice is written back to the JSON automatically, preventing future redundant prompts.
4. **Calibrated Frame refresh**: The auto-refresh algorithm now gates candidate terms using `existing_frames = auto_neg_frames | auto_pos_frames | review_pos_frames | review_neg_frames | blacklist_terms`, preventing re-discovery of quarantined or blacklisted terms. The lookup matching is normalized using a `_clean_term()` helper that strips any parentheses comments (e.g., `(insurance)coverage` counts as `coverage` during comparison), allowing you to annotate lists with inline comments without causing duplicate candidate prompts in future runs.

#### Consequence
- Re-extracted the corpus on 2 parquet shards to generate 20,612 kept sentences.
- Re-ran the pipeline and stability checks. Min PC2 LOO CosSim rose from **0.15** to **0.9608**, demonstrating that PC2 is now highly stable and no longer driven by single outlier groups.
- Bartlett's test became highly significant ($p = 0.00039$), KMO improved to 0.556, and Tucker's Congruence coefficient reached 0.9418.
- The net_atti cross-chunk Spearman rank correlation settled at a robust and non-inflated $\rho = 0.6850$ (previously 0.00 with 15+4 frames and 0.82 with 87 bloated frames), representing clean, stable, and genuine evaluative framing.
