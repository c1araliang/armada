---
title: 3. To-Dos
description: Research questions under investigation and broad plans for pipeline development.
tags:
  - methodology
  - pipeline
---

## Action Queue

**0. Incomplete filtering**

Accept incomplete filtering, but quantify and report it.

Reason:

- Changing the model doesn't solve the fundamental problem. 

   A larger model (e.g., replacing MiniLM with GTE ModernBERT for Phase 1) might improve some boundary cases, but it won't eliminate borderline regions—it simply moves the boundaries elsewhere. Furthermore, changing the model means recalibrating all thresholds, retraining the classifier, and rerunning the entire Phase 1, which is costly and the benefits are uncertain.

- For a 1.47M entries, 14k borderline sentences represent ~0.95%, somewhat negligible? it doesn't mean the filter has failed—it indicates that the corpus does indeed contain a fair number of candidates in a "potentially relevant but semantically weak" gray area.

- Methodologically, incomplete filtering can be justifiable:

  The  pipeline is precision-oriented (better to miss something than introduce noise).

  The review file itself is a transparency artifact—you didn't discard these sentences; you retained them and flagged the uncertainties.

  Report in the paper: N kept borderlines, N borderlines (not included in the analysis), rejection rate X%, and explain the borderline composition (which flag types account for what percentage).

**1. Calibrate encoder-specific thresholds before reporting new results**

Phase 1 extraction uses MiniLM for speed; Phase 2 analysis uses GTE ModernBERT for reported semantic association. Before treating new outputs as reported results, calibrate:

|Component|Values to check|Reference set|
|---|---|---|
|Phase 1 extraction encoder A/B|MiniLM vs. GTE ModernBERT runtime, false rejects, and kept-sentence quality|Same shard, same labeled examples, same random lexical-hit sample|
|Phase 1 strict semantic gate|`SEMANTIC_MIN`, `SEMANTIC_MARGIN_MIN` per extraction encoder|Held-out `RELEVANT / IRRELEVANT` sentences plus random lexical hits|
|Phase 1 rescue gate|`SEMANTIC_RESCUE_MIN`, `SEMANTIC_RESCUE_MARGIN_MIN`, `RESCUE_CAN_KEEP`, `RESCUE_CLASSIFIER_THRESHOLD`|Fresh `semantic_filter_report.txt` after rerun plus `semantic_rescue` kept/review rows|
|Phase 1 lexical-human rescue|`LEXICAL_HUMAN_CLASSIFIER_THRESHOLD`, `LEXICAL_HUMAN_REVIEW_PROB_MIN`|False rejects with demonym/ethnonym + human head, e.g. `American lady`, `Peruvian boys`, balanced against non-human lexical hits|
|Phase 1 classifier|`CLASSIFIER_THRESHOLD`, `BORDERLINE_PROB_MIN` per extraction encoder|`semantic_filter_review.tsv` rows, especially `high_semantic_low_classifier` vs. `reference_noise_like`|
|Phase 1 corpus-noise review|reference/index/citation flags|Rows flagged `reference_noise_like:*`; decide whether to add training negatives or a pre-classifier noise filter|
|Phase 1 sentence splitting|abbreviation and initial protection|Check semantic rejects for fragments ending in `Mr.`, `Ms.`, initials, `Fig.`, `No.`, or `Messrs.`|
|Phase 1 runtime|`ARMADA_DEVICE`, `ARMADA_EMB_BATCH_SIZE`, `ARMADA_SENT_BATCH_SIZE`, `ARMADA_PARQUET_BATCH_SIZE`|Compare elapsed seconds and output counts on the same shard; watch memory pressure on MacBook Air|
|Phase 2 runtime|`ARMADA_ANALYSIS_DEVICE`, `ARMADA_ANALYSIS_EMB_BATCH_SIZE`, `ARMADA_CEAT_FULL_MODE`, `ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP`, `ARMADA_CEAT_MAX_FRAME_CONTEXTS`|Use `reported` for normal local runs, `skip` for quick debugging, `all` only for exhaustive CEAT-full diagnostics; increase sampling caps only for final checks|
|Phase 2 GTE semantic group resolver|`positive_floor`, `positive_margin`|Mention-level human/non-human decisions for ambiguous tokens|
|Phase 2 GTE frame refresh|`FRAME_SIM_FLOOR`, `FRAME_SIM_MARGIN`|Human review of LLR candidates against F⁻/F⁺ seeds|
|Frame binding|binding distance, blocked-scope flags|Sentences with multiple groups, negation, quotation, contrast, and correction|
|Local AttI diagnostics|prototype floors/margins|Qualitative review only; not reported `netAttI`|

**2. Validate target-binding error types**

`X/group_mentions.py` now provides primary mentions, MWE metadata, scope flags, and target-bound frame binding for discourse association and frame-AttI. The next technical task is validation, not another conceptual redesign.

|Case to validate|Expected behavior|
|---|---|
|`Korean immigrant`|One primary anchor for association/frame-AttI; MWE child suppressed there|
|`Migrants are not a threat`|F⁻ frame is scope-blocked, not counted in reported frame-AttI|
|`Migrants were falsely accused of being a threat`|Correction/denial blocks reported frame-AttI and routes row to review|
|`The minister praised volunteers while blaming refugees for the crisis`|Negative frame binds to `refugees`, not to unrelated positive target|
|`Refugees arrived / suffered / feared / organized`|Subjecthood separated from AgI; affectedness and SI handled without automatic agency|

**3. Human-review frame auto-admissions**

Frame refresh still auto-admits generic terms too easily. Before treating expanded F⁻/F⁺ inventories as stable, review `candidate_terms.json` and classify high-LLR candidates.

## Broad Plans

### Scale through training-data levels

LLM training has three canonical stages, each with distinct data — our framework should generalize across all of them as long as the extraction skeleton holds:

| Level | Data type | Example open dataset | What bias looks like here |
|-|-|-|-|
| **1. Pretraining** | Raw web/book text (trillions of tokens) | **Dolma** [(*v1_6-sample, 16.4GB=10 billion tokens*)](https://huggingface.co/datasets/devingulliver/dolma-v1_6-sample) with a toolkit for curating datasets for language modeling | Distributional: co-occurrence patterns, metaphorical framing |
| **2. Instruction Tuning (SFT)** | (instruction, input, output) triples | **Stanford Alpaca 52k** (52k demos generated via self-instruct from text-davinci-003) | Instructional: biased task completions, stereotyped examples |
| **3. RLHF / Preference** | Human-ranked response pairs | Anthropic HH-RLHF, UltraFeedback | Evaluative: which "preferred" answers encode bias |

Start from Level 1 (basic pretraining data) with a manageable subset and scale up.

### Complex Contexts

Complex contexts should be treated as review flags, not as a promise that the automatic pipeline fully understands rhetorical intent.

|Phenomenon|Pipeline response|
|---|---|
|Negation / denial|Flag; avoid using local AttI as reported metric|
|Quotation / reported speech|Flag speaker/source distinction where possible|
|Concession / contrast|Flag contrastive frame; keep group associations separate|
|Anaphora / embedded clauses|Count AgI/PI/SI only when target binding is clear|
|Defended attacks|Represent frame co-occurrence at corpus level; inspect examples qualitatively|

The goal is not to solve all long and difficult sentences. The goal is to keep reported metrics interpretable and route low-confidence cases into review.
