---
title: 2. Changes
description: Important changes and pending actions from pipeline development.
tags:
  - tracking
  - pipeline
---


## Current Status

|Problem|Status & Impact|
|---|---|
|**Encoder calibration**|Open. Phase 1 extraction needs MiniLM vs. GTE A/B calibration; Phase 2 GTE resolver/frame-refresh thresholds need separate calibration before new outputs are reported.|
|**Extraction review diagnosis**|Corrected. Use `dolma/semantic_filter_review.tsv` and a same-run `semantic_filter_report.txt` for Phase 1 extraction diagnosis; `X/output_review.tsv` is Phase 2 role/frame review. Tune classifier/noise handling before treating extraction failures as an encoder-only problem.|
|**Extraction false rejects**|Partially addressed for future runs. `extract.py` now has a `SEMANTIC_RESCUE` lane for low-absolute-score but strong-margin lexical hits. High-confidence rescue rows can enter final output at a stricter classifier threshold; lower-confidence rescue rows stay in review.|
|**Mention layer**|Partially implemented. `X/group_mentions.py` now drives primary anchors, MWE suppression, scope flags, and target-bound frame binding for association/frame-AttI; Step 3 role extraction still uses its existing token loop.|
|**Frame auto-admission**|Open. Auto-admission can overfire on generic discourse words; human review of `candidate_terms.json` remains needed.|
|**AgI/PI/SI hard cases**|Substantially addressed. Subjecthood separated from AgI; all three dimensions now require prototype confirmation (`DIM_MARGIN=0.04`); negation-scope blocking suppresses role assignment when the governing predicate is negated; PI is symmetric with AgI (structural evidence alone no longer sufficient). Embedded clauses and cross-sentence coreference still need validation.|
|**Local prototype AttI**|Resolved as a reported-metric issue. It is diagnostic only; reported `netAttI` now comes from frame association.|
|**Scope/discourse flags**|Implemented for review routing and now also for role suppression. Negation-scope blocking prevents AgI/PI/SI assignment when the governing predicate is negated; frame-AttI scope blocking prevents negated/corrected frame terms from counting. Correction/denial, quotation, reported speech, contrast, and multi-group flags still route to review only.|
|**Duplicate sentence exports**|Low impact. Some sentences enter `semantic_filter_results.tsv` through multiple group triggers; negligible at sample scale but can be deduplicated later if it complicates review.|

Current thresholds:

```text
SEMANTIC_MIN = 0.34
SEMANTIC_MARGIN_MIN = 0.03
SEMANTIC_RESCUE_MIN = 0.27
SEMANTIC_RESCUE_MARGIN_MIN = 0.08
RESCUE_CAN_KEEP = True
RESCUE_CLASSIFIER_THRESHOLD = 0.60
BLOCK_REFERENCE_NOISE_KEEP = True
LEXICAL_HUMAN_CLASSIFIER_THRESHOLD = 0.65
LEXICAL_HUMAN_REVIEW_PROB_MIN = 0.30
ARMADA_DEVICE = auto (`mps` on Apple Silicon)
ARMADA_EMB_BATCH_SIZE = 64 for MiniLM on MPS, 32 on CPU, 256 on CUDA unless overridden
ARMADA_SENT_BATCH_SIZE = 4096
ARMADA_PARQUET_BATCH_SIZE = 10000
AttitudinalPrototypeMatcher.positive_floor = 0.24
AttitudinalPrototypeMatcher.positive_margin = 0.02
DIM_FLOOR = 0.60
DIM_MARGIN = 0.04
SemanticGroupResolver.positive_floor = 0.26
SemanticGroupResolver.positive_margin = 0.04
FRAME_SIM_FLOOR = 0.55
FRAME_SIM_MARGIN = 0.04
ARMADA_ANALYSIS_DEVICE = auto (`mps` on Apple Silicon)
ARMADA_ANALYSIS_EMB_BATCH_SIZE = 16 for GTE ModernBERT on MPS unless overridden
ARMADA_CEAT_FULL_MODE = reported
ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP = 500
ARMADA_CEAT_MIN_CONTEXTS_PER_GROUP = 10
ARMADA_CEAT_MAX_FRAME_CONTEXTS = 1000
```

## Implemented Changes

### 2026-05-13

1. **Association layer rewritten.**
`PPMI` was removed from the active collocate-discovery path and replaced with sentence-level, non-adjacent `LLR / LogDice` scoring to avoid inflated counts from adjacent lexicalized pairs.

2. **Role extraction upgraded from dependency-only to hybrid SRL.**
`AgI / PI / SI` are now extracted primarily through a Hugging Face SRL model.

3. **Demographic Lexicons Massively Expanded.**
`TARGET_TOKENS` expanded to >140 groups (Global South / historically marginalized), and `CONTRAST_TOKENS` expanded to >40 groups (European / Anglosphere / dominants). Broad tokens (`citizen`, `minority`, `native`) no longer trigger the gate independently to prevent noise.

4. **CEAT replaced SEAT for contextual association.**
WEAT and CEAT use the same GTE ModernBERT encoder for cross-corpus comparability. CEAT scores each sampled group context against F⁻/F⁺ sentence centroids and reports the mean with `N` and `SE`. `CEAT-full` samples from *all* raw lexical hits bypassing semantic filters, allowing `Δ-CEAT` (`CEAT-full` - `CEAT-filtered`) to quantify associative contamination from non-human or institutional usages in the raw corpus.

5. **Reporting tightened.**
Per-group summary outputs are now filtered to `N ≥ 50` for statistical stability at full scale.

6. **AttI moved to frame association; local prototypes kept as diagnostics.**
Reported `frame_negAttI`, `frame_posAttI`, and `netAttI` now come from group-level co-occurrence with final F⁻/F⁺ frame terms. The prototype matcher still writes local diagnostic signals, but it no longer defines the reported AttI dimension or EFI input.

7. **Frame inventory (`F⁻` / `F⁺`) refreshed from LLR candidates.**
`WEAT` and `CEAT` no longer anchor exclusively on the static manual frame taxonomy. At each pipeline run, we compare current LLR candidates against seed frame terms and augment `F⁻` / `F⁺` accordingly, writing the result into `candidate_terms.json`. The file is regenerated each run rather than accumulated. Manual seed sets are kept as polarity anchors.
Auto-admission currently overfires on generic discourse words (`apply`, `work`, `community`); the intended resolution is Step 1 (human baseline): annotators grouping high-LLR collocates into confirmed frame types provide exactly the community-validated gate that prevents generic candidates from entering `F⁻` / `F⁺` unrestricted.

8. **Classifiers & Disambiguation Hardened.**
The final sentence filter layer now uses PCA over target embeddings instead of TF-IDF, preventing sparse vector overfitting. The polysemous group resolution (`PROMPT_BANK`) was rebuilt into a universal template registry, ensuring consistent and scalable zero-shot disambiguation across ambiguous tokens like `polish`, `foreign`, or `black`.

9. **Encoder split introduced.**
Phase 1 extraction defaults to MiniLM for throughput but can be overridden with `ARMADA_EXTRACTION_PRESET` for A/B calibration. Phase 2 analysis uses GTE ModernBERT for semantic disambiguation, frame refresh, WEAT, CEAT, and CEAT-full. Sentence/context windows remain widened conservatively (`MAX_SENT_LEN=800`, semantic resolver/prototype context = 24 tokens).

10. **Lexicon prior surface tightened.**
Unused psych-verb and attitudinal-adjective inventories were removed from `lexicons.py`. Remaining lexicon sections are documented as active gates, disambiguation guards, or frame seeds.

11. **Reported AttI schema changed.**
`group_stats.tsv` now separates local diagnostic AttI from reported frame AttI: `local_negAttI`, `local_posAttI`, `frame_negAttI`, `frame_posAttI`, `netAttI`.

12. **Target-bound frame binding added.**
`X/group_mentions.py` now extracts primary group mentions, MWE metadata, sentence scope flags, and F⁻/F⁺ frame bindings. Reported frame-AttI counts only bound, non-blocked frame terms; negated/corrective frame terms are routed to review.

13. **Subjecthood separated from agency.**
`Subjecthood` is now a diagnostic output distinct from `AgI`. Low-control predicates suppress automatic agency, affectedness predicates can add `PI`, and non-volitional mental-state predicates can add `SI` without adding `AgI`.

14. **Political labels scoped separately.**
`soviet`, `ussr`, `communist`, and `conservatist` now resolve/report as `political` and are excluded from demographic frame-candidate discovery.

15. **Extraction rescue lane added.**
`extract.py` now distinguishes strict semantic passes from `SEMANTIC_RESCUE` candidates. Rescue rows are classifier-scored; high-confidence rescue rows can now enter final output at `RESCUE_CLASSIFIER_THRESHOLD=0.60`, while weaker rescue rows remain review candidates.

16. **Extraction review flags added.**
`semantic_filter_review.tsv` includes `review_flags` to distinguish low semantic margin, high semantic / low classifier disagreement, semantic rescue rows, and reference/index-like corpus noise.

17. **Sentence splitter hardened for initials and figure abbreviations.**
`split_sentences()` now protects common abbreviations, initials (`Bishop H. M. Turner`), acronyms, and figure labels (`Fig. 2`) before splitting. This reduces truncated fragments entering semantic rejects.

18. **Lexical-human extraction rescue added.**
Sentences with a group token structurally tied to a human head noun or person suffix now enter a classifier/review lane even when semantic retrieval assigns a low absolute score or weak margin. This targets false rejects such as `American lady` and `Peruvian boys` without rescuing non-human lexical hits such as `Italian coasts` or `white button`.

19. **Extraction runtime knobs added.**
`extract.py` now auto-selects GPU-like devices when available (`mps` on Apple Silicon, `cuda` on CUDA machines) and exposes batch-size/device overrides through environment variables. MiniLM defaults are tuned from a local micro-benchmark: MPS uses embedding batch 64, CPU uses 32, CUDA uses 256; sentence/parquet batches are increased to reduce per-call overhead.

20. **CEAT runtime mode added.**
`X/run_pipeline.py` now exposes analysis device/batch controls and `ARMADA_CEAT_FULL_MODE`. Default `reported` mode computes CEAT-full only for groups that will appear in `group_stats.tsv`; `all` keeps exhaustive lexical-hit diagnostics; `skip` is for development runs where `CEAT_full` / `delta_CEAT` are intentionally omitted. Context sampling is bounded by `ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP` and `ARMADA_CEAT_MAX_FRAME_CONTEXTS`.

### 2026-05-18

21. **Frame auto-admission thresholds tightened.**
`FRAME_SIM_FLOOR` raised from `0.22` to `0.55` and `FRAME_SIM_MARGIN` raised from `0.02` to `0.15` in `X/run_pipeline.py` to prevent background semantic embedding noise from auto-accepting neutral collocates (like `issue`, `police`, `school`) into the refreshed frame inventory (`candidate_terms.json`).

22. **Complete elimination of legacy evaluative verb lexicons.**
Residual verb lists (`SUBJECTIVE_VERBS`, `LOW_AGENCY_VERBS`, `AFFECTEDNESS_VERBS`, `VOLITIONAL_MENTAL_VERBS`) were deleted from `X/lexicons.py`. Both direct target mentions and anaphoric pronoun contexts (`_resolve_anaphora` for `they`/`them`/etc.) are now resolved using unified, relative-margin dimensional prototype similarity. Pronoun contexts are matched directly via `_ATTITUDE_MATCHER` on the fly, eliminating structural double-standards and standardizing all semantic role assignments.

23. **Elimination of static frame taxonomies (CLASSIFIED_FRAMES).**
To strictly enforce the unsupervised `Discovery -> Curate -> Refresh` paradigm, the static, hardcoded `CLASSIFIED_FRAMES` dictionary was removed from `X/lexicons.py`. The pipeline now uses `candidate_terms.json` as the sole source of truth for polarity seeds, ensuring that manual human curations and additions to the JSON file are preserved and actually applied during cosine similarity matching, rather than being silently overwritten on each run.

24. **Centroid-first architecture in candidate_terms.json.**
Seeds are now split into two layers: `seed_*_terms` (sentence-level Dolma prototypes) which are encoded by GTE ModernBERT to produce F-/F+ centroids used directly by WEAT and CEAT; and `auto_*_terms` (single-word terms auto-admitted from candidate discovery) which accumulate across runs and are used for AttI syntactic frame binding. There is no longer a manually maintained `frame_*_terms` wordlist, and WEAT/CEAT no longer require finding sentences containing frame words to compute centroids.

### 2026-05-19

25. **DIM_MARGIN tightened from 0.01 to 0.04.**
The relative margin required for a dimensional prototype to "win" over competing dimensions was raised from 0.01 (noise-level) to 0.04 (meaningful discrimination). This means AgI/PI/SI are only assigned when the winning dimension clearly dominates; ambiguous cases now route to review via `no_clear_semantic_role`.

26. **Negation-scope blocking for role assignment.**
When the governing predicate of a group mention is under negation (`not`, `n't`, `never`, `without` as dependency child or left-adjacent), all role assignment (AgI/PI/SI) is suppressed and the mention is routed to review with `negation_scope_blocked`. This mirrors the existing frame-AttI scope blocking and prevents "Refugees did not organize" from granting AgI.

27. **PI now requires prototype confirmation (symmetric with AgI).**
Previously, SRL PATIENT_LABELS, `dobj`, and `nsubjpass` granted PI unconditionally. Now all structural PI paths require `pi_wins` (prototype similarity confirms patienthood). Unconfirmed structural patients get review flags (`srl_patient_unconfirmed`, `dobj_patient_unconfirmed`, `passive_patient_unconfirmed`). This makes the "SRL is auxiliary" claim honest for all three dimensions.

28. **Broken PI prototype removed.**
The placeholder sentence `"something belong to from, of, done by the group"` was removed from `PI_PROTOTYPES` as it was not a coherent sentence and produced noisy embeddings.

