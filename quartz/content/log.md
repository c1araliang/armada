---
title: 6. Log
description: Chronological log of research activities and pipeline development starting from April 10th.
tags:
  - tracking
---

2026-04-10: Curation, creation of this page.

2026-04-13: Reading list expansion, focusing on novelty check; Presentation drafts.

2026-04-14: Taxonomy refinement, structural updates, reading group meeting.

2026-04-15: Reading list updates and careful polishing, presentation notes, graph alignment.

2026-04-16: Graph alighment, presentation slides, poster sketch.

2026-04-17: Reading list updates and presentation slides; how ceat correlates to my approach and informs the MWE issue.

2026-04-19: Poster design, more samples sentences for review, SRL design question.

2026-04-20: SRL vs Frame Semantic Transformer, Postponing human-review;

2026-04-22: Structure overview updates.

2026-04-27: Minor index edits; sample annotation; problems aligment; novelty check updates.

2026-05-02: @idea human-likeness and reliability being contrastive goals; instead of prob-COT-TOT, [adding another prob layer](https://arxiv.org/abs/2604.20022) as rec by Prof. Arora.

2026-05-05: Refining research milestones.

2026-05-06: Drawing annotated code map.

2026-05-07: @milestone May 16 AILC; June 4 Venice Conf; June 8 SRW Short Paper. Invite Prof. Arora for 1st year term paper supervision.

2026-05-08: @group WP2 report update; Abstract draft for AILC. [new anotation rule ATS introduced](https://c1araliang.github.io/armada/ats/)

2026-05-11: Polishing abstract draft for AILC; Short Paper draft for AACL.

2026-05-12 Code-cleaning, ensuring minimal reference; AILC draft (soften the tones for making claims);

2026-05-13 Split MiniLM extraction from GTE analysis; added extraction rescue review lane, target-bound frame binding, subjecthood/AgI separation, scope flags, and political-label scope.
2026-05-13 Corrected extraction review diagnosis to `semantic_filter_review.tsv`; added review flags for classifier-borderline, low-margin, rescue, and reference-noise cases.
2026-05-13 Updated `AGENTS.md` with caution-first coding discipline and explicit documentation/closeout update rules.
2026-05-13 Hardened sentence splitting for initials/figure abbreviations; allowed high-confidence rescue rows at a stricter threshold with reference-noise blocking.
2026-05-13 Added lexical-human rescue lane for semantic false rejects such as `American lady` and `Peruvian boys`.
2026-05-13 Added extraction runtime knobs and automatic MPS/CUDA device selection for faster Phase 1 runs.
2026-05-14 Replaced SEAT/SEAT-full with sampled CEAT/CEAT-full and Δ-CEAT, including CEAT N/SE outputs.

2026-05-18: Dimensional prototype scoring implemented; eliminated legacy verb lexicons completely (both primary and anaphora paths use relative-margin prototypes); raised auto-admission thresholds to combat frame candidate noise.

2026-05-19: Tightened DIM_MARGIN 0.01→0.04; added negation-scope blocking for role assignment; PI now requires prototype confirmation (symmetric with AgI); removed broken PI prototype sentence.

2026-05-23: Redesigned dimensional prototypes as polarity-balanced paraphrase sentences (AGI ↔ PI parallel ruling definitions); removed verb-grid expander and lemminflect dependency.
2026-05-23: SRL cache key now hashes prototype content; prototype edits auto-invalidate the cache.

2026-05-23: Frame admission switched to two-tier rule (sentence cosine + 4-word sentiment anchor); reset stale `auto_*` lists; cleaned seed prototypes.

2026-05-23: Anchor tier switched to direction-only; `FRAME_SIM_MARGIN` raised 0.04→0.06; `ANCHOR_SIM_MARGIN` removed.

2026-05-24: AgI/PI/SI made independent (no winner-take-all); per-dim empirical floors; PI/SI prototype rewrite; spaced-compound resolver (asylum-seeker, people-of-color); sentiment-anchor sims persisted on findings (veto removed); EFI orientation flip removed, PC1 + PC2 reported.

2026-05-25: Made `output_results.tsv` and `output_review.tsv` disjoint — review keeps rows with `no_clear_semantic_role` OR null targets; results keeps the rest. Together they cover all per-sentence rows. Reported group_stats numbers are unaffected (still aggregated from in-memory rows).

2026-05-27: Rewrote MWE/compound resolution in `lexicons.py`; lexicon adjustments; AACL SRW short paper draft continued.
2026-05-28: Removed Phase 1 PCA+LogReg classifier and rescue lanes after ablation showed 37% recall loss vs 6% precision gain; extraction gate now two semantic-margin lanes (STRICT, STRONG_MARGIN). `extract_PCA+LogReg.py` retained as debugging variant; NO rescue lane may be initiated when semantic_margin < 0.

2026-05-29: Reworked Phase 1 lexical-human-rescue lane: removed pure-regex paths 3 (article/determiner) and 4 (predicate-adjective+pronoun) and the inanimate-noun guard; rescue now tags lexical hits as `inherent` (plural non-color demonym, admit directly) or `candidate` (gate token + human head within ±4, must clear MiniLM rescue queries `pos>=0.30 AND margin>=0.06`); color-tone plurals (`whites`, `blacks`) routed to candidate path; kept rows on shard 0 dropped 11,099→7,284 (`german shepherd`/`german tv`/`white background` etc. eliminated from rescue);

2026-05-30: Queries cleaning and updates; dropped `-man/-woman/-boy/-girl/-people` inherent rescue path (forms still enter via GROUP_RE, scored by main 2-lane MiniLM); Rule 1b candidate window made bidirectional (±4); personal pronouns added to `_HUMAN_HEADS`; `RESCUE_POS_MIN` lowered 0.30→0.25 (rmargin floor 0.06 unchanged); updated documentation and cleaned stale comments.

2026-06-04: @milestone Venice abstract submitted.

2026-06-07: Removed polysemous civic tokens from demographic sets; added native-american compound; rewrote resolve_group_token as keyword extraction + inanimate suppression; removed SemanticGroupResolver; added Phase 1 inanimate-adjacency pre-filter; `non-` prefix now suppresses unconditionally.

2026-06-07: Expanded and organized INANIMATE_NOUNS and HUMAN_NOUNS in lexicons.py based on spaCy head analysis to resolve false positives (privilege, history, neighborhood, identity, right); successfully reran Phase 1 extraction showing pre-filter removals rose 29,865 -> 33,625 and kept sentences refined to 6,936.

2026-06-07: Implemented Rule 1c in lexical_human_rescue() to route det-preceded singular demonyms as candidates; bypassed margin check for candidates with very high absolute POS score (rp >= 0.35 MiniLM / 0.53 ModernBERT); allowed GTE ModernBERT main-lane screening to rescue STRICT/STRONG_MARGIN sentences. Rerun successfully rescued all targeted human sentences (white man, American slave, Jewish immigrants) while stabilizing kept count at 7,911 (filtering out the initial 19,823 bloat from inherent routing).

2026-06-07: Restored hard-blocking on bibliographic and index_page_ref noise flags in extract.py after tightening regexes; successfully ran Phase 1 extraction with final kept count settled at 7,933 sentences.

2026-06-07: Implemented adjacent hyphenated chain rule and active label filtering wrapper for resolved canonicals, and fixed [demo]-born bear/born lemmatization bug in lexicons.py.

2026-06-07: Added Phase 1 compound counting and non-[demo] negation discounting to pre-scan frequency analyzer; updated Phase 2 active label check to permit compound canonicals (e.g. native-american) if base parts are active; invalidated SRL cache to force full metrics re-computation.

2026-06-07: Removed `foreign` from TARGET_TOKENS; fixed compound part-match bypass in resolve_group_token; replaced foreign with jewish in active_labels.json; fixed conj chain recursion in _resolve_role; added SRL ARG0 patient-dep guard and filtered-roles SI gate; fixed AGI elif-blocking (guarded-if nsubj fallback); added gerund-pcomp and gerund-amod unwrapping in _resolve_role; added SI preference/affection prototype [6]; added Italian/black-players and Americans-love test sentences to dim_sanity.py.

2026-06-07: Added QUANTIFIER_NOUNS check to climb through partitive noun constructions; enhanced complement verb promotion for misclassified noun verbs and adjectival gerund structures; restricted prepositional PI path to recipient prepositions (for, to) and deprivation preposition (from with verbal-head guard); expanded AGI/SI prototype sentences (abuse/getting away, consuming medication, loving qualities); removed PI from si_target_eligible to prevent patient-only targets from triggering SI.

2026-06-08: Removed LLR differential constraint (diff > 0) from candidate frame word discovery, sorting candidates by max LLR of either side; restored the expected PC2 evaluative-thematic trade-off; added pipeline_run.log TeeLogger redirection.

2026-06-08: @milestone SRW mentorship submitted.

2026-06-08: @group WP2 overleaft due August 1st.

2026-06-09: SRL cache key now reads first 50 and last 50 hash value; updated index.md preprocessing flow, results table, and diagram to align with extract.py.

2026-06-09: @idea dimensional mean + per-sentence max >>> role attribution (Centroid Margin Guard/Hard Gating/**Soft Fusion**)

2026-06-12: Fixed AGENTS.md: `lexical_human_rescue()` description updated to document Rule 1c (DET+singular demonym) and personal-pronoun inclusion in `_HUMAN_HEADS`; Phase 1 purpose summary updated to list all three rules and context-aware threshold. Entries had stalled at 2026-05-29 state missing tracker #45 and #54.

2026-06-12: Tightened AGENTS.md Update Rules: added change-type → section mapping table and mandatory closeout self-check to prevent future Annotated Code Map drift.

2026-06-17: Added missing cache.tex citations to reading.md; grouped Hofmann and Ouyang (InstructGPT) under §2.2.2 Output; moved Bender et al. (2021) to §2.1 Foundational; removed Lucy et al. (2022) due to method discrepancies.

2026-06-18: @group meeting and survey udpates.

2026-06-19: short paper revision plan.

2026-06-22: Reorganized reading.md: moved WEAT/CEAT entries to §2.6 Tools & Methods, deleted the two Kurpicz-Briki papers, sorted all subsections by year ascending, updated relevance gap counts and cross-references.

2026-06-23: Created and executed `ablation_recall_audit.py` to evaluate Phase 1 MiniLM recall against GTE-ModernBERT; found 6.8% calibrated False Negative Rate, confirming the efficiency-recall trade-off.
2026-06-23: Implemented statistical robustness checks under X/stability/; computed 1000 bootstrap CIs, leave-one-out sensitivity, and cross-chunk rank correlations.

2026-06-25: Collective bibliography sorting and updates.

2026-06-26: @group new midterm milestone July 15th.

2026-06-27: @group Detailed literature drafting for Bias chapter (§§ Foundational Works to Bias as a Representation Problem) including mathematical formulations, empirical settings, and dataset details. Added sociolinguistics citations in intro paragraph.

2026-06-28: Evaluated 4-shard data run; resolved net_atti zeroing bug in robustness checks; confirmed major thematic role stability gains (Spearman >0.56); analyzed WEAT/netAttI decoupling.

2026-06-29: Short paper refinement with taxonomy table, visualizing one-sentence going through the pipeline (to be replaced by actual graph); two-test results analysis; PC2 problem -- fp autoadmission, now resolved by expanding top_n from 100 to 500.

2026-06-30: @group further polishing draft with CDA citations, eliminating exaggerated adjectives from original notes.
2026-06-30: Aiming for AACL SRW full paper.

2026-07-02: Bibliography updates; planning for end-of-year paper.

2026-07-07: Expanded group K to 14+14 and re-ran Phase 1 and 2 on 2 shards; introduced positive review category to quarantine topic-collocate FPs; validated PC2 stability using robustness checks.

2026-07-08: Re-ran statistical robustness checks on 4-shard data (20,612 sentences, 27 groups); updated paper draft and local documentation with 4-shard stability metrics.

2026-07-17: @group Overhaulling final section [Bias as a Linguistic Problem] to compress grounding taxonomy, integrate missing literature, and align citations with Zotero keys.

2026-07-20: @milestone July 26 THREE graphs added to AACL SRW full paper/thesis proposal; section reordered, redundant subsection cleaned.