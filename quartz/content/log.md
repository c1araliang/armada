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

2026-05-07: @milestone May 16 AILC; June 4 Venice Conf; June 8 AACL Short Paper. Invite Prof. Arora for 1st year term paper supervision.

2026-05-08: @group WP2 report update; Abstract draft for AILC. [new anotation rule ATS introduced](https://c1araliang.github.io/armada/ats/)

2026-05-11 Creation of [cache](https://c1araliang.github.io/armada/cache), a temporary storage for drafts under revision; Polishing abstract draft for AILC; Short Paper draft for AACL.

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
