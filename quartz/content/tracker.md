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
|**Extraction review diagnosis**|Corrected. Use `dolma/semantic_filter_review.tsv` and a same-run `semantic_filter_report.txt` for Phase 1 extraction diagnosis; `X/output_review.tsv` is Phase 2 role/frame review (routes sentences where `role_review_flags` contains `no_clear_semantic_role` OR `targets` is null).|
|**Extraction gate**|Two-lane (STRICT + STRONG_MARGIN) margin-only gate; classifier removed after ablation showed 37% recall loss vs. ~6% precision gain. Phase 2 mention resolution provides the second filter.|
|**Mention layer**|Keyword extraction + inanimate suppression. `resolve_group_token()` simplified to ~100 lines; no resolver, no AMBIGUOUS/STRONG/DUAL branching. Civic tokens removed from demographic sets. Phase 1 re-extraction needed.|
|**Frame auto-admission**|Open. Auto-admission can overfire on generic discourse words; human review of `candidate_terms.json` remains needed.|
|**AgI/PI/SI hard cases**|Substantially addressed. Subjecthood separated from AgI; all three dimensions now require prototype confirmation (`DIM_MARGIN=0.04`); negation-scope blocking suppresses role assignment when the governing predicate is negated; PI is symmetric with AgI (structural evidence alone no longer sufficient). Embedded clauses and cross-sentence coreference still need validation.|
|**Local prototype AttI**|Resolved as a reported-metric issue. It is diagnostic only; reported `netAttI` now comes from frame association.|
|**Scope/discourse flags**|Implemented for review routing and now also for role suppression. Negation-scope blocking prevents AgI/PI/SI assignment when the governing predicate is negated; frame-AttI scope blocking prevents negated/corrected frame terms from counting. Correction/denial, quotation, reported speech, contrast, and multi-group flags still route to review only.|
|**Duplicate sentence exports**|Low impact. Some sentences enter `semantic_filter_results.tsv` through multiple group triggers; negligible at sample scale but can be deduplicated later if it complicates review.|

Current thresholds:

```text
# Phase 1 extraction (no classifier)
SEMANTIC_MIN = 0.34
SEMANTIC_MARGIN_MIN = 0.03
SEMANTIC_STRONG_MARGIN = 0.10
REVIEW_MARGIN_MIN = 0.05
BLOCK_REFERENCE_NOISE_KEEP = True

# Runtime (Phase 1)
ARMADA_DEVICE = auto (`mps` on Apple Silicon)
ARMADA_EMB_BATCH_SIZE = 64 for MiniLM on MPS, 32 on CPU, 256 on CUDA unless overridden
ARMADA_SENT_BATCH_SIZE = 4096
ARMADA_PARQUET_BATCH_SIZE = 10000

# Phase 2 analysis
AttitudinalPrototypeMatcher.positive_floor = 0.24
AttitudinalPrototypeMatcher.positive_margin = 0.02
DIM_FLOOR = 0.60
DIM_MARGIN = 0.04
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


### 2026-05-23

28. **Dimensional prototypes redesigned as polarity-balanced paraphrase sentences.**
`X/step3_attitudinal_prototypes.py` now stores AGI / PI / SI as six hand-written paraphrase sentences per dimension. Index 0 is the ruling definition (AGI ↔ PI structurally parallel: "They are agents bringing about effects, physically or mentally, through their own decisions." / "They are patients being affected, physically or emotionally, by others' actions or attitudes."; SI: "They have and/or show mindedness, inner consciousness, and autonomous feeling."). Indices 1–5 cover present/past/perfect tense, simple/progressive aspect, active/passive/mixed/embedded-passive voice, he/she/none gender, and plural / singular / collective `the group` number directly in prose. Each list deliberately includes both pleasant and unpleasant outcomes ("for better or worse", "praised by some and blamed by others", "loves, hates, hopes, and believes") so dimensional similarity does not collapse into evaluative polarity (PI ≠ negAttI; AGI ≠ posAttI). PI paraphrases include indirect attitude-as-affectedness ("how others felt about her", "what others believe about them"), not just direct-action affectedness. AttI seeds keep the canonical-complement + copular-grid design (NEG/POS 36 each). An interim verb-grid expansion built on `lemminflect` was tried and removed: a verb list re-introduced the same evaluative-prior problem that motivated removing `SUBJECTIVE_VERBS` etc. from `lexicons.py`, and stative verbs (`know`, `believe`, `understand`) produced ungrammatical progressive cells. `lemminflect` was uninstalled. Dimensional prototype counts shrank from the verb-grid attempt (73/97/73) to 6 per dimension, so `DIM_MARGIN=0.04` may need light recalibration on a held-out sample. Reflexive constructions ("they organized themselves") remain a known limitation that whole-sentence cosine cannot fully separate from non-reflexive agency; target-bound SRL evidence handles those cases downstream.

29. **SRL cache key now tracks prototype content.**
`X/run_pipeline.py`'s `srl_cache.pkl` key was previously `sha1(sentences_path + mtime)`, but the cached findings include `dim_agi_sim` / `dim_pi_sim` / `dim_si_sim` and the binary `agi` / `pi` / `si` decisions derived from them. Prototype edits silently produced stale cached cosines. The key now also hashes the concatenated `AGI_PROTOTYPES + PI_PROTOTYPES + SI_PROTOTYPES + NEGATIVE_ATTITUDE_PROTOTYPES + POSITIVE_ATTITUDE_PROTOTYPES` so any prototype change auto-invalidates the cache. CEAT-full cache already hashed centroid bytes and was unaffected.

30. **Frame admission now uses sentence + direction-only anchor gate.**
The single-tier sentence-cosine admission at `FRAME_SIM_FLOOR=0.55, FRAME_SIM_MARGIN=0.04` was promoting topic markers (`asylum`, `racial`, `wall`, `humanitarian`, `water`, `protest` to NEG; `right`, `journey`, `story`, `organization`, `high`, `people`, `member` to POS) because long humanitarian-topic seed sentences produce diffuse positive-discourse vectors that any topical word lands near. `_refresh_frame_inventory` now applies a second tier: word-level cosine of each candidate against four abstract sentiment anchors (`bad`, `negative` vs. `good`, `positive`) loaded from `candidate_terms.json` `anchor_negative_terms` / `anchor_positive_terms`. After the seed sentences were cleaned (entries 28+), anchor `|diff|` clusters in the 0.01–0.05 range, well below the 0.05 absolute floor we initially tried; that floor blocked everything. The current rule is direction-only: anchor must merely **agree in sign** with the sentence-tier direction (`anchor_diff * sentence_diff > 0`). Sentence margin is tightened from 0.04 to `FRAME_SIM_MARGIN=0.06` so the magnitude check sits on the more reliable signal. Anchors never enter centroid geometry, so reported WEAT/CEAT stay pure corpus-derived. Stale `auto_*` lists were reset in `candidate_terms.json` since they accumulated under the noisier single-tier rule. Known limits: evaluative adjectives that are not framing terms (`right`, `high`) still pass anchor sentiment polarity, and POS candidates with weak sentence-margin (e.g., `eligible`, `willing` from prior runs) drop out at 0.06; both are inherent to the simple-cosine setup and would require POS-tag filtering or per-domain anchor extension.


31. **AgI / PI / SI promoted to independent dimensions; winner-take-all margin removed.**
The `_dim_wins(score, others)` rule treated AGI / PI / SI as exclusive: each mention could fire at most one dim, the one whose cosine beat the other two by `DIM_MARGIN=0.04`. This contradicted `decisions.md`'s "SI is independent from AgI" design (a target can be 100% agent and simultaneously have or lack subjectivity) and silently suppressed multi-role mentions like `Refugees protested the deportation orders that affected them` (AgI on `protested`, PI on `affected`). It also produced the EFI loadings shown above where PI dominated PC1 because AGI/SI were systematically pinned near zero by the relative gate, not by data. The function is now `_dim_passes(score, floor)` — each dim is judged on its own absolute floor and a single mention can pass any combination of AGI / PI / SI.

32. **Per-dimension floors calibrated empirically from cached cosines.**
`DIM_FLOOR=0.60` was a uniform value chosen for the previous winner-take-all rule. With independent gating, floors must be per-dim because PI's mean cosine sits ~0.04 above SI's against the 6-paraphrase prototype matrix; a uniform floor would let PI fire freely while SI almost never fires. The new floors come from the 70th percentile of each dim's distribution over 5,674 cached mentions: `AGI_FLOOR=0.626`, `PI_FLOOR=0.637`, `SI_FLOOR=0.597`. Each dim now has roughly a 30% pass-rate on its own; ~50% of mentions pass nothing, ~20% pass exactly one, ~28% pass two or three. `DIM_MARGIN` is removed.

33. **Syntactic evidence no longer requires prototype confirmation for PI; SRL ARG0 gate kept for AGI.**
Previously, SRL `PATIENT_LABELS`, `dobj`, `nsubjpass`, and `pcomp+auxpass` evidence all required `pi_wins` to fire PI. Because PI prototype geometry sits closer to AGI prototype on active-transitive `B verbs A` constructions, the old rule systematically rejected the most prototypical patienthood pattern in English. PI evidence from SRL/dep is now trusted on its own; prototype disagreement raises a `*_proto_disagrees` review flag without vetoing. The PI `nsubj` affected-subject fall-through was also tried (gated by `pi_passes`, briefly also by an anchor sentiment check) but over-fired on subjects of mental verbs (`immigrant believes`, `Migrants feared`, `Black hoped`, `Americans believe`) because PI prototype geometry covers general "people-in-events" semantics too broadly without syntactic evidence; it was removed. PI now requires direct syntactic evidence: SRL `PATIENT_LABELS`, `dobj`, `nsubjpass`, `pcomp+auxpass`, or relcl object-of-mental-state. On the AGI side, SRL ARG0 still requires AGI prototype confirmation (`agi_passes`) because SRL labels unaccusative subjects (`refugees suffered`) as ARG0 even though they are semantic patients; the prototype floor naturally blocks these (unaccusative verbs land below `AGI_FLOOR=0.626`). A sentiment-anchor veto was tried and removed — anchor margin tracks sentence-level polarity, not verb argument structure, so it was an unreliable unaccusativity proxy.

34. **SI gains a target-as-experiencer syntactic guard.**
Without any syntactic backstop SI was firing on bare modifier mentions like `Black` in `Activists from the Black community led the campaign`. The fix is minimal: SI requires the target to be `nsubj` / `nsubjpass`, or to have an SRL ARG0 / ARG1 label. This restricts SI to mentions structurally positioned to be the experiencer, without re-introducing competition with AGI / PI.

35. **PI / SI prototype rewrite to remove cross-dim mental-language leakage.**
The PI prototype contained `"how others felt about her"` and `"do, say, and believe about them"` as part of the indirect-attitude-as-affectedness design (entry 28). Empirically, those mental-verb phrases pulled PI cosine high on every clear SI sentence (`The immigrant believes` → PI_sim=0.711, SI_sim=0.627), so SI fired only ~3% of the time and was not separable from PI. PI's seeds 2, 4, 5, 6 dropped the mental-verb language and now describe physical or institutional affectedness only. SI gained two paraphrases that explicitly place the target as the subject of an active mental verb (`believes, hopes, worries`; `fear ... and remember`). After the rewrite, SI fires on the previously-failing probes (`believes / feared / hoped`) without retracting agency from `Migrants feared deportation` (SI now co-fires).

36. **Spaced and prepositional compounds resolved without a Matcher; `asylum` single-token coverage added.**
spaCy tokenizes `asylum seeker(s)` as two tokens and `people of color` as four; neither side resolved on its own under the old token-level inventory check. `lexicons.py` adds `COMPOUND_TARGET_HEADS`, a small dict keyed by the head lemma (`seeker`, `seekers`, `people`) that describes either a compound-modifier child (`asylum`) or a `prep + pobj` chain (`of` → `color`). `resolve_group_token` consults this dict before its other rules and emits the canonical compound lemma (`asylum-seeker`, `people-of-color`) on the head while suppressing the modifier child to avoid double-counting. `seeker / seekers` are also added to `HUMAN_NOUNS`. `asylum` is also added as a single-token target lemma so it resolves when the compound head is absent; this is occasionally over-broad (`applying for asylum` where the noun denotes the institution) — future work should add `asylum` to `SEMANTIC_DISAMBIGUATION_TOKENS` so the GTE resolver decides per-context. spaCy `PhraseMatcher` was avoided to keep the resolver token-by-token.

37. **SRL cache key now hashes floors as well as prototypes.**
The SRL cache key already hashed the prototype text contents (entry 29). Because the new floors are part of the role-assignment logic but the cached findings record final binary roles, a floor change without a prototype change would silently produce stale `agi/pi/si` columns. The key now also hashes `f"{AGI_FLOOR}|{PI_FLOOR}|{SI_FLOOR}"` so any floor adjustment auto-invalidates the cache. CEAT-full cache is unaffected.

### 2026-05-24

38. **AGI ruling-definition and paraphrase [5] rewritten; sentiment-anchor sims persisted on findings.**
`AGI_PROTOTYPES[0]` was rewritten to "They are agents intended to bring about effects, physically or mentally, on or not on others." and `AGI_PROTOTYPES[5]` to "Through deliberate action, members caused things to happen to themselves or to the others." — both broaden volitional efficacy past purely outward physical action so mental acts and self-directed acts qualify. In practice the change has marginal effect on cosine because flat max-pooling means the broadest narrative prototype wins on most active-subject focus_texts; the ruling definition rarely wins. `AttitudinalPrototypeMatcher.match()` now also encodes two abstract sentiment anchors (`bad / negative` vs. `good / positive`) and returns `anchor_neg_sim` / `anchor_pos_sim` per mention. These were added to support an SRL-ARG0 unaccusative veto that was subsequently removed (anchor margin tracks sentence-level polarity, not verb argument structure); the sims are still computed and persisted on findings for downstream review.

39. **EFI: `orientation_anchor` removed; PC1 + PC2 now reported.**
The PC1 sign-flip in `_compute_efi` presumed all six dims (AgI, PI, SI, frame-netAttI, WEAT, CEAT) covary toward "negative framing"; the empirical correlation matrix on the current corpus shows AgI / PI / SI cluster opposite frame-netAttI / CEAT (corr `PI ↔ CEAT = -0.30`, `SI ↔ CEAT = -0.43`, `AGI ↔ netAttI = -0.42`), indicating that bias surfaces as **erasure of personhood attribution** along one axis rather than as uniform co-variation across all dims. The flip silently re-imposed a fixed sign on a sign-arbitrary PCA component and was removed. `_compute_efi` now returns `pc1_loadings`, `pc1_scores`, `pc1_variance_explained`, `pc2_loadings`, `pc2_scores`, `pc2_variance_explained`. `group_stats.tsv` schema gains an `EFI_PC2` column. Substantive interpretation comes from the loadings of the run, not from a fixed convention. Full Phase 2 rerun result: PC1 explains 38.1% of cross-group variance (up from 28.9% under winner-take-all), PC2 adds 19.8% (57.9% total). All three attribution dims now have meaningful within-corpus variance (AGI mean=0.080, PI mean=0.369, SI mean=0.188; all 24/24 nonzero). Frame admission produced auto-NEG `border, deportation, drug, illegal, poverty, racial, slavery, war` and auto-POS `assistance, journey, leader, manage, marry, strength, volunteer, willing`. `violence` LLR with minority groups was below the `min_llr=3.0` candidate-discovery floor (only `violence ↔ black` LLR=1.6) — corpus-level fact, not an admission-rule artifact.

### 2026-05-27

41. **MWE/compound resolution rewritten; lexicon adjustments.**
`resolve_group_token()` in `X/lexicons.py` was substantially rewritten to replace the old composite-route hyphenation logic with a principled idiomaticity-based suppression scheme. Key changes:

- **`DUAL_GROUP_TOKENS`** added: the intersection of `TARGET_TOKENS` and `CONTRAST_TOKENS` (e.g. `native`, `spanish`, `citizen`). When a dual-group token modifies a head that is itself demographic, it defers to the head's side rather than picking by inventory order. When it is a head with a demographic modifier child, it defers to the modifier and suppresses itself.
- **`COMPOUND_TARGET_HEADS`** added: explicit spaced/prepositional compound specs keyed by head lemma (`seeker`/`seekers` → `asylum-seeker`; `people` + `of color` → `people-of-color`). The head emits the canonical compound lemma; the modifier child is suppressed to avoid double-counting. Replaces the old `COMPOUND_TARGET_HEADS` dict that was already present for `asylum seeker` / `people of color` — this version is now the sole compound-resolution path.
- **Sibling-modifier suppression**: when two modifiers of the same head both resolve to demographic groups, the dominant-side modifier is suppressed and the minority-side modifier carries the mention. `African American voters` → `african` (minority) only; `American` (dominant sibling) suppressed.
- **`X speaking Y` suppression**: when a demographic token modifies a `speak/speaking` head that itself modifies a demographic head, the language-modifier token is suppressed. `Spanish speaking immigrants` → `immigrants` only.
- **Old composite route removed**: the `f"{lemma}-{head_group}"` hyphenated-lemma emission path and the conjoined-modifier hyphenation (`Asian and Latino Americans` → `asian-latino`) were deleted. Conjoined modifiers now each emit as standalone mentions via their own token paths.
- **Lexicon adjustments**: `spanish` added to `TARGET_TOKENS`; `citizen` now appears in both `TARGET_TOKENS` (descriptive use) and `CONTRAST_TOKENS` (civic majority framing); `caucasian` removed from `TARGET_TOKENS`; `overseas` removed from `TARGET_TOKENS`; `aborigine` added to `TARGET_TOKENS`; `asylum` added as single-token target lemma; `nationalist` added to `CONTRAST_TOKENS`; `settlers`/`colonists` plural forms removed (singular kept); `HUMAN_NOUNS` expanded with occupation/role nouns (`guard`, `officer`, `clerk`, `chef`, `driver`, `athlete`, `actor`, etc.).
- **`DUAL_GROUP_TOKENS`** and **`COMPOUND_TARGET_HEADS`** added to `AGENTS.md` annotated code map for `lexicons.py`.

Verify: run syntax check; re-run `run_pipeline.py`; inspect `output_results.tsv` for `african-american` (should now appear as `african` only), `native african` (should appear as `african` only via sibling suppression), `asylum seeker` (should appear as `asylum-seeker`), `people of color` (should appear as `people-of-color`), `spanish speaking immigrants` (should appear as `immigrants` only).

### 2026-05-25

40. **`output_results.tsv` / `output_review.tsv` made disjoint.**
The previous behavior wrote *all* per-sentence rows into `output_results.tsv` and *also* wrote the flagged subset into `output_review.tsv`, so review was a duplicate-with-extra-noise of results. The two files are now disjoint and together cover all per-sentence rows: a row goes to `output_review.tsv` iff `role_review_flags` contains `no_clear_semantic_role` OR `targets` is null; everything else goes to `output_results.tsv`. Reported group statistics in `group_stats.tsv` still aggregate from the in-memory `rows` (unchanged), so this routing change does not affect reported AgI/PI/SI/AttI numbers — only which file you open to inspect a given sentence.


### 2026-05-28

42. **frequency-based precision-oriented reporting**
Instead of always scanning all ethnonyms present in `lexicons.py`, a lemma frequency report will be generated and cached for future reference (exception: changes to input corpus) when running `extract.py`. Only top 8 lemmas in contrast/target group, 16 in total, will be used for lexical hit scan and subsequent extraction/profiling. Now among color terms, only black and white are admitted for clarity.

43. **Phase 1 classifier removed; extraction gate simplified to two semantic-margin lanes.**
The PCA + logistic-regression classifier in `extract.py` was removed after a controlled ablation on a 10k sample of `semantic_filter_lexical_all.txt`. With the classifier (`prob >= 0.45`) the gate kept 287/458 candidates (margin >= 0.10); without it, all 458. The 171 classifier drops were almost entirely true RELEVANT sentences in non-immigration framing, dropped because the 258-row training set could not anchor PCA directions for the 184 demonyms in `GATE_TOKENS` and was strongly biased toward refugee/migrant/immigrant surface forms. Reverse ablation (100-row manual eval of margin-only kept set) showed FP rate ≈14% vs ≈7-8% with classifier — the classifier reduced FP by half but at a 37% recall cost. Trade was unfavourable for low-frequency demonyms in non-immigration framing, which is exactly the population we need for the dominant/minority contrast. The classifier, the rescue lanes that existed only to compensate for classifier conservatism (`SEMANTIC_RESCUE`, `LEXICAL_HUMAN_RESCUE`), and the `relevant_probability` column in `semantic_filter_results.tsv` were removed together. The extraction gate is now: `STRICT` (`pos >= 0.34 AND margin >= 0.03`) plus `STRONG_MARGIN` (`margin >= 0.10`). Phase 2 `resolve_group_token()` and `INANIMATE_NOUNS` provide the second filter at mention-resolution time. Schema change: `semantic_filter_results.tsv` columns are now `sentence | semantic_pos | semantic_neg | semantic_margin | semantic_bucket`; `semantic_filter_review.tsv` columns are now `bucket | review_flags | sentence | semantic_pos | semantic_neg | semantic_margin`. The `classifier_pass` stat in `semantic_filter_report.txt` is renamed to `kept`.


### 2026-05-29

44. **Phase 1 lexical-human-rescue lane reinstated under MiniLM control; pure-regex paths removed.**
Audit of shard 0 output showed 7,425 of 11,099 kept rows (67%) came from `LEXICAL_HUMAN_RESCUE`, dominated by surface-similar non-human demonym uses (`German Shepherd`, `German TV`, `German Navy`, `white background`, `white wines`/`whites`, `Chinese New Year`, `European Championship`, `white-headed boy`, `black and tan German Shepherd Dog mix`). Tracing the four pre-existing regex paths showed each was failing for an independent reason that lexicon padding could not fix:
- **Path 2 (right-side human noun)** treated `German` as a human head because `is_human_noun()` used a `endswith(("man","men","woman",...))` heuristic. The endswith path also misclassified `German / Roman / human / specimen`-shape words.
- **Path 3 (article/determiner prefix)** admitted any `the/an/a + demonym + head`, including `the German Navy` and `the white background`.
- **Path 4 (predicate-adjective + pronoun + verb)** admitted any sentence containing a demonym plus a pronoun plus an `-ed/-ing` token within five words to the left, which let `I am a spayed female, black and tan German Shepherd Dog mix` through.
- **Inanimate-veto** in path 2 short-circuited on `is_inanimate_noun()` matches, but that function reads `_INANIMATE_HEADS` and cannot tell the noun reading of `work / show / play / bear` apart from the verb reading. Adding more nouns to `INANIMATE_NOUNS` only created new false negatives elsewhere.

Final design (entry 42 had removed rescue entirely after the classifier ablation; this re-adds a much narrower form, with compound inherent rescue dropped on 2026-05-30):
- `lexical_human_rescue(sentence)` now returns one of `"inherent"` / `"candidate"` / `None`. `inherent` = gate token is a plural of a non-color gate token (`Germans`, `Americans`, `refugees`); admit without further checking. `candidate` = gate token has a human head noun within ±4 tokens and Rule 1a did not fire; this surface pattern still over-admits (`white-headed boy`, `German Shepherd Dog mix`, `Chinese New Year`), so a second MiniLM pass is required. `None` = no rescue path; sentence falls through to BORDERLINE/REJECTED based on the main lane margin.
- Color-tone gate plurals (`whites`, `blacks`, `browns`, `yellows`) are excluded from the `inherent` path because they often refer to wines, sports teams, or color categories. They route through the `candidate` path so MiniLM decides per-sentence.
- `process_batch()` adds two new query sets, `RESCUE_POS_QUERIES` (4 paraphrases of "person identified by nationality / ethnicity / race / skin color") and `RESCUE_NEG_QUERIES` (8 paraphrases covering animal breeds, ethnic-named cultural products / holidays / sports competitions, public events, color/material descriptions, geographically-named tech brands, languages, and country-named institutions). Sentences tagged `candidate` are admitted only if the rescue MiniLM scoring clears `RESCUE_POS_MIN=0.25 AND RESCUE_MARGIN_MIN=0.06`. `Path 1 (inherent)` is admitted directly without rescue scoring; this preserves throughput on the high-confidence cases.
- Rescue scoring reuses the per-sentence MiniLM embeddings already computed for the main lane (`semantic_scores()` returns `embeddings`); rescue only multiplies them by the rescue query matrices. No additional encoder pass.
- Helpers removed: `is_inanimate_noun`, `_INANIMATE_HEADS`, `_INANIMATE_WINDOW_RE`, `_HUMAN_WINDOW_RE`, `_GROUP_HUMAN_RE`, `_VERB_RE`, `_PRONOUN_RE`, `COLOR_OR_SPATIAL_TOKENS`, `_window`, `_nearest_distance`. `is_human_noun` no longer relies on endswith and now consults `_HUMAN_HEADS` plus a short hardcoded honorific/title set. `INANIMATE_NOUNS` is no longer imported by `extract.py` (still used by Phase 2 `resolve_group_token()`).

Effect on shard 0 (single-shard testing default): kept 11,099 → 7,284 (-34%); `lexical-human` 7,425 → 3,934 (-47%); `STRICT` (765) and `STRONG_MARGIN` (2,589) unchanged because rescue is downstream of the main two-lane gate. After the rerun, no `german shepherd` rows remain in `semantic_filter_results.tsv`. The four remaining `Chinese New Year` / `white background` / `German TV` hits are STRONG_MARGIN admissions of the main lane, not rescue admissions, and either contain genuine demographic content (Bangladeshi workers during Chinese New Year period) or would need `SEMANTIC_STRONG_MARGIN` retuning to address — out of scope for this rescue rework.

Verify: run syntax check; rerun `extract.py` (default `ARMADA_MAX_FILES=1` for shard 0); confirm `lexical-human` count in `semantic_filter_report.txt` dropped; spot-check that `german shepherd` / `german tv` / `white background` / `chinese new year` / `european championship` adversarial patterns no longer appear in the `LEXICAL_HUMAN_RESCUE` bucket of `semantic_filter_results.tsv`. The `LEXICAL_HUMAN_RESCUE` semantic_bucket label is unchanged; downstream Phase 2 readers do not need updating.

44. **-man suffix selective inclusion**

```
tr -d '\r' < dolma/semantic_filter_lexical_all.txt | grep -i -oE '\b[A-Za-z]+(man|men|woman|women|boy|boys|girl|girls)\b' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn | head -30
3523 german
 895 women
 402 human
 347 woman
 102 chairman
  87 roman
  69 frenchman
  66 ottoman
  38 spokesman
  35 cowboy
  35 businessman
  23 yemen
  23 gentleman
  22 blackman
  21 norman
  20 cowboys
  19 businessmen
  17 batman
  16 craftsmen
  16 coleman
  15 ramen
  15 herman
  14 truman
  14 statesman
  14 fishermen
  13 superman
  13 cayman
  12 spokeswoman
  12 oman
  12 gentlemen
  ```
statistically `nationality` + `-man` pattern makes very ignorable part of the first parquet; likewise for the whole corpus. even `_GROUP_PERSON_SUFFIX_RE`keeps only 3/69 for frenchman.

Note: Multi-word patterns like `citizen of [Nation]` (e.g., `citizen of France` which is semantically equivalent to `french`) represent a similar class of syntactic constructs that are currently not captured after the removal of the polysemous civic token `citizen`. Handling these requires dedicated multi-word dependency patterns or post-extraction mapping, which is deferred to prevent re-introducing civic token noise.


45. **Context-aware rescue thresholds and human lexicon expansion**
- **Problem**: Widening the candidate search window to bidirectional ±4 and lowering `RESCUE_POS_MIN` to `0.25` introduced physical color-adjective false positives (e.g. `white handle`, `white cap`, `striped shirts`, `black hair`) because they were close to pronouns/human nouns.
- **Solution**:
  - Implemented a context-aware threshold in `lexical_human_rescue()`: non-color demonyms and color terms immediately followed by a human noun (+1 or +2 tokens, e.g. `black man`, `white President`) use `RESCUE_POS_MIN` (0.25); other color-adjective candidates not directly modifying human nouns require a stricter `0.32` floor.
  - Added `"guy" / "guys"` to `HUMAN_NOUNS` in `X/lexicons.py` to ensure common phrases like "white guy" resolve as valid human-noun anchors.
- **Effect**: Dropped kept `lexical-human` rows on shard 0 by 112 (4,866 → 4,754), successfully filtering out color FPs (`white handle`, `white cap`, `black hair`) while preserving true demographic references (like `too white--all white` and `black man`).

### 2026-06-07

46. **Civic/polysemous token removal and keyword-extraction resolver.**
Removed `citizen`, `native`, `local`, `national`, `nationalist`, `domestic`, `majority`, `native-born` from `TARGET_TOKENS` / `CONTRAST_TOKENS`. These civic/institutional tokens caused systematic false positives ("national park", "local government", "native speaker") and consumed top-8 frequency slots that should go to genuinely demographic tokens. `citizen`, `native`, `majority` remain in `HUMAN_NOUNS` (valid human-referent head nouns); `national` excluded from `HUMAN_NOUNS` due to institutional dominance.

47. **`resolve_group_token()` rewritten as keyword extraction + inanimate suppression.**
The ~305-line, 14-branch resolver was replaced with ~100 lines: compound resolution → prefix suppression → X-speaking-Y → child/sibling modifier suppression → inanimate head suppression → emit. No `AMBIGUOUS_NOUNS`, `DUAL_GROUP_TOKENS`, `STRONG_TARGET/CONTRAST_TOKENS`, `AMBIGUOUS_TARGET/CONTRAST_MODIFIERS`, `SEMANTIC_DISAMBIGUATION_TOKENS`, or semantic resolver. Any active-set token whose head is not inanimate resolves.

48. **`SemanticGroupResolver` removed.**
The source `.py` was already deleted (only `.pyc` cache remained). All imports and references removed from `run_pipeline.py`, `dim_sanity.py`, and `lexicons.py`. GTE ModernBERT model is now loaded directly via `SentenceTransformer(ANALYSIS_EMBEDDING_MODEL)` in `run_pipeline.py`.

49. **`native-american` compound added.**
`COMPOUND_TARGET_HEADS` extended with `american`/`americans` + modifier-child `native` → canonical `native-american` (minority). Without this, "Native Americans" resolved as dominant `american` with `native` suppressed.

50. **`non-` prefix unconditional suppression.**
`non-white`, `non-japanese`, etc. now return `None` unconditionally. `anti-`/`pro-` prefixes retain the existing modifier+head guard (stance prefix, but the demographic referent is real).

51. **Phase 1 inanimate-adjacency pre-filter.**
`extract.py` now checks if every gate token in a sentence is adjacent (±2 whitespace tokens) to an inanimate noun before MiniLM scoring. If so, the sentence is skipped (no MiniLM compute). "German law", "black hole", "American government" are filtered before embedding. The all-lexical file still includes these for CEAT-full. `INANIMATE_NOUNS` was expanded to cover gaps previously handled by the resolver (dog breeds, statistics terms, `accent`, `parliament`, `engineering`, etc.).

52. **`foreign` added to `TARGET_TOKENS`.**
Was previously only in `AMBIGUOUS_TARGET_MODIFIERS` (removed). Needed for `_group_side()` to correctly classify it as minority-side.

53. **Inanimate and Human Lexicon Expansion.**
Based on empirical spaCy analysis of the extracted corpus, expanded `INANIMATE_NOUNS` in `X/lexicons.py` with 55 new entries (e.g., `privilege`, `history`, `heritage`, `identity`, `race`, `neighborhood`, `skin`, `citizenship`, `nationality`, `civilization`, `settlement`, `campus`, `force`, `troop`, `army`, `authority`, `commission`, `caucus`, `rule`, `colony`) to capture abstract concepts, military/corporate terms, and places. Expanded `HUMAN_NOUNS` with 13 new entries (e.g., `counterpart`, `male`, `female`, `author`, `explorer`, `individual`, `human`) to prevent false negatives. Rerun of Phase 1 extraction showed `inanimate_prefilter_removed` rose from 29,865 to 33,625, and final kept sentences dropped from 7,255 to 6,936, demonstrating significantly improved filtering precision.

54. **Det-Preceded Demonym Rescue and GTE ModernBERT Main-Lane Rescue.**
Resolved persistent false rejects of true human mentions (e.g., *An American gained entrance*, *white man*, *Jewish immigrants in Palestine*, *American slave*) under margin-based semantic scoring:
- **Noun Phrase Demonym Detection (Rule 1c)**: Tagged singular non-color demonyms preceded by determiners (e.g., *a/an/the/this*) and not followed by non-human nouns as `candidate` human mentions to undergo MiniLM semantic validation.
- **ModernBERT Main-Lane Screening**: Allowed the GTE ModernBERT fine screening pass to validate borderline sentences against the main STRICT/STRONG_MARGIN queries. GTE ModernBERT's superior semantic capacity successfully keeps sentences like *What to the American slave is your Fourth of July?* in the `STRICT` category.
- **Rescue POS Bypass**: Bypassed the strict relative margin check (`rm >= 0.06`) when the candidate's absolute POS query score `rp` is very high (`rp >= 0.35` under MiniLM, `rp >= 0.53` under ModernBERT). This prevents locations, events, and material nouns in the sentence from compressing the margin and causing false drops.
- **Rerun Stats**: Kept sentences stabilized at **7,911** (down from the initial **19,823** bloat when Rule 1c was routed as inherent), review candidates settled at **327**, and GTE ModernBERT successfully rescued **1,016** sentences from the review candidates.

55. **Bibliographic Reference Noise Hard-Blocking Restored.**
- **Rationale**: Re-enabled hard-blocking on `bibliographic` and `index_page_ref` noise patterns. We had disabled them to prevent year dates from blocking historical speeches (like *Speech to the Ottowas ... July 6th,1820.*). However, since the `index_page_ref` regex has been tightened to prevent matching 4-digit years, re-enabling the blocking safely excludes pure citation/book-title noise (such as *A Historical Atlas of the Jewish People*, *Oregon State University Press*, etc.) while keeping the true demographic historical speech sentences.
- **Rerun Stats**: Final kept sentences settled at **7,933** (a clean reduction of 15 bibliographic/citation lines from the previous run's **7,948**), borderline review candidates at **305**, and GTE ModernBERT successfully rescued **1,023** sentences from the review candidates.

56. **MWE Chain Compound Rule, Active Label Enforcer, and `[demo]-born` Fix.**
- **Hyphenated Chain Compound Rule**: Upgraded the compound resolution logic in `X/lexicons.py` to scan arbitrary length hyphenated chains (such as `Palestinian-Lebanese-American`) and suppress the dominant constituents if any minority constituent is present.
- **Active Label Gating for Resolved Canonicals**: Enforced `active_labels.json` validation on the final resolved canonical lemma in `resolve_group_token()`. This ensures that compounds like `native-american` (which previously resolved to `native-american` because the head `American` was active, despite the compound itself not being active) are now correctly suppressed and return `None` (solving the `output.md` native-american leak).
- **`[demo]-born` Fix**: Fixed a lemmatization conflict where the verb `born` (lemmatized to `bear` by spaCy) matched the animal `bear` in `INANIMATE_NOUNS`, incorrectly causing inanimate head suppression for words like `American-born`.
- **Methodological Note on MWE Proportional Info Loss**: Added explicit code comments detailing that we cannot guarantee how the language model distributes attention across MWE constituents (e.g. `African American`), meaning proportional info loss is certain (refer to `cache.tex` L125).

57. **Phase 1 Compound Counting, Negation Discounting, and Compound Active Gating.**
- **Phase 1 Compound Counting**: Updated the frequency pre-scan in `extract.py` to match and count compound terms (`native-american`, `asylum-seeker`, `asylum-seekers`, `non-white`) and decrement their constituents to prevent double-counting.
- **Negation Discounting**: Implemented `non-` prefix pattern matching for all demographic tokens to subtract negation instances from positive group counts.
- **Compound Active Gating**: Upgraded the active label check wrapper in `resolve_group_token()` to check if any constituent part of a compound canonical (e.g. `american` in `native-american`) is active, rather than suppressing it, preventing the native-american null target/review leakage.
- **SRL Cache Invalidation**: Deleted stale `srl_cache.pkl` and `ceat_full_cache.pkl` to force full semantic role labeling and contextual embedding metrics calculation from scratch.

**Impact**: Phase 1 extraction results are clean of bibliographic noise, and Phase 2 mention resolution now strictly conforms to the active labels set and handles chain compounds and `[demo]-born` constructs with high precision. Kept sentences rose to **8,060**, and CEAT-full successfully resolved `native-american` with $N = 181$ sentences and valid metrics (AgI, PI, SI, AttI, WEAT, CEAT) in the final reported metrics tables.

### 2026-06-07 (cont.)

58. **`foreign` removed from TOKEN list; compound part-match bypass fixed; `jewish` added to active labels.**
- **`foreign` removal**: Removed `foreign` from `TARGET_TOKENS` in `X/lexicons.py`. As a bare adjective it is too polysemous (`foreign policy`, `foreign land`, `foreign minister`) and inanimate-head suppression only catches modifier uses — `foreign` as a standalone predicative adjective or head-noun slot (e.g. `foreign objects`) was still leaking through. `foreigner`/`foreigners` remain in `TARGET_TOKENS` for cases where the person-referent reading is unambiguous.
- **Compound part-match bypass removed**: `resolve_group_token()` previously allowed a compound canonical (e.g. `native-american`) to pass the active-label check if any constituent part (`american`) was in the active set. This let `native-american` ride into `output_results.tsv` even though it was never in `active_labels.json`. The check is now exact: canonical must be the whole string present in the active set, otherwise `None` is returned.
- **`active_labels.json` updated**: `foreign` replaced by `jewish` (next highest-frequency uncolored target token in the demographic word count, 2799 occurrences on the current shard). `jewish` was already in `TARGET_TOKENS`; this change makes it eligible for Phase 2 mention resolution and reported group stats. `native-american` (495 occurrences) remains below the top-8 threshold and will no longer appear in output.

**Impact**: `foreign` and `native-american` rows no longer appear in Phase 2 output. `jewish` enters reported group stats. Compound part-match logic is now strict; the active-label gate is a whole-canonical equality check. No Phase 1 re-extraction required; `active_labels.json` will be overwritten on the next Phase 1 run by extract.py's frequency-based top-8 selector.

### 2026-06-07 (cont. 2)

59. **Recursive Coordinate Conjunction Resolution for Subjecthood.**
- **Conjunction Resolution**: Upgraded `_resolve_role()` in [step3_feature_extraction.py](file:///Users/l/projects/X/step3_feature_extraction.py) to recursively traverse coordinating conjunction (`conj`) chains. Previously, a single-step `if dep == "conj":` checked only the immediate head, causing downstream conjuncts in chains (like `Europeans -> German -> French -> Swedish -> Spanish`) to resolve to `"conj"` instead of propagating all the way up to the root conjunct's dependency (e.g. `"nsubj"`).
- **SRL Cache Invalidation**: Stale `srl_cache.pkl` is invalidated automatically when code / floor changes occur. (Note: deleted stale cache to force recalculation).

**Impact**: Correctly resolves and assigns `subjecthood=1` to all conjuncts in a coordinate conjunction list (e.g. `French` and `Spanish` in the user's probe sentence now correctly inherit `subjecthood=1`).

60. **SRL ARG0 patient-dep guard + SI eligibility uses filtered roles.**
- **ARG0 guard** (`effective_dep not in ("dobj", "nsubjpass", "no_inherit")`): SRL models sometimes assign ARG0 to structurally-patient tokens (double-object constructions, ergative verbs). Previously those ARG0 labels were accepted unconditionally, mislabelling patients as agents and — critically — setting `si_target_eligible = True` via the raw `srl_info["roles"]` check. The guard blocks ARG0 admission whenever the resolved dep tells us the token is in a patient or no-inherit position.
- **`si_target_eligible` uses filtered `roles`** (was: raw `srl_info["roles"]`): Even with the ARG0 guard, an ARG0 that was admitted to `srl_info` but then failed `agi_passes` (`srl_arg0_proto_disagrees`) would still trigger SI via the old raw check. Switching to `"AgI" in roles` / `"PI" in roles` closes that path — SI is now contingent on AGI/PI having actually been assigned.
- **`dim_sanity.py` updated**: Added the `use the Internet` coordination test sentence (`Europeans / German / French / Swedish / Spanish use the Internet …`) as a `dump=True` test case.

**Impact**: The `white`/`black` multi-group sentence now shows correct labels (white: AGI + SI as yeller/subject; black: PI + SI as object/patient). The Internet sentence correctly returns all dims as `fail` for all five group tokens (AGI winner ~0.61 < 0.626; PI winner ~0.55 < 0.637; SI winner ~0.56 < 0.597 — all clearly below floor, confirming the sentence has no evaluative framing signal). The two code changes together eliminate SI inflation on raw-ARG0 paths. SRL cache invalidated.


61. **AGI elif-blocking fix** — when SRL ARG0 exists but fails `agi_passes`, the old `elif nsubj` was silently blocked. Changed to a guarded `if "AgI" not in roles` so the nsubj fallback still fires (adding `subject_nonagentive` flag). Effect: `Americans love`, `Italian fans get away with` both now correctly reach the nsubj evidence path.

62. **Gerund-pcomp / gerund-amod unwrapping in `_resolve_role`** — two sub-cases:
- Sub-case (a): pcomp VERB is a child of the preposition → pobj reattaches as dobj of that VERB.
- Sub-case (b): spaCy attaches VERB as `amod` of the pobj noun (`players=pobj(with)`, `abusing=amod(players,VERB)`) → reattach as dobj of the VERB amod.
Effect: `black players` in "racially abusing black players" now resolves `[PRED:abusing]` and receives PI (0.606, marginal fail — PI_FLOOR 0.637) + SI (0.638, PASS). `srl_patient_proto_disagrees` flag correctly applied.

63. **SI preference/affection prototype [6] added** — `"They love, prefer, or appreciate what they find meaningful; they are drawn to or moved by what they encounter."` For `Americans love a British accent`, the new prototype scores 0.573 — below the existing winner [3] at 0.580 (itself 0.017 below floor 0.597). The sentence returns all-fail, which is correct — `love a British accent` is a generic cultural preference, not a demographic framing sentence in the pipeline sense. SI_FLOOR stays at 0.597.

**Residual**: `black players` PI = 0.606 misses PI_FLOOR by 0.031. Could be addressed by a targeted PI prototype for "subjected to abuse/harassment" but risks over-firing. Left open.

64. **Quantifier Noun Climbing, Complement Verb Promotion, Prepositional PI Path, and SI Eligibility.**
- **Quantifier Noun Climbing**: Upgraded `_resolve_role()` to climb through quantifier nouns (like `couple`, `group`, `majority`, etc.) when traversing preposition/quantifier chains (such as `a couple of white residents` -> `white` inherits the subject role of `yelling`).
- **Complement Verb Promotion**: Enhanced the light-verb complement promotion logic in `_resolve_role()` to handle misclassified ROOTs (e.g. `vote` parsed as NOUN in `vote to separate`) and adjectival gerund structures (e.g. `abusing` parsed as `amod` of `players` in `get away with racially abusing black players`). This correctly promotes the subject's head verb to the target action complement verb (`abusing` and `separate` respectively).
- **Prepositional PI Path**: Restricted the `pobj` PI path to recipient prepositions (`for`, `to`) and deprivation preposition (`from` with verbal-head syntactic guard, e.g. *taken from*), allowing prepositional targets (like *kit for Canadians* or *taken from his parents*) to receive the `PI` label if they pass the floor, while avoiding false-positive PI labels on locative/origin phrases (like *from the Black community*).
- **SI Eligibility Restructured**: Restricted `si_target_eligible` by removing `PI` from the eligibility clause so that pure direct object or prepositional object patient/recipients (who are not clausal subjects or agents) are excluded from triggering SI (resolving the false positive `SI` on `black` in `abusing black players`).

65. **Restoration of Original Balanced Prototypes.**
- **Restoration**: Restored the original 6-sentence balanced AGI/PI/SI prototype lists in `step3_attitudinal_prototypes.py` (which cover she/he/they/group across past/present/perfect/progressive tenses and active/passive/mixed voices) to preserve equal representation and eliminate ad-hoc additions, correcting two minor typographical/agreement issues in the original text.
- **Syntactic Check & Sanity Check**: Verified that compilation and sanity checks pass cleanly.

**Impact**: Correctly resolves and labels target semantic roles using the clean 6-sentence prototype baseline. Locative prepositional phrases are correctly suppressed while true deprivals (using `from` with a verbal head) and recipients (`for`, `to`) remain active.

66. **LLR Differential Constraint Removal in Frame Discovery.**
- **Removal**: Removed the `diff > 0` (minority_llr - dominant_llr > 0) asymmetric restriction from `_find_candidates` in `X/run_pipeline.py`. Collocates are now evaluated for frame candidate admission if their maximum LLR with either minority or dominant groups clears the `min_llr` floor (3.0), and candidates are sorted by their maximum LLR.
- **Impact**: Restored the expected PC2 evaluative-thematic trade-off where positive `net_atti` and `weat` load together against negative `PI` and `SI`. This enables the discovery of generic framing terms (e.g. `racial`, `race`, `attack`, `war`, `terrorist` as F⁻; `community`, `insurance` as F⁺) that co-occur strongly with both dominant and minority groups, resolving the decoupling of frame-level and embedding-level metrics.

### 2026-06-08

67. **Pipeline Log Redirection (TeeLogger).**
- **Changes**: Added a `TeeLogger` wrapper to `X/run_pipeline.py` to mirror all terminal output (stdout and stderr) to `X/pipeline_run.log` in append mode (`"a"`). Added a timestamped marker separator at the beginning of each run to make runs easy to contrast.
- **Impact**: Terminal outputs are now automatically recorded and persist in `pipeline_run.log` without overwriting prior runs.


