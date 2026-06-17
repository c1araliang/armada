# ARMADA Agent Guide

Academic computational linguistics project.

Research wiki:
https://c1araliang.github.io/armada/

Project Start: March 04, 2026

---

Note: output will be reviewed by LLM committee incl. but not limited to GPT, Claude, Gemini, DeepSeek and Qwen. Websearch allowed by default.

## Boot Summary

- Repo root: `/Users/l/projects`.
- Active pipeline: `extract.py` creates the filtered corpus; `X/run_pipeline.py` computes the analysis tables.
- Main research wiki: `quartz/content/` locally and https://c1araliang.github.io/armada/ online.
- Most important code: `extract.py`, `X/embedding_config.py`, `X/run_pipeline.py`, `X/lexicons.py`, `X/group_mentions.py`, `X/step3_feature_extraction.py`, `X/step4_metrics.py`.
- Protected research artifacts: `dolma/semantic_filter_results.tsv`, `dolma/semantic_filter_lexical_all.txt`, `dolma/semantic_filter_review.tsv`, `X/group_stats.tsv`, `X/output_results.tsv`, `X/output_review.tsv`, `X/association_discourse.tsv`, `X/candidate_terms.json`.
- Do not scan unless directly relevant: `X/venv/`, `quartz/public/`, `dolma/data/`, `__pycache__/`, `.git/`, `.DS_Store`.
- Syntax check: `cd /Users/l/projects && source X/venv/bin/activate && python -m py_compile extract.py X/*.py`.
- Phase 1 run: `cd /Users/l/projects && source X/venv/bin/activate && python extract.py`.
- Phase 2 run: `cd /Users/l/projects/X && source venv/bin/activate && python run_pipeline.py`.

---

## Purpose

ARMADA is a PhD-level computational linguistics project that detects and quantifies systematic **framing bias** toward minority/immigrant vs. dominant demographic groups in LLM pretraining corpora. Corpus family: **Dolma v1.6**.

The pipeline has two phases:

- **Phase 1** (`extract.py`): Offline, per-corpus streaming filter. Applies a 2-lane semantic gate (lexical mention regex → MiniLM semantic retrieval against POS/NEG queries; a row enters the corpus if it passes either the STRICT lane `pos >= 0.34 AND margin >= 0.03` or the STRONG_MARGIN lane `margin >= 0.10`) plus a constrained `LEXICAL_HUMAN_RESCUE` lane with three surface rules: Rule 1a admits plural non-color demonyms directly ("Germans", "refugees"); Rule 1c admits singular non-color demonyms preceded by a determiner ("an American", "the German who") as MiniLM-scored candidates; Rule 1b admits gate token + human-head noun or personal pronoun within ±4 tokens as MiniLM-scored candidates, with a context-aware threshold (0.25 for non-color demonyms or color+direct-human-head; 0.32 for other color adjacencies). An inanimate-adjacency pre-filter (added 2026-06-07) discards sentences where every gate token is adjacent to an inanimate noun before MiniLM scoring. Outputs a sentence-level TSV of demographically-relevant sentences. The earlier PCA+LogReg classifier was removed after a 2026-05-28 ablation; earlier pure-regex rescue paths (article-prefix, predicate-adjective, inanimate-noun veto) were removed 2026-05-29; see `decisions.md`.
- **Phase 2** (`X/run_pipeline.py`): Analysis. Reads the filtered TSV, runs preprocessing, feature extraction (Subjecthood/AgI/PI/SI plus local AttI diagnostics), LLR/LogDice collocate discovery, target-bound frame-derived AttI, WEAT, CEAT, CEAT-full/Δ-CEAT, PCA-based EFI, and regression. Writes per-sentence and per-group output tables.

Core per-group framing dimensions:
- **Subjecthood** — diagnostic syntactic subjecthood, separated from agency
- **AgI** — agency (proportion as semantic/social agent)
- **PI** — patienthood (proportion as patient/recipient of action)
- **SI** — subjectivity (proportion as subject of mental-state verbs)
- **frame_negAttI / frame_posAttI / netAttI** — target-bound corpus-level evaluative frame association; local prototype AttI is diagnostic only
- **WEAT** — type-level embedding association with F⁻/F⁺ frame sets (GTE ModernBERT)
- **CEAT-filtered** — sampled contextual embedding association on the filtered corpus; reports mean, N, and SE
- **CEAT-full / Δ-CEAT** — sampled contextual association on all lexical hits; difference quantifies "associative contamination"
- **EFI** — Evaluative Framing Index: PC1 of a PCA over the group × dimension matrix

---

## Agent Instructions

- Think before coding. State assumptions when they affect methodology, thresholds, schemas, artifacts, or interpretation. If two plausible interpretations would lead to different code or research claims, do not pick silently; name the fork and resolve it from local evidence or ask.
- Preserve research assumptions unless explicitly asked to revise them.
- Do not silently change thresholds, labels, lexicons, classifier features, prompts, schemas, evaluation logic, or reported-result logic.
- Prefer the minimum code and documentation that solves the requested problem. Do not add speculative features, unused configurability, or abstractions for one-off logic.
- Make surgical changes. Touch only files needed for the request, match existing style, and do not refactor adjacent code unless the request or verification requires it.
- Remove imports, variables, functions, or docs that your own changes make obsolete. Do not delete pre-existing dead code unless explicitly asked; mention it instead.
- When changing methodology-relevant code, note what changed and why.
- Define success criteria before substantial edits and verify them after implementation. For small code changes, at minimum run the syntax check unless the change is documentation-only.
- Do not rewrite unrelated code or reformat unrelated files.
- Treat generated outputs, cached artifacts, and reviewed annotations as research artifacts, not disposable build products. Do not delete or overwrite, except by re-running the respective pipeline step.
- Avoid broad repository scans over generated or dependency directories unless the task requires them.

---

## Documentation Update Contract

When code, schemas, thresholds, methodology, or pipeline behavior changes, update the smallest necessary documentation set:

- `quartz/content/log.md` — one-line daily activity note only. No prose, no rationale, no duplicated tracker content.
- `quartz/content/tracker.md` — current status, implemented changes, and remaining impact. Use this for "what changed and where we stand."
- `quartz/content/decisions.md` — methodological rationale. Use this for "why this design solves the research/pipeline problem."
- `quartz/content/todo.md` — unresolved action queue only. Do not duplicate implemented-change history or design rationale here; remove or resolve questions that the current code/docs already answer.
- `AGENTS.md` — operational instructions, active file map, schemas, validation commands, protected artifacts, and workflow rules for future agents.
- `quartz/content/index.md` — high-level public overview. `sync.py` updates the `**Latest Update**` date automatically; **content updates are the agent's responsibility**, not `sync.py`'s. Whenever a change touches the pipeline diagram, the dimensions table, the "Pipeline steps (Updated)" section, the EFI/PCA description, or any reported preliminary number visible at the top level, update `index.md` in the same closeout that updates `tracker.md` / `decisions.md`. Do not assume `sync.py` will catch substantive content drift — it will not.

Closeout responses after project edits must report:

- files changed and the reason for each group of changes;
- whether protected artifacts were regenerated or left untouched;
- verification commands run, or why verification was not run;
- any corrected assumption or artifact mismatch that affects interpretation;
- whether `quartz/content/index.md` was reviewed and updated when the change touches anything visible at the top-level overview (pipeline diagram, dimensions table, EFI description, reported preliminary numbers, schema mentions). `sync.py` only refreshes the `**Latest Update**` date — substantive content drift in `index.md` is the agent's responsibility.

## Update Rules

Update this file when:
- a new active pipeline file or workflow is added;
- a TSV/JSON schema changes;
- thresholds, prompts, lexicons, classifier features, frame taxonomy, or evaluation logic change;
- generated output paths change;
- a file is promoted from experimental/legacy status to active use, or the reverse;
- validation commands change;
- future-agent behavior changes, including documentation update policy or artifact-review policy.

**Which section to touch** — map each change type to the specific part of this file that must be updated:

| Change | Section(s) to update |
|---|---|
| Any function body or logic change in `extract.py` or `X/*.py` | The matching bullet under `Key sections:` in the Annotated Code Map |
| New/removed function in a pipeline file | Add/remove the bullet under that file's `Key sections:` |
| Threshold change | The matching `Key sections:` bullet AND the `Current thresholds` block in `tracker.md` |
| Rescue-lane rule change (any of Rules 1a/1b/1c, `_HUMAN_HEADS`, `_COLOR_GATE_TOKENS`) | `_HUMAN_HEADS`, `_COLOR_GATE_TOKENS`, and `lexical_human_rescue()` bullets AND the Phase 1 purpose summary in `## Purpose` |
| New output file or schema change | `Data/Output Conventions` section |
| New runtime env var | `Key sections:` bullet for the file that reads it |
| Lexicon change (`TARGET_TOKENS`, `CONTRAST_TOKENS`, `INANIMATE_NOUNS`, `HUMAN_NOUNS`, `COMPOUND_TARGET_HEADS`) | `Key sections:` bullet for `lexicons.py` |

**Closeout self-check** — if a function you modified has a `Key sections:` bullet in the Annotated Code Map below, verify that bullet still matches the new behaviour and update it in the same response if not. Do not scan functions that have no bullet.

Do not let `todo.md` and `tracker.md` drift into the same role. `todo.md` says what remains to do; `tracker.md` records status and changes.

---

## Annotated Code Map

### Phase 1 — Corpus Extraction

- `extract.py`
  - Role: Streaming 2-lane semantic filter plus MiniLM-controlled lexical-human rescue lane over Dolma parquet files. No classifier (removed 2026-05-28); pure-regex rescue paths removed 2026-05-29. Inanimate-adjacency pre-filter added 2026-06-07.
  - Key sections:
    - `Configuration constants` — semantic-margin thresholds (`SEMANTIC_MIN`, `SEMANTIC_MARGIN_MIN`, `SEMANTIC_STRONG_MARGIN`, `REVIEW_MARGIN_MIN`), rescue thresholds (`RESCUE_POS_MIN=0.25`, `RESCUE_MARGIN_MIN=0.06`), model preset, batch sizes, and parquet-shard limit.
    - Runtime speed knobs — `ARMADA_DEVICE`, `ARMADA_EMB_BATCH_SIZE`, `ARMADA_SENT_BATCH_SIZE`, `ARMADA_PARQUET_BATCH_SIZE`, `ARMADA_MAX_FILES`, and optional `ARMADA_TORCH_THREADS`. These alter throughput and device placement, not thresholds.
    - `POS_QUERIES` / `NEG_QUERIES` — hardcoded retrieval prompts used to compute main-lane semantic similarity; control what the 2-lane gate accepts.
    - `RESCUE_POS_QUERIES` / `RESCUE_NEG_QUERIES` — second query set used only on `lexical_human_rescue()` candidates. POS = paraphrases of "person identified by nationality / ethnicity / race / skin color"; NEG = paraphrases for animal breeds, ethnic-named cultural products / holidays / sports competitions, public events, color/material descriptions, geographically-named tech brands, languages, and country-named institutions.
    - `GROUP_RE` — compiled regex from `TARGET_TOKENS ∪ CONTRAST_TOKENS − GATE_EXCLUDE_TOKENS`; document-level lexical gate.
    - `_HUMAN_HEADS` — built from `HUMAN_NOUNS` plus regular-plural heuristics, four irregular plurals (`men`, `women`, `children`, `people`), and a full set of personal pronouns (`he`, `she`, `they`, `him`, `her`, `them`, `we`, `us`, `i`, `you`, etc.). A pronoun in the ±4 window of a gate token is a strong surface signal that the token refers to a person in context. Backs `is_human_noun()`.
    - `_INANIMATE_HEADS` — built from `INANIMATE_NOUNS` plus regular-plural heuristics; used by `_inanimate_adjacent_only()` pre-filter.
    - `_COLOR_GATE_TOKENS` — color-tone gate tokens (`black`, `white`, `brown`, `yellow`, `dark`, `darker`, `colored`) whose plurals are excluded from the inherent-rescue path because they often refer to wine/sports teams/colors rather than people.
    - `_inanimate_adjacent_only(sentence)` — returns True if every gate token in the sentence is adjacent (±2 whitespace tokens) to an inanimate noun. Used as a pre-filter before MiniLM scoring to discard sentences like "German law", "black hole", "American government". Sentences are still written to `semantic_filter_lexical_all.txt` before this filter.
    - `lexical_human_rescue(sentence)` — returns `"inherent"` / `"candidate"` / `None`. Three surface rules run in order per gate-token position: **Rule 1a** (`inherent`) — token is a +s/+es plural of a non-color gate token; admit directly without MiniLM. **Rule 1c** (`candidate`) — token is a non-color gate token preceded by a determiner within ±2 tokens and not followed by a known non-human noun (e.g. "an American gained", "the German who…"); route to MiniLM rescue at `RESCUE_POS_MIN=0.25`. **Rule 1b** (`candidate`) — gate token has a human-head noun or personal pronoun within ±4 tokens; route to MiniLM rescue with context-aware threshold: `0.25` if the token is a non-color demonym or a color token directly preceding a human noun (+1/+2), else `0.32` for other color-token candidates. `None` means no rule fired.
    - `split_sentences()` — sentence splitter that protects common abbreviations, figure labels, initials, and acronyms before regex splitting; keeps sentences with `MIN_SENT_LEN=40` and `MAX_SENT_LEN=800`.
    - `iter_sentences()` — streams parquet batches; applies `GROUP_RE` at document level, then yields sentences.
    - `process_batch()` — per-batch pipeline: (1) lexical re-check per sentence; (2) inanimate-adjacency pre-filter (discards before MiniLM); (3) MiniLM main-lane scores — `STRICT` or `STRONG_MARGIN` admit directly; (4) for non-passing rows, `lexical_human_rescue()` is consulted: `inherent` admits as `LEXICAL_HUMAN_RESCUE` directly; `candidate` is rescue-scored against rescue queries with context-aware thresholds. Reference-noise rows are blocked from final output and routed to review. (5) Review rows undergo GTE ModernBERT fine screening; validated rows are rescued into the kept set, the rest remain in review with diagnostic `review_flags`.
    - `write_report()` — writes `dolma/semantic_filter_report.txt` with stats and example sentences.
    - `main()` — top-level orchestrator.
  - Inputs: `dolma/data/train-*.parquet` (up to `MAX_FILES`), `X/lexicons.py` imports.
  - Outputs: `dolma/semantic_filter_results.tsv`, `dolma/semantic_filter_review.tsv`, `dolma/semantic_filter_report.txt`, `dolma/semantic_filter_lexical_all.txt`.
  - Edit risk: **High** — thresholds, main-lane and rescue prompts, `lexical_human_rescue()` rules, and `INANIMATE_NOUNS` coverage all directly control what enters the analysis corpus.
  - Status: Active.
  - Notes: `X/` is added to `sys.path` at startup. `ARMADA_MAX_FILES` defaults to `1` for single-shard testing; increase for full-corpus run. On Apple Silicon, extraction auto-selects `mps` unless `ARMADA_DEVICE=cpu` is set. Rescue scoring reuses the per-sentence MiniLM embeddings already computed for the main lane (no additional encoder pass).


### Phase 2 — Analysis Pipeline

- `X/dim_sanity.py`
  - Role: Diagnostic tool for prototype checks and group labeling sanity.
  - Status: Active diagnostic tool.

- `X/run_pipeline.py`
  - Role: Main analysis orchestrator. Reads Phase 1 output, runs all analysis steps, writes all Phase 2 outputs.
  - Key sections:
    - `main()` — top-level; loads spaCy `en_core_web_lg`, GTE ModernBERT via `SentenceTransformer`, `SrlRoleLabeler`, `AttitudinalPrototypeMatcher`; runs preprocessing → feature extraction → discourse association → frame refresh → WEAT/CEAT/CEAT-full → EFI → regression → TSV outputs.
    - `_write_discourse_association()` — writes `X/association_discourse.tsv` (target/collocate pairs with LLR and LogDice).
    - `_find_candidates()` — identifies high-LLR collocates of target groups not yet in frame taxonomy; candidates are sorted by max LLR of either side without differential constraint; accepts `existing_frames` to exclude known frame terms.
    - `_load_seeds()` — reads `candidate_terms.json`; returns `(auto_neg, auto_pos)` word-level sets accumulated from prior auto-refresh runs (used for AttI binding and candidate exclusion).
    - `_load_seed_sentences()` — reads `seed_negative_terms`/`seed_positive_terms` from `candidate_terms.json`; returns lists of sentence-level prototypes.
    - `_load_anchors()` — reads `anchor_negative_terms`/`anchor_positive_terms` from `candidate_terms.json` (defaults `['bad','negative']` / `['good','positive']`); used only by `_refresh_frame_inventory` as the second admission tier; never enters centroid geometry.
    - `_encode_seed_centroids()` — encodes seed sentences with GTE ModernBERT and returns `(neg_centroid, pos_centroid)` mean vectors; centroids are the F-/F+ reference points for WEAT and CEAT.
    - `_refresh_frame_inventory()` — two-tier admission. Tier 1 (magnitude): `cos(candidate, sentence-level seed)` requiring `max(neg_sim, pos_sim) >= FRAME_SIM_FLOOR=0.55` and `|neg_sim - pos_sim| >= FRAME_SIM_MARGIN=0.06`. Tier 2 (direction-only): `cos(candidate, sentiment anchor word)` must merely **agree in sign** with the sentence direction (`anchor_diff * sentence_diff > 0`); no anchor margin floor, because after seed cleanups anchor cosine differences cluster below 0.05 and any absolute floor blocks everything. Writes `X/candidate_terms.json` with seed + auto + anchor fields preserved.
    - `_compute_weat()` — type-level WEAT: encodes group lemmas and F⁻/F⁺ frame terms with GTE ModernBERT; mean cosine similarity difference per group.
    - `_compute_ceat()` — CEAT-style contextual association: samples filtered group contexts, scores each context as `cos(context, F⁻ centroid) - cos(context, F⁺ centroid)`, and reports per-group mean/N/SE plus centroids for CEAT-full reuse.
    - `_compute_ceat_full()` — CEAT on sampled rows from `dolma/semantic_filter_lexical_all.txt`; reuses centroids from `_compute_ceat()` for comparability. Runtime mode is controlled by `ARMADA_CEAT_FULL_MODE`: `reported` (default; only groups reported in `group_stats.tsv`), `all` (all lexical-hit groups), or `skip` (development runs only). `ARMADA_SEAT_FULL_MODE` remains only as a backward-compatible alias.
    - `_compute_efi()` — PCA on group × `[AGI, PI, SI, net_atti, weat, ceat]` matrix, where `net_atti` is frame-derived. Returns PC1 + PC2 loadings, scores, and variance-explained per component. No sign flip is applied (PCA component signs are mathematically arbitrary; substantive interpretation comes from the loading pattern). **HIGH-RISK**.
    - `_run_regression()` — OLS: WEAT/CEAT ~ AgI + PI + SI + frame-netAttI.
    - Analysis thresholds: `ASSOCIATION_MIN_COUNT=5`, `ANALYSIS_MIN_GROUP_COUNT=50`, `REPORT_MIN_GROUP_COUNT=50`.
  - Inputs: `dolma/semantic_filter_results.tsv` (auto-discovered at 3 candidate paths), `dolma/semantic_filter_lexical_all.txt`.
  - Outputs: `X/output_results.tsv`, `X/output_review.tsv`, `X/group_stats.tsv`, `X/association_discourse.tsv`, `X/candidate_terms.json`.
  - Edit risk: **High** — all reported results flow through here.
  - Status: Active.
  - Notes: Accepts optional CLI path argument: `python run_pipeline.py [path/to/sentences.tsv]`.
  - Runtime knobs: `ARMADA_ANALYSIS_DEVICE` / `ARMADA_DEVICE`, `ARMADA_ANALYSIS_EMB_BATCH_SIZE`, `ARMADA_CEAT_FULL_MODE`, `ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP`, `ARMADA_CEAT_MIN_CONTEXTS_PER_GROUP`, `ARMADA_CEAT_MAX_FRAME_CONTEXTS`, and `ARMADA_CEAT_FULL_PROGRESS_EVERY`.

### Shared Embedding Configuration

- `X/embedding_config.py`
  - Role: Central embedding-model configuration for Phase 1 and Phase 2.
  - Key sections:
    - `EMBEDDING_MODEL_CATALOG` — named model presets available for manual selection; catalog entries are options, not evidence that every model has been downloaded.
    - `EXTRACTION_EMBEDDING_PRESET = os.environ.get("ARMADA_EXTRACTION_PRESET", "minilm")` — active Phase 1 extraction preset.
    - `ANALYSIS_EMBEDDING_PRESET = os.environ.get("ARMADA_ANALYSIS_PRESET", "gte_modernbert_base")` — active Phase 2 analysis preset.
    - `DEFAULT_EMBEDDING_PRESET = ANALYSIS_EMBEDDING_PRESET` — default for reported analysis jobs.
    - `EXTRACTION_EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — active Phase 1 extraction encoder.
    - `ANALYSIS_EMBEDDING_MODEL = "Alibaba-NLP/gte-modernbert-base"` — active Phase 2 analysis encoder.
    - `DEFAULT_EMBEDDING_BATCH_SIZE = 32` — conservative local batch size for 16GB Apple Silicon.
  - Callers:
    - `extract.py` — Phase 1 semantic retrieval embeddings.
    - `X/run_pipeline.py` — loads GTE ModernBERT directly via `SentenceTransformer(ANALYSIS_EMBEDDING_MODEL)` for frame-refresh, WEAT, CEAT, CEAT-full, and AttI prototypes.
    - `X/dim_sanity.py` — loads GTE ModernBERT directly for diagnostic prototype checks.
  - Edit risk: **High** — changing `ARMADA_EXTRACTION_PRESET` changes Phase 1 corpus composition; changing `ARMADA_ANALYSIS_PRESET` changes frame refresh, WEAT, CEAT, and Δ-CEAT geometry. Existing outputs are not comparable across encoder changes without rerunning the affected phase.
  - Status: Active.

### Shared Lexicons

- `X/lexicons.py`
  - Role: Central lexical resource module. Defines all group token sets, compound specs, inanimate/human head nouns, and the `resolve_group_token()` keyword-extraction function. Imported by `extract.py`, `run_pipeline.py`, all `step*.py`, and `step4_metrics.py`.
  - Key sections:
    - `TARGET_TOKENS` — minority/immigrant/ethnic group lemmas (immigration status, ethnonyms, broad minority framing). Civic tokens (`citizen`, `native`, `local`, `national`, `domestic`, `majority`, `native-born`, `nationalist`) were removed 2026-06-07; see `decisions.md`.
    - `CONTRAST_TOKENS` — dominant/majority group lemmas (European, Anglosphere, historical dominant). Same civic tokens removed.
    - `GATE_EXCLUDE_TOKENS` — tokens too polysemous to trigger the gate alone (`minority`, `minorities`, `mainstream`); still resolved when present in sentences entering via other tokens.
    - `INANIMATE_NOUNS` — expanded inanimate head-noun set for Phase 1 pre-filter and Phase 2 mention suppression. Covers nature/geography, food, objects/technology, economic/policy, geographic units, media, physical objects/body parts/color carriers, dog breeds, statistics terms, and academic/institutional nouns.
    - `HUMAN_NOUNS` — human-referent head nouns for Phase 1 rescue and Phase 2 disambiguation guards.
    - `INANIMATE_ENTITY_TYPES` — spaCy entity types treated as inanimate (`GPE`, `LOC`, `EVENT`, `PRODUCT`, etc.).
    - `GROUP_CONTEXT_WINDOW=4` / `SEMANTIC_CONTEXT_WINDOW=24` — local rule-based mention context and wider context window.
    - `COMPOUND_TARGET_HEADS` — dict keyed by head lemma describing spaced/prepositional compound specs (`seeker`/`seekers` → `asylum-seeker`; `people` + `of color` → `people-of-color`; `american`/`americans` + `native` → `native-american`). The head emits the canonical compound lemma; the modifier child is suppressed.
    - `resolve_group_token(token, doc)` — keyword-extraction function (rewritten 2026-06-07). Returns `(group_type, canonical_lemma)` or `None`. Logic: compound resolution → compound-child suppression → `non-` unconditional suppression → `anti-`/`pro-` modifier+head guard → X-speaking-Y suppression → minority-child dominant-head suppression → sibling-modifier suppression → inanimate-head suppression → emit if group-resolvable. Active-label check is exact-canonical (no part-splitting — `native-american` only passes if it is itself in `_ACTIVE_EXTRACTION_TOKENS`). No semantic resolver, no AMBIGUOUS/STRONG/DUAL branching. **Most-called function in the pipeline.**
    - `POLITICAL_GROUP_TOKENS` — `{soviet, ussr, communist, conservatist}`. Resolved/reported separately from demographic contrast.
    - `set_active_extraction_tokens(tokens)` — restricts `resolve_group_token` to the given canonical-lemma set (populated by `run_pipeline` from `active_labels.json`). `None` lifts the restriction.
  - Active-use notes:
    - `TARGET_TOKENS`, `CONTRAST_TOKENS`, and `GATE_EXCLUDE_TOKENS` are used by `extract.py` to construct the Phase 1 lexical gate.
    - `TARGET_TOKENS`, `CONTRAST_TOKENS`, `HUMAN_NOUNS`, `INANIMATE_NOUNS`, and `INANIMATE_ENTITY_TYPES` are used by `resolve_group_token()` for mention resolution and inanimate-head suppression.
    - Legacy static frame taxonomies, ambiguous-token helper sets, semantic-resolver integration, and evaluative verb lexicons have all been removed. Frame polarity comes from `candidate_terms.json`.
  - Inputs: (module-level constants; no runtime inputs).
  - Outputs: (imported constants used by all callers).
  - Edit risk: **High** — changes to any token set, compound spec, `INANIMATE_NOUNS`, or `resolve_group_token()` propagate through all downstream metrics.
  - Status: Active.
  - Notes: Any token in `TARGET_TOKENS ∪ CONTRAST_TOKENS − GATE_EXCLUDE_TOKENS` contributes to `GROUP_RE` in `extract.py`. Adding a token that is also in `GATE_EXCLUDE_TOKENS` will not trigger the lexical gate. `foreign` was removed from `TARGET_TOKENS` (2026-06-07) due to high adjective-modifier polysemy; `foreigner`/`foreigners` remain for unambiguous person-referent coverage.

### Feature Extraction Layer

- `X/step2_preprocessing.py`
  - Role: Loads sentences, strips noise, runs spaCy parse.
  - Key sections:
    - `load_sentences()` — dual-format: TSV (`sentence` column from extract.py output) or legacy pipe-delimited `.txt`. Falls back to `bucket` column for category.
    - `remove_noise()` — strips HTML tags and non-printable characters.
    - `preprocess()` — runs `en_core_web_lg` on each sentence; returns list of dicts with `doc`, `tokens` (text/lemma/pos/dep/head/i), `category`, `cleaned_text`.
  - Inputs: file path to sentences (TSV or .txt).
  - Outputs: list of preprocessed sentence dicts (in-memory).
  - Edit risk: Low.
  - Status: Active.
  - Notes: Callers: `run_pipeline.main()`.

- `X/step3_feature_extraction.py`
  - Role: Assigns Subjecthood, AgI, PI, SI and local diagnostic negAttI/posAttI per resolved group mention.
  - Key sections:
    - `set_srl_role_labeler()` / `set_attitude_matcher()` — module-level dependency injection; called at pipeline startup.
    - `_resolve_role()` — dependency-parse role resolution: handles modifiers, `pobj-of` promotion, conjunctions, AUX chain traversal; fallback when SRL is unavailable or insufficient.
    - `_collect_srl_roles()` — calls `SrlRoleLabeler.annotate()` with predicate hints; yields raw `ARG0` and patient-label matches as structural evidence.
    - `_group_span_indices()` — computes the set of token indices belonging to a compound group mention (e.g., "Korean people"); allows SRL labels to be inherited from the head noun.
    - `extract_roles()` — per-sentence orchestrator: resolves all group mentions, detects negation scope on governing predicates (blocks role assignment when predicate is negated), collects SRL structural labels, calls `AttitudinalPrototypeMatcher.match()` early. Each dimension is independent — a single mention can pass any combination of AGI / PI / SI via `_dim_passes(score, floor)` against per-dim absolute floors. PI fires on SRL `PATIENT_LABELS` / `dobj` / `nsubjpass` / `pcomp+auxpass` without prototype confirmation; PI prototype disagreement is recorded as a `*_proto_disagrees` review flag, not a veto. AGI on SRL ARG0 requires AGI prototype confirmation (`agi_passes`) to filter unaccusative subjects (`refugees suffered` lands below `AGI_FLOOR`); a sentiment-anchor veto was tried and removed (anchor margin tracks sentence polarity, not verb argument structure). SI requires a target-as-experiencer guard (`nsubj` / `nsubjpass` or SRL ARG0/ARG1). Assigns role confidence and review flags, handles MWE deduplication, runs `_resolve_anaphora()` and `_resolve_adverbial_passive()`.
    - `_resolve_anaphora()` — transfers pronoun roles (they/them/their/he/she/her) to nearest preceding target mention within a sentence by applying on-the-fly per-dim independent prototype scoring directly to the pronoun's context.
    - `_resolve_adverbial_passive()` — propagates PI from adverbial passive clauses to main-clause subject.
    - `extract_all()` — iterates preprocessed sentences; prints SRL progress every 100 sentences.
  - Inputs: preprocessed sentence docs; SRL labeler + attitude matcher (injected).
  - Outputs: per-sentence list of per-token role dicts including `subjecthood`, `agi`, `pi`, `si`, `role_confidence`, and `role_review_flags` (in-memory).
  - Edit risk: **High** — role assignment logic directly controls AgI/PI/SI for every group mention.
  - Status: Active. Current implementation partially realizes the target-aware semantic attribution plan through predicate cues and review flags.
  - Notes: Callers: `run_pipeline.main()`.

- `X/step3_semantic_roles.py`
  - Role: HuggingFace BERT-based SRL model wrapper. Produces per-token ARG labels for each predicate in a doc.
  - Key sections:
    - `SRL_MODEL_NAME = "dannashao/bert-base-uncased-finetuned-advanced-srl_arg"` — HuggingFace model identifier.
    - `PATIENT_LABELS = {"ARG1", "ARG1-DSP", "ARG2", "ARG3", "ARG4", "ARG5"}` — determines what counts as PI; changing this affects patienthood for all groups.
    - `SrlRoleLabeler.__init__()` — loads tokenizer and model; auto-detects CUDA / MPS / CPU.
    - `_predicate_candidates()` — selects VERB and participial ADJ tokens as predicate candidates; optionally filtered by `predicate_indices` hints.
    - `_predict_word_labels_batch()` — inserts `[V]` marker before each predicate; runs batch inference; averages subword-piece logits per original token.
    - `annotate()` — returns list of frame dicts `{predicate_i, predicate_text, predicate_lemma, labels}`.
  - Inputs: spaCy Doc, optional `predicate_indices` hint set (from `_collect_srl_roles()`).
  - Outputs: list of frame dicts (in-memory).
  - Edit risk: Medium — changing `PATIENT_LABELS` affects PI pipeline-wide.
  - Status: Active.
  - Notes: Loaded once via `set_srl_role_labeler(SrlRoleLabeler())` in `run_pipeline.main()`.

- `X/step3_attitudinal_prototypes.py`
  - Role: Scores local group-centered context window against dimensional (AGI/PI/SI) and attitudinal (pos/neg) prototype sentences via GTE ModernBERT cosine similarity.
  - Key sections:
    - `NEGATIVE_ATTITUDE_PROTOTYPES` / `POSITIVE_ATTITUDE_PROTOTYPES` — diagnostic AttI seeds, generated at module load by expanding `_NEG_ATTI_CANON` / `_POS_ATTI_CANON` (copular complements) across a tense × number grid.
    - `AGI_PROTOTYPES` / `PI_PROTOTYPES` / `SI_PROTOTYPES` — six paraphrase sentences per dimension. Index 0 is the ruling definition (AGI ↔ PI structurally parallel: "They are agents bringing about ..." / "They are patients being affected ..."). Indices 1–5 diversify tense, aspect, voice, gender, and number, and deliberately include both pleasant and unpleasant outcomes so dimensional similarity does not collapse into evaluative polarity.
    - `_expand_copular()` / `_be_form()` / `_COPULAR_SPECS` — copular surface controller for AttI seeds (tense × number).
    - `AttitudinalPrototypeMatcher.__init__()` — encodes the prototypes plus four sentiment anchors (`bad`/`negative` vs. `good`/`positive`); thresholds: `positive_floor=0.24` (AttI), per-dim independent floors `AGI_FLOOR=0.626`, `PI_FLOOR=0.637`, `SI_FLOOR=0.597`. `match()` returns cosines for all five sets including `anchor_neg_sim` / `anchor_pos_sim`. `DIM_ANCHOR_VETO_MARGIN` is defined but currently unused at the role-assignment layer (the anchor veto was removed after calibration showed anchor margin tracks sentence polarity, not verb argument structure).
    - `_build_focus_text()` — builds `[GROUP:token][PRED:head_verb]`-annotated context window (default 24 tokens either side); unchanged by the surface-variation controller.
    - `match()` — returns `{label, focus_text, neg_sim, pos_sim, agi_sim, pi_sim, si_sim, anchor_neg_sim, anchor_pos_sim}`.
  - Inputs: token + spaCy doc + head_verb + span_indices.
  - Outputs: diagnostic attitudinal label dict, plus dimensional similarity scores.
  - Edit risk: **High** — dimensional paraphrase sentences and per-dim floors (`AGI_FLOOR`, `PI_FLOOR`, `SI_FLOOR`) directly control the primary attribution of AgI, PI, and SI for every group mention. Floors are 70th-percentile empirical against the cached cosine distribution and need recalibration if prototype text changes. Editing AttI canonical complements (`_NEG_ATTI_CANON` / `_POS_ATTI_CANON`) or the `_COPULAR_SPECS` grid changes the AttI prototype matrix size.
  - Status: Active.
  - Notes: Receives GTE ModernBERT encoder directly (loaded in `run_pipeline.main()` as `SentenceTransformer`). Callers: `step3_feature_extraction.extract_roles()`. No external inflection dependency; surface variation for AttI is grid-based, and dimensional paraphrases are written by hand to cover tense/aspect/voice/gender/number directly.

- `X/semantic_group_resolver.py`
  - Status: **Removed** (2026-06-07). Source `.py` was deleted earlier; only `.pyc` cache remained. All disambiguation is now handled by `resolve_group_token()` keyword extraction + inanimate suppression in `lexicons.py`. GTE ModernBERT model is loaded directly in `run_pipeline.py` and `dim_sanity.py`. No code imports from this module.

- `X/group_mentions.py`
  - Role: Target-binding helper layer for primary group anchors, MWE metadata, scope flags, and F⁻/F⁺ frame binding.
  - Key sections:
    - `GroupMention` — dataclass with token/span positions, canonical lemma, group type, MWE type, and flags.
    - `iter_primary_group_mentions()` — yields non-duplicative group anchors; suppresses same-head MWE children for association/frame-AttI.
    - `sentence_scope_flags()` — flags negation, correction/denial, quotation, reported speech, contrast, and multi-group sentences.
    - `bind_frame_terms_to_mentions()` — binds frame terms to the nearest plausible group mention via dependency, shared predicate, or bounded proximity.
    - `bound_frame_summary()` — per-sentence summary used by `compute_frame_attitude_indices()` and `aggregate_sentence_metrics()`.
  - Inputs: spaCy Doc plus final F⁻/F⁺ frame sets.
  - Outputs: in-memory mention/frame-binding structures.
  - Edit risk: **High** — controls target-bound frame-AttI and review routing for complex sentences.
  - Status: Active.

- `X/step4_metrics.py`
  - Role: LLR/LogDice co-occurrence scoring, signed association, cosine similarity utility, per-sentence and per-group index aggregation.
  - Key sections:
    - `_compute_llr()` — G² log-likelihood ratio (2×2 contingency table). **HIGH-RISK**: changes affect collocate discovery scores pipeline-wide.
    - `_compute_logdice()` — LogDice formula: `14 + log2(2·pair / (target + collocate))`.
    - `build_sentence_associations()` — sentence-level co-occurrence; pair counts only when at least one token pair has distance ≥ 2 (`_NON_ADJACENT_MIN_DISTANCE=2`).
    - `compute_association_scores()` — produces `{(target, collocate): {pair_count, target_count, collocate_count, llr, logdice}}`.
    - `compute_signed_association()` — multiplies LLR by the final frame sign map for frame terms.
    - `cosine_similarity()` — NumPy dot-product cosine; used by `run_pipeline._compute_weat()`.
    - `aggregate_sentence_metrics()` — per-sentence row assembly; non-MWE filtering; bound frame labels; role/frame review flags; max-|value| signed association selection.
    - `compute_frame_attitude_indices()` — computes group-level `frame_negAttI`, `frame_posAttI`, `netAttI`, and `frameReview` from target-bound, non-blocked F⁻/F⁺ frame terms.
    - `compute_group_indices()` — proportionalizes Subjecthood/AgI/PI/SI and local diagnostic AttI per lemma and category; excludes MWE children from category counts.
  - Inputs: preprocessed docs (in-memory), association dicts.
  - Outputs: TSV rows and dicts (in-memory, passed to `run_pipeline`).
  - Edit risk: **High** — formula changes and aggregation logic directly affect all reported group indices and association scores.
  - Status: Active.

### Data and Reference Files

- `X/filter_training_samples.txt`
  - Role: Legacy training data for the removed MiniLM+PCA+LogReg classifier (deprecated 2026-05-28). The file is preserved as a research annotation artifact and may be reactivated if a future pipeline reintroduces a classifier.
  - Content Constraint: **All sentences truly derive from the Dolma corpus to guarantee ecological validity.**
  - Format: `LABEL | sentence` (one per line); labels are `RELEVANT` or `IRRELEVANT`; `#`-prefixed lines are comments.
  - Edit risk: Low (currently unread by the active pipeline).
  - Status: Legacy (no active reader after 2026-05-28).

- `dolma/data/train-*.parquet`
  - Role: Dolma v1.6 corpus shards (read-only corpus data).
  - Edit risk: n/a (do not modify).
  - Status: Active corpus data.

- `dolma/active_labels.json`
  - Role: Written by Phase 1 to restrict Phase 2 extraction/metrics to the top N target and contrast lemmas.
  - Status: Generated artifact.

- `X/ceat_full_cache.pkl`
  - Role: Disk cache for CEAT-full computations to speed up analysis reruns.
  - Status: Generated artifact.

- `X/pipeline_run.log`
  - Role: Runtime log for Phase 2 pipeline execution.
  - Status: Generated artifact.

- `X/srl_cache.pkl`
  - Role: Disk cache for expensive HuggingFace SRL model outputs.
  - Status: Generated artifact.

- `dolma/demographic_word_counts.tsv`
  - Role: Phase 1 word frequency pre-scan counts. Schema: `label | category | count`.
  - Status: Generated artifact.

- `REference/`
  - Contents: NRC Emotion Lexicon (zipped), NRC VAD Lexicon v2.1 (zipped), NRC Emotion Intensity Lexicon (zipped), MECORE database (zipped), VerbNet 3.1 (tarred), WordNet 2025 (gzipped), DE-BIAS vocabulary (.rdf, .ttl, .pdf).
  - Not imported by any project code file.
  - Edit risk: Low (not used at runtime).
  - Status: Supplementary reference only.

### Wiki and Tooling

- `quartz/content/index.md`
  - Role: Primary research wiki page. Documents current pipeline status, methodology, preliminary results, and links to other wiki pages.
  - Edit risk: Low (documentation).
  - Status: Active.
  - Notes: `sync.py` updates the `**Latest Update**` date in this file on each sync.

- `quartz/content/decisions.md` — Design rationales (encoder standardization, LLR vs PPMI, WEAT vs CEAT, target-conditioned semantic dimensions, frame-AttI, EFI architecture, complex-sentence policy, SI independence). Active.
- `quartz/content/tracker.md` — Current status and implemented changes log. Active.
- `quartz/content/log.md` — Daily activity log; entries drive `sync.py` commit messages. Active.
- `quartz/content/todo.md` — Action queue only; do not duplicate design rationale or implemented-change history here. Active.
- `quartz/content/samples.md` — Qualitative example sentences for pipeline inspection. Active.
- `quartz/content/reading.md` — Literature review. Active.
- `quartz/public/` — Built static site (generated by Quartz; do not edit directly). Generated.

- `sync.py`
  - Role: Git automation helper. Reads today's `log.md` entries, updates "Latest Update" in `index.md`, commits and pushes.
  - Edit risk: Low.
  - Status: Active.

### Unrelated / Legacy Files

- `rebuild_conversations.py` — Fixes conversation index for the "Antigravity" IDE (protobuf/SQLite). Completely unrelated to ARMADA. Status: Unused / unrelated.
- `read.cursorrules` — IDE note referencing shared skill directory. Not project code. Status: IDE tooling.
- `dolma/dolma.py` — Official HuggingFace `datasets` loader script from the Dolma repository. Not imported or called by any ARMADA code. Status: Unused in active pipeline.

- `extract_PCA+LogReg.py` — Legacy extraction script with removed PCA+LogReg classifier. Kept for debugging/reference. Status: Legacy.


---

## Main Workflows

- **Corpus ingestion / Phase 1 filtering**: `extract.py` → `dolma/semantic_filter_results.tsv`
  - Inputs: `dolma/data/train-*.parquet`, `X/lexicons.py`.
  - Outputs: `dolma/semantic_filter_results.tsv` (kept), `dolma/semantic_filter_review.tsv` (borderline), `dolma/semantic_filter_lexical_all.txt` (all lexical hits), `dolma/semantic_filter_report.txt` (stats).
  - Notes: Run from project root: `python extract.py`. Default extraction encoder is MiniLM for throughput. Use `ARMADA_EXTRACTION_PRESET=gte_modernbert_base python extract.py` only for intentional A/B calibration or replacement runs. `ARMADA_MAX_FILES=1` limits to one shard by default; increase for full corpus. Speed-only knobs: `ARMADA_DEVICE` (`mps`, `cuda`, `cpu`), `ARMADA_EMB_BATCH_SIZE`, `ARMADA_SENT_BATCH_SIZE`, and `ARMADA_PARQUET_BATCH_SIZE`. Do not run while `semantic_filter_results.tsv` is open in a spreadsheet (file lock).
  - Current local speed baseline: MiniLM MPS batch 64 is faster than 128/256 on this MacBook; CPU batch 32 is fastest for CPU fallback.

- **Full analysis / Phase 2**: `X/run_pipeline.py` → `X/group_stats.tsv`
  - Inputs: `dolma/semantic_filter_results.tsv`, `dolma/semantic_filter_lexical_all.txt`.
  - Outputs: `X/group_stats.tsv` (primary reported table), `X/output_results.tsv` (per-sentence), `X/output_review.tsv` (outlier or role/frame-review sentences), `X/association_discourse.tsv` (collocate pairs), `X/candidate_terms.json` (auto-refreshed frame candidates).
  - Notes: Run from `X/`: `python run_pipeline.py`. Or pass explicit input path: `python run_pipeline.py path/to/sentences.tsv`. Requires `en_core_web_lg` (`python -m spacy download en_core_web_lg`). SRL model downloads from HuggingFace on first run. CEAT-full is bounded by deterministic sampling; use `ARMADA_CEAT_FULL_MODE=skip` for quick development runs and `ARMADA_CEAT_FULL_MODE=all` only when full all-group Δ-CEAT diagnostics are required.

- **Lexical / collocate discovery** (within Phase 2): `step4_metrics.build_sentence_associations()` → `compute_association_scores()` → `_find_candidates()` → `_refresh_frame_inventory()` → `X/candidate_terms.json` + augmented F⁻/F⁺
  - Outputs: `X/association_discourse.tsv`, `X/candidate_terms.json`.
  - Notes: Candidate terms gate: `min_llr=3.0`, `top_n=60`. Two-tier auto-admission: sentence cosine (`FRAME_SIM_FLOOR=0.55`, `FRAME_SIM_MARGIN=0.06`) for magnitude, plus word-level sentiment anchor cosine for direction (anchor list defaults `['bad','negative']` vs `['good','positive']`; anchor must agree in sign with sentence-tier direction, no anchor margin floor). Over-admission of generic topic markers and metaphor under-detection are known limits; see tracker entry 30.

- **WEAT + CEAT** (within Phase 2): `_compute_weat()` + `_compute_ceat()` + `_compute_ceat_full()`
  - Inputs: preprocessed docs, F⁻/F⁺ sets (seed + auto-admitted), `dolma/semantic_filter_lexical_all.txt`.
  - Outputs: in-memory score dicts → `X/group_stats.tsv` columns.
  - Notes: WEAT and CEAT both use the same GTE ModernBERT encoder (`Alibaba-NLP/gte-modernbert-base`) for cross-corpus comparability. Δ-CEAT = CEAT-full − CEAT-filtered; requires `semantic_filter_lexical_all.txt` to exist. Default CEAT-full mode computes only reported groups because `group_stats.tsv` does not write unreported groups; use `ARMADA_CEAT_FULL_MODE=all` for exhaustive diagnostics. The implementation is CEAT-style mean/N/SE, not the full original random-effects meta-analysis.

- **EFI via PCA** (within Phase 2): `_compute_efi()` on groups with `N ≥ 50`
  - Inputs: group profiles with `[AGI, PI, SI, net_atti, weat, ceat]`.
  - Outputs: PC1 + PC2 loadings, per-group PC1 / PC2 scores, variance explained per component → `X/group_stats.tsv` `EFI_PC1` and `EFI_PC2` columns; console prints both loading patterns and per-group PC1 / PC2 scores.
  - Notes: No sign flip is applied. PCA component signs are mathematically arbitrary; the substantive reading of each axis comes from the loading pattern of the run. The earlier `orientation_anchor` logic that flipped PC1 to make "negative framing" positive presumed all six dims covary toward one negative-framing axis; the empirical correlation matrix on the live corpus showed AgI / PI / SI cluster opposite frame-netAttI / CEAT, so the flip silently encoded an empirically wrong assumption and was removed.

- **Wiki sync**: `sync.py`
  - Inputs: `quartz/content/log.md` (today's entries).
  - Outputs: git commit + push to origin main; updated `**Latest Update**` date in `quartz/content/index.md`.
  - Notes: Run from project root: `python sync.py`. If no log entries found for today, uses timestamped fallback commit message.

---

## Data/Output Conventions

- **Raw corpus**: `dolma/data/train-*.parquet` — parquet files with a `text` column (document text).
- **Phase 1 outputs**: `dolma/` directory.
  - `semantic_filter_results.tsv` — primary bridge to Phase 2. Schema: `sentence | semantic_pos | semantic_neg | semantic_margin | semantic_bucket`. All numeric columns prefixed with a space (`_excel_safe()`). `semantic_bucket` is `"STRICT"` or `"STRONG_MARGIN"`.
  - `semantic_filter_review.tsv` — borderline sentences. Schema: `bucket | review_flags | sentence | semantic_pos | semantic_neg | semantic_margin`. `bucket` is one of `"STRICT"`, `"STRONG_MARGIN"`, `"BORDERLINE"`, or `"REJECTED"`.
    - `review_flags` meanings: `strong_margin` = row entered the STRONG_MARGIN lane (high pos-vs-neg differential, low absolute pos); `low_semantic_margin` = uncertain semantic contrast; `high_semantic_low_margin` = retrieval pos is high but margin is below the STRICT floor; `reference_noise_like:*` = index, URL/markup, bibliographic/citation-like text; `semantic_borderline` = no other flag fired.
  - `semantic_filter_lexical_all.txt` — one sentence per line, no headers or scores. Required by CEAT-full.
  - `semantic_filter_report.txt` — plaintext stats report.
- **Phase 2 outputs**: `X/` directory.
  - `output_results.tsv` — per-sentence; rows that have a clear semantic role AND a non-null target. Schema: `sentence_id | category | text | targets | subjecthood | agi | pi | si | role_confidence_min | role_review_flags | local_neg_atti | local_pos_atti | frames | bound_frames | frame_binding_flags | association`. All numeric columns space-prefixed.
  - `output_review.tsv` — same schema as output_results.tsv; rows where `role_review_flags` contains `no_clear_semantic_role` OR `targets` is null. `output_results.tsv` and `output_review.tsv` are disjoint and together cover all per-sentence rows.
  - `group_stats.tsv` — primary reported table. Schema: `Lemma | Type | N | Subjecthood | AgI | PI | SI | local_negAttI | local_posAttI | frame_negAttI | frame_posAttI | netAttI | frameReview | WEAT | CEAT | CEAT_N | CEAT_SE | CEAT_full | CEAT_full_N | CEAT_full_SE | delta_CEAT | EFI_PC1 | EFI_PC2`. Filtered to `N ≥ 50`. All numeric columns space-prefixed.
  - `association_discourse.tsv` — Schema: `target | group_type | collocate | pair_sentence_count | target_sentence_count | collocate_sentence_count | llr | logdice`.
  - `candidate_terms.json` — Centroid-first architecture with two-tier admission. Schema: `{last_updated, note, seed_negative_terms, seed_positive_terms, auto_negative_terms, auto_positive_terms, anchor_negative_terms, anchor_positive_terms, candidates: [{term, minority_llr, minority_logdice, dominant_llr, dominant_logdice, differential, found_with, frame_neg_sim, frame_pos_sim, anchor_neg_sim, anchor_pos_sim, suggested_frame_sign, suggested_frame_bucket, used_in_frame_inventory}]}`. `seed_*_terms` are sentence-level Dolma-sourced prototypes encoded by GTE ModernBERT to produce neg/pos centroids used directly by WEAT and CEAT. `auto_*_terms` are single-word terms auto-admitted from candidate discovery; they accumulate across runs and are used for AttI syntactic frame binding. `anchor_*_terms` are four abstract sentiment words (default `['bad','negative']` / `['good','positive']`) used only as the second admission tier; they never enter centroid geometry. No `frame_*_terms` wordlist is maintained.
- **Numeric space-prefix convention**: All numeric values written to TSV are prefixed with a single space via `_excel_safe()` to prevent Excel from interpreting them as formulas. Do not strip this prefix in downstream readers unless using CSV parsers that handle it correctly.
- **Label conventions**: Group type is `"minority"` (from `TARGET_TOKENS`), `"dominant"` (from `CONTRAST_TOKENS`), or `"political"` (from `POLITICAL_GROUP_TOKENS`). Category in per-sentence output is `"FILTERED"` for Phase 1 kept sentences, or the original `bucket` value.
- **Lemma conventions**: Canonical lemmas come from `resolve_group_token()`. Compound mentions produce hyphenated lemmas (e.g., `african-american`, `native-born`, `asylum-seeker`, `people-of-color`).
- **Models/checkpoints**: No local checkpoints saved. MiniLM (`all-MiniLM-L6-v2`) is used for Phase 1 extraction; GTE ModernBERT (`Alibaba-NLP/gte-modernbert-base`) is used for Phase 2 analysis; SRL model (`dannashao/bert-base-uncased-finetuned-advanced-srl_arg`) is downloaded from HuggingFace Hub on first use. spaCy model: `en_core_web_lg`.
- **Virtual environment**: `X/venv/` (Python 3.10). Activate with `source X/venv/bin/activate` before running.
- **Reporting threshold**: `REPORT_MIN_GROUP_COUNT` and `ANALYSIS_MIN_GROUP_COUNT` in `run_pipeline.py` control which groups enter reported and analysis-level outputs.

---

## High-Risk Files

- `extract.py` — `Configuration constants`
  - Why: `MAX_SENT_LEN`, `SEMANTIC_MIN`, `SEMANTIC_MARGIN_MIN`, `SEMANTIC_STRONG_MARGIN`, `REVIEW_MARGIN_MIN`, and `BLOCK_REFERENCE_NOISE_KEEP` directly control corpus composition or review coverage. Changing them invalidates `semantic_filter_results.tsv` and all downstream results unless the file is regenerated.
  - Verify after editing: Re-run `extract.py`; check `semantic_filter_report.txt` for a plausible `final_rate`. Inspect kept sentences and rejects for qualitative validity.

- `extract.py` — runtime speed knobs
  - Why: `ARMADA_DEVICE`, `ARMADA_EMB_BATCH_SIZE`, `ARMADA_SENT_BATCH_SIZE`, `ARMADA_PARQUET_BATCH_SIZE`, `ARMADA_MAX_FILES`, and `ARMADA_TORCH_THREADS` affect runtime and memory pressure. They should not change methodology, but device-level floating point differences can alter borderline rows near thresholds.
  - Verify after editing: Run syntax checks and a small helper/import smoke test. For replacement extraction artifacts, record device and batch settings from the report/console.

- `X/run_pipeline.py` — analysis runtime knobs and CEAT-full mode
  - Why: `ARMADA_ANALYSIS_DEVICE`, `ARMADA_ANALYSIS_EMB_BATCH_SIZE`, `ARMADA_CEAT_FULL_MODE`, `ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP`, `ARMADA_CEAT_MAX_FRAME_CONTEXTS`, and `ARMADA_CEAT_FULL_PROGRESS_EVERY` affect runtime, CEAT precision, and which CEAT-full groups are computed. `reported` mode preserves the primary `group_stats.tsv` output surface; `all` is required only for exhaustive unreported-group diagnostics; `skip` omits `CEAT_full` / `delta_CEAT` and is not a reporting mode.
  - Verify after editing: Run syntax checks. For reported runs, record `ARMADA_CEAT_FULL_MODE`, CEAT sampling caps, and device/batch settings in notes or logs.

- `extract.py` — `POS_QUERIES` / `NEG_QUERIES`
  - Why: Control what the semantic gate accepts as demographically relevant. Changes alter the kept corpus.
  - Verify after editing: Re-run `extract.py`; compare `kept` and `strong_margin_kept` counts with prior run; inspect new kept/rejected examples in report.

- `extract.py` — `RESCUE_POS_QUERIES` / `RESCUE_NEG_QUERIES` / `RESCUE_POS_MIN` / `RESCUE_MARGIN_MIN` / `lexical_human_rescue()` / `_COLOR_GATE_TOKENS`
  - Why: Control which lexical hits are admitted via the rescue lane. Loosening rescue queries or thresholds reintroduces non-human demonym noise (`German Shepherd`, `white background`, `Chinese New Year`); tightening them drops genuine `demonym + human-head` admissions. `_COLOR_GATE_TOKENS` controls which plural demonym surfaces (`whites`, `blacks`) bypass MiniLM rescue scoring.
  - Verify after editing: Re-run `extract.py`; check `lexical-human` count in `semantic_filter_report.txt`; spot-check `dolma/semantic_filter_results.tsv` for revived false positives (search for `german shepherd`, `german tv`, `white background`, `chinese new year`, `european championship`) and for missing true positives among plural demonyms.

- `X/embedding_config.py` — active extraction/analysis encoder presets and batch size
  - Why: Changing `EXTRACTION_EMBEDDING_PRESET` changes semantic retrieval, classifier embeddings, and Phase 1 corpus composition. Changing `ANALYSIS_EMBEDDING_PRESET` changes semantic disambiguation, frame refresh, WEAT, CEAT, and Δ-CEAT. This invalidates direct comparison with prior output artifacts unless the relevant pipeline phase is regenerated and the encoder change is reported.
  - Verify after editing: Run syntax checks first; then calibrate model-specific cosine thresholds before re-running protected outputs.

- `X/lexicons.py` — `TARGET_TOKENS` / `CONTRAST_TOKENS` / `CLASSIFIED_FRAMES`
  - Why: Token set changes alter the lexical gate (`GROUP_RE` in `extract.py`), which requires re-running Phase 1. Frame taxonomy changes alter F⁻/F⁺ and therefore WEAT, CEAT, and signed association scores.
  - Verify after editing: If gate tokens changed, re-run `extract.py`. If frame taxonomy changed, re-run `run_pipeline.py` and compare `X/group_stats.tsv` WEAT/CEAT columns with previous output.

- `X/lexicons.py` — `resolve_group_token()`
  - Why: All AgI/PI/SI/local AttI diagnostics, frame AttI, CEAT, and group-level aggregation depend on correct group token resolution. Edge cases in disambiguation logic affect all downstream metrics.
  - Verify after editing: Run `run_pipeline.py`; check `X/output_review.tsv` for new outlier patterns; manually inspect 10 random rows from `X/output_results.tsv` to confirm group assignment is correct.

- `X/run_pipeline.py` — `_compute_efi()` and 2D EFI reporting
  - Why: Loadings of PC1 and PC2 determine substantive interpretation of bias structure. Earlier sign-flip logic was empirically wrong and was removed; if reintroduced, it must be justified by a corpus-level finding rather than imposed as an a priori orientation.
  - Verify after editing: Confirm `EFI_PC1` and `EFI_PC2` columns in `group_stats.tsv` exist; inspect printed loading patterns to ensure they are interpretable.

- `X/run_pipeline.py` — `ANALYSIS_MIN_GROUP_COUNT` / `REPORT_MIN_GROUP_COUNT`
  - Why: Controls which groups appear in the reported output. Lowering produces less statistically stable results; raising may suppress legitimate groups.
  - Verify after editing: Count rows in `X/group_stats.tsv`; confirm expected groups are present.

- `X/step3_feature_extraction.py` — `extract_roles()` and role assignment logic
  - Why: Subjecthood/AgI/PI/SI and local AttI diagnostics for every group mention originate here. Bugs in predicate cues, dependency fallback, or SRL mapping silently corrupt the feature matrix.
  - Verify after editing: Run `run_pipeline.py`; check `X/output_review.tsv` (outlier filter); manually review 5–10 rows from `X/output_results.tsv` for expected role labels on simple sentences.

- `X/step3_attitudinal_prototypes.py` — `NEGATIVE_ATTITUDE_PROTOTYPES` / `POSITIVE_ATTITUDE_PROTOTYPES` and thresholds
  - Why: These directly determine local diagnostic AttI columns. They no longer determine reported `netAttI`. AttI lists are generated by expanding `_NEG_ATTI_CANON` / `_POS_ATTI_CANON` across the `_COPULAR_SPECS` grid; AGI / PI / SI are six hand-written paraphrase sentences per dimension. Edit canonical complements or paraphrase sentences directly; the surface grid only affects AttI matrix size.
  - Verify after editing: Run `run_pipeline.py`; compare `local_negAttI` and `local_posAttI` columns in `group_stats.tsv` with prior values; inspect new local attitudinal matches in `output_results.tsv`.

- `X/step4_metrics.py` — `_compute_llr()` / `_compute_logdice()`
  - Why: Collocate association scores drive candidate discovery and frame inventory refresh; errors propagate to WEAT/CEAT indirectly.
  - Verify after editing: Run `run_pipeline.py`; check `X/association_discourse.tsv` top LLR pairs are semantically plausible.

- `X/group_mentions.py` — target-bound mention and frame-binding logic
  - Why: Controls whether F⁻/F⁺ frame terms are counted for a group or routed to review. Errors here directly affect `frame_negAttI`, `frame_posAttI`, `netAttI`, `frameReview`, `bound_frames`, and `frame_binding_flags`.
  - Verify after editing: Run small sentence-level checks for negation/correction and multi-group contrast before any full pipeline rerun.

- `dolma/semantic_filter_results.tsv`, `dolma/semantic_filter_lexical_all.txt`
  - Why: These are irreplaceable unless `extract.py` is re-run (which requires the parquet corpus and significant compute time). They are the research artifact bridging Phase 1 and Phase 2.
  - Do not overwrite or delete without intent to regenerate.

---

## Unused / Redundant / Legacy Inventory

- **Unused:**
  - `dolma/dolma.py` — HuggingFace dataset loader script; not imported or called by any ARMADA code. Evidence: no `import dolma` or `from dolma` found in project Python files.
  - `REference/` — external lexicon files (NRC, VerbNet, WordNet, DE-BIAS); not imported by any code. Evidence: no file paths to `REference/` found in any `.py` file.
  - `rebuild_conversations.py` — Antigravity IDE utility; completely unrelated to ARMADA. Evidence: references `antigravity`, `BRAIN_DIR`, `CONVERSATIONS_DIR` with no connection to any ARMADA module.
  - `read.cursorrules` — IDE skill-directory reference; not project code.

  
- **Legacy:**  
  - `report_records.zip` stores past extraction records.
  - `2rp.md` — original research proposal containing the early PPMI-based EFI formula (`EFI = α·PPMI + (1-α)·WEAT/SEAT`), which was superseded by the LLR/LogDice + PCA approach. No code references it.
  - `X/filter_training_samples.txt` — labelled training data for the removed Phase 1 PCA+LogReg classifier (deprecated 2026-05-28). Preserved for possible future reactivation; no active reader.

- **Generated:**
  - `dolma/semantic_filter_results.tsv`, `dolma/semantic_filter_review.tsv`, `dolma/semantic_filter_report.txt`, `dolma/semantic_filter_lexical_all.txt` — output of `extract.py`.
  - `X/output_results.tsv`, `X/output_review.tsv`, `X/group_stats.tsv`, `X/association_discourse.tsv`, `X/candidate_terms.json` — output of `run_pipeline.py`.
  - `quartz/public/` — built static site (Quartz).
  - `X/__pycache__/` — Python bytecode cache.

- **Working notes:**
  - `build.md` — conceptual notes
  - `limitations.md` —limitation notes
  - `.gemini/` — directory present in repo root; contents unknown from inspection.

---

## Testing / Validation

There are no formal unit tests or test runner configuration in the repository. Validation is manual and pipeline-level.

**Syntax check (Phase 1):**
```bash
cd /Users/l/projects
source X/venv/bin/activate
python -m py_compile extract.py && echo "OK"
python -m py_compile X/embedding_config.py X/lexicons.py X/step2_preprocessing.py \
  X/step3_feature_extraction.py X/step3_attitudinal_prototypes.py \
  X/step3_semantic_roles.py X/step4_metrics.py X/group_mentions.py \
  X/dim_sanity.py X/run_pipeline.py && echo "OK"
```

**Phase 1 smoke test (extract.py):**
```bash
cd /Users/l/projects
source X/venv/bin/activate
python extract.py
# Expect: dolma/semantic_filter_report.txt updated; kept > 0
# Check: head -5 dolma/semantic_filter_results.tsv
# Check: cat dolma/semantic_filter_report.txt | grep -E "^kept|semantic_pass|strong_margin_kept|lexical_hits"
```

**Phase 2 smoke test (run_pipeline.py):**
```bash
cd /Users/l/projects/X
source venv/bin/activate
python run_pipeline.py
# Expect: group_stats.tsv written; console shows WEAT/CEAT/EFI table
# Check: wc -l group_stats.tsv
# Check: head -3 group_stats.tsv
```

**Schema checks after Phase 1:**
```bash
# Verify TSV column headers
head -1 dolma/semantic_filter_results.tsv
# Expected: sentence\tsemantic_pos\tsemantic_neg\tsemantic_margin\tsemantic_bucket
head -1 dolma/semantic_filter_review.tsv
# Expected current schema: bucket\treview_flags\tsentence\tsemantic_pos\tsemantic_neg\tsemantic_margin
wc -l dolma/semantic_filter_lexical_all.txt
# Must be > 0 for CEAT-full to compute
```

**Schema checks after Phase 2:**
```bash
head -1 X/group_stats.tsv
# Expected: Lemma\tType\tN\tSubjecthood\tAgI\tPI\tSI\tlocal_negAttI\tlocal_posAttI\tframe_negAttI\tframe_posAttI\tnetAttI\tframeReview\tWEAT\tCEAT\tCEAT_N\tCEAT_SE\tCEAT_full\tCEAT_full_N\tCEAT_full_SE\tdelta_CEAT\tEFI_PC1\tEFI_PC2
head -1 X/association_discourse.tsv
# Expected: target\tgroup_type\tcollocate\tpair_sentence_count\ttarget_sentence_count\tcollocate_sentence_count\tllr\tlogdice
```

**Manual inspection steps:**
- After any lexicon change: open `X/output_results.tsv` and verify 10 random rows — check that `targets` column contains expected lemmas and `subjecthood/agi/pi/si` values are 0 or 1 (not impossible values).
- After any threshold change in `extract.py`: check `dolma/semantic_filter_report.txt` for a plausible `final_rate`; read kept examples and rejects to confirm the gate is calibrated.
- For Phase 1 extraction calibration, use `dolma/semantic_filter_review.tsv`; `X/output_review.tsv` is Phase 2 role/frame review. If `semantic_filter_report.txt` is older than `semantic_filter_review.tsv` / `semantic_filter_results.tsv`, treat the report as stale until Phase 1 is rerun.
- After any change to `_compute_efi()`: inspect printed PC1 / PC2 loading patterns to ensure they remain interpretable; report changes in loadings as a methodology change rather than treating them as automatic.
- After any attitudinal prototype or threshold change: compare `local_negAttI` / `local_posAttI` columns in `group_stats.tsv` with prior values; inspect `local_neg_atti` / `local_pos_atti` in `output_results.tsv` for qualitative plausibility.

---

## External Context

Deeper methodological background, design rationale, literature review, and research notes live in the Quartz wiki:

**https://c1araliang.github.io/armada/**

Key pages:
- [Index / Overview](https://c1araliang.github.io/armada/) — pipeline diagram, current status, methodology summary, preliminary results.
- [Design Decisions](https://c1araliang.github.io/armada/decisions) — rationale for encoder choice, target-conditioned semantic dimensions, frame-AttI, WEAT/CEAT, EFI architecture, and SI independence from AgI.
- [Changes & Status](https://c1araliang.github.io/armada/tracker) — current open status and implemented changes.
- [To-Do](https://c1araliang.github.io/armada/todo) — unresolved action queue only.
