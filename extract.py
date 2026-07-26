"""
Streaming semantic relevance filter for ARMADA.

Two-lane gate (no classifier):
1. Lexical mention gate using the demographic lexicon (GROUP_RE).
2. Semantic retrieval with POS/NEG query sets (MiniLM).
   - STRICT lane:        pos >= SEMANTIC_MIN AND margin >= SEMANTIC_MARGIN_MIN
   - STRONG_MARGIN lane: margin >= SEMANTIC_STRONG_MARGIN

The PCA+LogReg classifier was removed 2026-05-28 after a controlled ablation
showed it dropped 37% of true RELEVANT sentences due to training-set imbalance
toward refugee/immigration framing. See decisions.md for the full rationale.
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(line_buffering=True)
# Auto-detect project root and add to path
PROJECT_ROOT = Path(__file__).parent
X_DIR = PROJECT_ROOT / "X"
if X_DIR.exists():
    sys.path.insert(0, str(X_DIR))

from embedding_config import (  # type: ignore
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EXTRACTION_EMBEDDING_MODEL,
    EXTRACTION_EMBEDDING_PRESET,
    EMBEDDING_MODEL_CATALOG,
)
from lexicons import TARGET_TOKENS, CONTRAST_TOKENS, GATE_EXCLUDE_TOKENS, HUMAN_NOUNS, INANIMATE_NOUNS  # type: ignore


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw_value!r}")
    return value


def _select_device() -> str:
    override = os.environ.get("ARMADA_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "dolma" / "data"
OUTPUT_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_results.tsv"
REVIEW_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_review.tsv"
REPORT_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_report.txt"
LEXICAL_ALL_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_lexical_all.txt"

# Encoder preset for Phase 1 extraction. This can be lighter than the analysis
# encoder because extraction is a recall-oriented corpus filter, not a reported
# embedding-association metric.
MODEL_PRESET = EXTRACTION_EMBEDDING_PRESET
MODEL_CATALOG = EMBEDDING_MODEL_CATALOG
MODEL_NAME = EXTRACTION_EMBEDDING_MODEL
MODEL_DEVICE = _select_device()
MAX_FILES = _env_int("ARMADA_MAX_FILES", 100)
PARQUET_BATCH_SIZE = _env_int("ARMADA_PARQUET_BATCH_SIZE", 10_000)
SENT_BATCH_SIZE = _env_int("ARMADA_SENT_BATCH_SIZE", 4_096)


def _default_embedding_batch_size() -> int:
    if MODEL_PRESET != "minilm":
        return DEFAULT_EMBEDDING_BATCH_SIZE
    if MODEL_DEVICE == "mps":
        return 64
    if MODEL_DEVICE == "cuda":
        return 256
    return DEFAULT_EMBEDDING_BATCH_SIZE


EMB_BATCH_SIZE = _env_int(
    "ARMADA_EMB_BATCH_SIZE",
    _default_embedding_batch_size(),
)
TORCH_THREADS = os.environ.get("ARMADA_TORCH_THREADS")
if TORCH_THREADS:
    torch.set_num_threads(_env_int("ARMADA_TORCH_THREADS", torch.get_num_threads()))

MIN_SENT_LEN = 40
MAX_SENT_LEN = 800

# Two-lane semantic gate (no classifier). The PCA+LogReg classifier was removed
# after a controlled ablation showed it dropped 37% of true RELEVANT sentences
# in the margin >= 0.10 candidate set due to training-set imbalance toward
# refugee/immigration framing, while reducing false positives only marginally
# (FP rate ~14% without classifier vs ~7-8% with classifier, but recall loss
# of 37% erased the precision gain). See decisions.md.
#
# Lane 1 — STRICT: high absolute semantic confidence on POS queries.
#   pos >= SEMANTIC_MIN AND margin >= SEMANTIC_MARGIN_MIN
# Lane 2 — STRONG_MARGIN: pos/neg margin alone is high enough.
#   margin >= SEMANTIC_STRONG_MARGIN
# Reference-noise patterns still block both lanes.
# Phase 2 resolve_group_token() and INANIMATE_NOUNS provide a second filter at
# mention-resolution time.
SEMANTIC_MIN = 0.34
SEMANTIC_MARGIN_MIN = 0.03
SEMANTIC_STRONG_MARGIN = 0.10

BLOCK_REFERENCE_NOISE_KEEP = True

# Review threshold: non-passing sentences with main-lane semantic_margin >= 0.05
# (or containing reference-noise flags) are routed to semantic_filter_review.tsv
# for ModernBERT fine screening (step 5) or human inspection, rather than being discarded.
REVIEW_MARGIN_MIN = 0.05

# ── Lexical-human rescue (MiniLM) ────────────────────────────────────────────
# Surface-form rescue admits sentences whose lexical hit is an unambiguous
# plural non-color demonym ("Germans", "Americans") without a second semantic
# check. Sentences with a "demonym + human-head" adjacency (the candidate
# path) get a second MiniLM pass against rescue queries that explicitly
# contrast human reference with animal/product/event/institution senses.
# The PCA+LogReg classifier is gone; this is the only disambiguation between
# "German workers" and "German Shepherd / German TV".
RESCUE_POS_MIN = 0.25
RESCUE_MARGIN_MIN = 0.06

RESCUE_POS_QUERIES = [
    "a person described by their nationality, ethnicity, race, or skin color",
    "people from a specific country or ethnic background",
    "an individual or group identified as belonging to a demographic",
    "someone treated, judged, or described based on where they are from or what they look like",
]

RESCUE_NEG_QUERIES = [
    "A tract of land, a place to live in, home",
    "a dog breed or animal species named after a place or people",
    "a cultural product, holiday, or tradition named after an ethnic group",
    "a sports competition, championship, tournament, or award named after a region",
    "a public event, ceremony, festival, or celebration",
    "a color, material, or physical object described by an adjective",
    "a technology product, brand, company, or model named after a place",
    "a language, script, or linguistic term",
    "an institution, organization, agency, or government body named after a country",
]

# Review diagnostics only. These do not decide whether a sentence is kept; they
# make future calibration passes distinguish semantic uncertainty and
# reference/index-like corpus noise.
REVIEW_LOW_MARGIN = 0.06
REVIEW_HIGH_SEMANTIC_POS = 0.65

POS_QUERIES = [
    "immigrant or refugee, or displaced people, as individual or groups",
    "ethnic majority or racial minority as individual or communities",
    "physical, bodily or mental state of people identified by nationality, ancestry, or ethnonym",
    "local or foreign workers, local or foreign residents",
    "people or individuals, with any skin color or racial identity, doing things",
    "how a demographic group is treated, affectd, or portrayed",
    "everyday life, family, or social condition of a person identified by their nationality, or ethnic, racial, or cultural background",
]

NEG_QUERIES = [
    "a subset of colors",
    "visa application, visa type", "a company's doing, a business, work conditions, wage, salary, employment",
    "culture tradition, festivals, celebrations"
    "folk religion practice, Christianity, Buddhism, Taoism, Hinduism, Islam",
    "cultural-specific medical pratice or traditional medicine",
    "a list of countries and regions", "countries like US or European as entity or entities",
    "a list of people's names",
    "a list of languages",
    "name of academic subject, studies, conference, association, society like like Psychological Society, Psychiatric Association, Sleep Foundation"
    "geopolitical organization, entiy, or international body",
    "ethnonyms such as black or white or brown used to describe colored objects or body parts or products, or as people's name",
    "weather, geopolitical or geographical locations, like black sea, italian coast, indian subcontinent",
    "language teaching or language study", "fluent in or native speaker of certain languages",
    "translation, definitions, dictionary entries", "bibliography, book title with author, page number or year or publisher",
    "movement, revolt, wars, or historical events, like French Revolution or Ukraine War",
    "software, industry, technology",
    "names with biographical labels or institutional relations or job titles",
    "policy, regulation, law, or tariffs",
    "regional or ethnic group's food, such as american cheese or black tea or japanese chopsticks or italian pizza",
    "natural phenomena such as a black hole, a white cloud, yellow moon",
    "pets, animals, plants, or wildlife such as black rhino, african elephant, or canadian maple, german shepherd",
    "brand, music, performance, films, cuisine, anime",
    "ecominics, finance or foreign exchange markets or labour market",
    "advertisement, techical instructions, or those for commercial use",
    "footwear, clothing, fabrics, cosmetics, phones, cars, products",
    "generic statistics, numbers, or data tables",
    "technical jargon or domain-specific terminology that uses geographic or ethnic words, such as indian ink, native resolution, western blot, black box, polish notation",
    "proper nouns, name of an organization or union, award, journal, or institution, act, law, like Workers Union, Pest Management Association, Rifle Association",
    "detailed regulation, rule, law, proposal",
    "government building, center, or political institute, like the white house, congress, embassy, consulate",
]

REFERENCE_NOISE_PATTERNS = [
    ("index_page_ref", re.compile(r"(?<!\d),\s*\b\d{1,3}\b\.$")),
    ("url_or_markup", re.compile(r"https?://|www\.|<[^>]+>")),
    (
        "bibliographic",
        re.compile(
            # Tightened to avoid false positives on natural language uses of
            # "Journal", "Proceedings", "chapter":
            #   - "Journal of X" (citation form), but not "her journal entry"
            #   - "Proceedings of the X" (conference), not "legal proceedings"
            #   - "chapter N" or "Chapter N" (book section), not "chapter on X"
            r"\b(ISBN|DOI|Journal\s+of\s+|Proceedings\s+of\s+the\s+|"
            r"University\s+Press|Cambridge\s+University\s+Press|"
            r"Oxford\s+University\s+Press|Princeton\s+University\s+Press|"
            r"Vol\.|No\.|Nr\.|"
            r"chapter\s+\d+|edited\s+by|published\s+by)\b",
            re.I,
        ),
    ),
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
# ── Lexical-human rescue helpers ──────────────────────────────────────────────
# Used by lexical_human_rescue() to decide whether a gate-token occurrence is
# anchored to a human referent. Inanimate-noun guessing was removed with the
# rule-based path #2/#3/#4: regex cannot reliably tell a noun "work / show /
# play / bear" apart from a verb, and adding more nouns to the inanimate set
# only created new false negatives. Disambiguation now happens semantically
# in the rescue MiniLM pass.

# Color/skin-tone gate tokens: their plurals ("blacks", "whites", "browns",
# "yellows") are NOT unambiguously human in English ("the whites" can mean
# wine; "the browns" can mean a sports team). These need MiniLM rescue
# scoring even when they appear as plurals; only the candidate path admits
# them.
_COLOR_GATE_TOKENS = {"black", "white", "brown", "yellow", "dark", "darker", "colored"}

# Human head nouns — auto-generated from HUMAN_NOUNS (lexicons.py) with common
# English plural heuristics for surface-text matching (no spaCy at this stage).
_HUMAN_HEADS: set[str] = set(HUMAN_NOUNS)
for _noun in list(HUMAN_NOUNS):
    if _noun.endswith("y") and len(_noun) > 2 and _noun[-2] not in "aeiou":
        _HUMAN_HEADS.add(_noun[:-1] + "ies")      # family → families
    elif _noun.endswith(("ss", "x", "sh", "ch")):
        _HUMAN_HEADS.add(_noun + "es")             # boss → bosses
    elif _noun.endswith("s"):
        pass                                        # already plural (parents, seekers)
    else:
        _HUMAN_HEADS.add(_noun + "s")              # worker → workers
# Irregular plurals that the heuristic cannot generate.
_HUMAN_HEADS.update({"men", "women", "children", "people"})
# Personal pronouns: a pronoun in the ±4 window of a demonym is a strong
# surface signal that the demonym refers to a person in context
# ("An American, he was proud", "she is Chinese"). MiniLM rescue scoring
# provides the second filter so pronoun-adjacent policy/institution sentences
# ("he said that the German policy") are still blocked by low rpos.
_HUMAN_HEADS.update({
    "he", "she", "they", "him", "her", "them",
    "his", "hers", "their", "theirs",
    "himself", "herself", "themselves",
    "we", "us", "our", "ours",
    "i", "me", "my", "mine",
    "you", "your", "yours", "yourself",
})

# Inanimate head nouns — auto-generated from INANIMATE_NOUNS (lexicons.py) with
# the same plural heuristics as _HUMAN_HEADS. Used by the inanimate-adjacency
# pre-filter to discard sentences where every gate token is next to an inanimate
# noun (e.g. "German law", "black hole", "American government").
_INANIMATE_HEADS: set[str] = set(INANIMATE_NOUNS)
for _noun in list(INANIMATE_NOUNS):
    if _noun.endswith("y") and len(_noun) > 2 and _noun[-2] not in "aeiou":
        _INANIMATE_HEADS.add(_noun[:-1] + "ies")    # policy → policies
    elif _noun.endswith(("ss", "x", "sh", "ch")):
        _INANIMATE_HEADS.add(_noun + "es")           # box → boxes
    elif _noun.endswith("s"):
        pass                                          # already plural-looking
    else:
        _INANIMATE_HEADS.add(_noun + "s")            # car → cars


def _is_inanimate_word(word: str) -> bool:
    word = word.lower()
    return word in _INANIMATE_HEADS


def _inanimate_adjacent_only(sentence: str) -> bool:
    """True if EVERY gate token in the sentence is adjacent (±2) to an inanimate noun.

    Used as a lightweight Phase 1 pre-filter before MiniLM scoring: sentences
    where the only demographic signals sit next to inanimate nouns are almost
    certainly non-demographic ("German law", "black hole", "American government").
    If at least one gate token has NO inanimate adjacency, the sentence proceeds
    to MiniLM scoring.
    """
    tokens = sentence.split()
    gate_positions = [
        i for i, t in enumerate(tokens)
        if re.search(r"(?<!\w)(?:" + GROUP_TOKEN_PATTERN + r")(?!\w)", t, re.I)
    ]
    if not gate_positions:
        return False  # no gate tokens → not applicable

    for pos in gate_positions:
        has_inanimate = False
        for offset in (-2, -1, 1, 2):
            idx = pos + offset
            if 0 <= idx < len(tokens):
                w_clean = re.sub(r"\W+", "", tokens[idx])
                if w_clean and _is_inanimate_word(w_clean):
                    has_inanimate = True
                    break
        if not has_inanimate:
            return False  # at least one gate token has no inanimate neighbor
    return True  # every gate token is inanimate-adjacent



def compile_gate_regexes(target_set: set[str], contrast_set: set[str]):
    global ALL_GROUP_TOKENS, GATE_TOKENS, GROUP_TOKEN_PATTERN, GROUP_PERSON_SUFFIX_PATTERN, GROUP_RE
    ALL_GROUP_TOKENS = target_set | contrast_set
    GATE_TOKENS = ALL_GROUP_TOKENS - GATE_EXCLUDE_TOKENS
    GROUP_TOKEN_PATTERN = (
        r"(?<!\w)(?:"
        + "|".join(sorted(map(re.escape, GATE_TOKENS), key=len, reverse=True))
        + r")s?(?!\w)"
    )
    # GROUP_PERSON_SUFFIX_PATTERN is kept in GROUP_RE so that compound forms
    # like "Frenchman" / "Englishwoman" still pass the document-level gate.
    # The separate compiled object (_GROUP_PERSON_SUFFIX_RE) was removed
    # 2026-05-30 when the compound inherent rescue path was dropped; these
    # forms now go through the main 2-lane MiniLM gate instead.
    GROUP_PERSON_SUFFIX_PATTERN = (
        r"(?<!\w)(?:"
        + "|".join(sorted(map(re.escape, GATE_TOKENS), key=len, reverse=True))
        + r")(?:man|men|woman|women|boy|boys|girl|girls|people)(?!\w)"
    )
    GROUP_RE = re.compile(
        GROUP_TOKEN_PATTERN + "|" + GROUP_PERSON_SUFFIX_PATTERN,
        re.I,
    )

# Initial compilation using the default demographic sets
compile_gate_regexes(TARGET_TOKENS, CONTRAST_TOKENS)


def is_human_noun(word: str) -> bool:
    word = word.lower()
    if word in _HUMAN_HEADS:
        return True
    # Honorific / role titles that are unambiguously human in surface form.
    # Kept as a short hardcoded set rather than baked into HUMAN_NOUNS to keep
    # the lexicon focused on canonical role nouns.
    if word in ("mr", "mrs", "ms", "dr", "prof", "sir", "lord", "lady",
                "president", "minister", "governor", "general", "captain",
                "chief", "officer"):
        return True
    return False


def lexical_human_rescue(sentence: str, return_threshold: bool = False) -> str | None | tuple[str | None, float]:
    """Classify a lexical hit for the rescue lane.

    Returns:
        If return_threshold is False:
            "inherent"  — gate token is a non-color plural demonym ("Germans",
                          "Americans", "refugees"). Admit without further
                          semantic check. Compound forms like "Frenchman" /
                          "Englishwoman" still enter via GROUP_RE but go through
                          the main 2-lane MiniLM gate instead of this path.
            "candidate" — gate token is within ±4 tokens of a human head noun
                          or personal pronoun ("white children", "Chinese
                          doctor", "she is black"). Needs a second MiniLM pass
                          against RESCUE queries because this surface pattern
                          also matches "white background", "German Shepherd",
                          "Chinese New Year", etc.
            None        — gate token has no surface-level human anchor; not a
                          rescue case.
        If return_threshold is True:
            A tuple of (tag, threshold).

    Per-token aggregation: any token in the sentence reaching "inherent"
    short-circuits to "inherent"; otherwise any token reaching "candidate"
    returns "candidate".
    """
    if not GROUP_RE.search(sentence):
        return (None, RESCUE_POS_MIN) if return_threshold else None

    tokens = sentence.split()

    gate_positions = [
        i for i, t in enumerate(tokens)
        if re.search(r"(?<!\w)(?:" + GROUP_TOKEN_PATTERN + r")(?!\w)", t, re.I)
    ]
    if not gate_positions:
        return (None, RESCUE_POS_MIN) if return_threshold else None

    saw_candidate = False
    has_non_color_demonym = False
    has_color_with_close_head = False

    for pos in gate_positions:
        matched_word = tokens[pos]
        clean_matched = re.sub(r"\W+", "", matched_word).lower()
        if not clean_matched:
            continue

        # Rule 1a — inherent human form: explicit +s/+es plural of a
        # non-color gate token ("Germans", "Americans", "refugees").
        # Color-tone plurals ("whites", "blacks") are excluded because
        # they often refer to wine, sports teams, or color categories;
        # those go through the candidate path so MiniLM can decide.
        # Suffix compound forms were removed from inherent rescue 2026-05-30:
        # GROUP_RE still catches them at the document gate but the main
        # 2-lane MiniLM gate handles their scoring.
        is_inherent = False
        for gt in GATE_TOKENS:
            gt_lower = gt.lower()
            if gt_lower in _COLOR_GATE_TOKENS:
                continue
            if clean_matched == gt_lower + "s" or clean_matched == gt_lower + "es":
                is_inherent = True
                break

        if is_inherent:
            return ("inherent", RESCUE_POS_MIN) if return_threshold else "inherent"

        # Rule 1c ── Singular non-color demonym used as a noun:
        # Preceded by a determiner (a, an, the, this, that, my, your, his, her, their, our) within 2 tokens,
        # and not followed by a noun that is not human (e.g. "An American gained...", "the German who...").
        is_singular_noun = False
        if clean_matched not in _COLOR_GATE_TOKENS:
            has_det = False
            for offset in (-1, -2):
                if pos + offset >= 0:
                    det = re.sub(r"\W+", "", tokens[pos + offset]).lower()
                    if det in ("a", "an", "the", "this", "that", "my", "your", "his", "her", "their", "our"):
                        has_det = True
                        break
            if has_det:
                is_next_non_noun = True
                if pos + 1 < len(tokens):
                    next_word = re.sub(r"\W+", "", tokens[pos + 1]).lower()
                    if next_word:
                        if next_word in INANIMATE_NOUNS or next_word in HUMAN_NOUNS:
                            is_next_non_noun = False
                if is_next_non_noun:
                    is_singular_noun = True

        if is_singular_noun:
            return ("candidate", RESCUE_POS_MIN) if return_threshold else "candidate"

        # Rule 1b — demonym within ±4 tokens of a human head noun or
        # personal pronoun.  Bidirectional so both "white man" and "the
        # man is white" / "she is Chinese" fire.  MiniLM rescue scoring
        # is the second filter for surface-similar non-human cases.
        window = tokens[max(0, pos - 4): pos] + tokens[pos + 1: pos + 5]
        local_candidate = False
        for w in window:
            w_clean = re.sub(r"\W+", "", w)
            if not w_clean:
                continue
            if is_human_noun(w_clean):
                local_candidate = True
                break

        if local_candidate:
            saw_candidate = True
            is_color = clean_matched in _COLOR_GATE_TOKENS
            if not is_color:
                has_non_color_demonym = True
            else:
                # Check if it has a human head at distance +1 or +2 to the right
                close_human = False
                for offset in (1, 2):
                    if pos + offset < len(tokens):
                        w_next = tokens[pos + offset]
                        w_next_clean = re.sub(r"\W+", "", w_next)
                        if w_next_clean and is_human_noun(w_next_clean):
                            close_human = True
                            break
                if close_human:
                    has_color_with_close_head = True

    if saw_candidate:
        if has_non_color_demonym or has_color_with_close_head:
            threshold = RESCUE_POS_MIN
        else:
            threshold = 0.32  # Stricter threshold for color adjectives not directly modifying human nouns
        return ("candidate", threshold) if return_threshold else "candidate"

    return (None, RESCUE_POS_MIN) if return_threshold else None

ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|St|Jr|Sr|vs|Fig|Figs|fig|figs|Messrs|"
    r"No|Nos|Vol|Ch|chap|pp|p)\."
)
INITIAL_RE = re.compile(r"\b[A-Z]\.")
ACRONYM_RE = re.compile(r"\b(?:[A-Z]\.){2,}")
SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\u201C\u2018])"
)
PROTECTED_PERIOD = "<prd>"


def _protect_sentence_internal_periods(text: str):
    def protect(match):
        return match.group(0).replace(".", PROTECTED_PERIOD)

    text = ACRONYM_RE.sub(protect, text)
    text = ABBREVIATION_RE.sub(protect, text)
    text = INITIAL_RE.sub(protect, text)
    return text


def _needs_following_fragment(sentence: str):
    if sentence.count("(") > sentence.count(")"):
        return True
    if sentence.count("[") > sentence.count("]"):
        return True
    if sentence.count("{") > sentence.count("}"):
        return True
    if re.search(r"\([A-Za-z]{1,8}\.$", sentence):
        return True
    return False


def split_sentences(text: str):
    protected_text = _protect_sentence_internal_periods(text)
    pending = ""
    for sentence in SENT_SPLIT.split(protected_text):
        sentence = sentence.replace(PROTECTED_PERIOD, ".").strip()
        if pending:
            sentence = f"{pending} {sentence}".strip()
            pending = ""
        if _needs_following_fragment(sentence):
            pending = sentence
            continue
        if not sentence:
            continue
        if not (MIN_SENT_LEN <= len(sentence) <= MAX_SENT_LEN):
            continue
        if not sentence[0].isupper():
            continue
        if sentence[-1] not in ".!?":
            continue
        if "\n" in sentence:
            continue
        yield sentence
    if pending and MIN_SENT_LEN <= len(pending) <= MAX_SENT_LEN and "\n" not in pending:
        yield pending


def iter_sentences(parquet_file: Path, stats: dict):
    parquet = pq.ParquetFile(parquet_file)
    for batch in parquet.iter_batches(columns=["text"], batch_size=PARQUET_BATCH_SIZE):
        for text in batch.column("text"):
            stats["documents_total"] += 1
            doc_text = str(text)
            if not GROUP_RE.search(doc_text):
                continue
            stats["documents_lexical"] += 1
            yield from split_sentences(doc_text)


def _excel_safe(value):
    if value == "":
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f" {value}"
    return value


def reference_noise_flags(sentence: str):
    return [
        label
        for label, pattern in REFERENCE_NOISE_PATTERNS
        if pattern.search(sentence)
    ]


def review_flags(row: dict):
    flags = []
    bucket = row.get("semantic_bucket")
    if bucket == "STRONG_MARGIN":
        flags.append("strong_margin")
    if row["semantic_margin"] < REVIEW_LOW_MARGIN:
        flags.append("low_semantic_margin")
    if row["semantic_pos"] >= REVIEW_HIGH_SEMANTIC_POS and row["semantic_margin"] < SEMANTIC_MARGIN_MIN:
        flags.append("high_semantic_low_margin")
    noise_flags = reference_noise_flags(row["sentence"])
    if noise_flags:
        flags.append("reference_noise_like:" + "+".join(noise_flags))
    if not flags:
        flags.append("semantic_borderline")
    return flags


def semantic_scores(embedder, pos_query_emb, neg_query_emb, sentences):
    sent_emb = embedder.encode(
        sentences,
        batch_size=EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    pos = (sent_emb @ pos_query_emb.T).max(axis=1)
    neg = (sent_emb @ neg_query_emb.T).max(axis=1)
    margin = pos - neg
    return pos, neg, margin, sent_emb


def write_report(
    report_path: Path,
    stats: dict,
    kept_examples,
    semantic_rejects,
    borderline_review,
):
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("SEMANTIC FILTER REPORT\n")
        handle.write("=" * 64 + "\n")
        for key in (
            "files_processed",
            "model_preset",
            "model_name",
            "model_device",
            "parquet_batch_size",
            "sent_batch_size",
            "emb_batch_size",
            "torch_threads",
            "documents_total",
            "documents_lexical",
            "total_sentences",
            "lexical_hits",
            "semantic_pass",
            "strong_margin_candidates",
            "strong_margin_kept",
            "lexical_human_rescue_candidates",
            "lexical_human_rescue_kept",
            "inanimate_prefilter_removed",
            "kept",
            "borderline_review",
            "reference_noise_blocked",
            "review_low_margin",
            "review_reference_noise_like",
            "elapsed_seconds",
        ):
            handle.write(f"{key}: {stats[key]}\n")
        total = stats["total_sentences"] or 1
        lexical = stats["lexical_hits"] or 1
        docs_total = stats["documents_total"] or 1
        handle.write(f"lexical_rate: {stats['lexical_hits'] / total:.3%}\n")
        handle.write(f"semantic_pass_rate: {stats['semantic_pass'] / total:.3%}\n")
        handle.write(f"strong_margin_rate: {stats['strong_margin_kept'] / total:.3%}\n")
        handle.write(f"final_rate: {stats['kept'] / total:.3%}\n")
        handle.write(f"document_gate_rate: {stats['documents_lexical'] / docs_total:.3%}\n")
        handle.write(f"kept_from_lexical: {stats['kept'] / lexical:.3%}\n")
        handle.write(f"review_from_lexical: {stats['borderline_review'] / lexical:.3%}\n")
        handle.write("\nKEPT EXAMPLES\n")
        handle.write("-" * 64 + "\n")
        for row in kept_examples:
            handle.write(
                f"[bucket={row['semantic_bucket']} sem={row['semantic_pos']:.3f} "
                f"margin={row['semantic_margin']:.3f}] {row['sentence']}\n"
            )
        handle.write("\nSEMANTIC REJECTS\n")
        handle.write("-" * 64 + "\n")
        for row in semantic_rejects:
            handle.write(
                f"[sem={row['semantic_pos']:.3f} neg={row['semantic_neg']:.3f} "
                f"margin={row['semantic_margin']:.3f}] {row['sentence']}\n"
            )
        handle.write("\nREVIEW CANDIDATES\n")
        handle.write("-" * 64 + "\n")
        for row in borderline_review:
            handle.write(
                f"[sem={row['semantic_pos']:.3f} margin={row['semantic_margin']:.3f} "
                f"flags={','.join(row.get('review_flags', []))}] {row['sentence']}\n"
            )


def process_batch(
    sentences,
    embedder,
    pos_query_emb,
    neg_query_emb,
    rescue_pos_emb,
    rescue_neg_emb,
    writer,
    review_writer,
    stats,
    kept_examples,
    semantic_rejects,
    borderline_review,
    lexical_all_handle,
):
    lexical_hits = [sentence for sentence in sentences if GROUP_RE.search(sentence)]
    stats["lexical_hits"] += len(lexical_hits)
    if not lexical_hits:
        return

    for sentence in lexical_hits:
        lexical_all_handle.write(sentence + "\n")

    # Pre-filter: if every gate token in the sentence is adjacent to an
    # inanimate noun, the sentence is almost certainly non-demographic
    # ("German law", "black hole").  Skip MiniLM scoring entirely.
    # The lexical_all file already has these for CEAT-full.
    filtered_hits = [s for s in lexical_hits if not _inanimate_adjacent_only(s)]
    stats["inanimate_prefilter_removed"] = stats.get("inanimate_prefilter_removed", 0) + (len(lexical_hits) - len(filtered_hits))
    if not filtered_hits:
        return
    lexical_hits = filtered_hits

    pos_scores, neg_scores, margins, embeddings = semantic_scores(
        embedder, pos_query_emb, neg_query_emb, lexical_hits
    )
    # Reuse the per-sentence embeddings to score against rescue queries — no
    # extra encoder pass.
    rescue_pos_scores = (embeddings @ rescue_pos_emb.T).max(axis=1)
    rescue_neg_scores = (embeddings @ rescue_neg_emb.T).max(axis=1)
    rescue_margins = rescue_pos_scores - rescue_neg_scores

    semantic_rows = []
    for i, (sentence, pos, neg, margin) in enumerate(
        zip(lexical_hits, pos_scores, neg_scores, margins)
    ):
        row = {
            "sentence": sentence,
            "semantic_pos": float(pos),
            "semantic_neg": float(neg),
            "semantic_margin": float(margin),
            "semantic_bucket": None,
        }
        if margin >= SEMANTIC_STRONG_MARGIN:
            row["semantic_bucket"] = "STRONG_MARGIN"
            stats["strong_margin_candidates"] += 1
            semantic_rows.append(row)
        elif pos >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN_MIN:
            row["semantic_bucket"] = "STRICT"
            semantic_rows.append(row)
        else:
            res = lexical_human_rescue(sentence, return_threshold=True)
            tag = res[0] if res else None
            threshold = res[1] if res else RESCUE_POS_MIN

            if tag == "inherent":
                # Plural non-color demonym: surface form is unambiguously
                # human; admit without rescue MiniLM scoring.
                row["semantic_bucket"] = "LEXICAL_HUMAN_RESCUE"
                stats["lexical_human_rescue_candidates"] += 1
                semantic_rows.append(row)
            elif tag == "candidate":
                # demonym + human-head adjacency: must clear the rescue
                # MiniLM thresholds (threshold / RESCUE_MARGIN_MIN) to
                # filter "German Shepherd / white background / Chinese
                # New Year"-style false positives.
                rp = float(rescue_pos_scores[i])
                rm = float(rescue_margins[i])
                row["rescue_pos"] = rp
                row["rescue_margin"] = rm
                if rp >= threshold and (rm >= RESCUE_MARGIN_MIN or rp >= 0.35):
                    row["semantic_bucket"] = "LEXICAL_HUMAN_RESCUE"
                    stats["lexical_human_rescue_candidates"] += 1
                    semantic_rows.append(row)
                elif margin >= REVIEW_MARGIN_MIN:
                    row["semantic_bucket"] = "BORDERLINE"
                    semantic_rows.append(row)
                elif len(semantic_rejects) < 15:
                    row["semantic_bucket"] = "REJECTED"
                    semantic_rejects.append(row)
            elif margin >= REVIEW_MARGIN_MIN:
                row["semantic_bucket"] = "BORDERLINE"
                semantic_rows.append(row)
            elif len(semantic_rejects) < 15:
                row["semantic_bucket"] = "REJECTED"
                semantic_rejects.append(row)

    strict_rows = sum(1 for row in semantic_rows if row["semantic_bucket"] == "STRICT")
    stats["semantic_pass"] += strict_rows
    if not semantic_rows:
        return

    for row in semantic_rows:
        flags = review_flags(row)
        noise_blocked = BLOCK_REFERENCE_NOISE_KEEP and any(
            flag.startswith("reference_noise_like") for flag in flags
        )
        keep = row["semantic_bucket"] in ("STRICT", "STRONG_MARGIN", "LEXICAL_HUMAN_RESCUE") and not noise_blocked

        if keep:
            writer.writerow(
                {
                    "sentence": row["sentence"],
                    "semantic_pos": _excel_safe(round(row["semantic_pos"], 4)),
                    "semantic_neg": _excel_safe(round(row["semantic_neg"], 4)),
                    "semantic_margin": _excel_safe(round(row["semantic_margin"], 4)),
                    "semantic_bucket": row["semantic_bucket"],
                }
            )
            stats["kept"] += 1
            if row["semantic_bucket"] == "STRONG_MARGIN":
                stats["strong_margin_kept"] += 1
            if row["semantic_bucket"] == "LEXICAL_HUMAN_RESCUE":
                stats["lexical_human_rescue_kept"] += 1
            if len(kept_examples) < 15:
                kept_examples.append(row)
            continue

        # Not kept — route to review if it has any margin signal or is noise-blocked.
        if row["semantic_margin"] >= REVIEW_MARGIN_MIN or noise_blocked:
            review_writer.writerow(
                {
                    "bucket": row["semantic_bucket"],
                    "review_flags": ",".join(flags),
                    "sentence": row["sentence"],
                    "semantic_pos": _excel_safe(round(row["semantic_pos"], 4)),
                    "semantic_neg": _excel_safe(round(row["semantic_neg"], 4)),
                    "semantic_margin": _excel_safe(round(row["semantic_margin"], 4)),
                }
            )
            if noise_blocked:
                stats["reference_noise_blocked"] += 1
            if "low_semantic_margin" in flags:
                stats["review_low_margin"] += 1
            if any(flag.startswith("reference_noise_like") for flag in flags):
                stats["review_reference_noise_like"] += 1
            stats["borderline_review"] += 1
            if len(borderline_review) < 15:
                row["review_flags"] = flags
                borderline_review.append(row)


def main():
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))[:MAX_FILES]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    # 1. Word frequency analysis
    import pyarrow.parquet as pq
    from collections import Counter
    import os

    freq_file = PROJECT_ROOT / "dolma" / "demographic_word_counts.tsv"

    cache_valid = False
    if freq_file.exists():
        freq_mtime = freq_file.stat().st_mtime
        parquet_mtimes = [p.stat().st_mtime for p in parquet_files]
        if parquet_mtimes and max(parquet_mtimes) < freq_mtime:
            cache_valid = True

    counts = {token: 0 for token in (TARGET_TOKENS | CONTRAST_TOKENS)}
    ALL_LABELS = TARGET_TOKENS | CONTRAST_TOKENS

    if cache_valid:
        print(f"Loading cached word counts from {freq_file}...")
        with freq_file.open("r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    token, cat, count = parts[:3]
                    if token in counts:
                        counts[token] = int(count)
    else:
        print("Pre-scanning Parquet files for demographic word counts...")
        ALL_LABELS_LOWER = {token.lower() for token in ALL_LABELS}
        plural_mapping = {}
        for token in ALL_LABELS:
            if not token.endswith("s"):
                plural_mapping[token.lower() + "s"] = token

        word_re = re.compile(r"\b\w+\b")

        na_re = re.compile(r"\bnative[- ]americans?\b", re.I)
        as_sg_re = re.compile(r"\basylum[- ]seeker\b", re.I)
        as_pl_re = re.compile(r"\basylum[- ]seekers\b", re.I)
        non_re = re.compile(
            r"\bnon[- ]("
            + "|".join(sorted(map(re.escape, ALL_LABELS), key=len, reverse=True))
            + r")s?\b",
            re.I
        )

        for p_file in parquet_files:
            parquet = pq.ParquetFile(p_file)
            # Read text column in batches
            for batch in parquet.iter_batches(columns=["text"], batch_size=PARQUET_BATCH_SIZE):
                for text in batch.column("text"):
                    doc_text = str(text).lower()
                    
                    na_matches = len(na_re.findall(doc_text))
                    as_sg_matches = len(as_sg_re.findall(doc_text))
                    as_pl_matches = len(as_pl_re.findall(doc_text))
                    non_matches = non_re.findall(doc_text)
                    
                    words = word_re.findall(doc_text)
                    doc_counts = Counter(words)
                    
                    if na_matches > 0:
                        counts["native-american"] += na_matches
                        doc_counts["american"] = max(0, doc_counts["american"] - na_matches)
                        doc_counts["americans"] = max(0, doc_counts["americans"] - na_matches)
                    if as_sg_matches > 0:
                        counts["asylum-seeker"] += as_sg_matches
                        doc_counts["asylum"] = max(0, doc_counts["asylum"] - as_sg_matches)
                    if as_pl_matches > 0:
                        counts["asylum-seekers"] += as_pl_matches
                        doc_counts["asylum"] = max(0, doc_counts["asylum"] - as_pl_matches)
                        
                    for matched_token in non_matches:
                        t_lower = matched_token.lower()
                        doc_counts[t_lower] = max(0, doc_counts[t_lower] - 1)
                        doc_counts[t_lower + "s"] = max(0, doc_counts[t_lower + "s"] - 1)
                        if t_lower.endswith("y"):
                            doc_counts[t_lower[:-1] + "ies"] = max(0, doc_counts[t_lower[:-1] + "ies"] - 1)
                        
                    for word, count in doc_counts.items():
                        if word in ALL_LABELS_LOWER:
                            counts[word] += count
                        elif word in plural_mapping:
                            counts[plural_mapping[word]] += count

        # 2. Output demographic_word_counts.tsv
        rows = []
        for token in ALL_LABELS:
            is_target = token in TARGET_TOKENS
            is_contrast = token in CONTRAST_TOKENS
            if is_target and is_contrast:
                cat = "both"
            elif is_target:
                cat = "target"
            else:
                cat = "contrast"
            rows.append((token, cat, counts[token]))
        # Sort descending by count, then alphabetically
        rows.sort(key=lambda x: (-x[2], x[0]))
        
        # Ensure parent dir exists (dolma/)
        freq_file.parent.mkdir(parents=True, exist_ok=True)
        with freq_file.open("w", encoding="utf-8") as f:
            f.write("label\tcategory\tcount\n")
            for token, cat, count in rows:
                f.write(f"{token}\t{cat}\t{count}\n")
        print(f"Demographic word counts written to {freq_file}")

    # 3. Limit to top 14 target and top 14 contrast (keep only black/white for color labels)
    EXCLUDED_COLOR_TOKENS = {"brown", "yellow", "colored", "whiter", "dark", "darker", "nonwhite", "non-white", "poc", "of color"}

    target_candidates = [t for t in TARGET_TOKENS if t not in EXCLUDED_COLOR_TOKENS]
    contrast_candidates = [c for c in CONTRAST_TOKENS if c not in EXCLUDED_COLOR_TOKENS]

    target_candidates.sort(key=lambda x: counts[x], reverse=True)
    contrast_candidates.sort(key=lambda x: counts[x], reverse=True)

    top_targets = set(target_candidates[:14])
    top_contrasts = set(contrast_candidates[:14])

    print("Selected Top 14 Target labels for extraction:")
    for t in target_candidates[:14]:
        print(f"  {t}: {counts[t]}")
    print("Selected Top 14 Contrast labels for extraction:")
    for c in contrast_candidates[:14]:
        print(f"  {c}: {counts[c]}")

    # Recompile gate regexes to restrict extraction
    compile_gate_regexes(top_targets, top_contrasts)

    print("Loading embedding model...")
    print(f"  preset: {MODEL_PRESET}")
    print(f"  model:  {MODEL_NAME}")
    print(f"  device: {MODEL_DEVICE}")
    print(f"  parquet_batch={PARQUET_BATCH_SIZE} sent_batch={SENT_BATCH_SIZE} emb_batch={EMB_BATCH_SIZE}")
    embedder = SentenceTransformer(MODEL_NAME, device=MODEL_DEVICE)
    pos_query_emb = embedder.encode(POS_QUERIES, normalize_embeddings=True)
    neg_query_emb = embedder.encode(NEG_QUERIES, normalize_embeddings=True)
    rescue_pos_emb = embedder.encode(RESCUE_POS_QUERIES, normalize_embeddings=True)
    rescue_neg_emb = embedder.encode(RESCUE_NEG_QUERIES, normalize_embeddings=True)

    stats = {
        "files_processed": len(parquet_files),
        "model_preset": MODEL_PRESET,
        "model_name": MODEL_NAME,
        "model_device": MODEL_DEVICE,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "sent_batch_size": SENT_BATCH_SIZE,
        "emb_batch_size": EMB_BATCH_SIZE,
        "torch_threads": torch.get_num_threads(),
        "documents_total": 0,
        "documents_lexical": 0,
        "total_sentences": 0,
        "lexical_hits": 0,
        "semantic_pass": 0,
        "strong_margin_candidates": 0,
        "strong_margin_kept": 0,
        "lexical_human_rescue_candidates": 0,
        "lexical_human_rescue_kept": 0,
        "inanimate_prefilter_removed": 0,
        "kept": 0,
        "borderline_review": 0,
        "reference_noise_blocked": 0,
        "review_low_margin": 0,
        "review_reference_noise_like": 0,
        "elapsed_seconds": 0.0,
    }
    kept_examples = []
    semantic_rejects = []
    borderline_review = []

    started = time.time()
    with (
        OUTPUT_FILE.open("w", encoding="utf-8", newline="") as handle,
        REVIEW_FILE.open("w", encoding="utf-8", newline="") as review_handle,
        LEXICAL_ALL_FILE.open("w", encoding="utf-8") as lexical_all_handle,
    ):
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence",
                "semantic_pos",
                "semantic_neg",
                "semantic_margin",
                "semantic_bucket",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        review_writer = csv.DictWriter(
            review_handle,
            fieldnames=[
                "bucket",
                "review_flags",
                "sentence",
                "semantic_pos",
                "semantic_neg",
                "semantic_margin",
            ],
            delimiter="\t",
        )
        review_writer.writeheader()

        for parquet_file in parquet_files:
            print(f"Processing {parquet_file.name} ...")
            batch = []
            for sentence in iter_sentences(parquet_file, stats):
                stats["total_sentences"] += 1
                batch.append(sentence)
                if stats["total_sentences"] % 50_000 == 0:
                    print(
                        f"  docs={stats['documents_total']:,} "
                        f"doc_gate={stats['documents_lexical']:,} "
                        f"seen={stats['total_sentences']:,} "
                        f"lexical={stats['lexical_hits']:,} "
                        f"semantic={stats['semantic_pass']:,} "
                        f"kept={stats['kept']:,}"
                    )
                if len(batch) >= SENT_BATCH_SIZE:
                    process_batch(
                        batch,
                        embedder,
                        pos_query_emb,
                        neg_query_emb,
                        rescue_pos_emb,
                        rescue_neg_emb,
                        writer,
                        review_writer,
                        stats,
                        kept_examples,
                        semantic_rejects,
                        borderline_review,
                        lexical_all_handle,
                    )
                    batch = []

            if batch:
                process_batch(
                    batch,
                    embedder,
                    pos_query_emb,
                    neg_query_emb,
                    rescue_pos_emb,
                    rescue_neg_emb,
                    writer,
                    review_writer,
                    stats,
                    kept_examples,
                    semantic_rejects,
                    borderline_review,
                    lexical_all_handle,
                )

            print(
                f"  completed {parquet_file.name}: "
                f"docs={stats['documents_total']:,} "
                f"doc_gate={stats['documents_lexical']:,} "
                f"seen={stats['total_sentences']:,} "
                f"lexical={stats['lexical_hits']:,} "
                f"semantic={stats['semantic_pass']:,} "
                f"kept={stats['kept']:,}"
            )
    # 4. ModernBERT fine screening of review candidates
    print("\nRunning ModernBERT fine screening on review candidates...")
    review_rows = []
    if REVIEW_FILE.exists():
        with REVIEW_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                review_rows.append(row)

    if review_rows:
        from embedding_config import ANALYSIS_EMBEDDING_MODEL
        print(f"Loading analysis embedding model {ANALYSIS_EMBEDDING_MODEL} for review candidate validation...")
        mb_embedder = SentenceTransformer(ANALYSIS_EMBEDDING_MODEL, device=MODEL_DEVICE)
        
        # Encode queries under ModernBERT (gte-modernbert-base).
        # Note: Fine screening re-evaluates borderline review candidates against the
        # active POS/NEG and RESCUE query sets using calibrated embedding thresholds
        # (reusing pipeline constants SEMANTIC_STRONG_MARGIN=0.10, SEMANTIC_MIN=0.34,
        # SEMANTIC_MARGIN_MIN=0.03, RESCUE_POS_MIN=0.25/0.32, with rp >= 0.53 margin bypass).
        mb_pos_query_emb = mb_embedder.encode(POS_QUERIES, normalize_embeddings=True)
        mb_neg_query_emb = mb_embedder.encode(NEG_QUERIES, normalize_embeddings=True)
        mb_rescue_pos_emb = mb_embedder.encode(RESCUE_POS_QUERIES, normalize_embeddings=True)
        mb_rescue_neg_emb = mb_embedder.encode(RESCUE_NEG_QUERIES, normalize_embeddings=True)
        
        # Batch encode all review sentences
        sentences_to_encode = [row["sentence"] for row in review_rows]
        print(f"Encoding {len(sentences_to_encode)} review candidates with ModernBERT...")
        mb_embeddings = mb_embedder.encode(sentences_to_encode, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        
        rescued_rows = []
        remaining_review_rows = []
        
        for idx, row in enumerate(review_rows):
            sentence = row["sentence"]
            emb = mb_embeddings[idx]
            
            # Check reference noise
            noise_flags = reference_noise_flags(sentence)
            noise_blocked = BLOCK_REFERENCE_NOISE_KEEP and bool(noise_flags)
            
            # Main lane scores
            pos_score = float((emb @ mb_pos_query_emb.T).max())
            neg_score = float((emb @ mb_neg_query_emb.T).max())
            margin = pos_score - neg_score
            
            # Rescue scores
            rp = float((emb @ mb_rescue_pos_emb.T).max())
            rm = rp - float((emb @ mb_rescue_neg_emb.T).max())
            
            # Check gates
            keep = False
            bucket = None
            
            if not noise_blocked:
                if margin >= SEMANTIC_STRONG_MARGIN:
                    keep = True
                    bucket = "STRONG_MARGIN"
                    stats["strong_margin_kept"] += 1
                elif pos_score >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN_MIN:
                    keep = True
                    bucket = "STRICT"
                    stats["semantic_pass"] += 1
                else:
                    res = lexical_human_rescue(sentence, return_threshold=True)
                    tag = res[0] if res else None
                    threshold = res[1] if res else RESCUE_POS_MIN
                    
                    if tag == "inherent":
                        keep = True
                        bucket = "LEXICAL_HUMAN_RESCUE"
                        stats["lexical_human_rescue_kept"] += 1
                    elif tag == "candidate":
                        if rp >= threshold and (rm >= RESCUE_MARGIN_MIN or rp >= 0.53):
                            keep = True
                            bucket = "LEXICAL_HUMAN_RESCUE"
                            stats["lexical_human_rescue_kept"] += 1
            
            if keep:
                # Rescued!
                r_dict = {
                    "sentence": sentence,
                    "semantic_pos": pos_score,
                    "semantic_neg": neg_score,
                    "semantic_margin": margin,
                    "semantic_bucket": bucket
                }
                rescued_rows.append(r_dict)
                stats["kept"] += 1
                stats["borderline_review"] -= 1
                
                # Update stats for low margin or other review flags
                flags = row["review_flags"].split(",")
                if "low_semantic_margin" in flags:
                    stats["review_low_margin"] -= 1
                if any(flag.startswith("reference_noise_like") for flag in flags):
                    stats["review_reference_noise_like"] -= 1
                    
                if len(kept_examples) < 15:
                    kept_examples.append(r_dict)
            else:
                remaining_review_rows.append(row)
        
        print(f"Rescued {len(rescued_rows)} sentences from review candidates using GTE ModernBERT!")
        
        # Append rescued rows to results.tsv
        if rescued_rows:
            with OUTPUT_FILE.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sentence",
                        "semantic_pos",
                        "semantic_neg",
                        "semantic_margin",
                        "semantic_bucket",
                    ],
                    delimiter="\t",
                )
                for r_row in rescued_rows:
                    writer.writerow({
                        "sentence": r_row["sentence"],
                        "semantic_pos": _excel_safe(round(r_row["semantic_pos"], 4)),
                        "semantic_neg": _excel_safe(round(r_row["semantic_neg"], 4)),
                        "semantic_margin": _excel_safe(round(r_row["semantic_margin"], 4)),
                        "semantic_bucket": r_row["semantic_bucket"],
                    })
                    
        # Overwrite review.tsv with remaining rows
        with REVIEW_FILE.open("w", encoding="utf-8", newline="") as review_handle:
            review_writer = csv.DictWriter(
                review_handle,
                fieldnames=[
                    "bucket",
                    "review_flags",
                    "sentence",
                    "semantic_pos",
                    "semantic_neg",
                    "semantic_margin",
                ],
                delimiter="\t",
            )
            review_writer.writeheader()
            for rem_row in remaining_review_rows:
                review_writer.writerow(rem_row)
                
        # Rebuild borderline_review list for the report
        borderline_review.clear()
        for rem_row in remaining_review_rows[:15]:
            borderline_review.append({
                "sentence": rem_row["sentence"],
                "semantic_pos": float(rem_row["semantic_pos"]),
                "semantic_neg": float(rem_row["semantic_neg"]),
                "semantic_margin": float(rem_row["semantic_margin"]),
                "semantic_bucket": rem_row["bucket"],
                "review_flags": rem_row["review_flags"].split(",") if rem_row["review_flags"] else []
            })

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    write_report(
        REPORT_FILE,
        stats,
        kept_examples,
        semantic_rejects,
        borderline_review,
    )


    # Write the active label sets so Phase 2 can restrict mention resolution
    # to the same top-8 target + top-8 contrast tokens used for extraction.
    active_labels_file = PROJECT_ROOT / "dolma" / "active_labels.json"
    import json as _json
    with active_labels_file.open("w", encoding="utf-8") as _f:
        _json.dump(
            {
                "target": sorted(top_targets),
                "contrast": sorted(top_contrasts),
            },
            _f,
            indent=2,
        )
    print(f"  active labels:   {active_labels_file}")

    print("\nDONE")
    print(f"  output: {OUTPUT_FILE}")
    print(f"  review: {REVIEW_FILE}")
    print(f"  report: {REPORT_FILE}")
    print(f"  documents seen:  {stats['documents_total']:,}")
    print(f"  device/batch:    {MODEL_DEVICE} emb={EMB_BATCH_SIZE} sent={SENT_BATCH_SIZE} parquet={PARQUET_BATCH_SIZE}")
    print(f"  doc gate pass:   {stats['documents_lexical']:,}")
    print(f"  total sentences: {stats['total_sentences']:,}")
    print(f"  lexical hits:    {stats['lexical_hits']:,}")
    print(f"  semantic pass:   {stats['semantic_pass']:,}")
    print(f"  strong-margin:   {stats['strong_margin_kept']:,}")
    print(f"  lexical-human:   {stats['lexical_human_rescue_kept']:,}")
    print(f"  inan-prefilter:  {stats['inanimate_prefilter_removed']:,}")
    print(f"  final kept:      {stats['kept']:,}")
    print(f"  borderline:      {stats['borderline_review']:,}")
    print(f"  elapsed:         {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
