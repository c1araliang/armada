"""
Streaming semantic relevance filter for ARMADA.

Current method:
1. Cheap lexical mention gate using the demographic lexicon.
2. Semantic retrieval with positive and negative prompt sets.
3. Binary embedding classifier trained for RELEVANT vs IRRELEVANT.

This script is meant to replace the older regex + spaCy corpus filter for
high-precision candidate extraction with lower memory pressure.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
from lexicons import TARGET_TOKENS, CONTRAST_TOKENS, GATE_EXCLUDE_TOKENS, HUMAN_NOUNS  # type: ignore


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
TRAIN_FILE = X_DIR / "filter_training_samples.txt"
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
MAX_FILES = _env_int("ARMADA_MAX_FILES", 1)
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

# High-precision defaults.
SEMANTIC_MIN = 0.34
SEMANTIC_MARGIN_MIN = 0.03

# Recall-oriented rescue lane. High-confidence rescue rows can enter the final
# corpus, but only behind a stricter classifier threshold than strict semantic
# passes. This preserves the current false-reject fix without letting every
# low-absolute-score rescue row bypass review.
SEMANTIC_RESCUE_MIN = 0.27
SEMANTIC_RESCUE_MARGIN_MIN = 0.08
RESCUE_CAN_KEEP = True
RESCUE_CLASSIFIER_THRESHOLD = 0.60
BLOCK_REFERENCE_NOISE_KEEP = True
LEXICAL_HUMAN_RESCUE_CAN_KEEP = True
LEXICAL_HUMAN_CLASSIFIER_THRESHOLD = 0.65
LEXICAL_HUMAN_REVIEW_PROB_MIN = 0.30

CLASSIFIER_THRESHOLD = 0.56
BORDERLINE_PROB_MIN = 0.45

# Review diagnostics only. These do not decide whether a sentence is kept; they
# make future calibration passes distinguish semantic uncertainty, classifier
# boundary cases, and reference/index-like corpus noise.
REVIEW_LOW_MARGIN = 0.06
REVIEW_HIGH_SEMANTIC_POS = 0.65
REVIEW_HIGH_SEMANTIC_MARGIN = 0.10

POS_QUERIES = [
    "sentence about immigrants or refugee groups",
    "sentence about ethnic or racial minority communities",
    "sentence about people identified by nationality, ancestry, or ethnonym",
    "sentence about foreign workers or foreign residents",
    "sentence about black people, asian people, or jewish people",
    "sentence about asylum seekers, migrants, or displaced families",
    "sentence describing how a demographic group is treated or portrayed",
    "sentence about citizens, locals, americans, or western people as social groups",
    "sentence about minority families facing discrimination or inclusion",
]

NEG_QUERIES = [
    "sentence about weather, geography, or a national park",
    "sentence about software, coding, or a foreign key",
    "sentence about food such as american cheese or black tea",
    "sentence about colored objects such as yellow flag, white hat",
    "sentence about natural phenomena such as a black hole, a white cloud, yellow moon",
    "sentence about movies, music, films, industries such as western films, asian music, or japanese anime",
    "sentence about finance or foreign exchange markets",
    "sentence about generic statistics with no people group involved",
    "sentence about technical terms using local, native, national, or western in a non-human sense",
]

REFERENCE_NOISE_PATTERNS = [
    ("index_page_ref", re.compile(r",\s*\d{1,4}\.$")),
    ("url_or_markup", re.compile(r"https?://|www\.|<[^>]+>")),
    (
        "bibliographic",
        re.compile(
            r"\b(ISBN|DOI|Journal|Proceedings|University Press|Cambridge University Press|"
            r"Oxford University Press|Princeton University Press|Vol\.|No\.|Nr\.|"
            r"chapter|edited by|published by)\b",
            re.I,
        ),
    ),
]

PERSON_SUFFIX_GENERIC_RE = re.compile(
    r"\b[A-Z]?[a-z]+(?:man|men|woman|women|boy|boys|girl|girls|people)\b"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
ALL_GROUP_TOKENS = TARGET_TOKENS | CONTRAST_TOKENS
GATE_TOKENS = ALL_GROUP_TOKENS - GATE_EXCLUDE_TOKENS
GROUP_TOKEN_PATTERN = (
    r"(?<!\w)(?:"
    + "|".join(sorted(map(re.escape, GATE_TOKENS), key=len, reverse=True))
    + r")(?!\w)"
)
GROUP_PERSON_SUFFIX_PATTERN = (
    r"(?<!\w)(?:"
    + "|".join(sorted(map(re.escape, GATE_TOKENS), key=len, reverse=True))
    + r")(?:man|men|woman|women|boy|boys|girl|girls|people)(?!\w)"
)
GROUP_RE = re.compile(
    GROUP_TOKEN_PATTERN + "|" + GROUP_PERSON_SUFFIX_PATTERN,
    re.I,
)


def _surface_variants(terms):
    variants = set(terms)
    for term in terms:
        if term.endswith("y") and len(term) > 1:
            variants.add(term[:-1] + "ies")
        if not term.endswith("s"):
            variants.add(term + "s")
        variants.add(term + "es")
    return variants


HUMAN_RESCUE_HEADS = _surface_variants(
    HUMAN_NOUNS
    | {
        "lady", "ladies", "gentleman", "gentlemen",
        "fellow", "fellows", "mother", "father", "parent", "parents",
        "wife", "husband", "son", "sons", "daughter", "daughters",
        "brother", "brothers", "sister", "sisters",
    }
)
HUMAN_HEAD_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(sorted(map(re.escape, HUMAN_RESCUE_HEADS), key=len, reverse=True))
    + r")(?!\w)",
    re.I,
)
GROUP_HUMAN_RE = re.compile(
    GROUP_TOKEN_PATTERN
    + r"(?:[\s,;'\"“”‘’()\[\]-]+\w+){0,3}?[\s,;'\"“”‘’()\[\]-]+(?:"
    + "|".join(sorted(map(re.escape, HUMAN_RESCUE_HEADS), key=len, reverse=True))
    + r")(?!\w)",
    re.I,
)
GROUP_PERSON_SUFFIX_RE = re.compile(GROUP_PERSON_SUFFIX_PATTERN, re.I)
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


def load_training_data(path: Path):
    texts = []
    labels = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            label, sentence = line.split("|", 1)
            label = label.strip().upper()
            if label not in {"RELEVANT", "IRRELEVANT"}:
                continue
            labels.append(label)
            texts.append(sentence.strip())
    if not texts:
        raise ValueError(f"No training rows found in {path}")
    return texts, labels


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


def lexical_human_rescue(sentence: str):
    if PERSON_SUFFIX_GENERIC_RE.search(sentence) and GROUP_PERSON_SUFFIX_RE.search(sentence):
        return True
    if HUMAN_HEAD_RE.search(sentence) and GROUP_HUMAN_RE.search(sentence):
        return True
    return False


def review_flags(row: dict):
    flags = []
    if row.get("semantic_bucket") == "SEMANTIC_RESCUE":
        flags.append("semantic_rescue")
    if row.get("semantic_bucket") == "LEXICAL_HUMAN_RESCUE":
        flags.append("lexical_human_rescue")
    if row["semantic_margin"] < REVIEW_LOW_MARGIN:
        flags.append("low_semantic_margin")
    if (
        row["semantic_pos"] >= REVIEW_HIGH_SEMANTIC_POS
        or row["semantic_margin"] >= REVIEW_HIGH_SEMANTIC_MARGIN
    ) and row["relevant_probability"] < CLASSIFIER_THRESHOLD:
        flags.append("high_semantic_low_classifier")
    noise_flags = reference_noise_flags(row["sentence"])
    if noise_flags:
        flags.append("reference_noise_like:" + "+".join(noise_flags))
    if not flags:
        flags.append("classifier_borderline")
    return flags


def train_classifier(path: Path, embedder):
    texts, labels = load_training_data(path)
    x = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    n_components = min(len(texts) // 3, x.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    x_reduced = pca.fit_transform(x)
    clf = LogisticRegression(
        max_iter=3_000,
        C=1.5,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_reduced, labels)
    return pca, clf, len(texts)


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


def relevant_probabilities(pca, clf, embeddings):
    x_reduced = pca.transform(embeddings)
    classes = list(clf.classes_)
    rel_idx = classes.index("RELEVANT")
    return clf.predict_proba(x_reduced)[:, rel_idx]


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
            "training_rows",
            "documents_total",
            "documents_lexical",
            "total_sentences",
            "lexical_hits",
            "semantic_pass",
            "semantic_rescue_candidates",
            "semantic_rescue_kept",
            "semantic_rescue_review",
            "lexical_human_rescue_candidates",
            "lexical_human_rescue_kept",
            "lexical_human_rescue_review",
            "classifier_pass",
            "borderline_review",
            "reference_noise_blocked",
            "review_low_margin",
            "review_high_semantic_low_classifier",
            "review_reference_noise_like",
            "elapsed_seconds",
        ):
            handle.write(f"{key}: {stats[key]}\n")
        total = stats["total_sentences"] or 1
        lexical = stats["lexical_hits"] or 1
        semantic = stats["semantic_pass"] or 1
        semantic_candidates = stats["semantic_pass"] + stats["semantic_rescue_candidates"] or 1
        docs_total = stats["documents_total"] or 1
        handle.write(f"lexical_rate: {stats['lexical_hits'] / total:.3%}\n")
        handle.write(f"semantic_rate: {stats['semantic_pass'] / total:.3%}\n")
        handle.write(f"semantic_candidate_rate: {semantic_candidates / total:.3%}\n")
        handle.write(f"final_rate: {stats['classifier_pass'] / total:.3%}\n")
        handle.write(f"document_gate_rate: {stats['documents_lexical'] / docs_total:.3%}\n")
        handle.write(f"semantic_keep_from_lexical: {stats['semantic_pass'] / lexical:.3%}\n")
        handle.write(f"semantic_candidate_from_lexical: {semantic_candidates / lexical:.3%}\n")
        handle.write(f"classifier_keep_from_semantic: {stats['classifier_pass'] / semantic:.3%}\n")
        handle.write(
            f"borderline_share_from_semantic: "
            f"{stats['borderline_review'] / semantic:.3%}\n"
        )
        handle.write("\nKEPT EXAMPLES\n")
        handle.write("-" * 64 + "\n")
        for row in kept_examples:
            handle.write(
                f"[sem={row['semantic_pos']:.3f} margin={row['semantic_margin']:.3f} "
                f"clf={row['relevant_probability']:.3f}] {row['sentence']}\n"
            )
        handle.write("\nSEMANTIC REJECTS\n")
        handle.write("-" * 64 + "\n")
        for row in semantic_rejects:
            handle.write(
                f"[sem={row['semantic_pos']:.3f} neg={row['semantic_neg']:.3f} "
                f"margin={row['semantic_margin']:.3f}] {row['sentence']}\n"
            )
        handle.write("\nBORDERLINE REVIEW CANDIDATES\n")
        handle.write("-" * 64 + "\n")
        for row in borderline_review:
            handle.write(
                f"[sem={row['semantic_pos']:.3f} margin={row['semantic_margin']:.3f} "
                f"clf={row['relevant_probability']:.3f} "
                f"flags={','.join(row.get('review_flags', []))}] {row['sentence']}\n"
            )


def process_batch(
    sentences,
    embedder,
    pos_query_emb,
    neg_query_emb,
    pca,
    clf,
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

    pos_scores, neg_scores, margins, embeddings = semantic_scores(
        embedder, pos_query_emb, neg_query_emb, lexical_hits
    )
    semantic_rows = []
    semantic_indices = []
    for i, (sentence, pos, neg, margin) in enumerate(
        zip(lexical_hits, pos_scores, neg_scores, margins)
    ):
        row = {
            "sentence": sentence,
            "semantic_pos": float(pos),
            "semantic_neg": float(neg),
            "semantic_margin": float(margin),
            "semantic_bucket": "STRICT",
        }
        if pos >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN_MIN:
            semantic_rows.append(row)
            semantic_indices.append(i)
        elif pos >= SEMANTIC_RESCUE_MIN and margin >= SEMANTIC_RESCUE_MARGIN_MIN:
            row["semantic_bucket"] = "SEMANTIC_RESCUE"
            semantic_rows.append(row)
            semantic_indices.append(i)
            stats["semantic_rescue_candidates"] += 1
        elif lexical_human_rescue(sentence):
            row["semantic_bucket"] = "LEXICAL_HUMAN_RESCUE"
            semantic_rows.append(row)
            semantic_indices.append(i)
            stats["lexical_human_rescue_candidates"] += 1
        elif len(semantic_rejects) < 15:
            semantic_rejects.append(row)

    strict_rows = sum(1 for row in semantic_rows if row["semantic_bucket"] == "STRICT")
    stats["semantic_pass"] += strict_rows
    if not semantic_rows:
        return

    semantic_embeddings = embeddings[semantic_indices]
    probabilities = relevant_probabilities(pca, clf, semantic_embeddings)
    for row, prob in zip(semantic_rows, probabilities):
        row["relevant_probability"] = float(prob)
        flags = review_flags(row)
        noise_blocked = BLOCK_REFERENCE_NOISE_KEEP and any(
            flag.startswith("reference_noise_like") for flag in flags
        )
        if row["semantic_bucket"] == "STRICT":
            keep_threshold = CLASSIFIER_THRESHOLD
        elif row["semantic_bucket"] == "SEMANTIC_RESCUE" and RESCUE_CAN_KEEP:
            keep_threshold = RESCUE_CLASSIFIER_THRESHOLD
        elif row["semantic_bucket"] == "LEXICAL_HUMAN_RESCUE" and LEXICAL_HUMAN_RESCUE_CAN_KEEP:
            keep_threshold = LEXICAL_HUMAN_CLASSIFIER_THRESHOLD
        else:
            keep_threshold = float("inf")

        if prob >= keep_threshold and not noise_blocked:
            writer.writerow(
                {
                    "sentence": row["sentence"],
                    "semantic_pos": _excel_safe(round(row["semantic_pos"], 4)),
                    "semantic_neg": _excel_safe(round(row["semantic_neg"], 4)),
                    "semantic_margin": _excel_safe(round(row["semantic_margin"], 4)),
                    "relevant_probability": _excel_safe(round(row["relevant_probability"], 4)),
                }
            )
            stats["classifier_pass"] += 1
            if row["semantic_bucket"] == "SEMANTIC_RESCUE":
                stats["semantic_rescue_kept"] += 1
            if row["semantic_bucket"] == "LEXICAL_HUMAN_RESCUE":
                stats["lexical_human_rescue_kept"] += 1
            if len(kept_examples) < 15:
                kept_examples.append(row)
            continue
        if row["semantic_bucket"] == "LEXICAL_HUMAN_RESCUE":
            review_threshold = LEXICAL_HUMAN_REVIEW_PROB_MIN
        else:
            review_threshold = BORDERLINE_PROB_MIN

        if prob >= review_threshold or noise_blocked:
            review_writer.writerow(
                {
                    "bucket": row["semantic_bucket"] if row["semantic_bucket"] != "STRICT" else "BORDERLINE",
                    "review_flags": ",".join(flags),
                    "sentence": row["sentence"],
                    "semantic_pos": _excel_safe(round(row["semantic_pos"], 4)),
                    "semantic_neg": _excel_safe(round(row["semantic_neg"], 4)),
                    "semantic_margin": _excel_safe(round(row["semantic_margin"], 4)),
                    "relevant_probability": _excel_safe(round(row["relevant_probability"], 4)),
                }
            )
            if row["semantic_bucket"] == "SEMANTIC_RESCUE":
                stats["semantic_rescue_review"] += 1
            if row["semantic_bucket"] == "LEXICAL_HUMAN_RESCUE":
                stats["lexical_human_rescue_review"] += 1
            if noise_blocked:
                stats["reference_noise_blocked"] += 1
            if "low_semantic_margin" in flags:
                stats["review_low_margin"] += 1
            if "high_semantic_low_classifier" in flags:
                stats["review_high_semantic_low_classifier"] += 1
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

    print("Loading embedding model...")
    print(f"  preset: {MODEL_PRESET}")
    print(f"  model:  {MODEL_NAME}")
    print(f"  device: {MODEL_DEVICE}")
    print(f"  parquet_batch={PARQUET_BATCH_SIZE} sent_batch={SENT_BATCH_SIZE} emb_batch={EMB_BATCH_SIZE}")
    embedder = SentenceTransformer(MODEL_NAME, device=MODEL_DEVICE)
    pos_query_emb = embedder.encode(POS_QUERIES, normalize_embeddings=True)
    neg_query_emb = embedder.encode(NEG_QUERIES, normalize_embeddings=True)

    print("Training embedding classifier...")
    pca, clf, n_train = train_classifier(TRAIN_FILE, embedder)

    stats = {
        "files_processed": len(parquet_files),
        "model_preset": MODEL_PRESET,
        "model_name": MODEL_NAME,
        "model_device": MODEL_DEVICE,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "sent_batch_size": SENT_BATCH_SIZE,
        "emb_batch_size": EMB_BATCH_SIZE,
        "torch_threads": torch.get_num_threads(),
        "training_rows": n_train,
        "documents_total": 0,
        "documents_lexical": 0,
        "total_sentences": 0,
        "lexical_hits": 0,
        "semantic_pass": 0,
        "semantic_rescue_candidates": 0,
        "semantic_rescue_kept": 0,
        "semantic_rescue_review": 0,
        "lexical_human_rescue_candidates": 0,
        "lexical_human_rescue_kept": 0,
        "lexical_human_rescue_review": 0,
        "classifier_pass": 0,
        "borderline_review": 0,
        "reference_noise_blocked": 0,
        "review_low_margin": 0,
        "review_high_semantic_low_classifier": 0,
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
                "relevant_probability",
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
                "relevant_probability",
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
                        f"kept={stats['classifier_pass']:,}"
                    )
                if len(batch) >= SENT_BATCH_SIZE:
                    process_batch(
                        batch,
                        embedder,
                        pos_query_emb,
                        neg_query_emb,
                        pca,
                        clf,
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
                    pca,
                    clf,
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
                f"kept={stats['classifier_pass']:,}"
            )

    stats["elapsed_seconds"] = round(time.time() - started, 1)
    write_report(
        REPORT_FILE,
        stats,
        kept_examples,
        semantic_rejects,
        borderline_review,
    )

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
    print(f"  rescue kept:     {stats['semantic_rescue_kept']:,}")
    print(f"  lexical-human kept: {stats['lexical_human_rescue_kept']:,}")
    print(f"  final kept:      {stats['classifier_pass']:,}")
    print(f"  borderline:      {stats['borderline_review']:,}")
    print(f"  elapsed:         {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
