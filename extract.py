"""
Streaming semantic relevance filter for ARMADA.

Current method:
1. Cheap lexical mention gate using the demographic lexicon.
2. Semantic retrieval with positive and negative prompt sets.
3. Binary TF-IDF classifier trained for RELEVANT vs IRRELEVANT.

This script is meant to replace the older regex + spaCy corpus filter for
high-precision candidate extraction with lower memory pressure.
"""

import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(line_buffering=True)
# Auto-detect project root and add to path
PROJECT_ROOT = Path(__file__).parent
X_DIR = PROJECT_ROOT / "X"
if X_DIR.exists():
    sys.path.insert(0, str(X_DIR))

from lexicons import TARGET_TOKENS, CONTRAST_TOKENS, GATE_EXCLUDE_TOKENS  # type: ignore


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "dolma" / "data"
TRAIN_FILE = X_DIR / "filter_training_samples.txt"
OUTPUT_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_results.tsv"
REVIEW_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_review.tsv"
REPORT_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_report.txt"
LEXICAL_ALL_FILE = PROJECT_ROOT / "dolma" / "semantic_filter_lexical_all.txt"

# Encoder presets:
# - minilm: best current CPU speed/quality tradeoff for this pipeline
# - bge_small: stronger retrieval semantics, somewhat slower
# - gte_small: reasonable middle-ground alternative
MODEL_PRESET = "minilm"
MODEL_CATALOG = {
    "minilm": "all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "gte_small": "thenlper/gte-small",
}
MODEL_NAME = MODEL_CATALOG[MODEL_PRESET]
MAX_FILES = 1
PARQUET_BATCH_SIZE = 5_000
SENT_BATCH_SIZE = 1_024
EMB_BATCH_SIZE = 128

MIN_SENT_LEN = 40
MAX_SENT_LEN = 300

# High-precision defaults.
SEMANTIC_MIN = 0.34
SEMANTIC_MARGIN_MIN = 0.03
CLASSIFIER_THRESHOLD = 0.56
BORDERLINE_PROB_MIN = 0.45

POS_QUERIES = [
    "sentence about immigrants or refugee groups",
    "sentence about ethnic or racial minority communities",
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


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
ALL_GROUP_TOKENS = TARGET_TOKENS | CONTRAST_TOKENS
GATE_TOKENS = ALL_GROUP_TOKENS - GATE_EXCLUDE_TOKENS
GROUP_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(sorted(map(re.escape, GATE_TOKENS), key=len, reverse=True))
    + r")(?!\w)",
    re.I,
)
SENT_SPLIT = re.compile(
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bRev\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bvs\.)"
    r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\u201C\u2018])"
)


def split_sentences(text: str):
    for sentence in SENT_SPLIT.split(text):
        sentence = sentence.strip()
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
            "training_rows",
            "documents_total",
            "documents_lexical",
            "total_sentences",
            "lexical_hits",
            "semantic_pass",
            "classifier_pass",
            "borderline_review",
            "elapsed_seconds",
        ):
            handle.write(f"{key}: {stats[key]}\n")
        total = stats["total_sentences"] or 1
        lexical = stats["lexical_hits"] or 1
        semantic = stats["semantic_pass"] or 1
        docs_total = stats["documents_total"] or 1
        handle.write(f"lexical_rate: {stats['lexical_hits'] / total:.3%}\n")
        handle.write(f"semantic_rate: {stats['semantic_pass'] / total:.3%}\n")
        handle.write(f"final_rate: {stats['classifier_pass'] / total:.3%}\n")
        handle.write(f"document_gate_rate: {stats['documents_lexical'] / docs_total:.3%}\n")
        handle.write(f"semantic_keep_from_lexical: {stats['semantic_pass'] / lexical:.3%}\n")
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
                f"clf={row['relevant_probability']:.3f}] {row['sentence']}\n"
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
        }
        if pos >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN_MIN:
            semantic_rows.append(row)
            semantic_indices.append(i)
        elif len(semantic_rejects) < 15:
            semantic_rejects.append(row)

    stats["semantic_pass"] += len(semantic_rows)
    if not semantic_rows:
        return

    semantic_embeddings = embeddings[semantic_indices]
    probabilities = relevant_probabilities(pca, clf, semantic_embeddings)
    for row, prob in zip(semantic_rows, probabilities):
        row["relevant_probability"] = float(prob)
        if prob >= CLASSIFIER_THRESHOLD:
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
            if len(kept_examples) < 15:
                kept_examples.append(row)
        elif prob >= BORDERLINE_PROB_MIN:
            review_writer.writerow(
                {
                    "bucket": "BORDERLINE",
                    "sentence": row["sentence"],
                    "semantic_pos": _excel_safe(round(row["semantic_pos"], 4)),
                    "semantic_neg": _excel_safe(round(row["semantic_neg"], 4)),
                    "semantic_margin": _excel_safe(round(row["semantic_margin"], 4)),
                    "relevant_probability": _excel_safe(round(row["relevant_probability"], 4)),
                }
            )
            stats["borderline_review"] += 1
            if len(borderline_review) < 15:
                borderline_review.append(row)


def main():
    parquet_files = sorted(DATA_DIR.glob("train-*.parquet"))[:MAX_FILES]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    print("Loading embedding model...")
    print(f"  preset: {MODEL_PRESET}")
    print(f"  model:  {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)
    pos_query_emb = embedder.encode(POS_QUERIES, normalize_embeddings=True)
    neg_query_emb = embedder.encode(NEG_QUERIES, normalize_embeddings=True)

    print("Training embedding classifier...")
    pca, clf, n_train = train_classifier(TRAIN_FILE, embedder)

    stats = {
        "files_processed": len(parquet_files),
        "model_preset": MODEL_PRESET,
        "model_name": MODEL_NAME,
        "training_rows": n_train,
        "documents_total": 0,
        "documents_lexical": 0,
        "total_sentences": 0,
        "lexical_hits": 0,
        "semantic_pass": 0,
        "classifier_pass": 0,
        "borderline_review": 0,
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
    print(f"  doc gate pass:   {stats['documents_lexical']:,}")
    print(f"  total sentences: {stats['total_sentences']:,}")
    print(f"  lexical hits:    {stats['lexical_hits']:,}")
    print(f"  semantic pass:   {stats['semantic_pass']:,}")
    print(f"  final kept:      {stats['classifier_pass']:,}")
    print(f"  borderline:      {stats['borderline_review']:,}")
    print(f"  elapsed:         {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()