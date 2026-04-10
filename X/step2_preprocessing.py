"""
STEP 2: Preprocessing Pipeline
Loads raw sentences (pipe-delimited .txt or filtered .tsv), cleans them,
and produces spaCy Doc objects for downstream feature extraction.
"""

import csv
import spacy
import re
from pathlib import Path


def load_sentences(filepath: str) -> list[dict]:
    """Load either legacy pipe-delimited text or filtered TSV output."""
    sentences = []
    path = Path(filepath)

    if path.suffix.lower() == ".tsv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("sentence") or row.get("text") or "").strip()
                if not text:
                    continue
                category = (row.get("category") or row.get("bucket") or "FILTERED").strip()
                sentences.append({
                    "category": category,
                    "raw_text": text,
                })
        return sentences

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                category, text = line.split("|", 1)
                sentences.append({
                    "category": category.strip(),
                    "raw_text": text.strip(),
                })
    return sentences


def remove_noise(text: str) -> str:
    """
    Strip HTML tags, non-printable characters, and excessive whitespace.
    Designed to handle the kind of noise found in raw LLM training corpora.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove non-printable / control characters (keep newlines and spaces)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def preprocess(nlp, sentences: list[dict]) -> list[dict]:
    """
    Run the full preprocessing pipeline:
    1. Noise removal
    2. spaCy parsing (tokenization, sentence splitting, lemmatization, POS, dep)
    3. Attach the Doc object to each record for downstream use
    """
    results = []
    for entry in sentences:
        cleaned = remove_noise(entry["raw_text"])
        doc = nlp(cleaned)
        results.append({
            "category": entry["category"],
            "raw_text": entry["raw_text"],
            "cleaned_text": cleaned,
            "doc": doc,
            "tokens": [
                {
                    "text": token.text,
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "i": token.i,
                }
                for token in doc
            ],
        })
    return results


