"""Shared embedding model configuration for ARMADA."""

import os

EMBEDDING_MODEL_CATALOG = {
    # Official sentence-transformers usage:
    # SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    # Model card: 149M params, 8192-token max input, 768-d output.
    "gte_modernbert_base": "Alibaba-NLP/gte-modernbert-base",
    "minilm": "all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "gte_small": "thenlper/gte-small",
}

# Fast, recall-biased corpus extraction can use MiniLM because Phase 1 is a
# filtering step, not a reported embedding-association metric.
EXTRACTION_EMBEDDING_PRESET = os.environ.get("ARMADA_EXTRACTION_PRESET", "minilm")

# Analysis jobs keep one stronger encoder for semantic disambiguation, frame
# refresh, WEAT, SEAT, and SEAT-full so reported scores share one geometry.
ANALYSIS_EMBEDDING_PRESET = os.environ.get("ARMADA_ANALYSIS_PRESET", "gte_modernbert_base")

DEFAULT_EMBEDDING_PRESET = ANALYSIS_EMBEDDING_PRESET
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_CATALOG[DEFAULT_EMBEDDING_PRESET]
EXTRACTION_EMBEDDING_MODEL = EMBEDDING_MODEL_CATALOG[EXTRACTION_EMBEDDING_PRESET]
ANALYSIS_EMBEDDING_MODEL = EMBEDDING_MODEL_CATALOG[ANALYSIS_EMBEDDING_PRESET]

# Conservative local default for 16GB Apple Silicon laptops. This only controls
# memory pressure during encoding; it does not change embedding geometry.
DEFAULT_EMBEDDING_BATCH_SIZE = 32
