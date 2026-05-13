"""Shared embedding model configuration for ARMADA."""

DEFAULT_EMBEDDING_PRESET = "gte_modernbert_base"

EMBEDDING_MODEL_CATALOG = {
    # Official sentence-transformers usage:
    # SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    # Model card: 149M params, 8192-token max input, 768-d output.
    "gte_modernbert_base": "Alibaba-NLP/gte-modernbert-base",
    "minilm": "all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "gte_small": "thenlper/gte-small",
}

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_CATALOG[DEFAULT_EMBEDDING_PRESET]

# Conservative local default for 16GB Apple Silicon laptops. This only controls
# memory pressure during encoding; it does not change embedding geometry.
DEFAULT_EMBEDDING_BATCH_SIZE = 32
