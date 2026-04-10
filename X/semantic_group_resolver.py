"""
Semantic backoff for ambiguous demographic/group tokens.

This is used only when rule-based phrase/context resolution cannot decide whether
an ambiguous token (e.g. black, foreign, native, western, polish) is a human-group
mention or an unrelated use ("black hole", "foreign exchange", "native app",
"polish the floor").
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from lexicons import SEMANTIC_CONTEXT_WINDOW


# ── Metadata registry (group type + canonical form) ───────────────────────────
# Adding a new ambiguous token = one line here. No custom prompts needed unless
# the universal templates are insufficient (see TOKEN_OVERRIDES below).

PROMPT_REGISTRY = {
    "foreign":   {"group": "minority",  "canonical": "foreign"},
    "black":     {"group": "minority",  "canonical": "black"},
    "polish":    {"group": "minority",  "canonical": "polish"},
    "american":  {"group": "dominant",  "canonical": "american"},
    "settler":   {"group": "dominant",  "canonical": "settler"},
    "british":   {"group": "dominant",  "canonical": "british"},
    "european":  {"group": "dominant",  "canonical": "european"},
    "western":   {"group": "dominant",  "canonical": "western"},
    "local":     {"group": "dominant",  "canonical": "local"},
    "native":    {"group": "minority",  "canonical": "native"},
    "national":  {"group": "dominant",  "canonical": "national"},
    "asylum":    {"group": "minority",  "canonical": "asylum"},
}

# ── Universal prompt templates ────────────────────────────────────────────────
# Applied to every token in PROMPT_REGISTRY via {token} interpolation,
# unless overridden in TOKEN_OVERRIDES.

UNIVERSAL_POSITIVE = [
    "sentence about {token} people, {token} community, or {token} residents as a demographic group",
    "sentence about {token} individuals in a social, racial, ethnic, or national context",
]

UNIVERSAL_NEGATIVE = [
    "sentence using {token} in a technical, financial, geographical, or non-human sense",
    "sentence about {token} objects, products, phenomena, or institutions rather than people",
]

# ── Per-token overrides (only for genuinely unique disambiguation) ────────────
# Keys: "positive" / "negative" fully replace the universal template.
#        "negative_extra" appends to (does not replace) the universal negatives.

TOKEN_OVERRIDES = {
    "asylum": {
        "positive": [
            "sentence about asylum seekers or people applying for refugee asylum",
        ],
        "negative": [
            "sentence about a psychiatric asylum, a place of refuge, or asylum law",
        ],
    },
    "american": {
        "negative_extra": [
            "sentence about non-white American people, or people of non-European origin",
        ],
    },
    "settler": {
        "positive": [
            "sentence about white or European people who settled in a new country",
        ],
        "negative": [
            "sentence about settling a dispute, settling down, or settler as a software term",
        ],
    },
    "native": {
        "negative_extra": [
            "sentence about native-born people as a dominant social group",
        ],
    },
}


class SemanticGroupResolver:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        context_window: int = SEMANTIC_CONTEXT_WINDOW,
        positive_margin: float = 0.04,
        positive_floor: float = 0.26,
    ):
        self.model_name = model_name
        self.context_window = context_window
        self.positive_margin = positive_margin
        self.positive_floor = positive_floor
        self.model = SentenceTransformer(model_name)
        self.prompt_vectors = {}
        for lemma, meta in PROMPT_REGISTRY.items():
            override = TOKEN_OVERRIDES.get(lemma, {})

            if "positive" in override:
                pos_texts = override["positive"]
            else:
                pos_texts = [t.format(token=lemma) for t in UNIVERSAL_POSITIVE]

            if "negative" in override:
                neg_texts = list(override["negative"])
            else:
                neg_texts = [t.format(token=lemma) for t in UNIVERSAL_NEGATIVE]
            neg_texts += [
                t.format(token=lemma) for t in override.get("negative_extra", [])
            ]

            self.prompt_vectors[lemma] = {
                "group": meta["group"],
                "canonical": meta["canonical"],
                "pos": self._encode_many(pos_texts),
                "neg": self._encode_many(neg_texts),
            }

    @lru_cache(maxsize=20000)
    def _encode_text(self, text: str):
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)

    def _encode_many(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def _focus_context(self, token, doc) -> str:
        left = max(0, token.i - self.context_window)
        right = min(len(doc), token.i + self.context_window + 1)
        pieces = []
        for i in range(left, right):
            tok = doc[i]
            if i == token.i:
                pieces.append(f"[{tok.text}]")
            else:
                pieces.append(tok.text)
        return " ".join(pieces)

    def __call__(self, token, doc):
        lemma = token.lemma_.lower()
        if lemma not in self.prompt_vectors:
            return None

        focus_text = self._focus_context(token, doc)
        sent_text = doc.text
        combined = f"{focus_text} || {sent_text}"
        vec = self._encode_text(combined)

        bank = self.prompt_vectors[lemma]
        pos_score = float(np.max(bank["pos"] @ vec))
        neg_score = float(np.max(bank["neg"] @ vec))
        margin = pos_score - neg_score

        if pos_score >= self.positive_floor and margin >= self.positive_margin:
            return (bank["group"], bank["canonical"])
        return None

