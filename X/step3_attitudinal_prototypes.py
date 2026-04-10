"""
Prototype-based local attitudinal matching for group mentions.

This replaces exact adjective / psych-verb lookup with local embedding matching
against positive and negative attribution prototypes.
"""

from __future__ import annotations

import numpy as np


NEGATIVE_ATTITUDE_PROTOTYPES = [
    "The group was unwanted and unwelcome.",
    "The group was seen as dangerous and threatening.",
    "The group was treated as a burden and a problem.",
    "The group was fearful, vulnerable, and powerless.",
    "The group was mistrusted, isolated, and excluded.",
]

POSITIVE_ATTITUDE_PROTOTYPES = [
    "The group was welcomed and respected.",
    "The group was seen as capable and resilient.",
    "The group was valued, included, and supported.",
    "The group was confident, hopeful, and empowered.",
    "The group was appreciated as beneficial and constructive.",
]


class AttitudinalPrototypeMatcher:
    def __init__(
        self,
        sentence_encoder,
        context_window: int = 8,
        positive_floor: float = 0.24,
        positive_margin: float = 0.02,
    ):
        self.sentence_encoder = sentence_encoder
        self.context_window = context_window
        self.positive_floor = positive_floor
        self.positive_margin = positive_margin
        self.neg_prototypes = self.sentence_encoder.encode(
            NEGATIVE_ATTITUDE_PROTOTYPES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.pos_prototypes = self.sentence_encoder.encode(
            POSITIVE_ATTITUDE_PROTOTYPES,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _build_focus_text(self, token, doc, head_verb=None, span_indices: set[int] | None = None) -> str:
        span_indices = span_indices or {token.i}
        anchor_indices = set(span_indices)
        if head_verb is not None and 0 <= head_verb.i < len(doc):
            anchor_indices.add(head_verb.i)
        left = max(0, min(anchor_indices) - self.context_window)
        right = min(len(doc), max(anchor_indices) + self.context_window + 1)

        pieces = []
        for i in range(left, right):
            tok = doc[i]
            text = tok.text
            if i in span_indices:
                text = f"[GROUP:{text}]"
            elif head_verb is not None and i == head_verb.i:
                text = f"[PRED:{text}]"
            pieces.append(text)
        return " ".join(pieces)

    def match(self, token, doc, head_verb=None, span_indices: set[int] | None = None) -> dict:
        focus_text = self._build_focus_text(token, doc, head_verb=head_verb, span_indices=span_indices)
        vec = self.sentence_encoder.encode(
            focus_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        neg_sim = float(np.max(self.neg_prototypes @ vec))
        pos_sim = float(np.max(self.pos_prototypes @ vec))
        margin = abs(pos_sim - neg_sim)
        best = max(pos_sim, neg_sim)

        label = None
        if best >= self.positive_floor and margin >= self.positive_margin:
            label = "posAttI" if pos_sim > neg_sim else "negAttI"

        return {
            "label": label,
            "focus_text": focus_text,
            "neg_sim": round(neg_sim, 4),
            "pos_sim": round(pos_sim, 4),
        }
