"""
Prototype-based local attitudinal and dimensional diagnostics for group mentions.

Reported group-level netAttI is computed downstream from frame association, not
from this module. The dimensional scores (agi_sim, pi_sim, si_sim) here replace
verb-set membership checks as the primary gate for AgI/PI/SI attribution:
instead of asking "is this verb in SUBJECTIVE_VERBS?", we ask "does this
[GROUP:x][PRED:verb]-annotated context resemble a context where the group is
thinking/acting/being affected?"
"""

from __future__ import annotations

import numpy as np


NEGATIVE_ATTITUDE_PROTOTYPES = [
    "The group or someone was unwanted and unwelcome.",
    "The group or someone was seen as dangerous and threatening.",
    "The group or someone was treated as a burden and a problem.",
    "The group or someone was fearful, vulnerable, and powerless.",
    "The group or someone was mistrusted, isolated, and excluded.",
    "The group or someone was sad"
]

POSITIVE_ATTITUDE_PROTOTYPES = [
    "The group or someone was welcomed and respected.",
    "The group or someone was seen as capable and resilient.",
    "The group or someone was valued, included, and supported.",
    "The group or someone was confident, hopeful, and empowered.",
    "The group or someone was appreciated as beneficial and constructive.",
    "The group or someone was happy"
]

# ── Dimensional prototypes ────────────────────────────────────────────────────
# Each set describes one semantic dimension from the perspective of the TARGET
# group.

AGI_PROTOTYPES = [
    "Someone organizes and launches a movement.",
    "Some group makes a decision and carries out a plan.",
    "They found something and lead the effort themselves.",
    "Someone acts and builds something.",
    "They take initiative and achieve their goal.",
    "Someone organized and launched a movement.",
    "Some group made a decision and carried out a plan.",
    "They founded something and led the effort themselves.",
    "They took initiative and achieved their goal.",
]

PI_PROTOTYPES = [
    "Someone is expelled by others.",
    "They are subjected to harm.",
    "Some people are detained, deported, and stripped of rights.",
    "They are attacked, displaced, or ignored.",
    "Someone is targeted by authorities.",
    "Someone loses, suffers, or hands something over.",
    "Someone was expelled by others.",
    "They were subjected to harm.",
    "Some people were detained, deported, and stripped of rights.",
    "They were attacked, displaced, or ignored.",
    "Someone was targeted by authorities.",
]

SI_PROTOTYPES = [
    "Some people consider something.",
    "They hope for something.",
    "The group believes in something.",
    "They worry about something.",
    "Someone feels or senses something.",
    "Some people considered something.",
    "They hoped for something.",
    "The group believed in something.",
    "They worried about something.",
    "Someone felt or sensed something.",
]

# Dimensional scoring uses relative margin, not absolute floors:
# the winning dimension must score highest AND exceed all others by DIM_MARGIN.
# DIM_FLOOR is a minimum absolute activation to avoid noise.
DIM_FLOOR: float = 0.60
DIM_MARGIN: float = 0.04

# Legacy names kept for import compatibility in step3_feature_extraction.py
AGI_FLOOR = DIM_FLOOR
PI_FLOOR  = DIM_FLOOR
SI_FLOOR  = DIM_FLOOR


class AttitudinalPrototypeMatcher:
    def __init__(
        self,
        sentence_encoder,
        context_window: int = 24,
        positive_floor: float = 0.24,
        positive_margin: float = 0.02,
        agi_floor: float = AGI_FLOOR,
        pi_floor: float = PI_FLOOR,
        si_floor: float = SI_FLOOR,
    ):
        self.sentence_encoder = sentence_encoder
        self.context_window = context_window
        self.positive_floor = positive_floor
        self.positive_margin = positive_margin
        self.agi_floor = agi_floor
        self.pi_floor = pi_floor
        self.si_floor = si_floor

        encode = lambda texts: self.sentence_encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        self.neg_prototypes = encode(NEGATIVE_ATTITUDE_PROTOTYPES)
        self.pos_prototypes = encode(POSITIVE_ATTITUDE_PROTOTYPES)
        self.agi_prototypes = encode(AGI_PROTOTYPES)
        self.pi_prototypes  = encode(PI_PROTOTYPES)
        self.si_prototypes  = encode(SI_PROTOTYPES)

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
        agi_sim = float(np.max(self.agi_prototypes @ vec))
        pi_sim  = float(np.max(self.pi_prototypes  @ vec))
        si_sim  = float(np.max(self.si_prototypes  @ vec))

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
            "agi_sim": round(agi_sim, 4),
            "pi_sim":  round(pi_sim,  4),
            "si_sim":  round(si_sim,  4),
        }
