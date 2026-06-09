"""
Prototype-based local attitudinal and dimensional diagnostics for group mentions.

Reported group-level netAttI is computed downstream from frame association, not
from this module. The dimensional scores (agi_sim, pi_sim, si_sim) here replace
verb-set membership checks as the primary gate for AgI/PI/SI attribution:
instead of asking "is this verb in SUBJECTIVE_VERBS?", we ask "does this
[GROUP:x][PRED:verb]-annotated context resemble a context where the group is
acting / being affected / having mental states?"

Seed design
-----------
AttI is **copular evaluative**: polarity lives in the complement, so the
NEG / POS lists are stored as canonical complements and surface-varied across
a fixed tense × number grid (`_COPULAR_SPECS`).

AGI / PI / SI are **paraphrase-based**, not verb-list-based. Each list consists of
hand-written sentences that together cover:

  - AGI: attributed volition, control, intentional efficacy, with effects
         (physical or mental) traceable to the group's own decisions.
  - PI:  attributed affectedness or vulnerability — physical or emotional,
         and by either others' actions or others' attitudes.
  - SI:  attributed mindedness, inner consciousness, and autonomous feeling,
         held internally and/or shown outwardly.

The first sentence of each dimensional list is the ruling definition (AGI / PI
parallel: "They are agents bringing about ..." / "They are patients being
affected ..."). The remaining paraphrases diversify surface
form: tense (present, past, present-perfect-progressive), aspect, voice
(active, passive, mixed, embedded passive in active main clause), gender (he,
she, none), and number (plural `they`, singular he/she, collective `the
group`). Each list deliberately includes both pleasant and unpleasant
outcomes so dimensional similarity does not collapse into evaluative polarity
(e.g., PI ≠ negAttI; AGI ≠ posAttI).

A bare verb-grid was tried earlier and removed: attaching specific verbs
re-introduced lexical priors that narrowed the construct, and BERT-class
encoders read the construct off whole-sentence semantics, not off lemma
membership. Reflexive constructions ("they organized themselves") remain a
known limitation that paraphrase scoring at this layer cannot fully resolve;
target-bound SRL evidence handles them downstream.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ── Copular surface controller (AttI only) ──────────────────────────────────
# AttI seeds are copular complements; polarity is in the complement, so we
# vary tense and number around `be` to keep the prototype matrix balanced
# across present/past/perfect and plural/singular subjects.

def _be_form(tense: str, number: str) -> str:
    if tense == "perfect":
        return "have been" if number == "plural" else "has been"
    if tense == "present":
        return "are" if number == "plural" else "is"
    return "were" if number == "plural" else "was"


# (tense, number, subject_pronoun)
_COPULAR_SPECS: list[tuple[str, str, str]] = [
    ("present", "plural",   "they"),
    ("past",    "plural",   "they"),
    ("perfect", "plural",   "they"),
    ("present", "singular", "he"),
    ("past",    "singular", "she"),
    ("perfect", "singular", "he"),
]


def _normspace(s: str) -> str:
    return " ".join(s.split())


def _expand_copular(complement: str) -> list[str]:
    out = []
    for tense, number, subj in _COPULAR_SPECS:
        be = _be_form(tense, number)
        out.append(_normspace(f"{subj} {be} {complement}"))
    return out


def _flatten(groups: Iterable[Iterable[str]]) -> list[str]:
    return [s for group in groups for s in group]


# ── Canonical seeds ──────────────────────────────────────────────────────────

_NEG_ATTI_CANON: list[str] = [
    "unwanted and unwelcome",
    "seen as dangerous and threatening",
    "treated as a burden and a problem",
    "fearful, vulnerable, and powerless",
    "mistrusted, isolated, and excluded",
    "sad or abhorrent",
]

_POS_ATTI_CANON: list[str] = [
    "welcomed and respected",
    "seen as capable and resilient",
    "valued, included, and supported",
    "confident, hopeful, and empowered",
    "appreciated as beneficial and constructive",
    "happy or pleasant",
]

# Dimensional paraphrases. Order convention (per dimension):
#   [0] ruling definition (AGI ↔ PI structurally parallel)
#   [1] past simple, female singular, mixed voice
#   [2] present perfect (progressive where it reads naturally), male singular
#   [3] present progressive, plural, active or active-with-embedded-passive
#   [4] present or past simple, "the group" / collective, polarity-mixed
#   [5] present simple, male singular, polarity-mixed
#   [6] past perfect, male singular (AGI/PI only)
#   [7] future simple, female singular (AGI/PI only)

AGI_PROTOTYPES: list[str] = [
    "They are intended to bring about effects, physically or mentally.",
    "She organized things and led the effort herself.",
    "He has been making good or bad decisions that influenced lives of others.",
    "They are using or doing things, or building things, legal or illegal.",
    "The group made or demanded change, or from authorities",
    "Members affects, influences, or causes things to happen to themselves or to others.",
    "He had taken things for himself or from others",
    "She will give something to others, or something was given by them",
]

PI_PROTOTYPES: list[str] = [
    "They are being affected, physically or emotionally, by others.",
    "She suffered from, underwent or experienced something.",
    "He has been welcomed, supported, or abused, rejected by other people.",
    "They are being influenced by what others do to them.",
    "The group was attacked by some or protected by others",
    "Members receive support, harm, or consequences",
    "He had lost something, or something was stripped from him",
    "She will receive something from others, or something was prepared for her",
]

SI_PROTOTYPES: list[str] = [
    "They have and/or show consciousness or feeling — what they love, prefer, appreciate, or hate, loathe, disgust",
    "She felt deeply, thought for herself, and held beliefs of her own.",
    "He has been sensing or expressing what he feels and thinks about what is happening.",
    "The group form negative opinions or positive judgments about some people or something.",
    "He is recognized as a thinking, sensible person with his own perspective.",
    "She will believe, hope, or fear, worry about some people or something.",
]

# Public AttI surface lists — built once at module load by expanding the
# canonical complements across the copular tense × number grid.
NEGATIVE_ATTITUDE_PROTOTYPES: list[str] = _flatten(
    _expand_copular(c) for c in _NEG_ATTI_CANON
)
POSITIVE_ATTITUDE_PROTOTYPES: list[str] = _flatten(
    _expand_copular(c) for c in _POS_ATTI_CANON
)

# Independent per-dimension floors. Each dim is now an independent assertion:
# a single target can be both AGI and PI, or both PI and SI, in the same
# mention. There is no winner-take-all margin; each cosine is checked against
# its own absolute threshold.
#
# Floors are calibrated empirically against the cached cosine distribution
# (5,674 mentions, 6-paraphrase prototype matrix). They correspond to roughly
# the 70th percentile of each dim's cosine distribution, giving each dim a
# ~30% pass-rate. PI's mean cosine sits ~0.04 above SI's, so SI_FLOOR is
# correspondingly lower; using a uniform floor would systematically suppress SI.
AGI_FLOOR: float = 0.626
PI_FLOOR:  float = 0.637
SI_FLOOR:  float = 0.597

# Backwards-compat re-exports for any external readers; deprecated and unused
# in the active pipeline. Kept until referencing imports are confirmed gone.
DIM_FLOOR: float = AGI_FLOOR
DIM_MARGIN: float = 0.0  # winner-take-all margin removed; dims are independent


# ── Sentiment anchors (persisted on findings for downstream review) ──────────
# Two single-word polarity anchors per side. Encoded as plain words so cosine
# against a candidate focus_text reflects the lexical polarity of the
# surrounding evaluative content, independent of syntactic role.
# `match()` returns `anchor_neg_sim` / `anchor_pos_sim` which are stored on
# findings for downstream review and future calibration.
# `DIM_ANCHOR_VETO_MARGIN` is defined here for reference; a veto that blocked
# AGI when anchor_diff exceeded this threshold was tried and removed because
# anchor margin tracks sentence-level polarity, not verb argument structure
# (unaccusative `suffered` had anchor_diff=+0.026, well below 0.05, while
# volitional `feared` had anchor_diff=+0.061, above 0.05 — the opposite of
# what the veto intended). The constant is kept in case the veto is revisited.
DIM_ANCHOR_NEG = ["bad", "negative"]
DIM_ANCHOR_POS = ["good", "positive"]
DIM_ANCHOR_VETO_MARGIN: float = 0.05  # calibrated value; currently unused


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
        self.anchor_neg_vecs = encode(DIM_ANCHOR_NEG)
        self.anchor_pos_vecs = encode(DIM_ANCHOR_POS)

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
        anchor_neg_sim = float(np.max(self.anchor_neg_vecs @ vec))
        anchor_pos_sim = float(np.max(self.anchor_pos_vecs @ vec))

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
            "anchor_neg_sim": round(anchor_neg_sim, 4),
            "anchor_pos_sim": round(anchor_pos_sim, 4),
        }
