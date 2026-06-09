"""Target-bound group mention and frame-binding helpers."""

from dataclasses import dataclass
from collections import defaultdict

from lexicons import (
    POLITICAL_GROUP_TOKENS,
    TARGET_TOKENS,
    CONTRAST_TOKENS,
    resolve_group_token,
)


_MODIFIER_DEPS = {"amod", "compound", "appos", "flat", "npadvmod"}
_QUOTE_CHARS = {'"', "'", "“", "”", "‘", "’"}
_NEGATION_TERMS = {"not", "n't", "never", "no", "without"}
_CONTRAST_TERMS = {
    "but", "however", "although", "though", "whereas", "while",
    "despite", "unlike", "yet", "nevertheless", "nonetheless",
}
_REPORTING_VERBS = {
    "say", "claim", "argue", "assert", "allege", "accuse", "report",
    "write", "tell", "warn", "call", "describe", "portray", "depict",
    "blame", "praise", "condemn", "criticize", "vilify",
}
_CORRECTION_TERMS = {
    "false", "falsely", "deny", "denial", "denied", "debunk", "refute",
    "reject", "myth", "hoax", "misleading",
}
_BINDING_BOUNDARY_TERMS = _CONTRAST_TERMS | {";", ":"}
_MAX_BINDING_DISTANCE = 8


@dataclass(frozen=True)
class GroupMention:
    token_i: int
    start_i: int
    end_i: int
    text: str
    lemma: str
    group_type: str
    is_mwe_child: bool
    mwe_type: str
    flags: tuple[str, ...]


def group_type_for_lemma(lemma: str) -> str:
    """Reporting type for a canonical lemma."""
    if lemma in POLITICAL_GROUP_TOKENS:
        return "political"
    if lemma in TARGET_TOKENS:
        return "minority"
    if lemma in CONTRAST_TOKENS:
        return "dominant"
    return "unknown"


def _has_same_group_head(token, doc) -> bool:
    if token.dep_ not in _MODIFIER_DEPS:
        return False
    resolved = resolve_group_token(token, doc)
    if resolved is None:
        return False
    head_resolved = resolve_group_token(token.head, doc)
    if head_resolved == resolved:
        return True
    if token.head.head != token.head:
        grand_resolved = resolve_group_token(token.head.head, doc)
        if grand_resolved == resolved:
            return True
    return False


def _mention_span(token) -> tuple[int, int]:
    indices = {token.i}
    if token.dep_ in _MODIFIER_DEPS:
        indices.add(token.head.i)
    for child in token.children:
        if child.dep_ in _MODIFIER_DEPS:
            child_resolved = resolve_group_token(child, token.doc)
            if child_resolved:
                indices.add(child.i)
    return min(indices), max(indices) + 1


def iter_group_mentions(doc, include_mwe_children: bool = True):
    """Yield resolved group mentions with MWE metadata."""
    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is None:
            continue

        group_type, canonical = resolved
        is_mwe_child = _has_same_group_head(token, doc)
        if is_mwe_child and not include_mwe_children:
            continue

        start_i, end_i = _mention_span(token)
        flags = set()
        if len({r[1] for r in (resolve_group_token(t, doc) for t in doc) if r}) > 1:
            flags.add("multi_group_sentence")
        if token.dep_ in _MODIFIER_DEPS:
            flags.add("modifier_group")
        if "-" in canonical:
            mwe_type = "compound_group"
        elif token.dep_ in _MODIFIER_DEPS:
            mwe_type = "modifier_head"
        else:
            mwe_type = "simple"

        yield GroupMention(
            token_i=token.i,
            start_i=start_i,
            end_i=end_i,
            text=doc[start_i:end_i].text,
            lemma=canonical,
            group_type=group_type,
            is_mwe_child=is_mwe_child,
            mwe_type=mwe_type,
            flags=tuple(sorted(flags)),
        )


def iter_primary_group_mentions(doc):
    """Yield non-duplicative group anchors for sentence-level metrics."""
    yield from iter_group_mentions(doc, include_mwe_children=False)


def sentence_scope_flags(doc) -> set[str]:
    """Cheap discourse/scope flags used for review routing."""
    flags = set()
    lowered = {t.text.lower() for t in doc}
    lemmas = {t.lemma_.lower() for t in doc}

    if lowered & _QUOTE_CHARS:
        flags.add("quotation")
    if lowered & _NEGATION_TERMS or any(t.dep_ == "neg" for t in doc):
        flags.add("negation")
    if lemmas & _REPORTING_VERBS:
        flags.add("reported_speech")
    if lowered & _CONTRAST_TERMS or lemmas & _CONTRAST_TERMS:
        flags.add("contrast")
    if lowered & _CORRECTION_TERMS or lemmas & _CORRECTION_TERMS:
        flags.add("correction_denial")

    primary_lemmas = {m.lemma for m in iter_primary_group_mentions(doc)}
    if len(primary_lemmas) > 1:
        flags.add("multi_group_sentence")

    return flags


def _frame_match(token, frame_terms: set[str]) -> str | None:
    lemma = token.lemma_.lower()
    text = token.text.lower()
    if lemma in frame_terms:
        return lemma
    if text in frame_terms:
        return text
    return None


def _governing_predicate(token):
    current = token
    for _ in range(6):
        if current.pos_ in {"VERB", "AUX"}:
            return current
        if current.head == current:
            return current if current.pos_ in {"VERB", "AUX"} else None
        current = current.head
    return None


def _has_boundary_between(doc, left_i: int, right_i: int) -> bool:
    lo, hi = sorted((left_i, right_i))
    for token in doc[lo + 1:hi]:
        if token.text.lower() in _BINDING_BOUNDARY_TERMS:
            return True
        if token.lemma_.lower() in _BINDING_BOUNDARY_TERMS:
            return True
    return False


def _binding_relation(mention: GroupMention, frame_token, doc) -> tuple[str | None, int]:
    mention_token = doc[mention.token_i]
    distance = abs(mention.token_i - frame_token.i)

    if mention_token.head == frame_token or frame_token.head == mention_token:
        return "direct_dependency", 0
    if mention_token.is_ancestor(frame_token) or frame_token.is_ancestor(mention_token):
        return "ancestor_dependency", 1

    mention_pred = _governing_predicate(mention_token)
    frame_pred = _governing_predicate(frame_token)
    if mention_pred is not None and mention_pred == frame_pred:
        return "shared_predicate", 2

    if distance <= _MAX_BINDING_DISTANCE and not _has_boundary_between(doc, mention.token_i, frame_token.i):
        return "bounded_proximity", distance + 3

    return None, 999


def _frame_scope_flags(frame_token, doc) -> set[str]:
    flags = set()
    for child in frame_token.children:
        if child.dep_ == "neg" or child.text.lower() in _NEGATION_TERMS:
            flags.add("frame_negated")

    left = max(0, frame_token.i - 4)
    local_terms = {t.text.lower() for t in doc[left:frame_token.i]}
    local_lemmas = {t.lemma_.lower() for t in doc[left:frame_token.i]}
    if local_terms & _NEGATION_TERMS or local_lemmas & _NEGATION_TERMS:
        flags.add("frame_negated")

    wide_left = max(0, frame_token.i - 8)
    wide_right = min(len(doc), frame_token.i + 9)
    wide_terms = {t.text.lower() for t in doc[wide_left:wide_right]}
    wide_lemmas = {t.lemma_.lower() for t in doc[wide_left:wide_right]}
    if wide_terms & _CORRECTION_TERMS or wide_lemmas & _CORRECTION_TERMS:
        flags.add("correction_denial")

    return flags


def bind_frame_terms_to_mentions(doc, neg_frames: set[str], pos_frames: set[str]) -> list[dict]:
    """Bind F-/F+ frame terms to the nearest plausible group mention."""
    mentions = list(iter_primary_group_mentions(doc))
    if not mentions:
        return []

    frame_terms = neg_frames | pos_frames
    frame_tokens = []
    for token in doc:
        term = _frame_match(token, frame_terms)
        if term:
            sign = -1 if term in neg_frames else 1
            frame_tokens.append((token, term, sign))

    bindings = []
    for frame_token, term, sign in frame_tokens:
        candidates = []
        for mention in mentions:
            relation, score = _binding_relation(mention, frame_token, doc)
            if relation:
                candidates.append((score, mention, relation))

        if not candidates:
            bindings.append({
                "lemma": None,
                "term": term,
                "sign": sign,
                "token_i": frame_token.i,
                "relation": "unbound",
                "blocked": False,
                "flags": ("unbound_frame_term",),
            })
            continue

        min_score = min(c[0] for c in candidates)
        nearest = [c for c in candidates if c[0] == min_score]
        scope_flags = _frame_scope_flags(frame_token, doc)
        blocked = bool({"frame_negated", "correction_denial"} & scope_flags)
        if len({c[1].lemma for c in nearest}) > 1:
            bindings.append({
                "lemma": None,
                "term": term,
                "sign": sign,
                "token_i": frame_token.i,
                "relation": "ambiguous",
                "blocked": blocked,
                "flags": tuple(sorted(scope_flags | {"ambiguous_frame_target"})),
            })
            continue

        _, mention, relation = nearest[0]
        bindings.append({
            "lemma": mention.lemma,
            "term": term,
            "sign": sign,
            "token_i": frame_token.i,
            "relation": relation,
            "blocked": blocked,
            "flags": tuple(sorted(scope_flags)),
        })

    return bindings


def bound_frame_summary(doc, neg_frames: set[str], pos_frames: set[str]) -> dict:
    """Sentence-level bound-frame summary keyed by group lemma."""
    summary = defaultdict(lambda: {"neg": set(), "pos": set(), "flags": set()})
    all_flags = set(sentence_scope_flags(doc))

    for binding in bind_frame_terms_to_mentions(doc, neg_frames, pos_frames):
        flags = set(binding["flags"])
        all_flags |= flags
        lemma = binding["lemma"]
        if lemma is None:
            continue
        summary[lemma]["flags"].update(flags)
        if binding["blocked"]:
            summary[lemma]["flags"].add("scope_blocked_frame")
            continue
        if binding["sign"] < 0:
            summary[lemma]["neg"].add(binding["term"])
        else:
            summary[lemma]["pos"].add(binding["term"])

    return {
        "by_lemma": summary,
        "flags": all_flags,
    }
