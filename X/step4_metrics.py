"""
STEP 4: Metrics — Sentence-level discourse association, signed association, WEAT mock,
and aggregation.

Primary association mode:
  - Sentence-level target/collocate association
  - Non-adjacent only: pairs count only when distance > 1 within a sentence
  - LLR + LogDice scoring for collocate discovery

Signed association reuses the classified frame taxonomy to summarize whether a
sentence is linked more strongly to positive or negative frame terms.
"""

import math
from collections import Counter, defaultdict
from lexicons import (
    TARGET_TOKENS, CONTRAST_TOKENS,
    ALL_FRAME_TERMS, FRAME_SIGN,
    resolve_group_token,
)

_ALL_GROUPS = TARGET_TOKENS | CONTRAST_TOKENS
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
_NON_ADJACENT_MIN_DISTANCE = 2


def _iter_resolved_anchors(doc):
    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is None:
            continue
        if token.dep_ in ("amod", "compound", "appos", "flat", "npadvmod"):
            head_resolved = resolve_group_token(token.head, doc)
            if head_resolved == resolved:
                continue
            if token.head.head != token.head:
                grand_resolved = resolve_group_token(token.head.head, doc)
                if grand_resolved == resolved:
                    continue
        yield token.i, resolved[1]


# ── Sentence-level discourse association (Step 0: collocate discovery) ──

def _g_stat_term(observed: float, expected: float) -> float:
    if observed <= 0 or expected <= 0:
        return 0.0
    return observed * math.log(observed / expected)


def _compute_llr(k11: int, k12: int, k21: int, k22: int) -> float:
    total = k11 + k12 + k21 + k22
    if total <= 0:
        return 0.0

    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22

    e11 = row1 * col1 / total
    e12 = row1 * col2 / total
    e21 = row2 * col1 / total
    e22 = row2 * col2 / total

    g2 = 2.0 * (
        _g_stat_term(k11, e11)
        + _g_stat_term(k12, e12)
        + _g_stat_term(k21, e21)
        + _g_stat_term(k22, e22)
    )
    return max(0.0, g2)


def _compute_logdice(pair_count: int, target_count: int, collocate_count: int) -> float:
    denom = target_count + collocate_count
    if pair_count <= 0 or denom <= 0:
        return 0.0
    return 14.0 + math.log2((2.0 * pair_count) / denom)


def build_sentence_associations(
    processed_data: list[dict],
    min_count: int = 1,
    min_distance: int = _NON_ADJACENT_MIN_DISTANCE,
) -> dict:
    """
    Count sentence-level (target, content-word) associations.

    A pair counts once per sentence if the sentence contains both the target lemma
    and the collocate lemma with at least one token pair whose linear distance is
    greater than 1.
    """
    pair_counts = Counter()
    target_counts = Counter()
    collocate_counts = Counter()
    total_sentences = 0

    for item in processed_data:
        doc = item["doc"]
        tokens = item["tokens"]
        total_sentences += 1

        anchor_positions = defaultdict(list)
        for token_i, anchor_lemma in _iter_resolved_anchors(doc):
            anchor_positions[anchor_lemma].append(token_i)

        collocate_positions = defaultdict(list)
        for tok in tokens:
            lemma = tok["lemma"]
            if tok["pos"] in _CONTENT_POS and lemma not in _ALL_GROUPS:
                collocate_positions[lemma].append(tok["i"])

        for anchor_lemma in anchor_positions:
            target_counts[anchor_lemma] += 1

        for collocate_lemma in collocate_positions:
            collocate_counts[collocate_lemma] += 1

        for anchor_lemma, anchor_idxs in anchor_positions.items():
            for collocate_lemma, collocate_idxs in collocate_positions.items():
                if any(abs(i - j) >= min_distance for i in anchor_idxs for j in collocate_idxs):
                    pair_counts[(anchor_lemma, collocate_lemma)] += 1

    pair_counts = Counter({
        key: value for key, value in pair_counts.items()
        if value >= min_count
    })

    return {
        "pair_counts": pair_counts,
        "target_counts": target_counts,
        "collocate_counts": collocate_counts,
        "total_sentences": total_sentences,
        "min_distance": min_distance,
    }


def compute_association_scores(cooc: dict) -> dict[tuple, dict[str, float]]:
    """Compute LLR and LogDice for all discovered sentence-level pairs."""
    scores: dict[tuple, dict[str, float]] = {}
    total = cooc["total_sentences"]
    if total <= 0:
        return scores

    for (target, collocate), pair_count in cooc["pair_counts"].items():
        target_count = cooc["target_counts"].get(target, 0)
        collocate_count = cooc["collocate_counts"].get(collocate, 0)

        k11 = pair_count
        k12 = max(0, target_count - pair_count)
        k21 = max(0, collocate_count - pair_count)
        k22 = max(0, total - k11 - k12 - k21)

        llr = _compute_llr(k11, k12, k21, k22)
        logdice = _compute_logdice(pair_count, target_count, collocate_count)
        scores[(target, collocate)] = {
            "pair_count": pair_count,
            "target_count": target_count,
            "collocate_count": collocate_count,
            "llr": round(llr, 4),
            "logdice": round(logdice, 4),
        }

    return scores


# ── Signed association (Step 5: applies classified frame taxonomy) ──

def compute_signed_association(
    association_scores: dict[tuple, dict[str, float]],
    score_key: str = "llr",
) -> dict[tuple, float]:
    """Apply frame-type sign to the selected association score."""
    signed = {}
    for (target, collocate), score_info in association_scores.items():
        if collocate in FRAME_SIGN:
            signed[(target, collocate)] = FRAME_SIGN[collocate] * score_info.get(score_key, 0.0)
    return signed


# ── WEAT utility (cosine similarity, used by run_pipeline._compute_weat) ──

def cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ── Sentence-level aggregation ──

def aggregate_sentence_metrics(
    extracted_data: list[dict],
    signed_association: dict[tuple, float],
    processed_data: list[dict],
) -> list[dict]:
    """Per-sentence summary row with indices + signed association."""
    rows = []

    for i, item in enumerate(extracted_data):
        targets_found = [f["token"] for f in item["findings"]] if item["findings"] else []

        doc = processed_data[i]["doc"]
        sentence_lemmas = [t["lemma"] for t in processed_data[i]["tokens"]]
        anchor_lemmas = [lemma for _, lemma in _iter_resolved_anchors(doc)]
        frame_lemmas = [l for l in sentence_lemmas if l in ALL_FRAME_TERMS]

        association_values = []
        for tl in anchor_lemmas:
            for fl in frame_lemmas:
                val = signed_association.get((tl, fl), 0.0)
                if val != 0.0:
                    association_values.append(val)
        net_association = max(association_values, key=abs) if association_values else 0.0

        frame_labels = []
        for fl in sorted(set(frame_lemmas)):
            sign = FRAME_SIGN.get(fl, 0)
            tag = "(-)" if sign < 0 else "(+)"
            frame_labels.append(f"{fl}{tag}")

        non_mwe = [f for f in item.get("findings", []) if not f.get("is_mwe_child")]
        sent_agi      = sum(f["agi"]      for f in non_mwe)
        sent_pi       = sum(f["pi"]       for f in non_mwe)
        sent_si       = sum(f["si"]       for f in non_mwe)
        sent_neg_atti = sum(f["neg_atti"] for f in non_mwe)
        sent_pos_atti = sum(f["pos_atti"] for f in non_mwe)

        rows.append({
            "sentence_id": i,
            "category": item["category"],
            "text": item["text"],
            "targets": ", ".join(targets_found) if targets_found else "null",
            "agi": sent_agi, "pi": sent_pi, "si": sent_si,
            "neg_atti": sent_neg_atti, "pos_atti": sent_pos_atti,
            "frames": ", ".join(frame_labels) if frame_labels else "null",
            "association": round(net_association, 3),
        })

    return rows


# ── Group-level aggregation ──

def compute_group_indices(extracted_data: list[dict]) -> dict:
    """Proportionalized indices per lemma and per category."""
    _KEYS = ("agi", "pi", "si", "neg_atti", "pos_atti")

    lemma_stats = defaultdict(lambda: {k: 0 for k in ("total", *_KEYS)} | {"type": ""})
    cat_stats = defaultdict(lambda: {k: 0 for k in ("total", *_KEYS)})

    for item in extracted_data:
        for f in item.get("findings", []):
            lemma, cat = f["lemma"], f["group"]

            lemma_stats[lemma]["total"] += 1
            lemma_stats[lemma]["type"] = cat
            for k in _KEYS:
                lemma_stats[lemma][k] += f[k]

            if not f.get("is_mwe_child"):
                cat_stats[cat]["total"] += 1
                for k in _KEYS:
                    cat_stats[cat][k] += f[k]

    def _proportionalize(stats, label_key, count_key="total"):
        results = []
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
            n = s["total"]
            entry = {label_key: name, "total": n}
            if "type" in s:
                entry["type"] = s["type"]
            for k in _KEYS:
                entry[k.upper() if k in ("agi", "pi", "si") else k] = (
                    round(s[k] / n, 3) if n > 0 else 0)
            results.append(entry)
        return results

    return {
        "lemmas": _proportionalize(lemma_stats, "lemma"),
        "categories": _proportionalize(cat_stats, "category"),
    }
