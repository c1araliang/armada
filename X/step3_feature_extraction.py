"""
STEP 3: Feature Extraction — Hybrid SRL + Prototype Role Labeling
Extracts AgI, PI, SI per target token via SRL (step3_semantic_roles) with
spaCy dependency fallback. negAttI / posAttI are assigned by the prototype
embedding matcher (step3_attitudinal_prototypes), not rule-based lookups.
"""

from lexicons import (
    TARGET_TOKENS, CONTRAST_TOKENS, SUBJECTIVE_VERBS,
    INANIMATE_NOUNS, INANIMATE_ENTITY_TYPES,
    resolve_group_token,
)
from step3_semantic_roles import PATIENT_LABELS


_SRL_ROLE_LABELER = None
_ATTITUDE_MATCHER = None


def set_srl_role_labeler(labeler) -> None:
    global _SRL_ROLE_LABELER
    _SRL_ROLE_LABELER = labeler


def set_attitude_matcher(matcher) -> None:
    global _ATTITUDE_MATCHER
    _ATTITUDE_MATCHER = matcher


def is_target(token, doc=None):
    if doc is None:
        return None
    resolved = resolve_group_token(token, doc)
    return resolved[0] if resolved else None


def _is_inanimate_noun(token) -> bool:
    """Guard: block role inheritance when head noun is clearly inanimate."""
    if token.ent_type_ in INANIMATE_ENTITY_TYPES:
        return True
    return token.lemma_.lower() in INANIMATE_NOUNS


def _resolve_role(token, doc):
    """Returns (governing_head, effective_dep, effective_noun)."""
    current = token

    if current.dep_ in ("amod", "compound", "appos", "flat", "npadvmod"):
        head_noun = current.head
        if head_noun.pos_ == "VERB" and head_noun.dep_ in ("amod", "compound"):
            head_noun = head_noun.head
        if _is_inanimate_noun(head_noun):
            return head_noun.head, "no_inherit", current
        current = head_noun

    if current.dep_ == "pobj" and current.head.text.lower() == "of":
        prep_of = current.head
        grandparent = prep_of.head
        if grandparent.dep_ in ("nsubj", "nsubjpass", "dobj"):
            return grandparent.head, grandparent.dep_, current

    dep = current.dep_
    head = current.head

    if dep == "conj":
        dep = current.head.dep_
        head = current.head.head

    if head.pos_ == "AUX":
        resolved = False
        for child in head.children:
            if child.pos_ == "VERB" and child.dep_ in ("ROOT", "ccomp", "xcomp", "conj", "advcl"):
                head = child
                resolved = True
                break
            if child.dep_ == "acomp":
                head = child
                if child.pos_ == "ADJ":
                    # Plain adjectival complements (e.g. "were desperate/proud")
                    # should stay subject-linked, not be coerced into passive PI.
                    dep = current.dep_
                else:
                    dep = "nsubjpass"
                resolved = True
                break
        if not resolved and any(c.dep_ == "auxpass" for c in head.children):
            dep = "nsubjpass"

    return head, dep, current


def _has_same_group_head(token, doc) -> bool:
    if token.dep_ not in ("amod", "compound", "appos", "flat", "npadvmod"):
        return False
    resolved = resolve_group_token(token, doc)
    if resolved is None:
        return False
    head = token.head
    head_resolved = resolve_group_token(head, doc)
    if head_resolved == resolved:
        return True
    if head.head != head:
        grand_resolved = resolve_group_token(head.head, doc)
        if grand_resolved == resolved:
            return True
    return False


def _group_span_indices(token, doc) -> set[int]:
    """
    Token indices that belong to the same local group mention.

    This lets adjective-like mentions such as "Korean people" inherit SRL labels
    from the human head noun when the SRL model attaches the argument label to
    the head rather than the modifier.
    """
    span = {token.i}
    if token.dep_ in ("amod", "compound", "appos", "flat", "npadvmod"):
        head = token.head
        if not _is_inanimate_noun(head):
            span.add(head.i)
        if head.head != head:
            grand = head.head
            grand_resolved = resolve_group_token(grand, doc)
            if grand_resolved:
                span.add(grand.i)
    return span


def _collect_srl_roles(doc) -> dict[int, dict]:
    """
    Aggregate SRL-derived roles for each token index in the doc.
    """
    token_roles: dict[int, dict] = {}
    if _SRL_ROLE_LABELER is None:
        return token_roles

    predicate_hints = set()
    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is None:
            continue
        head_verb, _, _ = _resolve_role(token, doc)
        if head_verb is not None and head_verb.i >= 0:
            predicate_hints.add(head_verb.i)
        predicate_hints.add(token.head.i)
        for child in token.children:
            predicate_hints.add(child.i)
        if token.head != token and token.head.head != token.head:
            predicate_hints.add(token.head.head.i)

    frames = _SRL_ROLE_LABELER.annotate(doc, predicate_indices=predicate_hints or None)
    if not frames:
        return token_roles

    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is None:
            continue

        span_indices = _group_span_indices(token, doc)
        roles = set()
        matched_predicates = []
        for frame in frames:
            labels = {frame["labels"].get(idx, "_") for idx in span_indices}
            if "ARG0" in labels:
                roles.add("AgI")
                matched_predicates.append(frame["predicate_lemma"])
                if frame["predicate_lemma"] in SUBJECTIVE_VERBS:
                    roles.add("SI")
            if labels & PATIENT_LABELS:
                roles.add("PI")
                matched_predicates.append(frame["predicate_lemma"])

        token_roles[token.i] = {
            "roles": roles,
            "predicates": sorted(set(matched_predicates)),
        }
    return token_roles


def extract_roles(doc) -> list[dict]:
    """Extract AgI, PI, SI, negAttI, posAttI for each target/contrast token."""
    findings = []
    resolved_mentions = []
    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is not None:
            resolved_mentions.append((token, resolved))
    if not resolved_mentions:
        return findings

    srl_roles = _collect_srl_roles(doc)

    mwe_children = {
        token.i for token, _ in resolved_mentions
        if _has_same_group_head(token, doc)
    }

    for token, resolved in resolved_mentions:
        group, canonical_lemma = resolved

        roles = []
        head_verb, effective_dep, effective_noun = _resolve_role(token, doc)
        srl_info = srl_roles.get(token.i, {"roles": set(), "predicates": []})
        srl_hit = bool(srl_info["roles"])

        if "AgI" in srl_info["roles"]:
            roles.append("AgI")
        elif effective_dep == "nsubj" and head_verb.pos_ == "VERB":
            roles.append("AgI")
        elif effective_dep == "pobj" and effective_noun.head.text.lower() == "by":
            roles.append("AgI")

        if "PI" in srl_info["roles"]:
            if "PI" not in roles:
                roles.append("PI")
        else:
            if effective_dep == "dobj":
                roles.append("PI")
            if effective_dep == "nsubjpass":
                roles.append("PI")
            if effective_dep == "pcomp" and any(c.dep_ == "auxpass" for c in head_verb.children):
                roles.append("PI")

        if "SI" in srl_info["roles"]:
            if "SI" not in roles:
                roles.append("SI")
        elif effective_dep == "nsubj" and head_verb.lemma_.lower() in SUBJECTIVE_VERBS:
            roles.append("SI")

        # Object-of-belief via relative clause (PI)
        for child in token.children:
            if child.dep_ == "relcl" and child.lemma_.lower() in SUBJECTIVE_VERBS:
                relcl_subjs = [c for c in child.children if c.dep_ == "nsubj"]
                if relcl_subjs and all(is_target(s, doc) is None for s in relcl_subjs):
                    if "PI" not in roles:
                        roles.append("PI")

        atti_info = {
            "label": None,
            "focus_text": "",
            "neg_sim": 0.0,
            "pos_sim": 0.0,
        }
        if _ATTITUDE_MATCHER is not None:
            atti_info = _ATTITUDE_MATCHER.match(
                token,
                doc,
                head_verb=head_verb,
                span_indices=_group_span_indices(token, doc),
            )
            if atti_info["label"] == "negAttI":
                roles.append("negAttI")
            elif atti_info["label"] == "posAttI":
                roles.append("posAttI")

        agi       = 1 if "AgI"      in roles else 0
        pi        = 1 if "PI"       in roles else 0
        si        = 1 if "SI"       in roles else 0
        neg_atti  = 1 if "negAttI"  in roles else 0
        pos_atti  = 1 if "posAttI"  in roles else 0

        findings.append({
            "token": token.text,
            "token_i": token.i,
            "lemma": canonical_lemma,
            "group": group,
            "dep": token.dep_,
            "is_mwe_child": token.i in mwe_children,
            "effective_dep": effective_dep,
            "head_verb": head_verb.text if head_verb else "null",
            "head_verb_lemma": head_verb.lemma_.lower() if head_verb else "null",
            "srl_predicates": ", ".join(srl_info["predicates"]) if srl_info["predicates"] else "null",
            "role_source": "srl+dep" if srl_hit else "dep",
            "atti_source": "prototype" if _ATTITUDE_MATCHER is not None else "none",
            "atti_neg_sim": atti_info["neg_sim"],
            "atti_pos_sim": atti_info["pos_sim"],
            "roles": roles if roles else ["present"],
            "agi": agi, "pi": pi, "si": si,
            "neg_atti": neg_atti, "pos_atti": pos_atti,
        })

    _resolve_anaphora(doc, findings)
    _resolve_adverbial_passive(doc, findings)

    return findings


def _resolve_adverbial_passive(doc, findings: list[dict]):
    """Propagate PI from adverbial passives to main-clause target subject."""
    for token in doc:
        if token.dep_ != "pcomp":
            continue
        if not any(c.dep_ == "auxpass" for c in token.children):
            continue
        ancestor = token.head
        while ancestor.dep_ in ("prep", "mark", "advcl") and ancestor.head != ancestor:
            ancestor = ancestor.head
        if ancestor.pos_ != "VERB":
            continue
        main_subj = None
        for c in ancestor.children:
            if c.dep_ == "nsubj":
                main_subj = c
                break
        if main_subj is None:
            continue
        for f in findings:
            if f["token_i"] == main_subj.i or f.get("head_verb") == ancestor.text:
                if f["pi"] == 0:
                    f["pi"] = 1
                    if "PI" not in f["roles"]:
                        f["roles"].append("PI")


_ANAPHORIC_PRONOUNS = {"they", "them", "their", "themselves",
                       "he", "him", "his", "she", "her", "himself", "herself"}


def _resolve_anaphora(doc, findings: list[dict]):
    """Transfer pronoun roles to nearest preceding target (within-sentence)."""
    for token in doc:
        if token.text.lower() not in _ANAPHORIC_PRONOUNS:
            continue
        preceding = [fi for fi in findings if fi["token_i"] < token.i]
        if not preceding:
            continue
        antecedent = preceding[-1]

        dep = token.dep_
        head = token.head

        if dep == "nsubj" and head.pos_ == "VERB":
            if antecedent["agi"] == 0:
                antecedent["agi"] = 1
                if "AgI" not in antecedent["roles"]:
                    antecedent["roles"].append("AgI")
            if head.lemma_.lower() in SUBJECTIVE_VERBS:
                if antecedent["si"] == 0:
                    antecedent["si"] = 1
                    if "SI" not in antecedent["roles"]:
                        antecedent["roles"].append("SI")

        if dep == "dobj":
            if antecedent["pi"] == 0:
                antecedent["pi"] = 1
                if "PI" not in antecedent["roles"]:
                    antecedent["roles"].append("PI")

        if dep == "nsubjpass":
            if antecedent["pi"] == 0:
                antecedent["pi"] = 1
                if "PI" not in antecedent["roles"]:
                    antecedent["roles"].append("PI")


def extract_all(processed_data: list[dict]) -> list[dict]:
    results = []
    for i, item in enumerate(processed_data):
        roles = extract_roles(item["doc"])
        results.append({
            "sentence_id": i,
            "category": item["category"],
            "text": item["cleaned_text"],
            "findings": roles,
        })
        if _SRL_ROLE_LABELER is not None and (i + 1) % 100 == 0:
            print(f"  SRL extraction: {i + 1}/{len(processed_data)} sentences")
    return results
