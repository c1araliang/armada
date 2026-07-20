"""
STEP 3: Feature Extraction — Hybrid SRL + Prototype Role Labeling
Extracts AgI, PI, SI per target token via SRL (step3_semantic_roles) with
spaCy dependency fallback and dimensional prototype scoring.

Primary AgI/PI/SI gate: target-bound dimensional prototype similarity via
_ATTITUDE_MATCHER (agi_sim/pi_sim/si_sim). This replaces verb-set membership
(SUBJECTIVE_VERBS, LOW_AGENCY_VERBS, AFFECTEDNESS_VERBS) as the primary gate.
Verb sets are retained only for anaphora resolution (residual path).

Local negAttI / posAttI remain diagnostic; reported netAttI is from frame
association downstream.
"""

from lexicons import (
    TARGET_TOKENS, CONTRAST_TOKENS,
    INANIMATE_NOUNS, INANIMATE_ENTITY_TYPES,
    resolve_group_token,
)
from step3_semantic_roles import PATIENT_LABELS
from step3_attitudinal_prototypes import AGI_FLOOR, PI_FLOOR, SI_FLOOR
from group_mentions import sentence_scope_flags


_SRL_ROLE_LABELER = None
_ATTITUDE_MATCHER = None


def _dim_passes(score: float, floor: float) -> bool:
    """True when score meets its independent floor.

    Each dimension is judged on its own absolute threshold; there is no
    winner-take-all margin. A single target can pass multiple dims (a group
    that acts AND is acted upon AND has mental states is plausible and
    expected — these are independent attribution layers, not exclusive
    syntactic roles).
    """
    return score >= floor


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

    if current.dep_ == "pobj":
        QUANTIFIER_NOUNS = {
            "couple", "group", "majority", "minority", "number", "percentage", "portion", 
            "part", "lot", "fraction", "half", "third", "quarter", "percent", "rest"
        }
        ancestor = current.head
        path = [current]
        while ancestor is not None and ancestor != ancestor.head:
            if (ancestor.pos_ in ("ADP", "NUM", "DET", "PRON")
                or ancestor.lemma_.lower() in QUANTIFIER_NOUNS
                or ancestor.dep_ in ("prep", "quantmod", "nummod", "det", "pobj")):
                path.append(ancestor)
                ancestor = ancestor.head
            else:
                break
        if ancestor is not None and len(path) > 1:
            connecting_token = path[-1]
            if connecting_token.dep_ in ("nsubj", "nsubjpass", "dobj"):
                return ancestor, connecting_token.dep_, current

    dep = current.dep_
    head = current.head

    if dep == "pobj" and head.pos_ == "ADP":
        # Gerund-pcomp unwrapping: "get away with abusing X".
        # spaCy sometimes attaches X as pobj of the preposition rather than
        # dobj of the gerund (pcomp VERB). Two sub-cases observed:
        #  (a) abusing = pcomp of prep: X is pobj of the same prep → reattach
        #  (b) abusing = amod of X (VERB modifier of the pobj noun): X is
        #      the semantic dobj of the verbal gerund, treat as dobj
        pcomp_verb = next(
            (c for c in head.children if c.dep_ == "pcomp" and c.pos_ == "VERB"),
            None,
        )
        if pcomp_verb is not None:
            dep = "dobj"
            head = pcomp_verb
        else:
            amod_verb = next(
                (c for c in current.children if c.dep_ == "amod" and c.pos_ == "VERB"),
                None,
            )
            if amod_verb is not None:
                dep = "dobj"
                head = amod_verb

    if dep == "conj":
        visited = {current.i}
        while dep == "conj" and head is not None and head != current:
            if head.i in visited:
                break
            visited.add(head.i)
            dep = head.dep_
            head = head.head

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

    if head is not None and head.pos_ in ("VERB", "AUX", "NOUN", "ADJ"):
        xcomp_verb = next((c for c in head.children if c.dep_ == "xcomp" and c.pos_ == "VERB"), None)
        if xcomp_verb is not None:
            head = xcomp_verb
        else:
            preps = [c for c in head.children if c.dep_ == "prep"]
            for c in head.children:
                if c.dep_ in ("prt", "advmod"):
                    preps.extend([gc for gc in c.children if gc.dep_ == "prep"])
            pcomp_verb = None
            for prep in preps:
                pcomp = next((c for c in prep.children if c.dep_ == "pcomp" and c.pos_ == "VERB"), None)
                if pcomp is not None:
                    pcomp_verb = pcomp
                    break
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    amod_verb = next((c for c in pobj.children if c.dep_ == "amod" and c.pos_ == "VERB"), None)
                    if amod_verb is not None:
                        pcomp_verb = amod_verb
                        break
            if pcomp_verb is not None:
                head = pcomp_verb

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


# Legacy verb-set helper functions deleted as they are no longer needed
# because both primary and anaphora paths use relative-margin prototype scoring.


def _collect_srl_roles(doc) -> dict[int, dict]:
    """
    Aggregate raw SRL labels for each token index in the doc.

    Returns ARG0→AgI and PATIENT_LABELS→PI as unfiltered structural evidence.
    Verb-set filtering (agency suppression, SI, affectedness) has been moved
    to extract_roles() where dimensional prototype scores serve as the primary
    gate. The field 'suppressed_agency_predicates' is kept for diagnostics but
    is no longer populated here.
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
            predicate_lemma = frame["predicate_lemma"]
            labels = {frame["labels"].get(idx, "_") for idx in span_indices}
            if "ARG0" in labels:
                # Raw AgI signal — not filtered by verb sets here.
                # Dimensional prototype score (agi_sim) in extract_roles() decides.
                roles.add("AgI")
                matched_predicates.append(predicate_lemma)
            if labels & PATIENT_LABELS:
                # Structural patient labels: kept as primary PI evidence.
                roles.add("PI")
                matched_predicates.append(predicate_lemma)

        token_roles[token.i] = {
            "roles": roles,
            "predicates": sorted(set(matched_predicates)),
            "suppressed_agency_predicates": [],  # now handled in extract_roles()
        }
    return token_roles


def extract_roles(doc, doc_id: int | None = None) -> list[dict]:
    """Extract AgI, PI, SI and local diagnostic AttI for each group token."""
    findings = []
    resolved_mentions = []
    for token in doc:
        resolved = resolve_group_token(token, doc)
        if resolved is not None:
            resolved_mentions.append((token, resolved))
    if not resolved_mentions:
        return findings

    srl_roles = _collect_srl_roles(doc)
    doc_scope_flags = sentence_scope_flags(doc)

    mwe_children = {
        token.i for token, _ in resolved_mentions
        if _has_same_group_head(token, doc)
    }

    for token, resolved in resolved_mentions:
        group, canonical_lemma = resolved

        roles = []
        head_verb, effective_dep, effective_noun = _resolve_role(token, doc)
        srl_info = srl_roles.get(
            token.i,
            {"roles": set(), "predicates": [], "suppressed_agency_predicates": []},
        )
        srl_hit = bool(srl_info["roles"])
        subjecthood = 1 if effective_dep in ("nsubj", "nsubjpass") else 0
        role_review_flags = set(doc_scope_flags)

        # ── Dimensional prototype scores (primary gate for AgI/PI/SI) ──
        # Call early so scores are available for all role decisions below.
        # The focus_text carries [GROUP:x][PRED:verb] annotations, making
        # scoring target-conditioned rather than verb-type-conditioned.
        atti_info = {
            "label": None, "focus_text": "",
            "neg_sim": 0.0, "pos_sim": 0.0,
            "agi_sim": 0.0, "pi_sim": 0.0, "si_sim": 0.0,
        }
        if _ATTITUDE_MATCHER is not None:
            atti_info = _ATTITUDE_MATCHER.match(
                token, doc,
                head_verb=head_verb,
                span_indices=_group_span_indices(token, doc),
                key=(doc_id, token.i) if doc_id is not None else None,
            )
        agi_sim = atti_info["agi_sim"]
        pi_sim  = atti_info["pi_sim"]
        si_sim  = atti_info["si_sim"]
        anchor_neg = atti_info.get("anchor_neg_sim", 0.0)
        anchor_pos = atti_info.get("anchor_pos_sim", 0.0)
        dim_src = _ATTITUDE_MATCHER is not None

        # ── Negation-scope detection ──
        # If the governing predicate is under negation, block role assignment
        # and route to review. Mirrors frame-AttI scope blocking.
        predicate_negated = False
        if head_verb is not None:
            for child in head_verb.children:
                if child.dep_ == "neg" or child.text.lower() in ("not", "n't", "never", "no"):
                    predicate_negated = True
                    break
            if not predicate_negated:
                # Check left-adjacent negation (e.g., "did not organize")
                left_start = max(0, head_verb.i - 3)
                for i in range(left_start, head_verb.i):
                    if doc[i].text.lower() in ("not", "n't", "never", "without"):
                        predicate_negated = True
                        break

        # ── Independent dimensional gates ──
        # Each dim is checked against its own floor (no winner-take-all).
        # SRL/dep evidence is now the primary syntactic source for AgI and
        # PI; the prototype gate is only required where syntactic evidence
        # is weaker (nsubj-only AgI, affected-subject PI, all SI).
        agi_passes = _dim_passes(agi_sim, AGI_FLOOR)
        pi_passes  = _dim_passes(pi_sim,  PI_FLOOR)
        si_passes  = _dim_passes(si_sim,  SI_FLOOR)

        if predicate_negated:
            # Negation blocks all role assignment from this predicate.
            role_review_flags.add("negation_scope_blocked")
        else:
            # ── AgI ──
            # SRL ARG0 is structural evidence of agent role, but SRL labels
            # unaccusative subjects (e.g. `refugees suffered`) as ARG0 even
            # though they are semantic patients. AGI prototype confirmation
            # (`agi_passes`) is the gate: unaccusative patient verbs land
            # below the AGI prototype floor because their focus-text geometry
            # sits closer to PI prototypes. The earlier sentiment-anchor veto
            # was an unreliable proxy — anchor margin tracks sentence-level
            # sentiment, not verb argument structure, so it both passed clear
            # unaccusatives (`suffered` anchor=+0.026 < 0.05) and blocked
            # clear volitional acts (`feared` anchor=+0.061 > 0.05).
            # PI is independent — even when AGI is rejected here, PI fires on
            # its own evidence below.
            # Guard: only trust ARG0 when the token is not structurally in a
            # patient position. SRL models sometimes label dobj / nsubjpass
            # tokens as ARG0 (e.g. in double-object or ergative constructions);
            # accepting ARG0 on such tokens mislabels patients as agents and
            # also leaks into SI eligibility via the srl_info roles check below.
            _agi_from_srl = (
                "AgI" in srl_info["roles"]
                and effective_dep not in ("dobj", "nsubjpass", "no_inherit")
            )
            if _agi_from_srl:
                if dim_src and not agi_passes:
                    role_review_flags.add("srl_arg0_proto_disagrees")
                else:
                    if "AgI" not in roles:
                        roles.append("AgI")
            # nsubj / by fallback — runs even when SRL gave ARG0 but its
            # prototype check failed (previously blocked by elif).  When SRL
            # ARG0 passed, ‘AgI’ is already in roles and the guard skips.
            if "AgI" not in roles:
                if effective_dep == "nsubj" and head_verb is not None and head_verb.pos_ == "VERB":
                    # No SRL ARG0 — prototype is the gate. AGI is checked
                    # independently of PI/SI (a single target can pass multiple
                    # dims; AgI fires whenever volitional-agency cosine is above
                    # AGI_FLOOR, regardless of patient/mental cosines).
                    if dim_src and agi_passes:
                        roles.append("AgI")
                    else:
                        role_review_flags.add("subject_nonagentive")
                elif effective_dep == "pobj" and effective_noun.head.text.lower() == "by":
                    roles.append("AgI")

            # ── PI ──
            # SRL PATIENT_LABELS, dobj, nsubjpass and pcomp-passive are direct
            # syntactic evidence of patient role and are trusted on their own.
            # Prototype disagreement is recorded as a flag, not a veto, because
            # PI prototype is biased toward unaccusative/passive surface and
            # systematically loses to AGI on active-transitive `B verbs A`
            # constructions even when A is the patient.
            if "PI" in srl_info["roles"]:
                if "PI" not in roles:
                    roles.append("PI")
                if dim_src and not pi_passes:
                    role_review_flags.add("srl_patient_proto_disagrees")
            else:
                if effective_dep == "dobj":
                    if "PI" not in roles:
                        roles.append("PI")
                    if dim_src and not pi_passes:
                        role_review_flags.add("dobj_patient_proto_disagrees")
                if effective_dep == "nsubjpass":
                    if "PI" not in roles:
                        roles.append("PI")
                    if dim_src and not pi_passes:
                        role_review_flags.add("passive_patient_proto_disagrees")
                if effective_dep == "pcomp" and head_verb is not None and any(
                    c.dep_ == "auxpass" for c in head_verb.children
                ):
                    if "PI" not in roles:
                        roles.append("PI")
                if effective_dep == "pobj":
                    prep = effective_noun.head.text.lower()
                    is_recipient = prep in ("for", "to")
                    is_deprivation = (
                        prep == "from"
                        and effective_noun.head.head is not None
                        and effective_noun.head.head.pos_ in ("VERB", "AUX")
                    )
                    if (is_recipient or is_deprivation) and dim_src and pi_passes:
                        if "PI" not in roles:
                            roles.append("PI")
                # Affected-subject PI is intentionally NOT assigned here.
                # Earlier attempts (`pi_passes` alone, `pi_passes AND
                # anchor_diff > 0`) over-fired on subjects of mental verbs
                # (`immigrant believes`, `Migrants feared`, `Black hoped`,
                # `Americans believe`) because PI prototype geometry overlaps
                # too broadly with general "people-in-events" semantics, and
                # sentiment anchor is not a reliable proxy for affectedness
                # vs. mental-state subject. Pure-nsubj PI requires syntactic
                # signal we don't have at this layer; leave the mention for
                # SRL ARG1 / dobj / nsubjpass evidence above instead.

            # ── PI: object-of-mental-state via relative clause ──
            # Target is object of a belief/perception held by a non-group subject.
            for child in token.children:
                if child.dep_ == "relcl":
                    relcl_subjs = [c for c in child.children if c.dep_ == "nsubj"]
                    if relcl_subjs and all(is_target(s, doc) is None for s in relcl_subjs):
                        if dim_src and pi_passes and "PI" not in roles:
                            roles.append("PI")

            # ── SI ──
            # No syntactic backstop, but a minimum target-as-experiencer guard:
            # SI fires only when the target is grammatically positioned to be
            # the experiencer (clausal subject or SRL-labeled core argument).
            # Without this guard, SI false-fires on bare modifiers and
            # prepositional-object mentions like "Activists from the Black
            # community" where 'Black' is amod of 'community' inside a 'from'
            # phrase. SI is independent of AgI/PI: a target can simultaneously
            # be agent, patient, and subject of mental states.
            #
            # Intentionally uses the filtered `roles` list (not raw
            # srl_info["roles"]) so that an ARG0/ARG1 label that failed the
            # AGI/PI prototype gate does not silently re-open the SI door.
            # This prevents the pattern: ARG0 fails agi_passes → srl_arg0_proto_disagrees
            # → but si_target_eligible = True via raw "AgI" in srl_info → SI fires.
            si_target_eligible = (
                effective_dep in ("nsubj", "nsubjpass")
                or "AgI" in roles
            )
            if dim_src and si_passes and si_target_eligible:
                if "SI" not in roles:
                    roles.append("SI")

        # ── AttI (diagnostic) ──
        if atti_info["label"] == "negAttI":
            roles.append("negAttI")
        elif atti_info["label"] == "posAttI":
            roles.append("posAttI")

        agi       = 1 if "AgI"      in roles else 0
        pi        = 1 if "PI"       in roles else 0
        si        = 1 if "SI"       in roles else 0
        neg_atti  = 1 if "negAttI"  in roles else 0
        pos_atti  = 1 if "posAttI"  in roles else 0
        if not any((agi, pi, si)):
            role_review_flags.add("no_clear_semantic_role")

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
            "srl_suppressed_agency": "null",  # moved to review flag
            "role_source": "srl+proto" if (srl_hit and dim_src) else ("proto" if dim_src else ("srl" if srl_hit else "dep")),
            "role_confidence": 0.9 if (srl_hit and dim_src) else (0.75 if (srl_hit or dim_src) else (0.7 if any((agi, pi, si)) else 0.4)),
            "role_review_flags": sorted(role_review_flags),
            "atti_source": "local_prototype_diagnostic" if dim_src else "none",
            "atti_neg_sim": atti_info["neg_sim"],
            "atti_pos_sim": atti_info["pos_sim"],
            "dim_agi_sim": agi_sim,
            "dim_pi_sim":  pi_sim,
            "dim_si_sim":  si_sim,
            "anchor_neg_sim": anchor_neg,
            "anchor_pos_sim": anchor_pos,
            "roles": roles if roles else ["present"],
            "subjecthood": subjecthood,
            "agi": agi, "pi": pi, "si": si,
            "neg_atti": neg_atti, "pos_atti": pos_atti,
        })

    _resolve_anaphora(doc, findings)
    _resolve_adverbial_passive(doc, findings)
    _finalize_review_flags(findings)

    return findings


def _finalize_review_flags(findings: list[dict]):
    for f in findings:
        flags = list(dict.fromkeys(f.get("role_review_flags", [])))
        if any((f.get("agi"), f.get("pi"), f.get("si"))):
            flags = [flag for flag in flags if flag != "no_clear_semantic_role"]
            if "present" in f.get("roles", []) and len(f["roles"]) > 1:
                f["roles"] = [role for role in f["roles"] if role != "present"]
        f["role_review_flags"] = sorted(flags)


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
                f.setdefault("role_review_flags", []).append("adverbial_passive")
                f["role_confidence"] = min(f.get("role_confidence", 0.7), 0.6)


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
            antecedent["subjecthood"] = 1
            if _ATTITUDE_MATCHER is not None:
                # Direct relative-margin scoring on pronoun context window
                span_indices = {token.i}
                p_info = _ATTITUDE_MATCHER.match(token, doc, head_verb=head, span_indices=span_indices)
                p_agi = p_info.get("agi_sim", 0.0)
                p_pi  = p_info.get("pi_sim", 0.0)
                p_si  = p_info.get("si_sim", 0.0)

                p_agi_passes = _dim_passes(p_agi, AGI_FLOOR)
                p_pi_passes  = _dim_passes(p_pi,  PI_FLOOR)
                p_si_passes  = _dim_passes(p_si,  SI_FLOOR)

                if p_agi_passes and antecedent["agi"] == 0:
                    antecedent["agi"] = 1
                    if "AgI" not in antecedent["roles"]:
                        antecedent["roles"].append("AgI")
                    antecedent.setdefault("role_review_flags", []).append("anaphora_resolved")
                    antecedent["role_confidence"] = min(antecedent.get("role_confidence", 0.7), 0.6)
                elif not p_agi_passes:
                    antecedent.setdefault("role_review_flags", []).extend(["anaphora_resolved", "subject_nonagentive"])

                if p_pi_passes and antecedent["pi"] == 0:
                    antecedent["pi"] = 1
                    if "PI" not in antecedent["roles"]:
                        antecedent["roles"].append("PI")
                    antecedent.setdefault("role_review_flags", []).append("anaphora_resolved")
                    antecedent["role_confidence"] = min(antecedent.get("role_confidence", 0.7), 0.6)

                if p_si_passes and antecedent["si"] == 0:
                    antecedent["si"] = 1
                    if "SI" not in antecedent["roles"]:
                        antecedent["roles"].append("SI")
                    antecedent.setdefault("role_review_flags", []).append("anaphora_resolved")
                    antecedent["role_confidence"] = min(antecedent.get("role_confidence", 0.7), 0.6)

        if dep == "dobj":
            if antecedent["pi"] == 0:
                antecedent["pi"] = 1
                if "PI" not in antecedent["roles"]:
                    antecedent["roles"].append("PI")
                antecedent.setdefault("role_review_flags", []).append("anaphora_resolved")
                antecedent["role_confidence"] = min(antecedent.get("role_confidence", 0.7), 0.6)

        if dep == "nsubjpass":
            if antecedent["pi"] == 0:
                antecedent["pi"] = 1
                if "PI" not in antecedent["roles"]:
                    antecedent["roles"].append("PI")
                antecedent.setdefault("role_review_flags", []).append("anaphora_resolved")
                antecedent["role_confidence"] = min(antecedent.get("role_confidence", 0.7), 0.6)


def extract_all(processed_data: list[dict]) -> list[dict]:
    # Batch-encode all target mention focus texts if attitude matcher is loaded
    if _ATTITUDE_MATCHER is not None:
        print("  Scanning target mentions for batch attitude matching...", flush=True)
        keys = []
        focus_texts = []
        for i, item in enumerate(processed_data):
            doc = item["doc"]
            for token in doc:
                resolved = resolve_group_token(token, doc)
                if resolved is not None:
                    head_verb, _, _ = _resolve_role(token, doc)
                    span_indices = _group_span_indices(token, doc)
                    focus_text = _ATTITUDE_MATCHER._build_focus_text(
                        token, doc, head_verb=head_verb, span_indices=span_indices
                    )
                    keys.append((i, token.i))
                    focus_texts.append(focus_text)

        if focus_texts:
            print(f"  Batch encoding {len(focus_texts)} target mention focus texts...", flush=True)
            match_results = _ATTITUDE_MATCHER.batch_match(focus_texts)
            _ATTITUDE_MATCHER.lookup = {k: res for k, res in zip(keys, match_results)}
            print("  Batch attitude matching complete.", flush=True)

    results = []
    for i, item in enumerate(processed_data):
        roles = extract_roles(item["doc"], doc_id=i)
        results.append({
            "sentence_id": i,
            "category": item["category"],
            "text": item["cleaned_text"],
            "findings": roles,
        })
        if _SRL_ROLE_LABELER is not None and (i + 1) % 100 == 0:
            print(f"  SRL extraction: {i + 1}/{len(processed_data)} sentences", flush=True)
    return results
