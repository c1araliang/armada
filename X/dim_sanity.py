"""
Dimensional prototype sanity check — NOT part of the pipeline.

Run from X/ with the venv active:
    python _dim_sanity.py

Prints per-target AGI/PI/SI pass/fail, raw cosines, and anchor diff (neg − pos).
For sentences flagged as tricky, also dumps the full per-prototype cosine
matrix with the winning prototype per dimension marked with `*`.

Does NOT write any output files or touch any research artifacts.
Add new test sentences to TEST_SENTENCES below; set the second tuple field
to True for sentences that should get the per-prototype breakdown.
"""

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
X_DIR = Path(__file__).parent
sys.path.insert(0, str(X_DIR))

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from embedding_config import ANALYSIS_EMBEDDING_MODEL
from step3_attitudinal_prototypes import (
    AttitudinalPrototypeMatcher,
    AGI_FLOOR, PI_FLOOR, SI_FLOOR,
    AGI_PROTOTYPES, PI_PROTOTYPES, SI_PROTOTYPES,
)
from step3_semantic_roles import SrlRoleLabeler
from step3_feature_extraction import (
    set_srl_role_labeler,
    set_attitude_matcher,
    extract_roles,
    _group_span_indices,
    _resolve_role,
)
from step2_preprocessing import remove_noise
from lexicons import resolve_group_token

# ── test sentences (sentence, dump_per_prototype) ─────────────────────────────
TEST_SENTENCES: list[tuple[str, bool]] = [
    ("The refugees organized a protest in the city square.", False),
    ("Activists from the Black community led the campaign.", False),
    ("The policy penalizes migrants by preventing them from applying for asylum.", False),
    ("Officials deported the migrants across the border.", False),
    ("The state expelled the refugees from their homes.", False),
    ("The migrants were detained without charge.", False),
    ("Refugees were attacked in the street by hostile crowds.", False),
    ("The refugees suffered for months in the camps.", False),
    ("The immigrant believes his children will have a better life.", False),
    ("Migrants feared deportation and avoided going outside.", False),
    ("Black communities have hoped for change for generations.", False),
    ("Refugees protested the deportation orders that affected them.", False),
    ("Migrants resisted the police who tried to detain them.", False),
    ("Despite being detained, refugees organized hunger strikes and hoped for release.", False),
    ("The refugees did not organize any protests.", False),
    ("Many Americans believe that immigrants are stealing jobs.", False),
    (
        "The main topic of his talk was how refugees were being suppressed "
        "to the limit that they naturally fight back.",
        True,
    ),
    (
        "But another neighbor said that a couple of white, adult residents of the "
        "neighborhood were yelling racial slurs and assaulting a black teenager.",
        False,
    ),
    (
        "all the Europeans or at least German, French, Swedish and Spanish people "
        "use the Internet more than two hours a day",
        True,
    ),
    ("Why do Italian football fans get away with racially abusing black players?", True),
    ("According to several studies, a little more than one out of every 10 Americans takes anti-depressant medications.", True),
    ("its also true that Americans love a British accent!", True),
    ("They also have a free kit for Canadians.", True),
    ("Americans in Texas territory vote to separate Texas from Mexico.", True),

    ("black and white residents clashed; french and american people watched.", True),
]

# ── model loading ──────────────────────────────────────────────────────────────
print("Loading models (this takes ~30s on first run)...")
nlp = spacy.load("en_core_web_lg")
sentence_encoder = SentenceTransformer(ANALYSIS_EMBEDDING_MODEL)
srl = SrlRoleLabeler()
set_srl_role_labeler(srl)
matcher = AttitudinalPrototypeMatcher(sentence_encoder)
set_attitude_matcher(matcher)

# Pre-cached prototype matrices on the matcher (numpy arrays of shape [n, dim]).
AGI_VECS = matcher.agi_prototypes
PI_VECS  = matcher.pi_prototypes
SI_VECS  = matcher.si_prototypes


def _per_prototype(focus_text: str) -> dict:
    """Encode focus_text once and return cosine vs. every prototype, per dim."""
    vec = matcher.sentence_encoder.encode(
        focus_text, normalize_embeddings=True, show_progress_bar=False
    )
    return {
        "AGI": (AGI_VECS @ vec).tolist(),
        "PI":  (PI_VECS  @ vec).tolist(),
        "SI":  (SI_VECS  @ vec).tolist(),
    }


def _print_per_prototype(focus_text: str):
    sims = _per_prototype(focus_text)
    matrices = {
        "AGI": (AGI_PROTOTYPES, sims["AGI"], AGI_FLOOR),
        "PI":  (PI_PROTOTYPES,  sims["PI"],  PI_FLOOR),
        "SI":  (SI_PROTOTYPES,  sims["SI"],  SI_FLOOR),
    }
    for dim, (texts, vals, floor) in matrices.items():
        winner = int(np.argmax(vals))
        max_val = vals[winner]
        flag = "PASS" if max_val >= floor else "fail"
        print(f"      {dim} (floor {floor}, winner cos {max_val:.3f}, {flag})")
        for i, (t, v) in enumerate(zip(texts, vals)):
            mark = " *" if i == winner else "  "
            short = (t[:88] + "…") if len(t) > 90 else t
            print(f"        [{i}]{mark} {v:.3f}  {short}")


# ── header ─────────────────────────────────────────────────────────────────────
COL_W = 70
print(f"\nFloors: AGI={AGI_FLOOR} PI={PI_FLOOR} SI={SI_FLOOR}")
print(
    f"{'sentence':<{COL_W}}  {'tgt':<14}  {'AGI':3} {'PI':3} {'SI':3}"
    f"   {'cos a/p/s':<22}  {'anchor n/p/n−p'}"
)
print("-" * 130)

# ── run ────────────────────────────────────────────────────────────────────────
detail_queue: list[tuple[str, str]] = []  # (sentence, focus_text) for each tricky finding

for sent, dump in TEST_SENTENCES:
    cleaned = remove_noise(sent)
    doc = nlp(cleaned)
    findings = extract_roles(doc)

    if not findings:
        print(f"{sent[:COL_W]:<{COL_W}}  {'(no targets)':<14}")
        continue

    first = True
    for f in findings:
        agi_s = "AGI" if f["agi"] else "."
        pi_s  = "PI"  if f["pi"]  else "."
        si_s  = "SI"  if f["si"]  else "."
        a, p, s = f["dim_agi_sim"], f["dim_pi_sim"], f["dim_si_sim"]
        an = f["anchor_neg_sim"]
        ap = f["anchor_pos_sim"]
        ad = an - ap
        cos_str    = f"{a:.3f}/{p:.3f}/{s:.3f}"
        anchor_str = f"{an:.3f}/{ap:.3f}/{ad:+.3f}"
        label = (sent[:COL_W] if first else "")
        tgt   = f["token"][:14]
        flags = [fl for fl in f.get("role_review_flags", [])
                 if fl != "no_clear_semantic_role"]
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"{label:<{COL_W}}  {tgt:<14}  {agi_s:3} {pi_s:3} {si_s:3}"
            f"   {cos_str:<22}  {anchor_str}{flag_str}"
        )
        first = False

        if dump:
            # Reconstruct the same focus_text the matcher used.
            for token in doc:
                if token.i == f["token_i"]:
                    head_verb, _, _ = _resolve_role(token, doc)
                    span = _group_span_indices(token, doc)
                    focus_text = matcher._build_focus_text(
                        token, doc, head_verb=head_verb, span_indices=span,
                    )
                    detail_queue.append((sent, f["token"], focus_text))
                    break

# ── per-prototype dumps for flagged sentences ─────────────────────────────────
if detail_queue:
    print("\n" + "=" * 130)
    print("Per-prototype cosines for flagged sentences (winner per dim marked *)")
    print("=" * 130)
    for sent, tok, focus in detail_queue:
        print(f"\n{sent}")
        print(f"  target: {tok}")
        print(f"  focus_text: {focus[:200]}{'…' if len(focus) > 200 else ''}")
        _print_per_prototype(focus)
