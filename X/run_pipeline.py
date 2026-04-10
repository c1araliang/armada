"""
Pipeline Runner — ARMADA bias detection framework.

Outputs:
  1. Sentence-level discourse collocate discovery (target + contrast groups)
  2. Per-group syntactic indices (AgI, PI, SI, negAttI, posAttI)
  3. WEAT scores (type-level, spaCy vocab vectors against refreshed F⁻/F⁺ frame sets)
  4. SEAT scores (token-level, from sentence embeddings)
  5. EFI via PCA on the group × dimension matrix
  6. Regression: WEAT/SEAT ~ indices
"""

import csv
import json
import re
import sys
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict

import spacy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

from step2_preprocessing import load_sentences, preprocess
from step3_feature_extraction import extract_all, set_attitude_matcher, set_srl_role_labeler
from step4_metrics import (
    build_sentence_associations,
    compute_association_scores,
    compute_signed_association,
    aggregate_sentence_metrics,
    compute_group_indices,
    cosine_similarity,
)
from lexicons import (
    TARGET_TOKENS, CONTRAST_TOKENS,
    ALL_FRAME_TERMS, FRAME_SIGN, CLASSIFIED_FRAMES, set_semantic_group_resolver,
    resolve_group_token,
)
from semantic_group_resolver import SemanticGroupResolver
from step3_attitudinal_prototypes import AttitudinalPrototypeMatcher
from step3_semantic_roles import SrlRoleLabeler

_ALL_GROUPS = TARGET_TOKENS | CONTRAST_TOKENS
_NEG_FRAMES = {t for f in CLASSIFIED_FRAMES.values() if f["sign"] < 0 for t in f["terms"]}
_POS_FRAMES = {t for f in CLASSIFIED_FRAMES.values() if f["sign"] > 0 for t in f["terms"]}
ASSOCIATION_MIN_COUNT = 5
ANALYSIS_MIN_GROUP_COUNT = 50
REPORT_MIN_GROUP_COUNT = 50
FRAME_REFRESH_TOP_N = 60
FRAME_SIM_FLOOR = 0.22
FRAME_SIM_MARGIN = 0.02


def _excel_safe(value):
    if value == "":
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f" {value}"
    return value


def _row_excel_safe(row: dict) -> dict:
    return {k: _excel_safe(v) for k, v in row.items()}


def _write_discourse_association(
    output_path: Path,
    cooc: dict,
    association_scores: dict[tuple, dict[str, float]],
) -> None:
    """Write sentence-level discourse association pairs to a TSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "target",
            "group_type",
            "collocate",
            "pair_sentence_count",
            "target_sentence_count",
            "collocate_sentence_count",
            "llr",
            "logdice",
        ])
        for (target, collocate), score_info in sorted(
            association_scores.items(),
            key=lambda x: (-x[1]["llr"], -x[1]["logdice"], x[0][0], x[0][1]),
        ):
            writer.writerow([
                target,
                "target" if target in TARGET_TOKENS else "contrast",
                collocate,
                _excel_safe(score_info.get("pair_count", cooc["pair_counts"].get((target, collocate), 0))),
                _excel_safe(score_info.get("target_count", cooc["target_counts"].get(target, 0))),
                _excel_safe(score_info.get("collocate_count", cooc["collocate_counts"].get(collocate, 0))),
                _excel_safe(round(score_info["llr"], 4)),
                _excel_safe(round(score_info["logdice"], 4)),
            ])


def _refresh_frame_inventory(
    project_dir: Path,
    candidates: list[dict],
    sentence_encoder,
) -> tuple[set[str], set[str], list[dict]]:
    """
    Refresh frame sets from current LLR candidates using the seed taxonomy as
    semantic anchors.
    """
    neg_terms = set(_NEG_FRAMES)
    pos_terms = set(_POS_FRAMES)

    if not candidates:
        payload = {
            "last_updated": str(date.today()),
            "note": "Current-run candidate frame suggestions derived from sentence-level LLR/LogDice using seed frame prototypes.",
            "seed_negative_terms": sorted(neg_terms),
            "seed_positive_terms": sorted(pos_terms),
            "auto_negative_terms": [],
            "auto_positive_terms": [],
            "candidates": [],
        }
        with open(project_dir / "candidate_terms.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return neg_terms, pos_terms, []

    seed_neg_vecs = sentence_encoder.encode(
        sorted(neg_terms),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    seed_pos_vecs = sentence_encoder.encode(
        sorted(pos_terms),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_vecs = sentence_encoder.encode(
        [c["term"] for c in candidates],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    annotated = []
    auto_negative = []
    auto_positive = []
    for candidate, vec in zip(candidates, candidate_vecs):
        neg_sim = float(np.max(seed_neg_vecs @ vec))
        pos_sim = float(np.max(seed_pos_vecs @ vec))
        diff = neg_sim - pos_sim

        suggested_sign = 0
        suggested_bucket = "neutral"
        use_in_inventory = False
        if max(neg_sim, pos_sim) >= FRAME_SIM_FLOOR and abs(diff) >= FRAME_SIM_MARGIN:
            suggested_sign = -1 if diff > 0 else 1
            suggested_bucket = "negative" if diff > 0 else "positive"
            use_in_inventory = True

        annotated_candidate = {
            "term": candidate["term"],
            "minority_llr": candidate["minority_llr"],
            "minority_logdice": candidate["minority_logdice"],
            "dominant_llr": candidate["dominant_llr"],
            "dominant_logdice": candidate["dominant_logdice"],
            "differential": candidate["differential"],
            "found_with": candidate["found_with"],
            "frame_neg_sim": round(neg_sim, 4),
            "frame_pos_sim": round(pos_sim, 4),
            "suggested_frame_sign": suggested_sign,
            "suggested_frame_bucket": suggested_bucket,
            "used_in_frame_inventory": use_in_inventory,
        }
        annotated.append(annotated_candidate)
        if use_in_inventory:
            if suggested_sign < 0:
                neg_terms.add(candidate["term"])
                auto_negative.append(candidate["term"])
            else:
                pos_terms.add(candidate["term"])
                auto_positive.append(candidate["term"])

    payload = {
        "last_updated": str(date.today()),
        "note": "Current-run candidate frame suggestions derived from sentence-level LLR/LogDice using seed frame prototypes.",
        "seed_negative_terms": sorted(_NEG_FRAMES),
        "seed_positive_terms": sorted(_POS_FRAMES),
        "auto_negative_terms": sorted(set(auto_negative)),
        "auto_positive_terms": sorted(set(auto_positive)),
        "candidates": annotated,
    }
    with open(project_dir / "candidate_terms.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return neg_terms, pos_terms, annotated


def _compute_weat(sentence_encoder, neg_frames: set[str], pos_frames: set[str]) -> dict[str, float]:
    """WEAT: type-level association via MiniLM word embeddings.

    Encodes group lemmas and frame terms through the same MiniLM encoder
    used by SEAT, ensuring methodological consistency. Both metrics now
    use the same frozen encoder — any cross-corpus difference in SEAT
    (but not WEAT) reflects corpus-specific framing.
    """
    all_terms = sorted(_ALL_GROUPS | neg_frames | pos_frames)
    all_vecs = sentence_encoder.encode(
        all_terms,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    term_to_vec = dict(zip(all_terms, all_vecs))

    scores = {}
    for lemma in sorted(_ALL_GROUPS):
        vec = term_to_vec.get(lemma)
        if vec is None:
            continue
        neg_sims = [cosine_similarity(vec, term_to_vec[f])
                    for f in neg_frames if f in term_to_vec]
        pos_sims = [cosine_similarity(vec, term_to_vec[f])
                    for f in pos_frames if f in term_to_vec]
        if neg_sims and pos_sims:
            scores[lemma] = round(np.mean(neg_sims) - np.mean(pos_sims), 4)
    return scores


def _compute_seat(processed: list[dict], sentence_encoder, neg_frames: set[str] | None = None, pos_frames: set[str] | None = None) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    """SEAT-style sentence association via MiniLM sentence embeddings.

    Returns (scores, neg_centroid, pos_centroid) so SEAT-full can reuse
    the same frame reference points for Δ-SEAT comparability.
    """
    group_vecs: dict[str, list] = defaultdict(list)
    frame_neg_vecs, frame_pos_vecs = [], []
    neg_frames = neg_frames or _NEG_FRAMES
    pos_frames = pos_frames or _POS_FRAMES
    sentence_texts = [item["cleaned_text"] for item in processed]
    sent_vecs = sentence_encoder.encode(
        sentence_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for item, sent_vec in zip(processed, sent_vecs):
        doc = item["doc"]
        group_lemmas = set()
        for token in doc:
            resolved = resolve_group_token(token, doc)
            if resolved:
                _, canonical = resolved
                group_lemmas.add(canonical)
        for lemma in group_lemmas:
            group_vecs[lemma].append(sent_vec)
        lemmas = {t["lemma"] for t in item["tokens"]}
        if lemmas & neg_frames:
            frame_neg_vecs.append(sent_vec)
        if lemmas & pos_frames:
            frame_pos_vecs.append(sent_vec)

    if not frame_neg_vecs or not frame_pos_vecs:
        return {}, None, None
    neg_centroid = np.mean(frame_neg_vecs, axis=0)
    pos_centroid = np.mean(frame_pos_vecs, axis=0)

    scores = {}
    for lemma, vecs in group_vecs.items():
        centroid = np.mean(vecs, axis=0)
        sim_neg = cosine_similarity(centroid, neg_centroid)
        sim_pos = cosine_similarity(centroid, pos_centroid)
        scores[lemma] = round(sim_neg - sim_pos, 4)
    return scores, neg_centroid, pos_centroid


def _compute_seat_full(
    lexical_all_path: Path,
    sentence_encoder,
    neg_centroid,
    pos_centroid,
    batch_size: int = 2048,
) -> dict[str, float]:
    """SEAT-full: association computed from ALL lexical-hit sentences.

    Unlike filtered SEAT (which only uses semantically-relevant, people-focused
    sentences), SEAT-full includes every sentence that matched the lexical gate:
    products, institutions, culture, etc.  Comparing the two reveals how
    non-human contexts shift a group's embedding association.

    Uses simple regex matching for group assignment (no spaCy needed) and
    reuses the F⁻/F⁺ centroids from filtered SEAT for comparability.
    """
    if not lexical_all_path.exists():
        return {}

    token_re = re.compile(
        r"(?<!\w)(?:"
        + "|".join(sorted(map(re.escape, _ALL_GROUPS), key=len, reverse=True))
        + r")(?!\w)",
        re.I,
    )

    group_vecs: dict[str, list] = defaultdict(list)
    buffer = []
    buffer_groups: list[set[str]] = []

    with open(lexical_all_path, encoding="utf-8") as fh:
        for line in fh:
            sentence = line.rstrip("\n")
            if not sentence:
                continue
            matches = {m.group().lower() for m in token_re.finditer(sentence)}
            if matches:
                buffer.append(sentence)
                buffer_groups.append(matches)

            if len(buffer) >= batch_size:
                vecs = sentence_encoder.encode(
                    buffer, normalize_embeddings=True, show_progress_bar=False,
                )
                for vec, groups in zip(vecs, buffer_groups):
                    for g in groups:
                        group_vecs[g].append(vec)
                buffer, buffer_groups = [], []

    if buffer:
        vecs = sentence_encoder.encode(
            buffer, normalize_embeddings=True, show_progress_bar=False,
        )
        for vec, groups in zip(vecs, buffer_groups):
            for g in groups:
                group_vecs[g].append(vec)

    scores = {}
    for lemma, vec_list in group_vecs.items():
        centroid = np.mean(vec_list, axis=0)
        sim_neg = cosine_similarity(centroid, neg_centroid)
        sim_pos = cosine_similarity(centroid, pos_centroid)
        scores[lemma] = round(sim_neg - sim_pos, 4)
    return scores


def _compute_efi(group_profiles: list[dict]) -> dict:
    """PCA on group × [AgI, PI, SI, netAttI, WEAT, SEAT] matrix."""
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "seat"]
    labels = [g["lemma"] for g in group_profiles]
    matrix = np.array([[g.get(d, 0.0) for d in dims] for g in group_profiles])

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    pca = PCA(n_components=min(len(dims), len(labels)))
    scores = pca.fit_transform(X)

    pc1 = pca.components_[0]
    raw_scores = scores[:, 0]

    # Orient PC1 so that "negative framing" is positive:
    # high PI + high netAttI + high WEAT/SEAT and low AgI/SI should → positive EFI.
    agi_idx = dims.index("AGI")
    pi_idx  = dims.index("PI")
    si_idx = dims.index("SI")
    net_atti_idx = dims.index("net_atti")
    weat_idx = dims.index("weat")
    seat_idx = dims.index("seat")
    orientation_anchor = (
        pc1[pi_idx]
        + pc1[net_atti_idx]
        + pc1[weat_idx]
        + pc1[seat_idx]
        - pc1[agi_idx]
        - pc1[si_idx]
    )
    if orientation_anchor < 0:
        pc1 = -pc1
        raw_scores = -raw_scores

    loadings = dict(zip(dims, pc1.round(3)))
    efi_scores = dict(zip(labels, raw_scores.round(3)))
    var_explained = round(pca.explained_variance_ratio_[0], 3)

    return {
        "loadings": loadings,
        "scores": efi_scores,
        "variance_explained": var_explained,
    }


def _run_regression(
    group_profiles: list[dict],
    target_key: str,
    predictors: list[str],
) -> dict | None:
    """OLS regression on group profiles."""
    y = np.array([g.get(target_key, 0.0) for g in group_profiles])
    X = np.array([[g.get(p, 0.0) for p in predictors] for g in group_profiles])

    if len(y) < len(predictors) + 2:
        return None

    X_aug = np.column_stack([np.ones(len(y)), X])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "target": target_key,
        "predictors": predictors,
        "intercept": round(beta[0], 4),
        "coefficients": {p: round(b, 4) for p, b in zip(predictors, beta[1:])},
        "r_squared": round(r_sq, 4),
        "n": len(y),
    }


def _find_candidates(
    association_scores: dict[tuple, dict[str, float]],
    top_n: int = 30,
    min_llr: float = 3.0,
) -> list[dict]:
    """High-association collocates of target groups not yet in the frame taxonomy."""
    minority_llr: dict[str, float] = {}
    minority_logdice: dict[str, float] = {}
    dominant_llr: dict[str, float] = {}
    dominant_logdice: dict[str, float] = {}

    for (target, collocate), score_info in association_scores.items():
        if collocate in ALL_FRAME_TERMS:
            continue
        llr = score_info.get("llr", 0.0)
        logdice = score_info.get("logdice", 0.0)
        if target in TARGET_TOKENS:
            if llr > minority_llr.get(collocate, 0):
                minority_llr[collocate] = llr
                minority_logdice[collocate] = logdice
        elif target in CONTRAST_TOKENS:
            if llr > dominant_llr.get(collocate, 0):
                dominant_llr[collocate] = llr
                dominant_logdice[collocate] = logdice

    candidates = []
    for collocate, minority_score in minority_llr.items():
        if minority_score < min_llr:
            continue
        dominant_score = dominant_llr.get(collocate, 0.0)
        diff = round(minority_score - dominant_score, 3)
        if diff > 0:
            targets = sorted({t for (t, c) in association_scores if c == collocate
                              and t in TARGET_TOKENS})
            candidates.append({
                "term": collocate,
                "minority_llr": round(minority_score, 3),
                "minority_logdice": round(minority_logdice.get(collocate, 0.0), 3),
                "dominant_llr": round(dominant_score, 3),
                "dominant_logdice": round(dominant_logdice.get(collocate, 0.0), 3),
                "differential": diff,
                "found_with": targets,
            })

    candidates.sort(key=lambda x: (-x["differential"], -x["minority_logdice"], x["term"]))
    return candidates[:top_n]



def main():
    project_dir = Path(__file__).parent
    _dolma_candidates = [
        project_dir.parent / "dolma" / "semantic_filter_results.tsv",  # relative (any OS)
        Path("d:/projects/dolma/semantic_filter_results.tsv"),          # Windows absolute
        Path.home() / "projects" / "dolma" / "semantic_filter_results.tsv",  # macOS ~/projects
    ]
    default_filtered = next((p for p in _dolma_candidates if p.exists()), None)
    if len(sys.argv) > 1:
        sentences_path = Path(sys.argv[1])
    elif default_filtered is not None:
        sentences_path = default_filtered
    else:
        sys.exit(
            "Error: no input file found.\n"
            "Pass a path as the first argument, or place the dolma TSV at one of:\n"
            + "\n".join(f"  {p}" for p in _dolma_candidates)
        )

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")
    print("Loading semantic group disambiguator...")
    semantic_resolver = SemanticGroupResolver()
    set_semantic_group_resolver(semantic_resolver)
    print("Loading SRL role labeler...")
    set_srl_role_labeler(SrlRoleLabeler())
    print("Loading attitudinal prototype matcher...")
    set_attitude_matcher(AttitudinalPrototypeMatcher(semantic_resolver.model))

    # ── Preprocessing ──
    print(f"Reading input from: {sentences_path}")
    raw = load_sentences(str(sentences_path))
    processed = preprocess(nlp, raw)
    print(f"Preprocessed {len(processed)} sentences.")

    # ── Feature extraction (AgI/PI/SI/AttI) ──
    extracted = extract_all(processed)
    n_findings = sum(1 for e in extracted if e["findings"])
    print(f"Target tokens in {n_findings}/{len(extracted)} sentences.\n")

    # ── Discourse association discovery ──
    cooc = build_sentence_associations(processed, min_count=ASSOCIATION_MIN_COUNT)
    association_scores = compute_association_scores(cooc)
    signed = compute_signed_association(association_scores, score_key="llr")
    rows = aggregate_sentence_metrics(extracted, signed, processed)

    association_tsv = project_dir / "association_discourse.tsv"
    _write_discourse_association(association_tsv, cooc, association_scores)
    print(
        f"Discourse association written to: {association_tsv.name} "
        f"(sentence-level, min pair count = {ASSOCIATION_MIN_COUNT}, min distance = {cooc['min_distance']})"
    )

    candidates = _find_candidates(association_scores, top_n=FRAME_REFRESH_TOP_N)
    neg_frames, pos_frames, annotated_candidates = _refresh_frame_inventory(
        project_dir,
        candidates,
        semantic_resolver.model,
    )
    print(
        f"Candidate terms refreshed: {len(annotated_candidates)} total, "
        f"{len(neg_frames - _NEG_FRAMES)} auto-negative, {len(pos_frames - _POS_FRAMES)} auto-positive"
    )

    # ── WEAT + SEAT (filtered) ──
    weat_scores = _compute_weat(semantic_resolver.model, neg_frames=neg_frames, pos_frames=pos_frames)
    seat_scores, seat_neg_centroid, seat_pos_centroid = _compute_seat(
        processed,
        sentence_encoder=semantic_resolver.model,
        neg_frames=neg_frames,
        pos_frames=pos_frames,
    )

    # ── SEAT-full (all lexical hits) + Δ-SEAT ──
    lexical_all_path = sentences_path.parent / "semantic_filter_lexical_all.txt"
    seat_full_scores = {}
    delta_seat_scores = {}
    if seat_neg_centroid is not None and seat_pos_centroid is not None:
        print("Computing SEAT-full from all lexical hits...")
        seat_full_scores = _compute_seat_full(
            lexical_all_path,
            sentence_encoder=semantic_resolver.model,
            neg_centroid=seat_neg_centroid,
            pos_centroid=seat_pos_centroid,
        )
        for lemma in set(seat_scores) | set(seat_full_scores):
            s_filtered = seat_scores.get(lemma)
            s_full = seat_full_scores.get(lemma)
            if s_filtered is not None and s_full is not None:
                delta_seat_scores[lemma] = round(s_full - s_filtered, 4)
        print(f"  SEAT-full computed for {len(seat_full_scores)} groups, Δ-SEAT for {len(delta_seat_scores)}")

    # ── Group indices ──
    summary = compute_group_indices(extracted)
    report_profiles = [g for g in summary["lemmas"] if g["total"] >= REPORT_MIN_GROUP_COUNT]

    print(f"\n{'='*80}")
    print(f"  WEAT / SEAT SCORES  (>0 = closer to F⁻ than F⁺, N >= {REPORT_MIN_GROUP_COUNT})")
    print(f"{'='*80}")
    print(f"  {'Lemma':<20} {'WEAT':>9}  {'SEAT':>9}  {'SEAT-f':>9}  {'Δ-SEAT':>9}")
    print(f"  {'-'*62}")
    for g in report_profiles:
        lemma = g["lemma"]
        w = weat_scores.get(lemma)
        s = seat_scores.get(lemma)
        sf = seat_full_scores.get(lemma)
        ds = delta_seat_scores.get(lemma)
        w_text = f"{w:+9.4f}" if w is not None else f"{'n/a':>9}"
        s_text = f"{s:+9.4f}" if s is not None else f"{'n/a':>9}"
        sf_text = f"{sf:+9.4f}" if sf is not None else f"{'n/a':>9}"
        ds_text = f"{ds:+9.4f}" if ds is not None else f"{'n/a':>9}"
        print(f"  {lemma:<20} {w_text}  {s_text}  {sf_text}  {ds_text}")

    print(f"\n{'='*70}")
    print(f"  PER-GROUP INDICES (proportionalized, N >= {REPORT_MIN_GROUP_COUNT})")
    print(f"{'='*70}")
    print(f"  {'Lemma':<14} {'Type':<9} {'N':>3}  "
          f"{'AgI':>5} {'PI':>5} {'SI':>5} {'-At':>5} {'+At':>5}")
    print(f"  {'-'*55}")
    for g in report_profiles:
        print(f"  {g['lemma']:<14} {g['type']:<9} {g['total']:>3}  "
              f"{g['AGI']:>5.2f} {g['PI']:>5.2f} {g['SI']:>5.2f} "
              f"{g['neg_atti']:>5.2f} {g['pos_atti']:>5.2f}")

    # ── EFI via PCA ──
    group_profiles = []
    for g in summary["lemmas"]:
        g["weat"] = weat_scores.get(g["lemma"], 0.0)
        g["seat"] = seat_scores.get(g["lemma"], 0.0)
        g["seat_full"] = seat_full_scores.get(g["lemma"], 0.0)
        g["delta_seat"] = delta_seat_scores.get(g["lemma"], 0.0)
        g["net_atti"] = round(g.get("neg_atti", 0.0) - g.get("pos_atti", 0.0), 3)
        group_profiles.append(g)

    analysis_profiles = [g for g in group_profiles if g["total"] >= ANALYSIS_MIN_GROUP_COUNT]

    if len(analysis_profiles) >= 3:
        efi = _compute_efi(analysis_profiles)

        print(f"\n{'='*60}")
        print(
            f"  EFI (PCA)  —  PC1 explains {efi['variance_explained']:.1%} of variance "
            f"(groups with N >= {ANALYSIS_MIN_GROUP_COUNT})"
        )
        print(f"{'='*60}")
        print("  Loadings (= what 'negative framing' consists of):")
        for dim, load in sorted(efi["loadings"].items(), key=lambda x: -abs(x[1])):
            bar = "+" if load > 0 else "-"
            print(f"    {dim:<10} {load:>+6.3f}  {bar * int(abs(load) * 20)}")
        print(f"\n  EFI scores (PC1, higher = more negatively framed):")
        for lemma, score in sorted(efi["scores"].items(), key=lambda x: -x[1]):
            grp = "T" if lemma in TARGET_TOKENS else "C"
            print(f"    [{grp}] {lemma:<16} {score:>+6.3f}")

    # ── Regression: WEAT ~ indices ──
    reg_weat = _run_regression(
        analysis_profiles,
        target_key="weat",
        predictors=["AGI", "PI", "SI", "neg_atti", "pos_atti"],
    )
    if reg_weat:
        print(f"\n{'='*60}")
        print(f"  REGRESSION: WEAT ~ AgI + PI + SI + AttI  (n={reg_weat['n']}, R²={reg_weat['r_squared']})")
        print(f"{'='*60}")
        print(f"  intercept: {reg_weat['intercept']:>+.4f}")
        for pred, coef in reg_weat["coefficients"].items():
            print(f"  β({pred}): {coef:>+.4f}")

    reg_seat = _run_regression(
        analysis_profiles,
        target_key="seat",
        predictors=["AGI", "PI", "SI", "neg_atti", "pos_atti"],
    )
    if reg_seat:
        print(f"\n{'='*60}")
        print(f"  REGRESSION: SEAT ~ AgI + PI + SI + AttI  (n={reg_seat['n']}, R²={reg_seat['r_squared']})")
        print(f"{'='*60}")
        print(f"  intercept: {reg_seat['intercept']:>+.4f}")
        for pred, coef in reg_seat["coefficients"].items():
            print(f"  β({pred}): {coef:>+.4f}")

    reg_seat_net = _run_regression(
        analysis_profiles,
        target_key="seat",
        predictors=["AGI", "PI", "SI", "net_atti"],
    )
    if reg_seat_net:
        print(f"\n{'='*60}")
        print(f"  REGRESSION: SEAT ~ AgI + PI + SI + netAttI  (n={reg_seat_net['n']}, R²={reg_seat_net['r_squared']})")
        print(f"{'='*60}")
        print(f"  intercept: {reg_seat_net['intercept']:>+.4f}")
        for pred, coef in reg_seat_net["coefficients"].items():
            print(f"  β({pred}): {coef:>+.4f}")

    # ── Write TSV outputs ──
    output_tsv = project_dir / "output_results.tsv"
    review_tsv = project_dir / "output_review.tsv"
    if rows:
        try:
            with open(output_tsv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(_row_excel_safe(row) for row in rows)
            print(f"\n→ {output_tsv.name}")
        except PermissionError:
            print(f"\n⚠ {output_tsv.name} is open — skipped")

        review_rows = [
            row for row in rows
            if (
                all(row[key] == 0 for key in ("agi", "pi", "si"))
                or any(row[key] > 2 for key in ("agi", "pi", "si"))
            )
        ]
        try:
            with open(review_tsv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(_row_excel_safe(row) for row in review_rows)
            print(f"→ {review_tsv.name}")
        except PermissionError:
            print(f"⚠ {review_tsv.name} is open — skipped")

    stats_tsv = project_dir / "group_stats.tsv"
    with open(stats_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Lemma", "Type", "N", "AgI", "PI", "SI",
                     "negAttI", "posAttI", "netAttI", "WEAT", "SEAT",
                     "SEAT_full", "delta_SEAT", "EFI_PC1"])
        efi_s = efi["scores"] if len(analysis_profiles) >= 3 else {}
        for g in report_profiles:
            weat_value = weat_scores.get(g["lemma"])
            seat_value = seat_scores.get(g["lemma"])
            seat_full_value = seat_full_scores.get(g["lemma"])
            delta_seat_value = delta_seat_scores.get(g["lemma"])
            w.writerow([
                g["lemma"], g["type"], _excel_safe(g["total"]),
                _excel_safe(g["AGI"]), _excel_safe(g["PI"]), _excel_safe(g["SI"]),
                _excel_safe(g["neg_atti"]), _excel_safe(g["pos_atti"]),
                _excel_safe(g["net_atti"]),
                _excel_safe("" if weat_value is None else weat_value),
                _excel_safe("" if seat_value is None else seat_value),
                _excel_safe("" if seat_full_value is None else seat_full_value),
                _excel_safe("" if delta_seat_value is None else delta_seat_value),
                _excel_safe(efi_s.get(g["lemma"], "")),
            ])
    print(f"→ {stats_tsv.name}")


if __name__ == "__main__":
    main()
