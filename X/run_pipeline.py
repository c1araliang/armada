"""
Pipeline Runner — ARMADA bias detection framework.

Outputs:
  1. Sentence-level discourse collocate discovery (target + contrast groups)
  2. Per-group syntactic indices (AgI, PI, SI) and frame-level AttI
  3. WEAT scores (type-level, embedding vectors against refreshed F⁻/F⁺ frame sets)
  4. CEAT scores (sampled contextual association from sentence embeddings)
  5. EFI via PCA on the group × dimension matrix
  6. Regression: WEAT/CEAT ~ indices
"""

import csv
import hashlib
import json
import os
import re
import sys
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict

import spacy
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

from step2_preprocessing import load_sentences, preprocess
from step3_feature_extraction import extract_all, set_attitude_matcher, set_srl_role_labeler
from step4_metrics import (
    build_sentence_associations,
    compute_association_scores,
    compute_signed_association,
    compute_frame_attitude_indices,
    aggregate_sentence_metrics,
    compute_group_indices,
    cosine_similarity,
)
from lexicons import (
    TARGET_TOKENS, CONTRAST_TOKENS, POLITICAL_GROUP_TOKENS,
    set_semantic_group_resolver,
    resolve_group_token,
)
from semantic_group_resolver import SemanticGroupResolver
from step3_attitudinal_prototypes import AttitudinalPrototypeMatcher
from step3_semantic_roles import SrlRoleLabeler
from embedding_config import ANALYSIS_EMBEDDING_MODEL, DEFAULT_EMBEDDING_BATCH_SIZE

_ALL_GROUPS = TARGET_TOKENS | CONTRAST_TOKENS

# Bootstrap sentence-level prototypes (cosine matching for auto-refresh)
_BOOTSTRAP_NEG_SEEDS = [
    "flood", "flooded", "wave", "surge", "tide", "deluge", "torrent", "overflow", "drown", "inundate",
    "swarm", "horde", "hordes", "flock", "pack", "breed", "nest", "infest", "infestation",
    "invade", "invader", "invasion", "incursion", "encroach", "penetrate",
    "threat", "threaten", "burden", "drain", "crisis", "collapse", "overrun", "overwhelm"
]
_BOOTSTRAP_POS_SEEDS = [
    "contribute", "enrich", "benefit", "boost", "strengthen", "provide",
    "integrate", "embrace", "welcome", "include", "accept",
    "build", "develop", "create", "innovate", "prosper", "thrive", "establish", "empower", "launch"
]
ASSOCIATION_MIN_COUNT = 5
ANALYSIS_MIN_GROUP_COUNT = 50
REPORT_MIN_GROUP_COUNT = 50
FRAME_REFRESH_TOP_N = 60
FRAME_SIM_FLOOR = 0.55
FRAME_SIM_MARGIN = 0.04   # low because seed sentences are long; cosine(word, sentence) < cosine(word, word)


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw_value!r}")
    return value


def _select_analysis_device() -> str:
    override = os.environ.get("ARMADA_ANALYSIS_DEVICE") or os.environ.get("ARMADA_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ANALYSIS_DEVICE = _select_analysis_device()
ANALYSIS_EMB_BATCH_SIZE = _env_int(
    "ARMADA_ANALYSIS_EMB_BATCH_SIZE",
    16 if ANALYSIS_DEVICE == "mps" else DEFAULT_EMBEDDING_BATCH_SIZE,
)
CEAT_FULL_MODE = os.environ.get(
    "ARMADA_CEAT_FULL_MODE",
    os.environ.get("ARMADA_SEAT_FULL_MODE", "reported"),
).lower()
if CEAT_FULL_MODE not in {"reported", "all", "skip"}:
    raise ValueError(
        "ARMADA_CEAT_FULL_MODE must be one of: reported, all, skip "
        f"(got {CEAT_FULL_MODE!r})"
    )
CEAT_MAX_CONTEXTS_PER_GROUP = _env_int("ARMADA_CEAT_MAX_CONTEXTS_PER_GROUP", 500)
CEAT_MIN_CONTEXTS_PER_GROUP = _env_int("ARMADA_CEAT_MIN_CONTEXTS_PER_GROUP", 10)
CEAT_MAX_FRAME_CONTEXTS = _env_int("ARMADA_CEAT_MAX_FRAME_CONTEXTS", 1000)
CEAT_FULL_PROGRESS_EVERY = _env_int("ARMADA_CEAT_FULL_PROGRESS_EVERY", 25_000)


def _excel_safe(value):
    if value == "":
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f" {value}"
    return value


def _row_excel_safe(row: dict) -> dict:
    return {k: _excel_safe(v) for k, v in row.items()}


def _group_report_type(lemma: str) -> str:
    if lemma in POLITICAL_GROUP_TOKENS:
        return "political"
    if lemma in TARGET_TOKENS:
        return "minority"
    if lemma in CONTRAST_TOKENS:
        return "dominant"
    return "unknown"


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
                _group_report_type(target),
                collocate,
                _excel_safe(score_info.get("pair_count", cooc["pair_counts"].get((target, collocate), 0))),
                _excel_safe(score_info.get("target_count", cooc["target_counts"].get(target, 0))),
                _excel_safe(score_info.get("collocate_count", cooc["collocate_counts"].get(collocate, 0))),
                _excel_safe(round(score_info["llr"], 4)),
                _excel_safe(round(score_info["logdice"], 4)),
            ])


def _load_seeds(project_dir: Path) -> tuple[set[str], set[str]]:
    """Load polarity seeds from candidate_terms.json, or fallback to bootstrap.

    Returns (seed_neg, seed_pos, auto_neg, auto_pos):
      - seed_neg/pos: sentence-level prototypes for auto-refresh cosine matching
        and for encoding seed centroids used by WEAT/CEAT.
      - auto_neg/pos: word-level terms accumulated from prior auto-refresh runs;
        used for AttI frame binding (exact lemma match) and candidate exclusion.
    """
    json_path = project_dir / "candidate_terms.json"
    seed_neg: set[str] = set(_BOOTSTRAP_NEG_SEEDS)
    seed_pos: set[str] = set(_BOOTSTRAP_POS_SEEDS)
    auto_neg: set[str] = set()
    auto_pos: set[str] = set()
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
                if "seed_negative_terms" in data:
                    seed_neg = set(data["seed_negative_terms"])
                if "seed_positive_terms" in data:
                    seed_pos = set(data["seed_positive_terms"])
                auto_neg = set(data.get("auto_negative_terms", []))
                auto_pos = set(data.get("auto_positive_terms", []))
                print(f"  Loaded from candidate_terms.json: "
                      f"{len(auto_neg)} accumulated neg frames, {len(auto_pos)} accumulated pos frames")
        except Exception as e:
            print(f"  Warning: could not parse candidate_terms.json ({e}), using bootstrap seeds.")
    return auto_neg, auto_pos


def _load_seed_sentences(project_dir: Path) -> tuple[list[str], list[str]]:
    """Load sentence-level seed prototypes from candidate_terms.json for centroid encoding."""
    json_path = project_dir / "candidate_terms.json"
    neg_seeds = list(_BOOTSTRAP_NEG_SEEDS)
    pos_seeds = list(_BOOTSTRAP_POS_SEEDS)
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if "seed_negative_terms" in data:
                neg_seeds = list(data["seed_negative_terms"])
            if "seed_positive_terms" in data:
                pos_seeds = list(data["seed_positive_terms"])
        except Exception:
            pass
    return neg_seeds, pos_seeds


def _encode_seed_centroids(
    sentence_encoder, neg_seeds: list[str], pos_seeds: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Encode seed sentences and return their mean vectors as F-/F+ centroids."""
    all_seeds = list(dict.fromkeys(neg_seeds + pos_seeds))
    all_vecs = sentence_encoder.encode(
        all_seeds,
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    seed_to_vec = {s: np.asarray(v, dtype=np.float64) for s, v in zip(all_seeds, all_vecs)}
    neg_centroid = np.mean([seed_to_vec[s] for s in neg_seeds if s in seed_to_vec], axis=0)
    pos_centroid = np.mean([seed_to_vec[s] for s in pos_seeds if s in seed_to_vec], axis=0)
    return neg_centroid, pos_centroid


def _refresh_frame_inventory(
    project_dir: Path,
    candidates: list[dict],
    sentence_encoder,
    seed_neg_terms: set[str],
    seed_pos_terms: set[str],
    prior_auto_neg: set[str],
    prior_auto_pos: set[str],
) -> tuple[set[str], set[str], list[dict]]:
    """
    Refresh frame sets from current LLR candidates using seed sentence prototypes.

    seed_neg/pos_terms: sentence-level prototypes (may be long strings); used for
    GTE cosine matching to score candidates.

    Prior auto_neg/pos accumulated from previous runs are carried forward and
    extended with newly admitted candidates. These word-level sets are used by
    AttI (syntactic binding) and as exclusion guard for _find_candidates.
    """
    json_path = project_dir / "candidate_terms.json"
    # Start frame sets from accumulated auto terms (carry forward across runs)
    neg_terms = set(prior_auto_neg)
    pos_terms = set(prior_auto_pos)

    seed_neg_list = sorted(seed_neg_terms)
    seed_pos_list = sorted(seed_pos_terms)

    if not candidates:
        payload = {
            "last_updated": str(date.today()),
            "note": "Current-run candidate frame suggestions derived from sentence-level LLR/LogDice using seed frame prototypes.",
            "seed_negative_terms": seed_neg_list,
            "seed_positive_terms": seed_pos_list,
            "auto_negative_terms": sorted(neg_terms),
            "auto_positive_terms": sorted(pos_terms),
            "candidates": [],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return neg_terms, pos_terms, []

    seed_neg_vecs = sentence_encoder.encode(
        seed_neg_list,
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    seed_pos_vecs = sentence_encoder.encode(
        seed_pos_list,
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_vecs = sentence_encoder.encode(
        [c["term"] for c in candidates],
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    annotated = []
    new_auto_negative: list[str] = []
    new_auto_positive: list[str] = []
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
                if candidate["term"] not in neg_terms:
                    neg_terms.add(candidate["term"])
                    new_auto_negative.append(candidate["term"])
            else:
                if candidate["term"] not in pos_terms:
                    pos_terms.add(candidate["term"])
                    new_auto_positive.append(candidate["term"])

    all_auto_neg = sorted(neg_terms)
    all_auto_pos = sorted(pos_terms)
    payload = {
        "last_updated": str(date.today()),
        "note": "Current-run candidate frame suggestions derived from sentence-level LLR/LogDice using seed frame prototypes.",
        "seed_negative_terms": seed_neg_list,
        "seed_positive_terms": seed_pos_list,
        "auto_negative_terms": all_auto_neg,
        "auto_positive_terms": all_auto_pos,
        "candidates": annotated,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if new_auto_negative:
        print(f"    New auto-negative: {new_auto_negative}")
    if new_auto_positive:
        print(f"    New auto-positive: {new_auto_positive}")

    return neg_terms, pos_terms, annotated


def _compute_weat(
    sentence_encoder,
    neg_centroid: np.ndarray,
    pos_centroid: np.ndarray,
) -> dict[str, float]:
    """WEAT: type-level association using seed-derived centroids.

    Encodes group lemmas and computes cos(lemma, F- centroid) - cos(lemma, F+ centroid).
    Centroids are derived from the seed sentences, not from a flat word list.
    """
    group_terms = sorted(_ALL_GROUPS)
    group_vecs = sentence_encoder.encode(
        group_terms,
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = {}
    for lemma, vec in zip(group_terms, group_vecs):
        vec = np.asarray(vec, dtype=np.float64)
        scores[lemma] = round(
            cosine_similarity(vec, neg_centroid) - cosine_similarity(vec, pos_centroid),
            4,
        )
    return scores


def _stable_context_sample(texts: list[str], key: str, max_items: int) -> list[str]:
    if len(texts) <= max_items:
        return texts
    return sorted(
        texts,
        key=lambda text: hashlib.blake2b(
            f"{key}\t{text}".encode("utf-8"), digest_size=8
        ).digest(),
    )[:max_items]


def _encode_text_map(sentence_encoder, texts: list[str]) -> dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        return {}
    vecs = sentence_encoder.encode(
        unique_texts,
        batch_size=ANALYSIS_EMB_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {text: np.asarray(vec, dtype=np.float64) for text, vec in zip(unique_texts, vecs)}


def _association_summary(
    context_texts: dict[str, list[str]],
    sentence_encoder,
    neg_centroid: np.ndarray,
    pos_centroid: np.ndarray,
    max_contexts: int = CEAT_MAX_CONTEXTS_PER_GROUP,
    min_contexts: int = CEAT_MIN_CONTEXTS_PER_GROUP,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    sampled = {
        lemma: _stable_context_sample(texts, lemma, max_contexts)
        for lemma, texts in context_texts.items()
        if len(texts) >= min_contexts
    }
    text_to_vec = _encode_text_map(
        sentence_encoder,
        [text for texts in sampled.values() for text in texts],
    )

    scores: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for lemma, texts in sampled.items():
        values = []
        for text in texts:
            vec = text_to_vec[text]
            values.append(cosine_similarity(vec, neg_centroid) - cosine_similarity(vec, pos_centroid))
        values_arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(values_arr))
        n = int(len(values_arr))
        sd = float(np.std(values_arr, ddof=1)) if n > 1 else 0.0
        se = sd / float(np.sqrt(n)) if n > 0 else 0.0
        scores[lemma] = round(mean, 4)
        details[lemma] = {
            "n": n,
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "se": round(se, 4),
        }
    return scores, details


def _compute_ceat(
    processed: list[dict],
    sentence_encoder,
    neg_centroid: np.ndarray,
    pos_centroid: np.ndarray,
    target_groups: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """CEAT-style contextual association over sampled group contexts.

    Centroids come from seed sentences (pre-computed), not from scanning the
    corpus for frame words.  Each group sentence is encoded and scored as
    cos(sentence, F- centroid) - cos(sentence, F+ centroid).
    """
    context_texts: dict[str, list[str]] = defaultdict(list)

    for item in processed:
        doc = item["doc"]
        for token in doc:
            resolved = resolve_group_token(token, doc)
            if resolved:
                _, canonical = resolved
                if target_groups is None or canonical in target_groups:
                    context_texts[canonical].append(item["cleaned_text"])

    scores, details = _association_summary(
        context_texts,
        sentence_encoder,
        neg_centroid,
        pos_centroid,
    )
    return scores, details


def _compute_ceat_full(
    lexical_all_path: Path,
    sentence_encoder,
    neg_centroid,
    pos_centroid,
    target_groups: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """CEAT-full over sampled all-lexical-hit contexts."""
    if not lexical_all_path.exists():
        return {}, {}

    groups_to_score = _ALL_GROUPS if target_groups is None else target_groups
    if not groups_to_score:
        return {}, {}

    group_lookup: dict[str, str] = {}
    alternatives = []
    for i, lemma in enumerate(sorted(groups_to_score, key=len, reverse=True)):
        group_name = f"G{i}"
        group_lookup[group_name] = lemma
        pattern = re.escape(lemma).replace(r"\-", r"[\s-]+").replace(r"\ ", r"[\s-]+")
        alternatives.append(f"(?P<{group_name}>{pattern})")
    token_re = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)

    context_texts: dict[str, list[str]] = defaultdict(list)
    seen = 0
    matched = 0
    with open(lexical_all_path, encoding="utf-8") as fh:
        for line in fh:
            seen += 1
            if seen % CEAT_FULL_PROGRESS_EVERY == 0:
                print(f"  CEAT-full scanned {seen:,} lexical-hit lines...")
            sentence = line.rstrip("\n")
            if not sentence:
                continue
            matches = {
                group_lookup[m.lastgroup]
                for m in token_re.finditer(sentence)
                if m.lastgroup is not None
            }
            if not matches:
                continue
            matched += 1
            for lemma in matches:
                context_texts[lemma].append(sentence)

    scores, details = _association_summary(
        context_texts,
        sentence_encoder,
        neg_centroid,
        pos_centroid,
    )
    encoded = sum(detail["n"] for detail in details.values())
    print(
        f"  CEAT-full scanned {seen:,} lines, matched {matched:,}, "
        f"encoded {encoded:,} sampled contexts."
    )
    return scores, details


def _compute_efi(group_profiles: list[dict]) -> dict:
    """PCA on group × [AgI, PI, SI, frame-netAttI, WEAT, CEAT] matrix."""
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    labels = [g["lemma"] for g in group_profiles]
    matrix = np.array([[g.get(d, 0.0) for d in dims] for g in group_profiles])

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    pca = PCA(n_components=min(len(dims), len(labels)))
    scores = pca.fit_transform(X)

    pc1 = pca.components_[0]
    raw_scores = scores[:, 0]

    # Orient PC1 so that "negative framing" is positive:
    # high PI + high frame-netAttI + high WEAT/CEAT and low AgI/SI should → positive EFI.
    agi_idx = dims.index("AGI")
    pi_idx  = dims.index("PI")
    si_idx = dims.index("SI")
    net_atti_idx = dims.index("net_atti")
    weat_idx = dims.index("weat")
    ceat_idx = dims.index("ceat")
    orientation_anchor = (
        pc1[pi_idx]
        + pc1[net_atti_idx]
        + pc1[weat_idx]
        + pc1[ceat_idx]
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
    existing_frames: set[str] | None = None,
) -> list[dict]:
    """High-association collocates of target groups not yet in the frame taxonomy."""
    if existing_frames is None:
        existing_frames = set()
    minority_llr: dict[str, float] = {}
    minority_logdice: dict[str, float] = {}
    dominant_llr: dict[str, float] = {}
    dominant_logdice: dict[str, float] = {}

    for (target, collocate), score_info in association_scores.items():
        if target in POLITICAL_GROUP_TOKENS:
            continue
        if collocate in existing_frames:
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
                              and t in TARGET_TOKENS
                              and t not in POLITICAL_GROUP_TOKENS})
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
    print(f"  analysis device: {ANALYSIS_DEVICE}")
    print(f"  analysis embedding batch: {ANALYSIS_EMB_BATCH_SIZE}")
    print(f"  CEAT-full mode: {CEAT_FULL_MODE}")
    semantic_resolver = SemanticGroupResolver(device=ANALYSIS_DEVICE)
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

    # ── Feature extraction (AgI/PI/SI + local AttI diagnostics) ──
    # Cache SRL results keyed on input file mtime so downstream-only changes
    # don't require a full re-run of the ~10-min SRL extraction step.
    import pickle, hashlib as _hl
    _cache_path = project_dir / "srl_cache.pkl"
    _cache_key = _hl.sha1(
        f"{sentences_path}:{sentences_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    if _cache_path.exists():
        try:
            with open(_cache_path, "rb") as _f:
                _cached = pickle.load(_f)
            if _cached.get("key") == _cache_key:
                extracted = _cached["extracted"]
                print(f"SRL cache hit — loaded {len(extracted)} extraction results.")
            else:
                raise ValueError("stale cache")
        except Exception as _e:
            print(f"SRL cache miss ({_e}), running extraction...")
            extracted = extract_all(processed)
            with open(_cache_path, "wb") as _f:
                pickle.dump({"key": _cache_key, "extracted": extracted}, _f)
    else:
        extracted = extract_all(processed)
        with open(_cache_path, "wb") as _f:
            pickle.dump({"key": _cache_key, "extracted": extracted}, _f)
    n_findings = sum(1 for e in extracted if e["findings"])
    print(f"Target tokens in {n_findings}/{len(extracted)} sentences.\n")

    # ── Discourse association discovery ──
    cooc = build_sentence_associations(processed, min_count=ASSOCIATION_MIN_COUNT)
    association_scores = compute_association_scores(cooc)

    association_tsv = project_dir / "association_discourse.tsv"
    _write_discourse_association(association_tsv, cooc, association_scores)
    print(
        f"Discourse association written to: {association_tsv.name} "
        f"(sentence-level, min pair count = {ASSOCIATION_MIN_COUNT}, min distance = {cooc['min_distance']})"
    )

    # Load seeds and pre-compute seed centroids (used by WEAT, CEAT, CEAT-full)
    auto_neg_frames, auto_pos_frames = _load_seeds(project_dir)
    seed_neg_terms, seed_pos_terms = _load_seed_sentences(project_dir)
    seed_neg_centroid, seed_pos_centroid = _encode_seed_centroids(
        semantic_resolver.model, seed_neg_terms, seed_pos_terms
    )

    existing_frames = auto_neg_frames | auto_pos_frames
    candidates = _find_candidates(association_scores, top_n=FRAME_REFRESH_TOP_N, existing_frames=existing_frames)
    neg_frames, pos_frames, annotated_candidates = _refresh_frame_inventory(
        project_dir,
        candidates,
        semantic_resolver.model,
        seed_neg_terms,
        seed_pos_terms,
        auto_neg_frames,
        auto_pos_frames,
    )
    n_auto_neg = len(neg_frames - auto_neg_frames)
    n_auto_pos = len(pos_frames - auto_pos_frames)
    print(
        f"Candidate terms refreshed: {len(annotated_candidates)} total, "
        f"{n_auto_neg} auto-negative, {n_auto_pos} auto-positive"
    )
    frame_atti_scores = compute_frame_attitude_indices(
        processed,
        neg_frames=neg_frames,
        pos_frames=pos_frames,
    )
    frame_sign = {term: -1 for term in neg_frames} | {term: 1 for term in pos_frames}
    signed = compute_signed_association(association_scores, score_key="llr", frame_sign=frame_sign)
    rows = aggregate_sentence_metrics(
        extracted,
        signed,
        processed,
        neg_frames=neg_frames,
        pos_frames=pos_frames,
    )

    # ── WEAT + CEAT (seed-centroid based) ──
    weat_scores = _compute_weat(semantic_resolver.model, seed_neg_centroid, seed_pos_centroid)

    # ── Group indices ──
    summary = compute_group_indices(extracted)
    report_profiles = [g for g in summary["lemmas"] if g["total"] >= REPORT_MIN_GROUP_COUNT]
    reported_group_lemmas = {g["lemma"] for g in report_profiles}

    ceat_scores, ceat_details = _compute_ceat(
        processed,
        sentence_encoder=semantic_resolver.model,
        neg_centroid=seed_neg_centroid,
        pos_centroid=seed_pos_centroid,
        target_groups=reported_group_lemmas,
    )
    ceat_neg_centroid, ceat_pos_centroid = seed_neg_centroid, seed_pos_centroid

    # ── CEAT-full (sampled all lexical hits) + Δ-CEAT ──
    lexical_all_path = sentences_path.parent / "semantic_filter_lexical_all.txt"
    ceat_full_scores = {}
    ceat_full_details = {}
    delta_ceat_scores = {}
    if CEAT_FULL_MODE == "skip":
        print("Skipping CEAT-full (ARMADA_CEAT_FULL_MODE=skip).")
    elif ceat_neg_centroid is not None and ceat_pos_centroid is not None:
        if CEAT_FULL_MODE == "all":
            ceat_full_target_groups = None
            mode_note = "all lexical-hit groups"
        else:
            ceat_full_target_groups = reported_group_lemmas
            mode_note = f"reported groups only (N >= {REPORT_MIN_GROUP_COUNT})"
        print(f"Computing CEAT-full from sampled lexical hits: {mode_note}...")
        # Cache CEAT-full results keyed on inputs that affect the output:
        # lexical_all mtime, target groups, sampling caps, encoder, and the
        # seed centroid bytes. Cache miss triggers a fresh compute; subsequent
        # runs with identical inputs skip straight to the cached scores.
        import pickle, hashlib as _hl
        _ceat_cache_path = project_dir / "ceat_full_cache.pkl"
        _centroid_digest = _hl.sha1(
            ceat_neg_centroid.tobytes() + ceat_pos_centroid.tobytes()
        ).hexdigest()
        _ceat_cache_key = _hl.sha1(
            "|".join([
                str(lexical_all_path),
                str(lexical_all_path.stat().st_mtime_ns) if lexical_all_path.exists() else "missing",
                "all" if ceat_full_target_groups is None else ",".join(sorted(ceat_full_target_groups)),
                str(CEAT_MAX_CONTEXTS_PER_GROUP),
                str(CEAT_MIN_CONTEXTS_PER_GROUP),
                ANALYSIS_EMBEDDING_MODEL,
                _centroid_digest,
            ]).encode()
        ).hexdigest()
        ceat_full_scores = None
        ceat_full_details = None
        if _ceat_cache_path.exists():
            try:
                with open(_ceat_cache_path, "rb") as _f:
                    _cached = pickle.load(_f)
                if _cached.get("key") == _ceat_cache_key:
                    ceat_full_scores = _cached["scores"]
                    ceat_full_details = _cached["details"]
                    print(
                        f"  CEAT-full cache hit — loaded {len(ceat_full_scores)} group scores."
                    )
            except Exception as _e:
                print(f"  CEAT-full cache unreadable ({_e}), recomputing...")
        if ceat_full_scores is None:
            ceat_full_scores, ceat_full_details = _compute_ceat_full(
                lexical_all_path,
                sentence_encoder=semantic_resolver.model,
                neg_centroid=ceat_neg_centroid,
                pos_centroid=ceat_pos_centroid,
                target_groups=ceat_full_target_groups,
            )
            try:
                with open(_ceat_cache_path, "wb") as _f:
                    pickle.dump(
                        {"key": _ceat_cache_key, "scores": ceat_full_scores, "details": ceat_full_details},
                        _f,
                    )
            except Exception as _e:
                print(f"  CEAT-full cache write failed ({_e})")
        for lemma in set(ceat_scores) | set(ceat_full_scores):
            s_filtered = ceat_scores.get(lemma)
            s_full = ceat_full_scores.get(lemma)
            if s_filtered is not None and s_full is not None:
                delta_ceat_scores[lemma] = round(s_full - s_filtered, 4)
        print(f"  CEAT-full computed for {len(ceat_full_scores)} groups, Δ-CEAT for {len(delta_ceat_scores)}")

    print(f"\n{'='*80}")
    print(f"  WEAT / CEAT SCORES  (>0 = closer to F⁻ than F⁺, N >= {REPORT_MIN_GROUP_COUNT})")
    print(f"{'='*80}")
    print(f"  {'Lemma':<20} {'WEAT':>9}  {'CEAT':>9}  {'CEAT-f':>9}  {'Δ-CEAT':>9} {'N':>5}")
    print(f"  {'-'*70}")
    for g in report_profiles:
        lemma = g["lemma"]
        w = weat_scores.get(lemma)
        s = ceat_scores.get(lemma)
        sf = ceat_full_scores.get(lemma)
        ds = delta_ceat_scores.get(lemma)
        cn = ceat_details.get(lemma, {}).get("n")
        w_text = f"{w:+9.4f}" if w is not None else f"{'n/a':>9}"
        s_text = f"{s:+9.4f}" if s is not None else f"{'n/a':>9}"
        sf_text = f"{sf:+9.4f}" if sf is not None else f"{'n/a':>9}"
        ds_text = f"{ds:+9.4f}" if ds is not None else f"{'n/a':>9}"
        n_text = f"{cn:>5}" if cn is not None else f"{'n/a':>5}"
        print(f"  {lemma:<20} {w_text}  {s_text}  {sf_text}  {ds_text} {n_text}")

    print(f"\n{'='*70}")
    print(f"  PER-GROUP INDICES (proportionalized, N >= {REPORT_MIN_GROUP_COUNT})")
    print(f"{'='*70}")
    print(f"  {'Lemma':<14} {'Type':<9} {'N':>3}  "
          f"{'Subj':>5} {'AgI':>5} {'PI':>5} {'SI':>5} {'F-':>5} {'F+':>5} {'netAt':>6}")
    print(f"  {'-'*69}")
    for g in report_profiles:
        frame_atti = frame_atti_scores.get(g["lemma"], {})
        print(f"  {g['lemma']:<14} {g['type']:<9} {g['total']:>3}  "
              f"{g['subjecthood']:>5.2f} {g['AGI']:>5.2f} {g['PI']:>5.2f} {g['SI']:>5.2f} "
              f"{frame_atti.get('frame_neg_atti', 0.0):>5.2f} "
              f"{frame_atti.get('frame_pos_atti', 0.0):>5.2f} "
              f"{frame_atti.get('net_atti', 0.0):>6.2f}")

    # ── EFI via PCA ──
    group_profiles = []
    for g in summary["lemmas"]:
        frame_atti = frame_atti_scores.get(g["lemma"], {})
        g["local_neg_atti"] = g.get("neg_atti", 0.0)
        g["local_pos_atti"] = g.get("pos_atti", 0.0)
        g["frame_neg_atti"] = frame_atti.get("frame_neg_atti", 0.0)
        g["frame_pos_atti"] = frame_atti.get("frame_pos_atti", 0.0)
        g["net_atti"] = frame_atti.get("net_atti", 0.0)
        g["frame_review"] = frame_atti.get("frame_review", 0.0)
        g["weat"] = weat_scores.get(g["lemma"], 0.0)
        g["ceat"] = ceat_scores.get(g["lemma"], 0.0)
        g["ceat_full"] = ceat_full_scores.get(g["lemma"], 0.0)
        g["delta_ceat"] = delta_ceat_scores.get(g["lemma"], 0.0)
        g["ceat_n"] = ceat_details.get(g["lemma"], {}).get("n", 0)
        g["ceat_se"] = ceat_details.get(g["lemma"], {}).get("se", 0.0)
        g["ceat_full_n"] = ceat_full_details.get(g["lemma"], {}).get("n", 0)
        g["ceat_full_se"] = ceat_full_details.get(g["lemma"], {}).get("se", 0.0)
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
            grp = "P" if lemma in POLITICAL_GROUP_TOKENS else ("T" if lemma in TARGET_TOKENS else "C")
            print(f"    [{grp}] {lemma:<16} {score:>+6.3f}")

    # ── Regression: WEAT/CEAT ~ indices ──
    reg_weat = _run_regression(
        analysis_profiles,
        target_key="weat",
        predictors=["AGI", "PI", "SI", "net_atti"],
    )
    if reg_weat:
        print(f"\n{'='*60}")
        print(f"  REGRESSION: WEAT ~ AgI + PI + SI + frame-netAttI  (n={reg_weat['n']}, R²={reg_weat['r_squared']})")
        print(f"{'='*60}")
        print(f"  intercept: {reg_weat['intercept']:>+.4f}")
        for pred, coef in reg_weat["coefficients"].items():
            print(f"  β({pred}): {coef:>+.4f}")

    reg_ceat = _run_regression(
        analysis_profiles,
        target_key="ceat",
        predictors=["AGI", "PI", "SI", "net_atti"],
    )
    if reg_ceat:
        print(f"\n{'='*60}")
        print(f"  REGRESSION: CEAT ~ AgI + PI + SI + frame-netAttI  (n={reg_ceat['n']}, R²={reg_ceat['r_squared']})")
        print(f"{'='*60}")
        print(f"  intercept: {reg_ceat['intercept']:>+.4f}")
        for pred, coef in reg_ceat["coefficients"].items():
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
                or row.get("role_review_flags") != "null"
                or row.get("frame_binding_flags") != "null"
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
        w.writerow(["Lemma", "Type", "N", "Subjecthood", "AgI", "PI", "SI",
                     "local_negAttI", "local_posAttI",
                     "frame_negAttI", "frame_posAttI", "netAttI", "frameReview",
                     "WEAT", "CEAT", "CEAT_N", "CEAT_SE",
                     "CEAT_full", "CEAT_full_N", "CEAT_full_SE", "delta_CEAT",
                     "EFI_PC1"])
        efi_s = efi["scores"] if len(analysis_profiles) >= 3 else {}
        for g in report_profiles:
            weat_value = weat_scores.get(g["lemma"])
            ceat_value = ceat_scores.get(g["lemma"])
            ceat_full_value = ceat_full_scores.get(g["lemma"])
            delta_ceat_value = delta_ceat_scores.get(g["lemma"])
            ceat_detail = ceat_details.get(g["lemma"], {})
            ceat_full_detail = ceat_full_details.get(g["lemma"], {})
            w.writerow([
                g["lemma"], g["type"], _excel_safe(g["total"]),
                _excel_safe(g["subjecthood"]),
                _excel_safe(g["AGI"]), _excel_safe(g["PI"]), _excel_safe(g["SI"]),
                _excel_safe(g["local_neg_atti"]), _excel_safe(g["local_pos_atti"]),
                _excel_safe(g["frame_neg_atti"]), _excel_safe(g["frame_pos_atti"]),
                _excel_safe(g["net_atti"]), _excel_safe(g["frame_review"]),
                _excel_safe("" if weat_value is None else weat_value),
                _excel_safe("" if ceat_value is None else ceat_value),
                _excel_safe(ceat_detail.get("n", "")),
                _excel_safe(ceat_detail.get("se", "")),
                _excel_safe("" if ceat_full_value is None else ceat_full_value),
                _excel_safe(ceat_full_detail.get("n", "")),
                _excel_safe(ceat_full_detail.get("se", "")),
                _excel_safe("" if delta_ceat_value is None else delta_ceat_value),
                _excel_safe(efi_s.get(g["lemma"], "")),
            ])
    print(f"→ {stats_tsv.name}")


if __name__ == "__main__":
    main()
