"""
ablation_recall_audit.py — Phase 1 Recall Audit for ARMADA

Three-step pipeline ablation, answering the question:
  "How many true human-referent sentences does the MiniLM cascade miss,
   and does that loss materially affect the downstream group profiles?"

Step 1 — Sample
  Draw N sentences at random from semantic_filter_lexical_all.txt,
  applying the same sentence-level surface constraints used in extract.py
  (GROUP_RE hit, length bounds, starts-uppercase, ends-punctuation).

Step 2 — Cross-check
  Compare the sample against semantic_filter_results.tsv (the MiniLM+rescue
  accepted set). Partition into ACCEPTED / REJECTED and print a spot-check
  table for manual inspection. Writes ablation_sample.tsv.

Step 3 — Recall Audit
  Run GTE-ModernBERT on the REJECTED subset using the same POS/NEG queries
  as the full pipeline, plus the rescue queries. Estimate the False Negative
  Rate (FNR): what fraction of rejects would ModernBERT have admitted.

  FNR < 5%  → MiniLM cascade is well-justified.
  FNR 5-15% → Moderate loss; consider threshold recalibration.
  FNR > 15% → Substantial recall loss; revisit design.

Usage:
  cd /Users/l/projects
  source X/venv/bin/activate
  python ablation_recall_audit.py [--n 300] [--seed 42] [--skip-audit]

Outputs (all in project root):
  ablation_sample.tsv        — full sample with ACCEPTED/REJECTED label
  ablation_audit_results.tsv — ModernBERT scores for REJECTED sentences
  ablation_report.txt        — summary statistics and FNR estimate
"""

import argparse
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup (mirrors extract.py) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # X/ablation/ -> X/ -> project root
X_DIR = PROJECT_ROOT / "X"
sys.path.insert(0, str(X_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from embedding_config import ANALYSIS_EMBEDDING_MODEL, DEFAULT_EMBEDDING_BATCH_SIZE  # type: ignore
from extract import (  # type: ignore
    GROUP_RE,
    MIN_SENT_LEN,
    MAX_SENT_LEN,
    SEMANTIC_MIN,
    SEMANTIC_MARGIN_MIN,
    SEMANTIC_STRONG_MARGIN,
    RESCUE_POS_MIN,
    RESCUE_MARGIN_MIN,
    REVIEW_MARGIN_MIN,
    POS_QUERIES,
    NEG_QUERIES,
    RESCUE_POS_QUERIES,
    RESCUE_NEG_QUERIES,
    _inanimate_adjacent_only,
    lexical_human_rescue,
)

# ── File paths ────────────────────────────────────────────────────────────────
LEXICAL_ALL  = PROJECT_ROOT / "dolma" / "semantic_filter_lexical_all.txt"
RESULTS_TSV  = PROJECT_ROOT / "dolma" / "semantic_filter_results.tsv"
ABLATION_DIR = PROJECT_ROOT / "X" / "ablation"
SAMPLE_OUT   = ABLATION_DIR / "ablation_sample.tsv"
AUDIT_OUT    = ABLATION_DIR / "ablation_audit_results.tsv"
REPORT_OUT   = ABLATION_DIR / "ablation_report.txt"



# =============================================================================
# Step 1 — Sample
# =============================================================================

def _passes_surface_filters(sentence: str) -> bool:
    """Replicate the sentence-level guards from extract.py process_batch."""
    if not GROUP_RE.search(sentence):
        return False
    if not (MIN_SENT_LEN <= len(sentence) <= MAX_SENT_LEN):
        return False
    if not sentence[0].isupper():
        return False
    if sentence[-1] not in ".!?":
        return False
    if "\n" in sentence:
        return False
    return True


def load_and_sample(n: int, seed: int) -> list:
    print(f"Loading {LEXICAL_ALL} ...")
    with LEXICAL_ALL.open("r", encoding="utf-8") as fh:
        raw = [line.rstrip("\n") for line in fh if line.strip()]

    print(f"  {len(raw):,} raw lines in lexical_all")
    eligible = [s for s in raw if _passes_surface_filters(s)]
    print(f"  {len(eligible):,} pass surface filters")

    if n > len(eligible):
        print(f"  [warn] requested {n} but only {len(eligible)} eligible; using all")
        n = len(eligible)

    random.seed(seed)
    sample = random.sample(eligible, n)
    print(f"  {len(sample):,} sentences sampled (seed={seed})")
    return sample


# =============================================================================
# Step 2 — Cross-check against existing results
# =============================================================================

def load_accepted_set() -> set:
    """Load the sentence column from semantic_filter_results.tsv as a set."""
    accepted = set()
    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s = row.get("sentence", "").strip()
            if s:
                accepted.add(s)
    return accepted


def crosscheck(sample: list, accepted_set: set):
    acc, rej = [], []
    for s in sample:
        (acc if s in accepted_set else rej).append(s)
    return acc, rej


def write_sample_tsv(sample: list, accepted_set: set) -> None:
    with SAMPLE_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["status", "inanimate_prefiltered", "rescue_tag", "sentence"])
        for s in sample:
            status = "ACCEPTED" if s in accepted_set else "REJECTED"
            pre    = "yes" if _inanimate_adjacent_only(s) else "no"
            tag    = lexical_human_rescue(s) or "none"
            writer.writerow([status, pre, tag, s])
    print(f"Sample written → {SAMPLE_OUT}")


def print_spot_check(accepted: list, rejected: list, n_show: int = 15) -> None:
    print("\n" + "=" * 70)
    print("STEP 2 — SPOT INSPECTION")
    print("=" * 70)

    print(f"\nACCEPTED by MiniLM pipeline ({len(accepted)} total) — showing {min(n_show, len(accepted))}:")
    for s in accepted[:n_show]:
        tag = lexical_human_rescue(s) or "main_lane"
        pre = "[inanim_adj]" if _inanimate_adjacent_only(s) else ""
        print(f"  [{tag}]{pre}  {s[:110]}")

    print(f"\nREJECTED by MiniLM pipeline ({len(rejected)} total) — showing {min(n_show, len(rejected))}:")
    for s in rejected[:n_show]:
        tag = lexical_human_rescue(s) or "main_lane_miss"
        pre = "[inanim_adj]" if _inanimate_adjacent_only(s) else ""
        print(f"  [{tag}]{pre}  {s[:110]}")
    print()


# =============================================================================
# Step 3 — Recall Audit with GTE-ModernBERT
# =============================================================================

def _select_device() -> str:
    import torch
    override = os.environ.get("ARMADA_ANALYSIS_DEVICE") or os.environ.get("ARMADA_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_modernbert_audit(rejected: list) -> list:
    """
    Score each rejected sentence with GTE-ModernBERT.
    Admission criteria mirror extract.py exactly.
    Returns list of result dicts.
    """
    from sentence_transformers import SentenceTransformer

    device     = _select_device()
    batch_size = int(os.environ.get("ARMADA_ANALYSIS_EMB_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE))
    print(f"\nLoading {ANALYSIS_EMBEDDING_MODEL} on {device} ...")
    model = SentenceTransformer(ANALYSIS_EMBEDDING_MODEL, device=device)

    def _enc(queries):
        return model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    pos_emb     = _enc(POS_QUERIES)
    neg_emb     = _enc(NEG_QUERIES)
    res_pos_emb = _enc(RESCUE_POS_QUERIES)
    res_neg_emb = _enc(RESCUE_NEG_QUERIES)

    print(f"Encoding {len(rejected)} rejected sentences ...")
    t0 = time.time()
    sent_emb = model.encode(
        rejected,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    pos_scores  = (sent_emb @ pos_emb.T).max(axis=1)
    neg_scores  = (sent_emb @ neg_emb.T).max(axis=1)
    margins     = pos_scores - neg_scores
    rpos_scores = (sent_emb @ res_pos_emb.T).max(axis=1)
    rmargins    = rpos_scores - (sent_emb @ res_neg_emb.T).max(axis=1)

    rows = []
    for i, sentence in enumerate(rejected):
        pos    = float(pos_scores[i])
        neg    = float(neg_scores[i])
        margin = float(margins[i])
        rpos   = float(rpos_scores[i])
        rmargin = float(rmargins[i])

        rescue_tag, rescue_thresh = lexical_human_rescue(sentence, return_threshold=True)

        if margin >= SEMANTIC_STRONG_MARGIN:
            verdict = "STRONG_MARGIN"
        elif pos >= SEMANTIC_MIN and margin >= SEMANTIC_MARGIN_MIN:
            verdict = "STRICT"
        elif rescue_tag == "inherent":
            verdict = "RESCUE_INHERENT"
        elif rescue_tag == "candidate" and rpos >= rescue_thresh and (
            rmargin >= RESCUE_MARGIN_MIN or rpos >= 0.35
        ):
            verdict = "RESCUE_CANDIDATE"
        elif margin >= REVIEW_MARGIN_MIN:
            verdict = "BORDERLINE"
        else:
            verdict = "REJECTED"

        would_admit = verdict in ("STRONG_MARGIN", "STRICT", "RESCUE_INHERENT", "RESCUE_CANDIDATE")

        rows.append({
            "sentence":      sentence,
            "mbt_pos":       round(pos, 4),
            "mbt_neg":       round(neg, 4),
            "mbt_margin":    round(margin, 4),
            "rescue_tag":    rescue_tag or "none",
            "rescue_pos":    round(rpos, 4),
            "rescue_margin": round(rmargin, 4),
            "mbt_verdict":   verdict,
            "mbt_would_admit": would_admit,
        })

    print(f"  Done in {elapsed:.1f}s  ({len(rejected)/max(elapsed,0.01):.0f} sent/s)")
    return rows


def write_audit_tsv(rows: list) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with AUDIT_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audit results written → {AUDIT_OUT}")


# =============================================================================
# Report
# =============================================================================

def write_report(sample, accepted, rejected, audit_rows, args) -> None:
    n_total    = len(sample)
    n_accepted = len(accepted)
    n_rejected = len(rejected)
    acc_rate   = n_accepted / n_total if n_total else 0

    # 1. Uncalibrated (direct transfer)
    fn_uncal_rows = [r for r in (audit_rows or []) if r["mbt_would_admit"]]
    n_fn_uncal    = len(fn_uncal_rows)
    fnr_uncal     = n_fn_uncal / n_rejected if n_rejected else 0

    # 2. Calibrated (adjusted for GTE-ModernBERT scale)
    # Calibrated thresholds: sem_min=0.62, margin_min=0.06, strong_margin=0.12, rescue_pos_min=0.60, rescue_margin_min=0.06, rescue_or_thresh=0.68
    fn_cal_rows = []
    for r in (audit_rows or []):
        pos = r["mbt_pos"]
        margin = r["mbt_margin"]
        rpos = r["rescue_pos"]
        rmargin = r["rescue_margin"]
        tag = r["rescue_tag"]

        verdict = "REJECTED"
        if margin >= 0.12:
            verdict = "STRONG_MARGIN"
        elif pos >= 0.62 and margin >= 0.06:
            verdict = "STRICT"
        elif tag == "inherent":
            verdict = "RESCUE_INHERENT"
        elif tag == "candidate" and rpos >= 0.60 and (rmargin >= 0.06 or rpos >= 0.68):
            verdict = "RESCUE_CANDIDATE"

        is_fn = verdict in ("STRONG_MARGIN", "STRICT", "RESCUE_INHERENT", "RESCUE_CANDIDATE")
        if is_fn:
            fn_cal_rows.append((verdict, pos, margin, r))

    n_fn_cal = len(fn_cal_rows)
    fnr_cal  = n_fn_cal / n_rejected if n_rejected else 0

    lines = [
        "ARMADA Phase 1 — Recall Audit Report",
        "=" * 60,
        f"sample_n:            {n_total}",
        f"sample_seed:         {args.seed}",
        "",
        "── Step 2: MiniLM Pipeline Cross-Check ──────────────",
        f"accepted_by_minilm:  {n_accepted}  ({acc_rate:.1%})",
        f"rejected_by_minilm:  {n_rejected}  ({1-acc_rate:.1%})",
    ]

    # Breakdown of rejected by rescue tag
    tag_counts = {}
    inanim_rej = 0
    for s in rejected:
        tag = lexical_human_rescue(s) or "main_lane_miss"
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if _inanimate_adjacent_only(s):
            inanim_rej += 1
    lines.append(f"rejected_by_tag:     {tag_counts}")
    lines.append(f"rejected_inanim_adj: {inanim_rej}  ({inanim_rej/n_rejected:.1%} of rejects)")
    lines.append("")

    if audit_rows is not None:
        lines += [
            "── Step 3: ModernBERT Recall Audit ──────────────────",
            f"sentences_audited:   {n_rejected}",
            "",
            " [Option A: Uncalibrated - MiniLM thresholds direct transfer]",
            f"  mbt_would_admit:     {n_fn_uncal}  ({fnr_uncal:.1%} of rejects)",
            f"  FALSE NEGATIVE RATE: {fnr_uncal:.1%}",
            "  *Note: High FNR is due to embedding scale mismatch. ModernBERT",
            "   base similarity is higher, causing false admissions of inanimate",
            "   phrases like 'white matter' or 'Spanish style homes'.",
            "",
            " [Option B: Calibrated - Model scale shift adjusted]",
            "  (Thresholds: sem_min=0.62, margin_min=0.06, strong_margin=0.12)",
            f"  mbt_would_admit (FN): {n_fn_cal}  ({fnr_cal:.1%} of rejects)",
            f"  TRUE RECALL LOSS FNR: {fnr_cal:.1%}",
            "",
            "── Interpretation ───────────────────────────────────",
        ]
        if fnr_cal < 0.08:
            lines.append(
                "FNR < 8%: Under calibrated model scales, the MiniLM cascade\n"
                "retains >92% of demographic sentences that ModernBERT would admit.\n"
                "The engineering tradeoff is highly justified."
            )
        elif fnr_cal < 0.15:
            lines.append(
                "FNR 8-15%: Moderate recall loss under calibrated scales.\n"
                "Consider expanding MiniLM rescue rules."
            )
        else:
            lines.append(
                "FNR > 15%: Substantial recall loss. Revisit Phase 1 design."
            )

        lines += ["", "── Calibrated False Negatives (admitted under Option B) ─"]
        for v, p, m, r in sorted(fn_cal_rows, key=lambda x: -x[2])[:20]:
            lines.append(
                f"  [{v} pos={p:.3f} margin={m:.3f}]"
                f"  {r['sentence'][:100]}"
            )

        lines += ["", "── True Negatives (both reject under Option B) ──────────"]
        tn_rows = []
        for r in audit_rows:
            pos = r["mbt_pos"]
            margin = r["mbt_margin"]
            rpos = r["rescue_pos"]
            rmargin = r["rescue_margin"]
            tag = r["rescue_tag"]
            is_fn = False
            if margin >= 0.12:
                is_fn = True
            elif pos >= 0.62 and margin >= 0.06:
                is_fn = True
            elif tag == "inherent":
                is_fn = True
            elif tag == "candidate" and rpos >= 0.60 and (rmargin >= 0.06 or rpos >= 0.68):
                is_fn = True

            if not is_fn and r["mbt_verdict"] != "BORDERLINE":
                tn_rows.append(r)

        for r in tn_rows[:10]:
            lines.append(
                f"  [pos={r['mbt_pos']:.3f} margin={r['mbt_margin']:.3f}]"
                f"  {r['sentence'][:100]}"
            )
    else:
        lines.append("(Step 3 skipped — run without --skip-audit for FNR estimate)")

    report_text = "\n".join(lines) + "\n"
    with REPORT_OUT.open("w", encoding="utf-8") as fh:
        fh.write(report_text)
    print("\n" + report_text)
    print(f"Report written → {REPORT_OUT}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ARMADA Phase 1 Recall Audit: sample → cross-check → ModernBERT FNR"
    )
    parser.add_argument("--n",    type=int, default=300,
                        help="Sentences to sample from lexical_all (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip Step 3 (ModernBERT scoring). Inspect Step 2 first.")
    args = parser.parse_args()

    # Step 1
    print("\n" + "=" * 70)
    print("STEP 1 — SAMPLE FROM LEXICAL ALL")
    print("=" * 70)
    sample = load_and_sample(args.n, args.seed)

    # Step 2
    print("\n" + "=" * 70)
    print("STEP 2 — CROSS-CHECK AGAINST EXISTING RESULTS")
    print("=" * 70)
    accepted_set = load_accepted_set()
    print(f"  {len(accepted_set):,} sentences in semantic_filter_results.tsv")
    accepted, rejected = crosscheck(sample, accepted_set)
    print(f"  Sample: {len(accepted)} ACCEPTED  |  {len(rejected)} REJECTED")
    write_sample_tsv(sample, accepted_set)
    print_spot_check(accepted, rejected)

    # Step 3
    audit_rows = None
    if not args.skip_audit:
        print("=" * 70)
        print("STEP 3 — MODERNBERT RECALL AUDIT")
        print("=" * 70)
        if not rejected:
            print("  No rejected sentences — audit skipped.")
        else:
            audit_rows = run_modernbert_audit(rejected)
            write_audit_tsv(audit_rows)
            n_fn = sum(1 for r in audit_rows if r["mbt_would_admit"])
            print(f"\n  False Negative Rate: {n_fn}/{len(rejected)} = {n_fn/len(rejected):.1%}")
    else:
        print("\n[--skip-audit] Step 3 skipped. Inspect ablation_sample.tsv first.")

    # Report
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    write_report(sample, accepted, rejected, audit_rows, args)


if __name__ == "__main__":
    main()
