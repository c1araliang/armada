#!/usr/bin/env python3
"""
Robustness Checks for the framing analysis pipeline.
Performs:
1. Bootstrap CIs (B=1000) for group-level metrics, PCA (EFI) loadings, and OLS regressions.
2. Leave-One-Out (LOO) sensitivity for demographic groups and dimensions.
3. Cross-Chunk rank stability using Spearman rank correlation.
4. Scaling-choice sensitivity: PCA under z-score / min-max / raw scaling.
5. Factor analysis comparison: ML factor analysis vs. PCA loadings.
6. CEAT SE uncertainty propagation: Monte Carlo perturbation of CEAT values.
"""

import sys
import os
import csv
import json
import pickle
import hashlib
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import spearmanr

# Add the X/ directory to sys.path to import pipeline modules
stability_dir = Path(__file__).resolve().parent
project_dir = stability_dir.parent
sys.path.append(str(project_dir))

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Import modules from the main pipeline
from run_pipeline import (
    _load_seed_sentences,
    _encode_seed_centroids,
    _compute_efi,
    _run_regression,
    _load_seeds,
    preprocess,
    load_sentences,
    TeeLogger,
    ANALYSIS_DEVICE,
    ANALYSIS_EMB_BATCH_SIZE,
    ANALYSIS_MIN_GROUP_COUNT,
)
from embedding_config import ANALYSIS_EMBEDDING_MODEL
from lexicons import (
    TARGET_TOKENS,
    CONTRAST_TOKENS,
    POLITICAL_GROUP_TOKENS,
    resolve_group_token,
    set_active_extraction_tokens,
)
from step3_attitudinal_prototypes import (
    AGI_PROTOTYPES,
    PI_PROTOTYPES,
    SI_PROTOTYPES,
    NEGATIVE_ATTITUDE_PROTOTYPES,
    POSITIVE_ATTITUDE_PROTOTYPES,
    AGI_FLOOR,
    PI_FLOOR,
    SI_FLOOR,
)
from group_mentions import bound_frame_summary
from step4_metrics import compute_group_indices, cosine_similarity

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

def compute_metrics_for_sample(sampled_indices, sentence_metrics, target_groups, original_weat):
    """Aggregate sentence-level stats for the given sampled sentence indices."""
    role_sums = defaultdict(lambda: defaultdict(float))
    role_counts = defaultdict(int)
    
    frame_sums = defaultdict(lambda: defaultdict(float))
    frame_counts = defaultdict(int)
    
    ceat_sums = defaultdict(float)
    ceat_counts = defaultdict(int)
    
    for idx in sampled_indices:
        metrics = sentence_metrics[idx]
        for g, m in metrics.items():
            if g not in target_groups:
                continue
            # Role indices
            if m["count"] > 0:
                role_counts[g] += m["count"]
                role_sums[g]["subjecthood"] += m["subjecthood"]
                role_sums[g]["agi"] += m["agi"]
                role_sums[g]["pi"] += m["pi"]
                role_sums[g]["si"] += m["si"]
            
            # Frame indices
            if m["has_mention"]:
                frame_counts[g] += 1
                frame_sums[g]["neg"] += m["frame_neg"]
                frame_sums[g]["pos"] += m["frame_pos"]
                
                # CEAT index
                ceat_sums[g] += m["ceat_val"]
                ceat_counts[g] += 1
                
    # Build group profiles
    profiles = []
    for g in target_groups:
        rc = role_counts.get(g, 0)
        fc = frame_counts.get(g, 0)
        cc = ceat_counts.get(g, 0)
        
        profiles.append({
            "lemma": g,
            "total": rc,
            "subjecthood": round(role_sums[g]["subjecthood"] / rc, 3) if rc > 0 else 0.0,
            "AGI": round(role_sums[g]["agi"] / rc, 3) if rc > 0 else 0.0,
            "PI": round(role_sums[g]["pi"] / rc, 3) if rc > 0 else 0.0,
            "SI": round(role_sums[g]["si"] / rc, 3) if rc > 0 else 0.0,
            
            "frame_neg_atti": round(frame_sums[g]["neg"] / fc, 3) if fc > 0 else 0.0,
            "frame_pos_atti": round(frame_sums[g]["pos"] / fc, 3) if fc > 0 else 0.0,
            "net_atti": round((frame_sums[g]["neg"] - frame_sums[g]["pos"]) / fc, 3) if fc > 0 else 0.0,
            
            "weat": original_weat.get(g, 0.0),
            "ceat": round(ceat_sums[g] / cc, 4) if cc > 0 else 0.0,
        })
    return profiles

def run_bootstrap_analysis(sentence_metrics, target_groups, original_weat, observed_efi, observed_reg_weat, observed_reg_ceat, baseline_profiles, B=1000):
    """Run bootstrap resampling and compute stats."""
    N = len(sentence_metrics)
    print(f"\nRunning bootstrap resampling (B={B} iterations)...")
    rng = np.random.default_rng(42)
    bootstrap_indices = [rng.choice(N, size=N, replace=True).tolist() for _ in range(B)]
    
    # Track statistics
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    predictors = ["AGI", "PI", "SI", "net_atti"]
    
    # Store profiles per iteration
    boot_profiles = defaultdict(list) # group -> metric -> list of values
    boot_pc1_loadings = defaultdict(list)
    boot_pc2_loadings = defaultdict(list)
    boot_pc1_var = []
    boot_pc2_var = []
    
    boot_reg_weat = defaultdict(list) # coeff/r_squared -> list
    boot_reg_ceat = defaultdict(list) # coeff/r_squared -> list
    
    # Baseline loadings for sign alignment
    base_pc1 = observed_efi["pc1_loadings"]
    base_pc2 = observed_efi["pc2_loadings"]
    
    # Baseline observed profile lookup map
    obs_profile_map = {}
    for p in baseline_profiles:
        lemma = p["lemma"]
        for d in dims:
            obs_profile_map[(lemma, d)] = p.get(d, 0.0)

    for b in range(B):
        if (b + 1) % (max(1, B // 10)) == 0:
            print(f"  Iteration {b + 1}/{B}...")
            
        sampled_indices = bootstrap_indices[b]
        profiles = compute_metrics_for_sample(sampled_indices, sentence_metrics, target_groups, original_weat)
        
        # Save group metrics
        for p in profiles:
            lemma = p["lemma"]
            for d in dims:
                boot_profiles[(lemma, d)].append(p[d])
            
        # Run PCA (EFI)
        efi = _compute_efi(profiles)
        
        # Resolve sign flip of PC1 and PC2 using dot product
        pc1_load = efi["pc1_loadings"]
        dot1 = sum(base_pc1[d] * pc1_load[d] for d in dims)
        if dot1 < 0:
            for d in dims:
                efi["pc1_loadings"][d] = -efi["pc1_loadings"][d]
                
        pc2_load = efi["pc2_loadings"]
        dot2 = sum(base_pc2[d] * pc2_load[d] for d in dims)
        if dot2 < 0:
            for d in dims:
                efi["pc2_loadings"][d] = -efi["pc2_loadings"][d]
                
        # Save PCA loadings
        for d in dims:
            boot_pc1_loadings[d].append(efi["pc1_loadings"][d])
            boot_pc2_loadings[d].append(efi["pc2_loadings"][d])
        boot_pc1_var.append(efi["pc1_variance_explained"])
        boot_pc2_var.append(efi["pc2_variance_explained"])
        
        # Run regressions
        reg_weat = _run_regression(profiles, target_key="weat", predictors=predictors)
        if reg_weat:
            boot_reg_weat["r_squared"].append(reg_weat["r_squared"])
            boot_reg_weat["intercept"].append(reg_weat["intercept"])
            for p_name in predictors:
                boot_reg_weat[f"b_{p_name}"].append(reg_weat["coefficients"].get(p_name, 0.0))
                
        reg_ceat = _run_regression(profiles, target_key="ceat", predictors=predictors)
        if reg_ceat:
            boot_reg_ceat["r_squared"].append(reg_ceat["r_squared"])
            boot_reg_ceat["intercept"].append(reg_ceat["intercept"])
            for p_name in predictors:
                boot_reg_ceat[f"b_{p_name}"].append(reg_ceat["coefficients"].get(p_name, 0.0))
                
    # Helper to calculate stats
    def get_stats(vals, obs_val):
        arr = np.array(vals)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        ci_lower = np.percentile(arr, 2.5)
        ci_upper = np.percentile(arr, 97.5)
        return {
            "observed": obs_val,
            "mean": round(mean, 4),
            "se": round(se, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }
        
    return {
        "profiles": {k: get_stats(v, obs_profile_map.get(k, 0.0)) for k, v in boot_profiles.items()},
        "pc1_loadings": {d: get_stats(boot_pc1_loadings[d], base_pc1[d]) for d in dims},
        "pc2_loadings": {d: get_stats(boot_pc2_loadings[d], base_pc2[d]) for d in dims},
        "pc1_var": get_stats(boot_pc1_var, observed_efi["pc1_variance_explained"]),
        "pc2_var": get_stats(boot_pc2_var, observed_efi["pc2_variance_explained"]),
        "reg_weat": {k: get_stats(v, observed_reg_weat["intercept"] if k == "intercept" else (observed_reg_weat["r_squared"] if k == "r_squared" else observed_reg_weat["coefficients"].get(k[2:], 0.0))) for k, v in boot_reg_weat.items()},
        "reg_ceat": {k: get_stats(v, observed_reg_ceat["intercept"] if k == "intercept" else (observed_reg_ceat["r_squared"] if k == "r_squared" else observed_reg_ceat["coefficients"].get(k[2:], 0.0))) for k, v in boot_reg_ceat.items()},
    }

def run_loo_sensitivity(baseline_profiles, target_groups):
    """Run Leave-One-Out sensitivity checks."""
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    predictors = ["AGI", "PI", "SI", "net_atti"]
    
    baseline_efi = _compute_efi(baseline_profiles)
    baseline_reg_weat = _run_regression(baseline_profiles, "weat", predictors)
    baseline_reg_ceat = _run_regression(baseline_profiles, "ceat", predictors)
    
    base_pc1_vec = np.array([baseline_efi["pc1_loadings"][d] for d in dims])
    base_pc2_vec = np.array([baseline_efi["pc2_loadings"][d] for d in dims])
    
    report_lines = []
    report_lines.append("=== LEAVE-ONE-GROUP-OUT PCA & REGRESSION SENSITIVITY ===")
    
    for g in sorted(target_groups):
        reduced_profiles = [p for p in baseline_profiles if p["lemma"] != g]
        efi = _compute_efi(reduced_profiles)
        
        # Align loadings
        pc1_vec = np.array([efi["pc1_loadings"][d] for d in dims])
        dot1 = np.dot(base_pc1_vec, pc1_vec)
        if dot1 < 0:
            pc1_vec = -pc1_vec
            
        pc2_vec = np.array([efi["pc2_loadings"][d] for d in dims])
        dot2 = np.dot(base_pc2_vec, pc2_vec)
        if dot2 < 0:
            pc2_vec = -pc2_vec
            
        # Cosine similarity to baseline loadings
        cos_sim1 = np.dot(base_pc1_vec, pc1_vec) / (np.linalg.norm(base_pc1_vec) * np.linalg.norm(pc1_vec))
        cos_sim2 = np.dot(base_pc2_vec, pc2_vec) / (np.linalg.norm(base_pc2_vec) * np.linalg.norm(pc2_vec))
        
        reg_weat = _run_regression(reduced_profiles, "weat", predictors)
        reg_ceat = _run_regression(reduced_profiles, "ceat", predictors)
        
        weat_r2_diff = reg_weat["r_squared"] - baseline_reg_weat["r_squared"] if reg_weat else 0.0
        ceat_r2_diff = reg_ceat["r_squared"] - baseline_reg_ceat["r_squared"] if reg_ceat else 0.0
        
        report_lines.append(
            f"  Removed group '{g:<12}': "
            f"PC1 CosSim={cos_sim1:.4f}, PC2 CosSim={cos_sim2:.4f}, "
            f"WEAT ΔR²={weat_r2_diff:+.4f}, CEAT ΔR²={ceat_r2_diff:+.4f}"
        )
        
    report_lines.append("\n=== LEAVE-ONE-DIMENSION-OUT PCA SENSITIVITY ===")
    for removed_dim in dims:
        sub_dims = [d for d in dims if d != removed_dim]
        # Re-run custom PCA on submatrix
        matrix = np.array([[g.get(d, 0.0) for d in sub_dims] for g in baseline_profiles])
        scaler = StandardScaler()
        X = scaler.fit_transform(matrix)
        pca = PCA(n_components=min(len(sub_dims), len(baseline_profiles)))
        pca.fit(X)
        pc1_var = pca.explained_variance_ratio_[0]
        pc2_var = pca.explained_variance_ratio_[1] if pca.n_components_ >= 2 else 0.0
        
        report_lines.append(
            f"  Removed dimension '{removed_dim:<10}': "
            f"PC1 VarExplained={pc1_var:.3f} (baseline was {baseline_efi['pc1_variance_explained']:.3f}), "
            f"PC2 VarExplained={pc2_var:.3f} (baseline was {baseline_efi['pc2_variance_explained']:.3f})"
        )
        
    return "\n".join(report_lines)

def run_cross_chunk_stability(sentence_metrics, target_groups, original_weat, K=3):
    """Split the sentence list into K chunks, compute rankings, and check Spearman correlation."""
    N = len(sentence_metrics)
    chunk_size = N // K
    
    chunk_profiles = []
    print(f"\nComputing profiles for K={K} chunks (chunk size ≈ {chunk_size} sentences)...")
    for k in range(K):
        start = k * chunk_size
        end = (k + 1) * chunk_size if k < K - 1 else N
        chunk_indices = list(range(start, end))
        prof = compute_metrics_for_sample(chunk_indices, sentence_metrics, target_groups, original_weat)
        chunk_profiles.append(prof)
        
    metrics = ["AGI", "PI", "SI", "net_atti", "ceat"]
    
    correlation_results = []
    
    for metric in metrics:
        # Get rank vectors for each chunk
        rank_vectors = []
        for prof in chunk_profiles:
            # Sort target_groups by their value of this metric to establish rankings
            val_map = {p["lemma"]: p[metric] for p in prof}
            rank_vec = [val_map[g] for g in sorted(target_groups)]
            rank_vectors.append(rank_vec)
            
        # Compute pairwise correlations
        corrs = []
        for i in range(K):
            for j in range(i + 1, K):
                corr, _ = spearmanr(rank_vectors[i], rank_vectors[j])
                if not np.isnan(corr):
                    corrs.append(corr)
                    
        avg_corr = np.mean(corrs) if corrs else 0.0
        correlation_results.append({
            "Metric": metric,
            "AvgPairwiseSpearman": round(avg_corr, 4),
            "MinPairwiseSpearman": round(np.min(corrs), 4) if corrs else 0.0,
            "MaxPairwiseSpearman": round(np.max(corrs), 4) if corrs else 0.0,
        })
        
    return correlation_results


def run_scaling_sensitivity(baseline_profiles):
    """Compare PCA results under z-score (StandardScaler), min-max, and raw (no) scaling.
    Addresses Pastra point (i.3): sensitivity to scaling choices.
    """
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    labels = [g["lemma"] for g in baseline_profiles]
    matrix = np.array([[g.get(d, 0.0) for d in dims] for g in baseline_profiles])

    scalers = {
        "zscore": StandardScaler(),
        "minmax": MinMaxScaler(),
        "raw": None,
    }

    results = {}
    for scaler_name, scaler in scalers.items():
        X = scaler.fit_transform(matrix) if scaler else matrix.copy()
        pca = PCA(n_components=min(len(dims), len(labels)))
        scores = pca.fit_transform(X)

        pc1_loadings = dict(zip(dims, pca.components_[0].round(4)))
        pc2_loadings = dict(zip(dims, pca.components_[1].round(4))) if pca.n_components_ >= 2 else {d: 0.0 for d in dims}
        pc1_var = round(pca.explained_variance_ratio_[0], 4)
        pc2_var = round(pca.explained_variance_ratio_[1], 4) if pca.n_components_ >= 2 else 0.0

        # EFI scores per group
        group_scores = dict(zip(labels, scores[:, 0].round(4)))

        results[scaler_name] = {
            "pc1_loadings": pc1_loadings,
            "pc2_loadings": pc2_loadings,
            "pc1_var": pc1_var,
            "pc2_var": pc2_var,
            "group_scores": group_scores,
        }

    # Align sign of minmax/raw loadings to zscore using dot product
    ref_pc1 = np.array([results["zscore"]["pc1_loadings"][d] for d in dims])
    ref_pc2 = np.array([results["zscore"]["pc2_loadings"][d] for d in dims])
    for alt in ["minmax", "raw"]:
        alt_pc1 = np.array([results[alt]["pc1_loadings"][d] for d in dims])
        if np.dot(ref_pc1, alt_pc1) < 0:
            results[alt]["pc1_loadings"] = {d: round(-v, 4) for d, v in results[alt]["pc1_loadings"].items()}
            results[alt]["group_scores"] = {g: round(-v, 4) for g, v in results[alt]["group_scores"].items()}
        alt_pc2 = np.array([results[alt]["pc2_loadings"][d] for d in dims])
        if np.dot(ref_pc2, alt_pc2) < 0:
            results[alt]["pc2_loadings"] = {d: round(-v, 4) for d, v in results[alt]["pc2_loadings"].items()}

    # Cross-scaling Spearman: do group EFI rankings agree?
    zscore_ranks = [results["zscore"]["group_scores"][g] for g in sorted(labels)]
    rank_corrs = {}
    for alt in ["minmax", "raw"]:
        alt_ranks = [results[alt]["group_scores"][g] for g in sorted(labels)]
        corr, pval = spearmanr(zscore_ranks, alt_ranks)
        rank_corrs[alt] = {"spearman": round(corr, 4), "pval": round(pval, 6)}

    results["rank_correlations"] = rank_corrs
    return results


def run_factor_analysis(baseline_profiles):
    """Compare Maximum Likelihood factor analysis with PCA loadings.
    Addresses Pastra point (i.5): factor analysis / clustering comparison.
    """
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    labels = [g["lemma"] for g in baseline_profiles]
    matrix = np.array([[g.get(d, 0.0) for d in dims] for g in baseline_profiles])

    # Standardize for comparability with PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    report_lines = []
    report_lines.append("=== FACTOR ANALYSIS VS. PCA COMPARISON ===")

    # Bartlett's test of sphericity
    try:
        chi_sq, p_val = calculate_bartlett_sphericity(X)
        report_lines.append(f"  Bartlett's test: χ²={chi_sq:.2f}, p={p_val:.6f}")
        if p_val > 0.05:
            report_lines.append("  ⚠ p > 0.05: correlations may be insufficient for factor extraction")
    except Exception as e:
        report_lines.append(f"  Bartlett's test failed: {e}")

    # KMO test
    try:
        kmo_all, kmo_model = calculate_kmo(X)
        report_lines.append(f"  KMO measure of sampling adequacy: {kmo_model:.4f}")
        if kmo_model < 0.5:
            report_lines.append("  ⚠ KMO < 0.5: data may not be suitable for factor analysis")
    except Exception as e:
        report_lines.append(f"  KMO test failed: {e}")

    # Run FA with 2 factors (matching PCA n_components interpretation)
    n_factors = 2
    fa_results = {}
    for rotation in [None, "varimax"]:
        rot_label = rotation or "none"
        try:
            fa = FactorAnalyzer(
                n_factors=n_factors,
                rotation=rotation,
                method="ml",  # Maximum Likelihood
                is_corr_matrix=False,
            )
            fa.fit(X)
            loadings = fa.loadings_
            var_explained = fa.get_factor_variance()
            # var_explained returns (SS Loadings, Proportion Var, Cumulative Var)

            fa_results[rot_label] = {
                "loadings": {dims[i]: {f"F{j+1}": round(loadings[i, j], 4) for j in range(n_factors)} for i in range(len(dims))},
                "ss_loadings": [round(v, 4) for v in var_explained[0]],
                "proportion_var": [round(v, 4) for v in var_explained[1]],
                "cumulative_var": [round(v, 4) for v in var_explained[2]],
            }

            report_lines.append(f"\n  Factor Analysis (ML, rotation={rot_label}):")
            report_lines.append(f"    Proportion variance: F1={var_explained[1][0]:.3f}, F2={var_explained[1][1]:.3f}")
            report_lines.append(f"    Cumulative: {var_explained[2][-1]:.3f}")
            report_lines.append(f"    {'Dimension':<10}  {'F1':>8}  {'F2':>8}")
            for i, d in enumerate(dims):
                report_lines.append(f"    {d:<10}  {loadings[i, 0]:>8.4f}  {loadings[i, 1]:>8.4f}")
        except Exception as e:
            report_lines.append(f"\n  Factor Analysis (ML, rotation={rot_label}) failed: {e}")
            fa_results[rot_label] = None

    # Compare with PCA
    pca = PCA(n_components=n_factors)
    pca.fit(X)
    report_lines.append(f"\n  PCA (baseline, z-score):")
    report_lines.append(f"    Proportion variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    report_lines.append(f"    {'Dimension':<10}  {'PC1':>8}  {'PC2':>8}")
    for i, d in enumerate(dims):
        report_lines.append(f"    {d:<10}  {pca.components_[0, i]:>8.4f}  {pca.components_[1, i]:>8.4f}")

    # Congruence coefficients between PCA and FA(no rotation)
    if fa_results.get("none") is not None:
        report_lines.append("\n  Tucker's congruence (PCA PC1 vs FA F1, no rotation):")
        pca_pc1 = pca.components_[0]
        fa_f1 = np.array([fa_results["none"]["loadings"][d]["F1"] for d in dims])
        tucker = np.dot(pca_pc1, fa_f1) / (np.linalg.norm(pca_pc1) * np.linalg.norm(fa_f1))
        report_lines.append(f"    Tucker's φ = {tucker:.4f}")
        if abs(tucker) > 0.95:
            report_lines.append("    → Factor structure replicates PCA structure well")
        elif abs(tucker) > 0.85:
            report_lines.append("    → Fair replication")
        else:
            report_lines.append("    → Poor replication — PCA and FA disagree on latent structure")

    return "\n".join(report_lines), fa_results


def run_ceat_uncertainty_propagation(
    baseline_profiles,
    group_ceat_se,
    baseline_reg_ceat=None,
    baseline_reg_weat=None,
    n_mc=2000,
):
    """Monte Carlo propagation of CEAT standard errors into EFI and regression.
    Addresses Pastra point (i.6): uncertainty propagation from CEAT SE.

    For each MC iteration, perturb each group's CEAT by sampling from
    N(observed_ceat, ceat_se), then re-run PCA and regression.
    """
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    predictors = ["AGI", "PI", "SI", "net_atti"]
    groups = [g["lemma"] for g in baseline_profiles]

    np.random.seed(123)

    # Storage
    mc_pc1_loadings = defaultdict(list)
    mc_pc2_loadings = defaultdict(list)
    mc_pc1_var = []
    mc_pc2_var = []
    mc_reg_weat = defaultdict(list)
    mc_reg_ceat = defaultdict(list)
    mc_efi_scores = defaultdict(list)

    # Baseline for sign alignment
    baseline_efi = _compute_efi(baseline_profiles)
    base_pc1 = np.array([baseline_efi["pc1_loadings"][d] for d in dims])
    base_pc2 = np.array([baseline_efi["pc2_loadings"][d] for d in dims])

    print(f"\nRunning CEAT uncertainty propagation (MC={n_mc})...")
    for it in range(n_mc):
        if (it + 1) % (max(1, n_mc // 10)) == 0:
            print(f"  MC iteration {it + 1}/{n_mc}...")

        # Perturb CEAT values
        perturbed = []
        for p in baseline_profiles:
            pp = dict(p)
            se = group_ceat_se.get(p["lemma"], 0.0)
            pp["ceat"] = float(np.random.normal(p["ceat"], se))
            perturbed.append(pp)

        # Re-run PCA
        efi = _compute_efi(perturbed)

        # Sign alignment
        pc1_vec = np.array([efi["pc1_loadings"][d] for d in dims])
        if np.dot(base_pc1, pc1_vec) < 0:
            pc1_vec = -pc1_vec
            for d in dims:
                efi["pc1_loadings"][d] = -efi["pc1_loadings"][d]
            for g in groups:
                efi["pc1_scores"][g] = -efi["pc1_scores"][g]
        pc2_vec = np.array([efi["pc2_loadings"][d] for d in dims])
        if np.dot(base_pc2, pc2_vec) < 0:
            for d in dims:
                efi["pc2_loadings"][d] = -efi["pc2_loadings"][d]

        for d in dims:
            mc_pc1_loadings[d].append(efi["pc1_loadings"][d])
            mc_pc2_loadings[d].append(efi["pc2_loadings"][d])
        mc_pc1_var.append(efi["pc1_variance_explained"])
        mc_pc2_var.append(efi["pc2_variance_explained"])

        for g in groups:
            mc_efi_scores[g].append(efi["pc1_scores"][g])

        # Re-run regressions (CEAT is also a target)
        reg_weat = _run_regression(perturbed, "weat", predictors)
        reg_ceat = _run_regression(perturbed, "ceat", predictors)
        if reg_weat:
            mc_reg_weat["r_squared"].append(reg_weat["r_squared"])
            for pn in predictors:
                mc_reg_weat[f"b_{pn}"].append(reg_weat["coefficients"].get(pn, 0.0))
        if reg_ceat:
            mc_reg_ceat["r_squared"].append(reg_ceat["r_squared"])
            for pn in predictors:
                mc_reg_ceat[f"b_{pn}"].append(reg_ceat["coefficients"].get(pn, 0.0))

    # Compile results
    def _stats(vals, obs):
        arr = np.array(vals)
        return {
            "observed": round(obs, 4),
            "mc_mean": round(np.mean(arr), 4),
            "mc_se": round(np.std(arr, ddof=1), 4),
            "mc_ci_lower": round(np.percentile(arr, 2.5), 4),
            "mc_ci_upper": round(np.percentile(arr, 97.5), 4),
        }

    def _reg_obs(reg_result, key):
        """Extract observed value from a regression result dict for a given key."""
        if not reg_result:
            return 0.0
        if key == "r_squared":
            return reg_result.get("r_squared", 0.0)
        if key == "intercept":
            return reg_result.get("intercept", 0.0)
        # key is like "b_AGI"
        param = key[2:]  # strip "b_"
        return reg_result.get("coefficients", {}).get(param, 0.0)

    return {
        "pc1_loadings": {d: _stats(mc_pc1_loadings[d], baseline_efi["pc1_loadings"][d]) for d in dims},
        "pc2_loadings": {d: _stats(mc_pc2_loadings[d], baseline_efi["pc2_loadings"][d]) for d in dims},
        "pc1_var": _stats(mc_pc1_var, baseline_efi["pc1_variance_explained"]),
        "pc2_var": _stats(mc_pc2_var, baseline_efi["pc2_variance_explained"]),
        "efi_scores": {g: _stats(mc_efi_scores[g], baseline_efi["pc1_scores"].get(g, 0.0)) for g in groups},
        "reg_weat": {k: _stats(v, _reg_obs(baseline_reg_weat, k)) for k, v in mc_reg_weat.items()},
        "reg_ceat": {k: _stats(v, _reg_obs(baseline_reg_ceat, k)) for k, v in mc_reg_ceat.items()},
    }

def main():
    parser = argparse.ArgumentParser(description="Statistical Robustness Checks")
    parser.add_argument("sentences_path", nargs="?", default="dolma/semantic_filter_results.tsv", help="Path to input TSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for robustness artifacts")
    parser.add_argument("--bootstrap-iter", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks for cross-shard proxy check")
    args = parser.parse_args()
    
    project_dir = Path(__file__).resolve().parents[2]
    if os.path.isabs(args.sentences_path):
        sentences_path = Path(args.sentences_path)
    else:
        sentences_path = project_dir / args.sentences_path

    out_dir = Path(args.output_dir) if args.output_dir else (project_dir / "X" / "stability")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up TeeLogger to save logs
    sys.stdout = TeeLogger(str(out_dir / "robustness_checks.log"))
    sys.stderr = sys.stdout
    
    print("=== STATISTICAL ROBUSTNESS AND STABILITY CHECKS ===")
    print(f"Input path: {sentences_path}")
    
    # Load spaCy
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_lg")
    
    # Load active labels
    _active_labels_path = sentences_path.parent / "active_labels.json"
    if _active_labels_path.exists():
        with _active_labels_path.open("r", encoding="utf-8") as _f:
            _al = json.load(_f)
        _active_set = set(_al.get("target", [])) | set(_al.get("contrast", []))
        set_active_extraction_tokens(_active_set)
        print(f"Active extraction labels loaded: {len(_active_set)} tokens")
        
    # Preprocess (check preprocess_cache.pkl first)
    raw = load_sentences(str(sentences_path))
    _prep_cache_path = project_dir / "preprocess_cache.pkl"
    if not _prep_cache_path.exists():
        _prep_cache_path = project_dir / "X" / "preprocess_cache.pkl"

    processed = None
    if _prep_cache_path.exists():
        try:
            print(f"Loading preprocessing cache from {_prep_cache_path.name}...")
            with open(_prep_cache_path, "rb") as _f:
                _cached = pickle.load(_f)
            processed = _cached.get("processed")
            if processed and len(processed) == len(raw):
                print(f"Preprocessing cache hit — loaded {len(processed)} parsed sentences.")
            else:
                processed = None
        except Exception as _e:
            print(f"Preprocessing cache unreadable ({_e}), falling back to spaCy...")
            processed = None

    if processed is None:
        processed = preprocess(nlp, raw)
        print(f"Preprocessed {len(processed)} sentences.")
    
    # Load cache
    _cache_path = project_dir / "X" / "srl_cache.pkl"
    if not _cache_path.exists():
        sys.exit(f"Error: {_cache_path.name} not found. Please run run_pipeline.py first to build the cache.")
        
    with open(_cache_path, "rb") as f:
        _cached = pickle.load(f)
    extracted = _cached["extracted"]
    print(f"SRL cache loaded: {len(extracted)} entries.")
    
    # Load seeds and centroids
    auto_neg_frames, auto_pos_frames, _, _, _ = _load_seeds(project_dir / "X")
    seed_neg_terms, seed_pos_terms = _load_seed_sentences(project_dir / "X")
    
    print("Loading SentenceTransformer for centroid encoding...")
    sentence_encoder = SentenceTransformer(ANALYSIS_EMBEDDING_MODEL, device=ANALYSIS_DEVICE)
    neg_centroid, pos_centroid = _encode_seed_centroids(sentence_encoder, seed_neg_terms, seed_pos_terms)
    
    # Read group stats to load original WEAT and baseline target groups
    group_stats_path = project_dir / "X" / "group_stats.tsv"
    if not group_stats_path.exists():
        sys.exit(f"Error: {group_stats_path.name} not found. Please run run_pipeline.py first.")
        
    original_weat = {}
    target_groups = set()
    with open(group_stats_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lemma = row["Lemma"]
            original_weat[lemma] = float(row["WEAT"]) if row["WEAT"] else 0.0
            # Only include groups that pass the analysis count floor
            if int(row["N"]) >= ANALYSIS_MIN_GROUP_COUNT:
                target_groups.add(lemma)
                
    print(f"Active target groups for stability analysis (N >= {ANALYSIS_MIN_GROUP_COUNT}): {sorted(target_groups)}")

    # Filter processed & extracted to only sentences containing at least one active target group
    active_processed = []
    active_extracted = []
    for proc_item, ext_item in zip(processed, extracted):
        doc = proc_item["doc"]
        has_active = False
        for token in doc:
            resolved = resolve_group_token(token, doc)
            if resolved:
                _, canonical = resolved
                if canonical in target_groups:
                    has_active = True
                    break
        if has_active:
            active_processed.append(proc_item)
            active_extracted.append(ext_item)
            
    processed = active_processed
    extracted = active_extracted
    print(f"Filtered dataset to {len(processed)} active sentences containing target group mentions.")

    # Pre-encode active sentences (with caching to avoid re-encoding)
    _emb_cache_path = project_dir / "X" / "text_vec_cache.pkl"
    all_texts = list(dict.fromkeys(item["cleaned_text"] for item in processed))
    text_to_vec = {}

    if _emb_cache_path.exists():
        try:
            print(f"Loading embedding cache from {_emb_cache_path.name}...")
            with open(_emb_cache_path, "rb") as _f:
                text_to_vec = pickle.load(_f)
            print(f"  Loaded {len(text_to_vec)} cached sentence vectors.")
        except Exception as _e:
            print(f"Embedding cache unreadable ({_e}), re-encoding...")
            text_to_vec = {}

    missing_texts = [t for t in all_texts if t not in text_to_vec]
    if missing_texts:
        print(f"Pre-encoding {len(missing_texts)} sentences with GTE ModernBERT...")
        new_vecs = _encode_text_map(sentence_encoder, missing_texts)
        text_to_vec.update(new_vecs)
        try:
            with open(_emb_cache_path, "wb") as _f:
                pickle.dump(text_to_vec, _f)
            print(f"Wrote embedding cache ({len(text_to_vec)} vectors).")
        except Exception as _e:
            print(f"Failed to write embedding cache ({_e})")
    else:
        print("Embedding cache hit — skipped GTE ModernBERT encoding!")
    
    # Pre-build sentence-level database (with disk cache)
    _metrics_cache_path = project_dir / "X" / "stability" / "sentence_metrics_cache.pkl"
    _metrics_cache_key = None
    sentence_metrics = None

    # Build a cache key from: sentences file mtime, frame sets, neg/pos centroid fingerprint
    try:
        _mtime = str(sentences_path.stat().st_mtime)
        _frames_sig = hashlib.md5(
            json.dumps(sorted(auto_neg_frames) + sorted(auto_pos_frames), ensure_ascii=False).encode()
        ).hexdigest()
        _centroid_sig = hashlib.md5(
            neg_centroid.tobytes() + pos_centroid.tobytes()
        ).hexdigest()
        _metrics_cache_key = f"{_mtime}|{_frames_sig}|{_centroid_sig}|{len(processed)}"
    except Exception as _e:
        print(f"Could not build metrics cache key ({_e}), will rebuild.")

    if _metrics_cache_path.exists() and _metrics_cache_key is not None:
        try:
            print(f"Loading sentence metrics cache from {_metrics_cache_path.name}...")
            with open(_metrics_cache_path, "rb") as _f:
                _mc = pickle.load(_f)
            if _mc.get("key") == _metrics_cache_key:
                sentence_metrics = _mc["data"]
                print(f"Sentence metrics cache hit — loaded {len(sentence_metrics)} entries. Skipping rebuild.")
            else:
                print("Sentence metrics cache key mismatch, rebuilding...")
                sentence_metrics = None
        except Exception as _e:
            print(f"Sentence metrics cache unreadable ({_e}), rebuilding...")
            sentence_metrics = None

    if sentence_metrics is None:
        print("\nPre-building sentence-level metrics database...")
        sentence_metrics = []
        _report_every = max(1, len(processed) // 10)
        for i in range(len(processed)):
            if (i + 1) % _report_every == 0:
                print(f"  {i + 1}/{len(processed)} sentences processed...")
            item_proc = processed[i]
            item_ext = extracted[i]
            doc = item_proc["doc"]

            frame_summary = bound_frame_summary(doc, auto_neg_frames, auto_pos_frames)

            resolved_lemmas = set()
            for token in doc:
                resolved = resolve_group_token(token, doc)
                if resolved:
                    _, canonical = resolved
                    resolved_lemmas.add(canonical)

            vec = text_to_vec[item_proc["cleaned_text"]]
            ceat_val = cosine_similarity(vec, neg_centroid) - cosine_similarity(vec, pos_centroid)

            lemmas_in_sentence = resolved_lemmas | {f["lemma"] for f in item_ext.get("findings", [])}
            findings_by_lemma = defaultdict(list)
            for f in item_ext.get("findings", []):
                findings_by_lemma[f["lemma"]].append(f)

            by_lemma = {}
            for g in lemmas_in_sentence:
                findings = findings_by_lemma.get(g, [])
                sub = sum(f.get("subjecthood", 0) for f in findings)
                agi = sum(f.get("agi", 0) for f in findings)
                pi = sum(f.get("pi", 0) for f in findings)
                si = sum(f.get("si", 0) for f in findings)
                count = len(findings)

                bound = frame_summary["by_lemma"].get(g)
                neg_bound = 1 if (bound and bound["neg"]) else 0
                pos_bound = 1 if (bound and bound["pos"]) else 0

                by_lemma[g] = {
                    "subjecthood": sub,
                    "agi": agi,
                    "pi": pi,
                    "si": si,
                    "count": count,
                    "frame_neg": neg_bound,
                    "frame_pos": pos_bound,
                    "ceat_val": ceat_val,
                    "has_mention": 1 if g in resolved_lemmas else 0,
                }
            sentence_metrics.append(by_lemma)

        # Save to cache
        if _metrics_cache_key is not None:
            try:
                with open(_metrics_cache_path, "wb") as _f:
                    pickle.dump({"key": _metrics_cache_key, "data": sentence_metrics}, _f)
                print(f"Wrote sentence metrics cache ({len(sentence_metrics)} entries) to {_metrics_cache_path.name}.")
            except Exception as _e:
                print(f"Failed to write sentence metrics cache ({_e})")
        
    # Baseline observed profiles
    baseline_profiles = compute_metrics_for_sample(list(range(len(processed))), sentence_metrics, target_groups, original_weat)
    
    # Run EFI and regressions on baseline
    baseline_efi = _compute_efi(baseline_profiles)
    predictors = ["AGI", "PI", "SI", "net_atti"]
    baseline_reg_weat = _run_regression(baseline_profiles, "weat", predictors)
    baseline_reg_ceat = _run_regression(baseline_profiles, "ceat", predictors)
    
    # 1. RUN BOOTSTRAP
    boot = run_bootstrap_analysis(
        sentence_metrics,
        target_groups,
        original_weat,
        baseline_efi,
        baseline_reg_weat,
        baseline_reg_ceat,
        baseline_profiles,
        B=args.bootstrap_iter
    )
    
    # Write bootstrap results TSV
    boot_tsv = out_dir / "bootstrap_results.tsv"
    with open(boot_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Type", "Identifier", "Metric", "Observed", "Bootstrap_Mean", "Bootstrap_SE", "CI_Lower", "CI_Upper"])
        
        # 1.1 Group profiles
        for (g, metric), stats in sorted(boot["profiles"].items()):
            w.writerow(["GroupProfile", g, metric, round(stats["observed"], 4), stats["mean"], stats["se"], stats["ci_lower"], stats["ci_upper"]])
            
        # 1.2 PCA loadings
        for dim, stats in sorted(boot["pc1_loadings"].items()):
            w.writerow(["PC1_Loading", "EFI", dim, round(stats["observed"], 4), stats["mean"], stats["se"], stats["ci_lower"], stats["ci_upper"]])
        for dim, stats in sorted(boot["pc2_loadings"].items()):
            w.writerow(["PC2_Loading", "EFI", dim, round(stats["observed"], 4), stats["mean"], stats["se"], stats["ci_lower"], stats["ci_upper"]])
            
        # 1.3 PCA Explained Variance
        w.writerow(["PC1_Variance", "EFI", "var_explained", round(boot["pc1_var"]["observed"], 4), boot["pc1_var"]["mean"], boot["pc1_var"]["se"], boot["pc1_var"]["ci_lower"], boot["pc1_var"]["ci_upper"]])
        w.writerow(["PC2_Variance", "EFI", "var_explained", round(boot["pc2_var"]["observed"], 4), boot["pc2_var"]["mean"], boot["pc2_var"]["se"], boot["pc2_var"]["ci_lower"], boot["pc2_var"]["ci_upper"]])
        
        # 1.4 Regressions
        for k, stats in sorted(boot["reg_weat"].items()):
            w.writerow(["Regression_WEAT", "WEAT_Model", k, round(stats["observed"], 4), stats["mean"], stats["se"], stats["ci_lower"], stats["ci_upper"]])
        for k, stats in sorted(boot["reg_ceat"].items()):
            w.writerow(["Regression_CEAT", "CEAT_Model", k, round(stats["observed"], 4), stats["mean"], stats["se"], stats["ci_lower"], stats["ci_upper"]])
            
    print(f"\n→ Bootstrap results written to {boot_tsv.name}")
    
    # 2. RUN LEAVE-ONE-OUT SENSITIVITY
    loo_report = run_loo_sensitivity(baseline_profiles, target_groups)
    loo_txt = out_dir / "loo_sensitivity_results.txt"
    loo_txt.write_text(loo_report, encoding="utf-8")
    print(f"→ Leave-one-out sensitivity report written to {loo_txt.name}")
    print("\n" + loo_report)
    
    # 3. RUN CROSS-CHUNK STABILITY
    cross_chunk = run_cross_chunk_stability(sentence_metrics, target_groups, original_weat, K=args.chunks)
    cross_tsv = out_dir / "cross_chunk_stability.tsv"
    with open(cross_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Metric", "AvgPairwiseSpearman", "MinPairwiseSpearman", "MaxPairwiseSpearman"])
        for r in cross_chunk:
            w.writerow([r["Metric"], r["AvgPairwiseSpearman"], r["MinPairwiseSpearman"], r["MaxPairwiseSpearman"]])
            
    print(f"\n→ Cross-chunk stability metrics written to {cross_tsv.name}")
    print("\n=== CROSS-CHUNK RANK STABILITY ===")
    for r in cross_chunk:
        print(f"  {r['Metric']:<10}: AvgSpearman={r['AvgPairwiseSpearman']:.4f} (Min={r['MinPairwiseSpearman']:.4f}, Max={r['MaxPairwiseSpearman']:.4f})")

    # 4. SCALING-CHOICE SENSITIVITY (Pastra i.3)
    print("\n" + "=" * 60)
    print("=== SCALING-CHOICE SENSITIVITY (i.3) ===")
    scaling = run_scaling_sensitivity(baseline_profiles)
    dims = ["AGI", "PI", "SI", "net_atti", "weat", "ceat"]
    scaling_tsv = out_dir / "scaling_sensitivity.tsv"
    with open(scaling_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Scaler", "Component", "Dimension", "Loading", "VarExplained"])
        for sc_name in ["zscore", "minmax", "raw"]:
            sc = scaling[sc_name]
            for d in dims:
                w.writerow([sc_name, "PC1", d, sc["pc1_loadings"][d], sc["pc1_var"]])
                w.writerow([sc_name, "PC2", d, sc["pc2_loadings"][d], sc["pc2_var"]])
    with open(scaling_tsv, "a", encoding="utf-8") as f:
        f.write("\n# Rank correlation of EFI scores (zscore vs. alternative)\n")
        for alt, rc in scaling["rank_correlations"].items():
            f.write(f"# zscore_vs_{alt}: Spearman={rc['spearman']}, p={rc['pval']}\n")

    print(f"→ Scaling sensitivity written to {scaling_tsv.name}")
    for sc_name in ["zscore", "minmax", "raw"]:
        sc = scaling[sc_name]
        print(f"  {sc_name}: PC1 VarExpl={sc['pc1_var']:.3f}, PC2 VarExpl={sc['pc2_var']:.3f}")
        load_str = ", ".join(f"{d}={sc['pc1_loadings'][d]:.3f}" for d in dims)
        print(f"    PC1 loadings: {load_str}")
    for alt, rc in scaling["rank_correlations"].items():
        print(f"  EFI rank corr (zscore vs {alt}): ρ={rc['spearman']:.4f} (p={rc['pval']:.4f})")

    # 5. FACTOR ANALYSIS COMPARISON (Pastra i.5)
    print("\n" + "=" * 60)
    fa_report, fa_results = run_factor_analysis(baseline_profiles)
    fa_txt = out_dir / "factor_analysis_comparison.txt"
    fa_txt.write_text(fa_report, encoding="utf-8")
    print(f"→ Factor analysis comparison written to {fa_txt.name}")
    print("\n" + fa_report)

    # 6. CEAT SE UNCERTAINTY PROPAGATION (Pastra i.6)
    print("\n" + "=" * 60)
    print("=== CEAT SE UNCERTAINTY PROPAGATION (i.6) ===")
    group_ceat_se = {}
    with open(group_stats_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lemma = row["Lemma"]
            se_val = row.get("CEAT_SE", "0").strip()
            group_ceat_se[lemma] = float(se_val) if se_val else 0.0

    mc_results = run_ceat_uncertainty_propagation(
        baseline_profiles,
        group_ceat_se,
        baseline_reg_ceat=baseline_reg_ceat,
        baseline_reg_weat=baseline_reg_weat,
        n_mc=2000,
    )

    mc_tsv = out_dir / "ceat_uncertainty_propagation.tsv"
    with open(mc_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Type", "Identifier", "Observed", "MC_Mean", "MC_SE", "MC_CI_Lower", "MC_CI_Upper"])
        for d in dims:
            s = mc_results["pc1_loadings"][d]
            w.writerow(["PC1_Loading", d, s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])
        for d in dims:
            s = mc_results["pc2_loadings"][d]
            w.writerow(["PC2_Loading", d, s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])
        s = mc_results["pc1_var"]
        w.writerow(["PC1_Variance", "var_explained", s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])
        s = mc_results["pc2_var"]
        w.writerow(["PC2_Variance", "var_explained", s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])
        for g, s in sorted(mc_results["efi_scores"].items()):
            w.writerow(["EFI_Score", g, s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])
        for k, s in sorted(mc_results["reg_ceat"].items()):
            w.writerow(["Regression_CEAT", k, s["observed"], s["mc_mean"], s["mc_se"], s["mc_ci_lower"], s["mc_ci_upper"]])

    print(f"→ CEAT uncertainty propagation written to {mc_tsv.name}")
    print("\n  PCA loadings sensitivity to CEAT SE perturbation:")
    for d in dims:
        s = mc_results["pc1_loadings"][d]
        print(f"    PC1 {d:<10}: obs={s['observed']:.4f}, MC 95% CI=[{s['mc_ci_lower']:.4f}, {s['mc_ci_upper']:.4f}]")
    print("\n  EFI score sensitivity:")
    for g, s in sorted(mc_results["efi_scores"].items()):
        print(f"    {g:<12}: obs={s['observed']:.3f}, MC 95% CI=[{s['mc_ci_lower']:.3f}, {s['mc_ci_upper']:.3f}]")
    if mc_results["reg_ceat"]:
        print("\n  CEAT regression sensitivity:")
        for k, s in sorted(mc_results["reg_ceat"].items()):
            print(f"    {k:<12}: MC 95% CI=[{s['mc_ci_lower']:.4f}, {s['mc_ci_upper']:.4f}]")

if __name__ == "__main__":
    main()
