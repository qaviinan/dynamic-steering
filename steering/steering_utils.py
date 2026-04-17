"""Steering vector extraction and diagnostics utilities.

Provides difference-in-means extraction, probe-derived vector normalization,
and projection diagnostics (Cohen's d, means, ROC AUC on projections).
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import roc_auc_score


def steering_vector_diff_of_means(unsafe_acts: np.ndarray, safe_acts: np.ndarray) -> np.ndarray:
    """Return μ_unsafe − μ_safe at native activation-space scale (not unit-normalised).

    Preserving native scale matters: unit-normalising strips the magnitude that
    calibrates the vector to the model's internal representation, forcing you to
    compensate with a large arbitrary alpha that collapses LayerNorm variance.
    With native scale, alpha=1.0 is a natural starting point; tune from there.
    """
    mean_unsafe = np.mean(unsafe_acts, axis=0)
    mean_safe   = np.mean(safe_acts,   axis=0)
    return (mean_unsafe - mean_safe).astype(np.float32)


def steering_vector_from_probe(probe) -> np.ndarray:
    coef = probe.coef_[0]
    norm = np.linalg.norm(coef)
    if norm == 0:
        return coef
    return coef / norm


def diagnostics_projection(steer_vec: np.ndarray, unsafe_acts: np.ndarray, safe_acts: np.ndarray) -> Dict[str, Any]:
    # project
    proj_unsafe = unsafe_acts.dot(steer_vec)
    proj_safe = safe_acts.dot(steer_vec)

    mu_u = float(np.mean(proj_unsafe))
    mu_s = float(np.mean(proj_safe))
    sd_u = float(np.std(proj_unsafe, ddof=1)) if len(proj_unsafe) > 1 else float(np.std(proj_unsafe))
    sd_s = float(np.std(proj_safe, ddof=1)) if len(proj_safe) > 1 else float(np.std(proj_safe))

    # pooled std
    pooled_std = np.sqrt(((len(proj_unsafe) - 1) * sd_u ** 2 + (len(proj_safe) - 1) * sd_s ** 2) / max(1, (len(proj_unsafe) + len(proj_safe) - 2)))
    cohen_d = (mu_u - mu_s) / (pooled_std + 1e-12)

    # ROC AUC
    y = np.concatenate([np.ones(len(proj_unsafe)), np.zeros(len(proj_safe))])
    scores = np.concatenate([proj_unsafe, proj_safe])
    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = None

    return {
        "mean_unsafe": mu_u,
        "mean_safe": mu_s,
        "std_unsafe": sd_u,
        "std_safe": sd_s,
        "cohen_d": cohen_d,
        "roc_auc": auc,
    }
