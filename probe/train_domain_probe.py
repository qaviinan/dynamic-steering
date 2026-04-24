#!/usr/bin/env python3
"""Train a non-linear 3-class domain probe on residual-stream activations.

Classes:
  0  safe               — benign prompts (from XSTest safe rows)
  1  unsafe + cyber     — security/IT content that safety-tuned models refuse
  2  unsafe + general   — general harmful content (from XSTest unsafe rows)

The probe is an MLPClassifier trained on final-token activations at the
(optimal_layer, optimal_stream) identified by run_refusal_extraction.py.
A non-linear probe is needed because the boundary between cyber and general
unsafe content is not linearly separable in the refusal direction alone.

Usage:
  python probe/train_domain_probe.py

Prerequisites:
  1. python run_refusal_extraction.py -n 64  (populate steering_state cache)
  2. data/steering_vectors.npz must exist with optimal_layer + optimal_stream

Output:
  data/domain_probe.pkl   — trained MLPClassifier via joblib
"""
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

from activations.cache_activations import STREAM_TYPES, get_activations_for_prompts
from data.cyber_prompts import get_cyber_prompts_formatted
from model_config import load_instruct_model

load_dotenv()

VECTORS_FILE  = os.getenv("VECTORS_FILE", "data/steering_vectors.npz")
STATE_DIR     = os.getenv("STATE_DIR",    "data/steering_state")
PROBE_FILE    = "data/domain_probe.pkl"

LABEL_NAMES   = {0: "safe", 1: "unsafe_cyber", 2: "unsafe_general"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cached_activations(
    side: str,
    progress: dict,
    stream_idx: int,
    optimal_layer: int,
) -> np.ndarray:
    """Load activations at (stream_idx, optimal_layer) for all cached prompts on `side`.

    Handles both cache formats:
      v2: [n_stream_types, n_layers, d_model]  (current)
      v1: [n_layers, d_model]                  (legacy — stream_idx ignored)
    """
    arrays = []
    for sha, meta in progress["prompts"].items():
        if meta["type"] != side:
            continue
        path = os.path.join(STATE_DIR, side, f"{sha}.npy")
        if not os.path.exists(path):
            print(f"  Warning: missing {path} — skipped.")
            continue
        arr = np.load(path)
        if arr.ndim == 3:
            arrays.append(arr[stream_idx, optimal_layer, :])
        else:
            # v1 legacy format
            arrays.append(arr[optimal_layer, :])
    if not arrays:
        raise RuntimeError(
            f"No cached '{side}' activations found in {STATE_DIR}. "
            f"Run run_refusal_extraction.py first."
        )
    return np.array(arrays, dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load extraction metadata ──────────────────────────────────────────
    if not os.path.exists(VECTORS_FILE):
        print(f"ERROR: {VECTORS_FILE} not found. Run run_refusal_extraction.py first.")
        sys.exit(1)

    vec_data       = np.load(VECTORS_FILE, allow_pickle=True)
    optimal_layer  = int(vec_data["optimal_layer"])
    optimal_stream = str(vec_data["optimal_stream"])
    stream_idx     = list(STREAM_TYPES).index(optimal_stream)

    print(f"Optimal layer={optimal_layer}  stream={optimal_stream}  stream_idx={stream_idx}")

    progress_file = os.path.join(STATE_DIR, "progress.json")
    if not os.path.exists(progress_file):
        print(f"ERROR: {progress_file} not found. Run run_refusal_extraction.py first.")
        sys.exit(1)

    with open(progress_file) as f:
        progress = json.load(f)

    # ── Load safe (label 0) and unsafe_general (label 2) from cache ───────
    print("\nLoading safe activations (label 0)...")
    safe_acts = load_cached_activations("safe", progress, stream_idx, optimal_layer)
    print(f"  safe: {safe_acts.shape}")

    print("Loading unsafe_general activations (label 2)...")
    unsafe_general_acts = load_cached_activations("unsafe", progress, stream_idx, optimal_layer)
    print(f"  unsafe_general: {unsafe_general_acts.shape}")

    # ── Collect cyber activations via live model forward pass (label 1) ───
    print("\nLoading model to collect cyber activations (label 1)...")
    model = load_instruct_model()

    cyber_formatted = get_cyber_prompts_formatted(model.tokenizer)
    print(f"Collecting activations for {len(cyber_formatted)} cyber prompts...")
    cyber_acts = get_activations_for_prompts(
        model,
        cyber_formatted,
        layer_idx=optimal_layer,
        stream_type=optimal_stream,
    )
    print(f"  unsafe_cyber: {cyber_acts.shape}")

    # ── Assemble dataset ──────────────────────────────────────────────────
    X = np.vstack([safe_acts, cyber_acts, unsafe_general_acts])
    y = np.concatenate([
        np.zeros(len(safe_acts),           dtype=int),   # 0 — safe
        np.ones( len(cyber_acts),          dtype=int),   # 1 — unsafe_cyber
        np.full( len(unsafe_general_acts), 2, dtype=int),# 2 — unsafe_general
    ])

    print(f"\nDataset  total={X.shape[0]}  d={X.shape[1]}")
    for label, name in LABEL_NAMES.items():
        print(f"  label {label} ({name}): {(y == label).sum()}")

    # ── Train / eval split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining MLPClassifier(hidden=[256, 128])  "
          f"train={len(X_train)}  test={len(X_test)}...")
    probe = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds = probe.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.3f}")

    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=target_names, zero_division=0))

    try:
        probs = probe.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        print(f"Macro ROC AUC (one-vs-rest): {auc:.3f}")
    except Exception as e:
        print(f"ROC AUC: {e}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    joblib.dump(probe, PROBE_FILE)
    print(f"\nSaved domain probe → {PROBE_FILE}")
    print("  Next: python demo_business_cases.py")


if __name__ == "__main__":
    main()
