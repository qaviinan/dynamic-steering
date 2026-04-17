#!/usr/bin/env python3
"""Incremental steering vector extraction pipeline.

Source: data/xstest_full.csv  (every XSTest prompt, one row each, with
        focus / label / prompt / formatted_prompt columns)

Each prompt is stored individually under a SHA of its raw text so every
activation file is traceable back to the CSV.  Runs are resumable: already-
processed prompts are skipped and the steering vector is recomputed from all
accumulated activations at the end of each run.

Usage:
  python run_steering_extraction.py           # process DEFAULT_N_ROWS new prompts
  python run_steering_extraction.py -n 20     # process 20 new prompts (10 each side)
  python run_steering_extraction.py --status  # show per-focus progress, no model needed
  python run_steering_extraction.py --reset   # wipe cached activations, keep vectors

Storage layout:
  data/steering_state/
    progress.json          — registry: sha12 → {focus, type, csv_row, raw_preview}
    safe/{sha12}.npy       — [n_layers, d_model] float32 per cached safe prompt
    unsafe/{sha12}.npy     — [n_layers, d_model] float32 per cached unsafe prompt

Environment variables (all optional):
  STATE_DIR        activation cache root (default: data/steering_state)
  MODEL_ID_INSTRUCT safe model HF repo id to extract refusal from
  MODEL_ID_BASE    base model HF repo id (architecture)

Quality guide — diff-of-means Cohen's d:
  < 0.5   WEAK     add more prompts
  0.5–1.0 MODERATE may be unstable
  1.0–1.5 GOOD     ready to test
  > 1.5   STRONG   well-defined direction
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.format_xstest import load_and_save_full_dataset, load_full_dataset
from activations.cache_activations import generate_and_cache_prompt
from probe.train_probe import train_probe
from steering.steering_utils import (
    steering_vector_diff_of_means,
    steering_vector_from_probe,
    diagnostics_projection,
)

# ── Sampling configuration ────────────────────────────────────────────────────
# New prompts to process per run, split evenly between safe and unsafe sides.
DEFAULT_N_ROWS = 32

FULL_DATASET_PATH = "data/xstest_full.csv"
VECTORS_FILE = "data/steering_vectors.npz"


# ── Identifiers ────────────────────────────────────────────────────────────────

def prompt_sha(raw_text: str) -> str:
    """12-char hex digest of the raw prompt — stable, derivable from the CSV."""
    return hashlib.sha256(raw_text.strip().encode()).hexdigest()[:12]


# ── State paths ────────────────────────────────────────────────────────────────

def state_dir() -> str:
    return os.getenv("STATE_DIR", "data/steering_state")


def progress_path() -> str:
    return os.path.join(state_dir(), "progress.json")


def activation_path(sha: str, side: str) -> str:
    """side is 'safe' or 'unsafe'."""
    return os.path.join(state_dir(), side, f"{sha}.npy")


# ── Progress I/O ───────────────────────────────────────────────────────────────

def load_progress() -> dict:
    p = progress_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"model_id": None, "prompts": {}}


def save_progress(state: dict):
    os.makedirs(state_dir(), exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(progress_path(), "w") as f:
        json.dump(state, f, indent=2)


def reset_state():
    sd = state_dir()
    if os.path.exists(sd):
        shutil.rmtree(sd)
        print(f"Cleared: {sd}")
    else:
        print("Nothing to reset.")


# ── Activation helpers ─────────────────────────────────────────────────────────

def extract_all_layers(per_layer_dict: dict) -> np.ndarray:
    """Extract all post-LayerNorm residual streams into a [num_layers, d_model] array."""
    resid_keys = sorted(
        [k for k in per_layer_dict if "resid_post" in k], 
        key=lambda x: int(x.split('.')[1])
    )
    if not resid_keys:
        raise RuntimeError("No resid_post keys found in cache.")
    return np.stack([per_layer_dict[k] for k in resid_keys])


def load_activations_for_side(side: str, prompts: dict) -> tuple:
    """Return (arrays [N, num_layers, d_model], sha_list) for all cached prompts of `side`."""
    arrays, shas = [], []
    for sha, meta in prompts.items():
        if meta["type"] != side:
            continue
        p = activation_path(sha, side)
        if os.path.exists(p):
            arrays.append(np.load(p))
            shas.append(sha)
        else:
            print(f"  Warning: missing file sha={sha} focus={meta.get('focus')} — skipped.")
    return arrays, shas


# ── Status display ─────────────────────────────────────────────────────────────

def print_status(prompts: dict, all_rows: list):
    total = defaultdict(lambda: {"safe": 0, "unsafe": 0})
    cached = defaultdict(lambda: {"safe": 0, "unsafe": 0})

    for row in all_rows:
        total[row["focus"]][row["label"]] += 1
    for meta in prompts.values():
        cached[meta["focus"]][meta["type"]] += 1

    all_focuses = sorted(total.keys())
    t_safe  = sum(v["safe"]   for v in total.values())
    t_unsafe = sum(v["unsafe"] for v in total.values())
    c_safe  = sum(v["safe"]   for v in cached.values())
    c_unsafe = sum(v["unsafe"] for v in cached.values())

    print(f"\n{'Focus':<28}  {'safe':>14}  {'unsafe':>14}")
    print("─" * 60)
    for focus in all_focuses:
        s = f"{cached[focus]['safe']}/{total[focus]['safe']}"
        u = f"{cached[focus]['unsafe']}/{total[focus]['unsafe']}"
        print(f"  {focus:<26}  {s:>14}  {u:>14}")
    print("─" * 60)
    print(f"  {'TOTAL':<26}  {f'{c_safe}/{t_safe}':>14}  {f'{c_unsafe}/{t_unsafe}':>14}")
    print()


# ── Quality diagnostics ────────────────────────────────────────────────────────

def print_quality(label: str, steer_vec, unsafe_acts, safe_acts):
    d = diagnostics_projection(steer_vec, unsafe_acts, safe_acts)
    auc = f"{d['roc_auc']:.3f}" if d["roc_auc"] is not None else "n/a"
    print(f"  [{label:14s}]  n_unsafe={len(unsafe_acts):3d}  n_safe={len(safe_acts):3d}"
          f"  cohen_d={d['cohen_d']:6.3f}  roc_auc={auc}")


def quality_verdict(cohen_d: float) -> str:
    if cohen_d < 0.5:
        return "WEAK — add more prompts before using this vector"
    if cohen_d < 1.0:
        return "MODERATE — may be unstable; consider more prompts"
    if cohen_d < 1.5:
        return "GOOD — ready to test"
    return "STRONG — well-defined direction"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incremental steering vector extraction")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe cached activations and progress (vectors file kept)")
    parser.add_argument("--status", action="store_true",
                        help="Show per-focus cached/total counts and exit (no model needed)")
    parser.add_argument("-n", "--n-rows", type=int, default=DEFAULT_N_ROWS,
                        help=f"New prompts to process this run, split evenly safe/unsafe "
                             f"(default: {DEFAULT_N_ROWS})")
    args = parser.parse_args()

    if args.reset:
        reset_state()
        return

    load_dotenv()

    # ── Load existing state ────────────────────────────────────────────────
    progress = load_progress()
    stored_model = progress.get("model_id")
    current_model = os.getenv("MODEL_ID_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
    prompts: dict = progress.get("prompts", {})

    if stored_model and stored_model != current_model:
        print(
            f"Warning: stored activations are from '{stored_model}' but "
            f"current instruct target is '{current_model}'. Mixing models corrupts the vector.\n"
            f"Run --reset if you have switched base models."
        )

    # ── --status: load CSV only, no model ─────────────────────────────────
    if args.status:
        all_rows = load_full_dataset(FULL_DATASET_PATH)
        print_status(prompts, all_rows)
        return

    # ── Load model and dataset ─────────────────────────────────────────────
    print(f"Loading safe/instruct model ({current_model}) to extract refusal...")
    # Load directly to guarantee base/instruct weights — bypasses setup_model.py
    # so MODEL_ID_ABL env var cannot accidentally route to abliterated weights.
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.bfloat16 if _device.startswith("cuda") else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(current_model, use_fast=True, trust_remote_code=True)
    _hf_model = AutoModelForCausalLM.from_pretrained(current_model, torch_dtype=_dtype, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(
        current_model,
        hf_model=_hf_model,
        tokenizer=_tokenizer,
        device=_device,
        default_padding_side="left",
        trust_remote_code=True,
    )
    del _hf_model

    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    load_and_save_full_dataset(model, save_path=FULL_DATASET_PATH, overwrite=False)
    all_rows = load_full_dataset(FULL_DATASET_PATH)

    unsafe_rows = [(i, r) for i, r in enumerate(all_rows) if r["label"] == "unsafe"]
    safe_rows   = [(i, r) for i, r in enumerate(all_rows) if r["label"] == "safe"]
    print(f"Dataset: {len(all_rows)} prompts total  "
          f"({len(unsafe_rows)} unsafe, {len(safe_rows)} safe)\n")

    # ── Determine pending prompts ──────────────────────────────────────────
    already_processed = set(prompts.keys())

    pending_unsafe = [(i, r) for i, r in unsafe_rows
                      if prompt_sha(r["prompt"]) not in already_processed]
    pending_safe   = [(i, r) for i, r in safe_rows
                      if prompt_sha(r["prompt"]) not in already_processed]

    print(f"Progress: {len(already_processed)} prompts cached  "
          f"({len(pending_unsafe)} unsafe pending, {len(pending_safe)} safe pending)\n")

    # ── Process new batch ──────────────────────────────────────────────────
    n_per_side = max(1, args.n_rows // 2)
    to_process = pending_unsafe[:n_per_side] + pending_safe[:n_per_side]

    if not to_process:
        print("All prompts already processed. Use --reset to start over.")
    else:
        print(f"Processing {len(to_process)} prompts "
              f"(up to {n_per_side} unsafe + {n_per_side} safe)...\n")

        for csv_row, row in to_process:
            label  = row["label"]            # "safe" or "unsafe"
            focus  = row.get("focus", "unknown")
            raw    = row.get("prompt") or ""
            formatted = row.get("formatted_prompt") or raw
            sha    = prompt_sha(raw)

            print(f"  [row {csv_row:04d}] {label:6s}  focus={focus!r}")

            _, per_layer, _ = generate_and_cache_prompt(model, formatted)
            # act is now [num_layers, d_model]
            act = extract_all_layers(per_layer).astype(np.float32)

            os.makedirs(os.path.join(state_dir(), label), exist_ok=True)
            np.save(activation_path(sha, label), act)

            prompts[sha] = {
                "type": label,
                "focus": focus,
                "csv_row": csv_row,
                "raw_preview": raw[:80],
            }
            print(f"    saved  sha={sha}  ({raw[:60]!r})")

        progress["prompts"]  = prompts
        progress["model_id"] = current_model
        save_progress(progress)
        print(f"\nProgress saved. Total cached: {len(prompts)}\n")

    # ── Recompute steering vector from all stored activations ──────────────
    if not prompts:
        print("No activations stored yet.")
        return

    unsafe_arrays, _ = load_activations_for_side("unsafe", prompts)
    safe_arrays,   _ = load_activations_for_side("safe",   prompts)

    if not unsafe_arrays or not safe_arrays:
        print("Need at least one safe and one unsafe activation to compute a vector.")
        return

    # Arrays are [N, num_layers, d_model]
    unsafe_acts_all = np.array(unsafe_arrays)
    safe_acts_all   = np.array(safe_arrays)

    print(f"Evaluating {unsafe_acts_all.shape[1]} layers to locate optimal Refusal Feature...")
    
    num_layers = unsafe_acts_all.shape[1]
    best_layer = -1
    max_norm = 0

    for l in range(num_layers):
        layer_diff = np.mean(unsafe_acts_all[:, l, :], axis=0) - np.mean(safe_acts_all[:, l, :], axis=0)
        norm = np.linalg.norm(layer_diff)
        if norm > max_norm:
            max_norm = norm
            best_layer = l

    print(f"Optimal Refusal Direction found at Layer {best_layer} (Magnitude: {max_norm:.3f})\n")

    # Isolate the data for the best layer to generate the final vectors
    unsafe_acts = unsafe_acts_all[:, best_layer, :]
    safe_acts   = safe_acts_all[:, best_layer, :]

    print(f"Computing vectors for Layer {best_layer}  "
          f"n_unsafe={len(unsafe_acts)}  n_safe={len(safe_acts)}\n")

    steer_dim = steering_vector_diff_of_means(unsafe_acts, safe_acts)

    steer_probe = steer_dim
    if len(unsafe_acts) >= 6 and len(safe_acts) >= 6:
        res = train_probe(unsafe_acts, safe_acts)
        steer_probe = steering_vector_from_probe(res["probe"])
        cos_sim = float(
            np.dot(steer_dim, steer_probe)
            / (np.linalg.norm(steer_dim) * np.linalg.norm(steer_probe) + 1e-12)
        )
        roc = res["metrics"]["roc_auc"]
        roc_str = f"{roc:.3f}" if roc is not None else "n/a"
        print(f"  Probe  accuracy={res['metrics']['accuracy']:.3f}  "
              f"roc_auc={roc_str}  "
              f"cos_sim(dim↔probe)={cos_sim:.3f}")
    else:
        print(f"  (Probe skipped — need ≥6 of each side; "
              f"have {len(unsafe_acts)} unsafe / {len(safe_acts)} safe)")

    print()
    print_quality("diff-of-means", steer_dim,   unsafe_acts, safe_acts)
    if steer_probe is not steer_dim:
        print_quality("probe",         steer_probe, unsafe_acts, safe_acts)

    diag = diagnostics_projection(steer_dim, unsafe_acts, safe_acts)
    print(f"\n  Verdict: {quality_verdict(diag['cohen_d'])}")

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(VECTORS_FILE, 
                        diff_of_means=steer_dim, 
                        probe=steer_probe, 
                        optimal_layer=best_layer)
    print(f"\n  Saved: {VECTORS_FILE} (Includes optimal_layer index for intervention hook)")

    print("\n─── Next steps ───────────────────────────────────────────────────────")
    if pending_unsafe or pending_safe:
        print(f"  Add more:  python run_steering_extraction.py -n {args.n_rows}")
    print(f"  Status:    python run_steering_extraction.py --status")
    print(f"  Test:      python tests/test_steering_intervention.py")


if __name__ == "__main__":
    main()