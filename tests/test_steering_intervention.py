"""Steering vector intervention test.

Compares model responses on one safe and one unsafe prompt from the full dataset:
  Control:   abliterated model, no hook
  Treatment: abliterated model with diff-of-means steering vector subtracted
             at the target layer (negative alpha = concept erasure)

Expected outcome:
  Unsafe prompt -> control complies, treatment refuses or deflects
  Safe  prompt  -> control and treatment responses are substantively identical

Environment variables (all optional):
  LAYER_TO_PROBE    residual layer index to hook (default: 14)
  STEER_ALPHA       scale factor on the native-magnitude diff vector
                    (default: 1.0 = natural scale; tune in 0.5–2.0 first)
  MAX_NEW_TOKENS    tokens to generate per call (default: 200)
  UNSAFE_ROW_INDEX  index into unsafe rows of xstest_full.csv (default: 0)
  SAFE_ROW_INDEX    index into safe rows of xstest_full.csv (default: 0)
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
import numpy as np

from model_utils import load_model, generate_response
from data.format_xstest import load_full_dataset
from steering.apply_steering import run_with_steering
from steering.steering_utils import steering_vector_diff_of_means

SEP = "=" * 72
STATE_DIR = os.getenv("STATE_DIR", "data/steering_state")


def load_steer_vec_from_activations() -> np.ndarray:
    """Recompute diff-of-means at native scale from the cached .npy files.

    Always recomputes from raw activations so the vector reflects the current
    (fixed, non-normalised) steering_vector_diff_of_means implementation,
    regardless of what was previously written to steering_vectors.npz.
    """
    import json

    progress_path = os.path.join(STATE_DIR, "progress.json")
    if not os.path.exists(progress_path):
        raise FileNotFoundError(
            f"{progress_path} not found — run run_steering_extraction.py first"
        )

    with open(progress_path) as f:
        prompts = json.load(f).get("prompts", {})

    safe_acts, unsafe_acts = [], []
    for sha, meta in prompts.items():
        side = meta["type"]
        p = os.path.join(STATE_DIR, side, f"{sha}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        if side == "safe":
            safe_acts.append(arr)
        else:
            unsafe_acts.append(arr)

    if not unsafe_acts or not safe_acts:
        raise RuntimeError(
            f"No activations found in {STATE_DIR} — run run_steering_extraction.py first"
        )

    print(f"Recomputed steer_vec from {len(unsafe_acts)} unsafe + {len(safe_acts)} safe activations")
    return steering_vector_diff_of_means(np.array(unsafe_acts), np.array(safe_acts))


def print_result(tag: str, prompt: str, response: str):
    print(f"\n{SEP}")
    print(f"  {tag}")
    print(SEP)
    print(f"PROMPT:\n{prompt.strip()}\n")
    print(f"RESPONSE:\n{response.strip()}")
    print(SEP)


def main():
    load_dotenv()

    model = load_model(
        model_id=None,
        model_id_abl=os.getenv("MODEL_ID_ABL"),
        model_id_base=os.getenv("MODEL_ID_BASE"),
    )

    steer_vec = load_steer_vec_from_activations()   # native-scale [d_model]

    layer_idx = int(os.getenv("LAYER_TO_PROBE", "18"))
    alpha_magnitude = float(os.getenv("STEER_ALPHA", "-5.0"))
    alpha = -alpha_magnitude          # subtract the harmful direction
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "200"))
    unsafe_row_index = int(os.getenv("UNSAFE_ROW_INDEX", "12"))
    safe_row_index   = int(os.getenv("SAFE_ROW_INDEX",   "12"))

    print(f"\nConfig  layer={layer_idx}  alpha={alpha}  max_new_tokens={max_new_tokens}")

    all_rows = load_full_dataset("data/xstest_full.csv")
    unsafe_rows = [r for r in all_rows if r["label"] == "unsafe"]
    safe_rows   = [r for r in all_rows if r["label"] == "safe"]

    if not unsafe_rows or not safe_rows:
        raise RuntimeError("data/xstest_full.csv missing or empty — run run_steering_extraction.py first")
    if unsafe_row_index >= len(unsafe_rows):
        raise IndexError(f"UNSAFE_ROW_INDEX={unsafe_row_index} out of range ({len(unsafe_rows)} unsafe rows)")
    if safe_row_index >= len(safe_rows):
        raise IndexError(f"SAFE_ROW_INDEX={safe_row_index} out of range ({len(safe_rows)} safe rows)")

    unsafe_row = unsafe_rows[unsafe_row_index]
    safe_row   = safe_rows[safe_row_index]
    unsafe_prompt = unsafe_row.get("formatted_prompt") or unsafe_row.get("prompt") or ""
    safe_prompt   = safe_row.get("formatted_prompt")   or safe_row.get("prompt")   or ""

    print(f"Unsafe focus: {unsafe_row.get('focus')}  prompt: {unsafe_row.get('prompt', '')[:60]!r}")
    print(f"Safe   focus: {safe_row.get('focus')}    prompt: {safe_row.get('prompt', '')[:60]!r}")

    # ── UNSAFE PROMPT ──────────────────────────────────────────────────────
    print("\n\n" + "█" * 72)
    print("  UNSAFE PROMPT")
    print("█" * 72)

    ctrl_unsafe = generate_response(model, unsafe_prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    print_result("CONTROL — abliterated model, no intervention", unsafe_prompt, ctrl_unsafe)

    treat_unsafe = run_with_steering(model, unsafe_prompt, steer_vec, layer_idx, alpha, max_new_tokens)
    print_result("TREATMENT — steering vector applied (concept erasure)", unsafe_prompt, treat_unsafe)
    

    # # ── SAFE PROMPT ────────────────────────────────────────────────────────
    # print("\n\n" + "█" * 72)
    # print("  SAFE PROMPT")
    # print("█" * 72)

    # ctrl_safe = generate_response(model, safe_prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    # print_result("CONTROL — abliterated model, no intervention", safe_prompt, ctrl_safe)

    # treat_safe = run_with_steering(model, safe_prompt, steer_vec, layer_idx, alpha, max_new_tokens)
    # print_result("TREATMENT — steering vector applied (concept erasure)", safe_prompt, treat_safe)


if __name__ == "__main__":
    main()
