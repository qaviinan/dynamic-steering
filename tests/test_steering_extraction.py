"""End-to-end steering extraction smoke test.

Steps:
- Load / format XSTest (saved to `data/matched_xstest_prompts.csv`).
- Use 10 matched prompt pairs for a quick run.
- For each pair: cache prompt-final activations (all layers) then generate response.
- Write CSV with prompts + responses for manual inspection.
- Save all-layer activations as [N, num_layers, d_model] NPZ.
- Pick target layer, train a linear probe, and evaluate.
- Extract steering vector (diff-of-means) and print diagnostics.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
import csv
import numpy as np

from model_utils import load_model
from data.format_xstest import format_and_save_matched_pairs, load_formatted_pairs
from activations.cache_activations import generate_and_cache_prompt
from probe.train_probe import train_probe
from steering.steering_utils import (
    steering_vector_diff_of_means,
    steering_vector_from_probe,
    diagnostics_projection,
)


def main():
    load_dotenv()

    model = load_model(
        model_id=None,
        model_id_abl=os.getenv("MODEL_ID_ABL"),
        model_id_base=os.getenv("MODEL_ID_BASE"),
    )

    csv_path = "data/matched_xstest_prompts.csv"
    print("Formatting XSTest (if not already saved)...")
    format_and_save_matched_pairs(model, save_path=csv_path, overwrite=False)

    rows = load_formatted_pairs(csv_path)
    if len(rows) == 0:
        raise RuntimeError("No matched pairs found in dataset")

    n_pairs = min(10, len(rows))
    print(f"Using {n_pairs} matched pairs for this smoke test.")

    pairs = rows[:n_pairs]
    resp_csv = "data/matched_xstest_with_responses.csv"
    os.makedirs("data", exist_ok=True)

    safe_per_example = []
    unsafe_per_example = []
    global_keys = None

    print("Generating responses and caching prompt-final activations (all layers)...")
    with open(resp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["focus", "safe_prompt", "safe_response", "unsafe_prompt", "unsafe_response"],
        )
        writer.writeheader()

        for pair in pairs:
            focus = pair.get("focus", "")
            safe_prompt = pair.get("formatted_safe") or pair.get("safe_raw") or ""
            unsafe_prompt = pair.get("formatted_unsafe") or pair.get("unsafe_raw") or ""

            safe_resp, safe_per_layer, keys = generate_and_cache_prompt(model, safe_prompt)
            unsafe_resp, unsafe_per_layer, _ = generate_and_cache_prompt(model, unsafe_prompt)

            if global_keys is None:
                global_keys = keys

            writer.writerow({
                "focus": focus,
                "safe_prompt": safe_prompt,
                "safe_response": safe_resp,
                "unsafe_prompt": unsafe_prompt,
                "unsafe_response": unsafe_resp,
            })

            safe_per_example.append(safe_per_layer)
            unsafe_per_example.append(unsafe_per_layer)

    print(f"Saved responses to {resp_csv}")

    # Stack all-layer activations: [N, num_layers, d_model]
    safe_array = np.array([[ex[k] for k in global_keys] for ex in safe_per_example])
    unsafe_array = np.array([[ex[k] for k in global_keys] for ex in unsafe_per_example])
    all_layers_path = "data/xstest_all_layers_activations.npz"
    np.savez_compressed(all_layers_path, keys=global_keys, safe=safe_array, unsafe=unsafe_array)
    print(f"Saved all-layer activations to {all_layers_path}  shape={safe_array.shape}")

    # Select target layer by matching cache key pattern (e.g. 'blocks.14.hook_resid_post')
    layer_to_probe = int(os.getenv("LAYER_TO_PROBE", "14"))
    target_idx = next(
        (i for i, k in enumerate(global_keys) if f"blocks.{layer_to_probe}." in k),
        None,
    )
    if target_idx is None:
        raise RuntimeError(f"Layer {layer_to_probe} not found in cache keys: {global_keys[:5]}")
    print(f"Using layer {layer_to_probe} (index {target_idx} of {len(global_keys)} cached layers).")

    safe_acts = safe_array[:, target_idx, :]
    unsafe_acts = unsafe_array[:, target_idx, :]

    layer_path = f"data/xstest_layer{layer_to_probe}_activations.npz"
    np.savez_compressed(layer_path, safe=safe_acts, unsafe=unsafe_acts)
    print(f"Saved layer-{layer_to_probe} activations to {layer_path}")

    print("Training linear probe on activations...")
    res = train_probe(unsafe_acts, safe_acts)
    print("Probe accuracy:", res["metrics"]["accuracy"])
    print("Probe ROC AUC:", res["metrics"]["roc_auc"])
    print("Classification report:\n", res["report"])

    print("Extracting steering vectors...")
    steer_dim = steering_vector_diff_of_means(unsafe_acts, safe_acts)
    steer_probe = steering_vector_from_probe(res["probe"])

    cos_sim = float(
        np.dot(steer_dim, steer_probe)
        / ((np.linalg.norm(steer_dim) * np.linalg.norm(steer_probe)) + 1e-12)
    )
    print("Cosine similarity between diff-of-means and probe vector:", cos_sim)

    print("Diagnostics for diff-of-means steering vector:")
    diag_dim = diagnostics_projection(steer_dim, unsafe_acts, safe_acts)
    for k, v in diag_dim.items():
        print(f"  {k}: {v}")

    print("Diagnostics for probe-derived steering vector:")
    diag_probe = diagnostics_projection(steer_probe, unsafe_acts, safe_acts)
    for k, v in diag_probe.items():
        print(f"  {k}: {v}")

    np.savez_compressed("data/steering_vectors.npz", diff_of_means=steer_dim, probe=steer_probe)
    print("Saved steering vectors to data/steering_vectors.npz")


if __name__ == "__main__":
    main()
