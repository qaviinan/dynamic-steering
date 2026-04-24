"""Cache final-token residual-stream activations for a list of prompts.

Captures all three residual stream hook points per layer:
  hook_resid_pre  — before attention
  hook_resid_mid  — between attention and MLP
  hook_resid_post — after MLP

Callers (e.g. run_refusal_extraction.py) can then select among stream types
when ranking candidate refusal directions.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model_utils import generate_response

# Canonical ordering for the stream-type axis of cached arrays.
STREAM_TYPES: Tuple[str, ...] = ("resid_pre", "resid_mid", "resid_post")


def generate_and_cache_prompt(model, prompt: str, max_new_tokens: int = 64):
    """Causally extract prompt-final activations, then generate a response.

    Captures hook_resid_pre, hook_resid_mid, and hook_resid_post for every
    layer (last prompt token only). If a stream type is absent for a given
    architecture the dict simply won't contain its keys.

    Returns: (response_str, per_layer_dict, resid_keys)
      - per_layer_dict: {hook_key: np.array([d_model])}  one entry per layer per stream type
      - resid_keys:     list of the captured hook key strings
    """
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = model.to_tokens([prompt], prepend_bos=True).to(device)

    try:
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: any(st in name for st in STREAM_TYPES),
        )
    except Exception:
        out = model.run_with_cache(tokens)
        if isinstance(out, tuple) and len(out) == 2:
            logits, cache = out
        else:
            logits = None
            cache = getattr(out, "cache", None)

    if cache is None:
        raise RuntimeError("run_with_cache did not return a cache object")

    resid_keys = [k for k in cache.keys() if any(st in k for st in STREAM_TYPES)]

    per_layer: Dict[str, np.ndarray] = {}
    for k in resid_keys:
        t = cache[k]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        per_layer[k] = t[0, -1, :].detach().cpu().numpy()

    del cache, logits
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    response = generate_response(model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    return response, per_layer, resid_keys


def get_activations_for_prompts(
    model,
    prompts: List[str],
    layer_idx: int = 14,
    stream_type: str = "resid_post",
    batch_size: int = 8,
) -> np.ndarray:
    """Return final-token activations for `prompts` at one (layer_idx, stream_type) pair.

    Output shape: [len(prompts), d_model]

    Args:
        model:       HookedTransformer.
        prompts:     Pre-formatted prompt strings.
        layer_idx:   Which layer to read from (0-indexed).
        stream_type: One of STREAM_TYPES ("resid_pre", "resid_mid", "resid_post").
        batch_size:  Prompts per forward pass.
    """
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hook_key = f"blocks.{layer_idx}.hook_{stream_type}"

    all_acts: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Activations"):
            batch = prompts[i : i + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True).to(device)

            out = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_key,
            )
            cache = getattr(out, "cache", None)
            if cache is None and isinstance(out, tuple) and len(out) == 2:
                _, cache = out
            if cache is None:
                raise RuntimeError("run_with_cache did not return a cache object")

            if hook_key not in cache:
                # Fallback: take the first resid key at this layer
                fallback = [k for k in cache.keys() if f"blocks.{layer_idx}" in k]
                if not fallback:
                    raise RuntimeError(
                        f"No cache key found for layer {layer_idx}  stream={stream_type}"
                    )
                hook_key_used = fallback[0]
            else:
                hook_key_used = hook_key

            t = cache[hook_key_used]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            all_acts.append(t[:, -1, :].detach().cpu().numpy())

            del cache
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return np.vstack(all_acts)


def save_activation_pairs(save_path: str, unsafe_acts: np.ndarray, safe_acts: np.ndarray):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, unsafe=unsafe_acts, safe=safe_acts)


def load_activation_pairs(save_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)
    arr = np.load(save_path)
    return arr["unsafe"], arr["safe"]
