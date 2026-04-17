"""Cache final-token residual-stream activations for a list of prompts.

Provides utilities to extract activations for a specific layer index and save
them for later use.
"""
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model_utils import find_resid_keys, get_layer_key, generate_response


def generate_and_cache_prompt(model, prompt: str, max_new_tokens: int = 64):
    """Causally extract prompt-final activations, then generate a response.

    Steps:
    1) Run `run_with_cache()` on the prompt tokens only and extract the
       prompt-final residual-stream activations (post-LN keys).
    2) Free the cache and GPU memory.
    3) Generate the model response (deterministic by default) for logging.

    Returns: (response_str, per_layer_dict, ordered_keys)
      - per_layer_dict: {key: np.array(d_model)}
      - ordered_keys: list of keys (same order used to build arrays)
    """
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize prompt only so we capture h_T^{(l)} (final prompt token)
    tokens = model.to_tokens([prompt], prepend_bos=True).to(device)

    # run_with_cache can return (logits, cache) or an object with .cache
    try:
        logits, cache = model.run_with_cache(tokens)
    except Exception:
        out = model.run_with_cache(tokens)
        if isinstance(out, tuple) and len(out) == 2:
            logits, cache = out
        else:
            logits = None
            cache = getattr(out, "cache", None)

    if cache is None:
        raise RuntimeError("run_with_cache did not return a cache object")

    # Prefer post-LN residual stream keys for steering
    resid_keys = [k for k in cache.keys() if "resid_post" in k]

    per_layer = {}
    for k in resid_keys:
        t = cache[k]  # expected shape [B, T, d_model]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        # batch 0, final token of prompt
        per_layer[k] = t[0, -1, :].detach().cpu().numpy()

    # free the cache before generating to avoid holding both cache + generation KV
    del cache
    del logits
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # generate the response deterministically for logging
    response = generate_response(model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    return response, per_layer, resid_keys


def get_activations_for_prompts(model, prompts: List[str], layer_idx: int = 14, batch_size: int = 8) -> np.ndarray:
    """Return an array of final-token activations for `prompts` at `layer_idx`.

    Output shape: [len(prompts), d_model]
    """
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_acts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_texts = prompts[i : i + batch_size]
            tokens = model.to_tokens(batch_texts, prepend_bos=True)
            tokens = tokens.to(device)

            out = model.run_with_cache(tokens)
            cache = getattr(out, "cache", None)
            if cache is None and isinstance(out, tuple) and len(out) == 2:
                _, cache = out

            if cache is None:
                raise RuntimeError("run_with_cache did not return a cache object")

            resid_keys = find_resid_keys(cache)
            key = get_layer_key(cache, layer_idx)
            if key is None:
                raise RuntimeError("Could not find residual-stream key in cache")

            t = cache[key]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)

            final_token_activations = t[:, -1, :].detach().cpu().numpy()
            all_acts.append(final_token_activations)

            # cleanup
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
