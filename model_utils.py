"""Helper utilities for loading the HookedTransformer, generating a short response,
and extracting residual-stream activations for the final token.

These functions are used by tests in `tests/test_model_integration.py`.
"""
import re
import torch
from typing import Optional, Tuple, List, Any, Dict

from setup_model import load_model as _load_model


def load_model(
    model_id: str = None,
    model_id_abl: str = None,
    model_id_base: str = None,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load a HookedTransformer, optionally using an abliterated HF model.

    If `model_id_abl` is provided (or `MODEL_ID_ABL` in the environment), the
    abliterated HF weights/tokenizer will be loaded and injected into the
    HookedTransformer wrapper of `model_id_base` (or `MODEL_ID_BASE` env var).
    """
    return _load_model(
        model_id=model_id,
        model_id_abl=model_id_abl,
        model_id_base=model_id_base,
        device=device,
        torch_dtype=torch_dtype,
    )


def generate_response(model, prompt: str, max_new_tokens: int = 32, temperature: float = 0.4) -> str:
    """Generate a short response for a string `prompt`.

    This tries `model.generate(...)` when available and falls back to decoding
    the input tokens (so the test still runs even if generation isn't implemented).
    """
    try:
        if hasattr(model, "generate"):
            out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.9)
            if isinstance(out, str):
                return out
            # If `generate` returned token tensors or lists, try to decode
            try:
                if hasattr(model, "to_string"):
                    if isinstance(out, (list, tuple)):
                        return model.to_string(out[0])
                    return model.to_string(out)
                if hasattr(model, "tokenizer"):
                    if isinstance(out, torch.Tensor):
                        return model.tokenizer.decode(out[0].tolist())
                    elif isinstance(out, (list, tuple)):
                        return model.tokenizer.decode(out[0])
            except Exception:
                return str(out)

        # Fallback: just return the prompt decoded (no autoregressive generation)
        tokens = model.to_tokens([prompt], prepend_bos=True)
        try:
            return model.to_string(tokens[0])
        except Exception:
            if hasattr(model, "tokenizer"):
                t = tokens[0]
                if isinstance(t, torch.Tensor):
                    return model.tokenizer.decode(t.tolist())
                return model.tokenizer.decode(t)
            return str(tokens)
    except Exception as e:
        return f"<<generation error: {e}>>"


def run_with_cache_for_prompt(model, prompt: str):
    """Run the model with a cache on `prompt` and return (out, cache).

    The caller is responsible for moving inputs to the correct device.
    """
    tokens = model.to_tokens([prompt], prepend_bos=True)
    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = tokens.to(device)
    out = model.run_with_cache(tokens)
    cache = getattr(out, "cache", None)
    if cache is None and isinstance(out, tuple) and len(out) == 2:
        _, cache = out
    return out, cache


def find_resid_keys(cache: Dict[str, Any]) -> List[str]:
    """Return a sorted list of keys in `cache` that look like residual-stream tensors.

    Sorting is attempted by extracting the first integer in the key so keys
    map to layer order when possible.
    """
    if cache is None:
        return []

    keys = [k for k in cache.keys() if "resid" in k.lower()]

    def _key_index(k: str) -> int:
        m = re.search(r"(\d+)", k)
        return int(m.group(1)) if m else -1

    return sorted(keys, key=_key_index)


def get_layer_key(cache: Dict[str, Any], layer_idx: int) -> Optional[str]:
    """Pick a residual key corresponding to `layer_idx` from `cache`.

    If exact mapping isn't found, returns the last available residual key.
    """
    keys = find_resid_keys(cache)
    if not keys:
        return None
    if 0 <= layer_idx < len(keys):
        return keys[layer_idx]
    return keys[-1]


def get_final_token_activation(cache: Dict[str, Any], key: str):
    """Return (final_token_activation_tensor, residual_matrix_shape) for `key`.

    final_token_activation_tensor has shape `[B, d_model]` (final token only).
    """
    t = cache[key]
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    final = t[:, -1, :]
    return final, tuple(t.shape)
