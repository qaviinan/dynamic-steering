"""Hook-based activation addition for TransformerLens HookedTransformer.

Applies h̃_t^{(l)} = h_t^{(l)} + α * v at the last token position only.
Use negative α to subtract the concept direction (concept erasure / safety).

Alpha interpretation (steer_vec is at native activation-space scale):
  alpha =  1.0  natural-magnitude push toward the concept
  alpha = -1.0  natural-magnitude erasure away from the concept
  Tune in the range [-2, 2] before reaching for larger values.
"""
import numpy as np
import torch

from model_utils import generate_response


def make_steering_hook(steer_vec: np.ndarray, alpha: float):
    """Return a TransformerLens hook_fn that adds alpha * steer_vec to the
    last token position of the residual stream only.

    Targeting only position [-1] prevents corrupting the prompt's structural
    tokens (punctuation, system-prompt, etc.) whose self-attention keys were
    computed without the intervention, which would break causality and produce
    gibberish.  The direction is cast to the model's device/dtype on first call.
    """
    direction = torch.tensor(steer_vec, dtype=torch.float32)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        # value shape: [batch, seq_len, d_model]
        # Cast once; write back only to the last token position
        d = direction.to(device=value.device, dtype=value.dtype)
        value[:, -1, :] = value[:, -1, :] + alpha * d
        return value

    return hook_fn


def run_with_steering(
    model,
    prompt: str,
    steer_vec: np.ndarray,
    layer_idx: int,
    alpha: float,
    max_new_tokens: int = 200,
) -> str:
    """Generate a response with the steering vector injected at `layer_idx`.

    Registers a hook on `blocks.{layer_idx}.hook_resid_post`, generates, then
    clears all hooks via reset_hooks() regardless of success or failure.
    """
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    hook_fn = make_steering_hook(steer_vec, alpha)

    try:
        model.add_hook(hook_name, hook_fn)
        return generate_response(model, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    finally:
        model.reset_hooks()
