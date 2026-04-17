"""Model loader for TransformerLens HookedTransformer.

Loads a HookedTransformer instance and returns it. Reads `HUGGINGFACE_HUB_TOKEN`
from the environment (use a .env file).
"""
import os
from dotenv import load_dotenv
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_id: str | None = None,
    model_id_abl: str | None = None,
    model_id_base: str | None = None,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
    fallback_model: str = "gpt2",
):
    """Load a HookedTransformer.

    - `MODEL_ID` environment variable is honored when present.
    - On gated-repo / auth errors, this will attempt `fallback_model` (default: `gpt2`).
    """
    load_dotenv()

    # allow overriding model via env var for easy testing
    env_model = os.getenv("MODEL_ID")
    env_abl = os.getenv("MODEL_ID_ABL")
    env_base = os.getenv("MODEL_ID_BASE")

    if model_id is None and env_model:
        model_id = env_model
    if model_id_abl is None and env_abl:
        model_id_abl = env_abl
    if model_id_base is None and env_base:
        model_id_base = env_base

    print(f"Using model_id: {model_id} (abl: {model_id_abl}, base: {model_id_base})")

    if model_id is None:
        model_id = model_id_abl

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_TOKEN"] = hf_token

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Loading {model_id} onto {device} with dtype {torch_dtype}...")

    # If an abliterated model is provided, try loading its HF weights/tokenizer
    # and inject them into a HookedTransformer wrapper of the base architecture.
    if model_id_abl:
        base_arch = model_id_base or model_id or None
        try:
            print(f"Loading tokenizer and raw HF weights from '{model_id_abl}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_id_abl, use_fast=True, trust_remote_code=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id_abl,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

            if base_arch is None:
                raise RuntimeError("MODEL_ID_BASE or MODEL_ID (base architecture) must be provided when using MODEL_ID_ABL")

            print(f"Wrapping HF weights into HookedTransformer base '{base_arch}'...")
            model = HookedTransformer.from_pretrained(
                base_arch,
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=device,
                default_padding_side="left",
                trust_remote_code=True,
            )
            print(f"Abliterated model loaded into HookedTransformer (d_model={model.cfg.d_model})")
            return model
        except Exception as e:
            print(f"Abliterated model load failed: {e}")
            print("Falling back to normal HookedTransformer loading path...")

    # Standard hooked-transformer loading path
    try:
        model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
            default_padding_side="left",
            trust_remote_code=True,
        )
        print(f"Model loaded. Residual stream dimension: {model.cfg.d_model}")
        return model
    except Exception as e:
        print(f"Failed to load model '{model_id}': {e}")
        if fallback_model and fallback_model != model_id:
            print(f"Attempting fallback model '{fallback_model}' for smoke test...")
            try:
                model = HookedTransformer.from_pretrained(
                    fallback_model,
                    device=device,
                    torch_dtype=torch.float32,
                    default_padding_side="left",
                    trust_remote_code=True,
                )
                print(f"Fallback model loaded: {fallback_model}. Residual stream dimension: {model.cfg.d_model}")
                return model
            except Exception as e2:
                print(f"Fallback model load failed: {e2}")
        raise


if __name__ == "__main__":
    m = load_model()
    print("Model ready.")
