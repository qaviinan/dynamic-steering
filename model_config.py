"""Central model configuration.

All scripts import MODEL_ID and load_instruct_model() from here so the
model can be switched globally via a single env var (MODEL_ID_BASE) or by
editing this file's default.

Default: Qwen/Qwen2.5-0.5B-Instruct (fast, CPU-friendly for development).
Swap to a larger model (e.g. meta-llama/Meta-Llama-3-8B-Instruct) by
setting MODEL_ID_BASE in your .env file.
"""
import os

import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID: str = os.getenv("MODEL_ID_BASE", "Qwen/Qwen2.5-0.5B-Instruct")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    d = device or get_device()
    return torch.bfloat16 if d.startswith("cuda") else torch.float32


def load_instruct_model(model_id: str | None = None) -> HookedTransformer:
    """Load the instruct model directly into HookedTransformer.

    Loads HF weights first then wraps them, bypassing setup_model.py so that
    MODEL_ID_ABL cannot accidentally redirect to abliterated weights.

    Args:
        model_id: Optional override; falls back to module-level MODEL_ID
                  (which reads MODEL_ID_BASE from the environment).

    Returns:
        HookedTransformer ready for run_with_cache and forward hooks.
    """
    mid = model_id or MODEL_ID
    device = get_device()
    dtype = get_dtype(device)

    print(f"Loading {mid!r} onto {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=dtype, trust_remote_code=True
    )
    model = HookedTransformer.from_pretrained(
        mid,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        default_padding_side="left",
        trust_remote_code=True,
    )
    del hf_model
    print(f"Model ready  n_layers={model.cfg.n_layers}  d_model={model.cfg.d_model}")
    return model
