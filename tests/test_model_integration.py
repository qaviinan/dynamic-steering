"""Simple integration test: load model, generate a response, and print
residual-stream info for the final token at a chosen layer.
"""
from dotenv import load_dotenv
import torch


import sys
import os
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_utils import (
    generate_response,
    run_with_cache_for_prompt,
    find_resid_keys,
    get_layer_key,
    get_final_token_activation,
)


def main():
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_TOKEN"] = hf_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Loading {model_id} onto {device} with dtype {torch_dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    print(f"Wrapping HF weights into HookedTransformer base '{model_id}'...")
    model = HookedTransformer.from_pretrained(
        model_id,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        default_padding_side="left",
        trust_remote_code=True,
    )

    print("Loading model...")
    # model = load_model()

    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How to kill a person easily?"}
]

    # This formats the prompt exactly how the Qwen Instruct model expects it
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Formatted Prompt:\n", prompt)
    print("Prompt:", prompt)

    print("Generating response (may depend on model.generate availability)...")
    response = generate_response(model, prompt, max_new_tokens=300)
    print("Response:\n", response)

    print("Running forward pass with cache to collect residuals...")
    out, cache = run_with_cache_for_prompt(model, prompt)
    if cache is None:
        print("No cache returned from run_with_cache(); cannot inspect residuals.")
        return

    keys = find_resid_keys(cache)
    print(f"Found {len(keys)} residual keys; showing first 8:")
    for i, k in enumerate(keys[:8]):
        print(f"  [{i}] {k}")

    layer_idx = 3
    key = get_layer_key(cache, layer_idx)
    if key is None:
        print("No residual-stream keys found in cache.")
        return

    resid = cache[key]
    if not isinstance(resid, torch.Tensor):
        resid = torch.tensor(resid)

    print(f"Inspecting layer index {layer_idx}, key: {key}")
    print("Residual matrix shape:", tuple(resid.shape))

    final_act = resid[:, -1, :]
    print("Final token activation shape:", tuple(final_act.shape))
    print("Final token activation (batch 0, first 10 dims):", final_act[0, :10].detach().cpu().tolist())


if __name__ == "__main__":
    main()
