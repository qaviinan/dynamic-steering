# LEGACY: This is an early proof-of-concept extraction script from the
# prototype dataloader flow. The current repo path is
# `run_refusal_extraction.py`, which incrementally caches XSTest activations and
# packages vectors for the newer intervention pipeline.
"""Run the model with a cache and extract residual-stream vectors h^{(l)} for the final token.

Saves a small pickle of per-layer final-token vectors for each batch processed.
"""
import os
import re
import pickle
import sys
from dotenv import load_dotenv
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from setup_model import load_model
from legacy.data_prep import prepare_dataloader


def extract_layer_index(key: str) -> int:
    m = re.search(r"(\d+)", key)
    return int(m.group(1)) if m else -1


def find_resid_keys(cache):
    # conservative: look for keys containing 'resid'
    keys = [k for k in cache.keys() if 'resid' in k.lower()]
    # stable sort by numeric token if present
    keys = sorted(keys, key=extract_layer_index)
    return keys


def run_and_extract(model, dataloader, max_batches=1, out_file="h_vectors.pkl"):
    all_h = []
    for i, (pos_tokens, neg_tokens) in enumerate(dataloader):
        device = next(model.parameters()).device
        pos_tokens = pos_tokens.to(device)

        out = model.run_with_cache(pos_tokens)
        cache = getattr(out, 'cache', None)
        if cache is None:
            try:
                _, cache = out
            except Exception:
                raise RuntimeError("Couldn't extract cache from run_with_cache() result")

        resid_keys = find_resid_keys(cache)
        batch_h = {}
        for k in resid_keys:
            t = cache[k]  # expected shape [B, T, d_model]
            if isinstance(t, torch.Tensor):
                batch_h[k] = t[:, -1, :].detach().cpu().numpy()
            else:
                batch_h[k] = torch.tensor(t)[:, -1, :].detach().cpu().numpy()

        all_h.append(batch_h)
        if i + 1 >= max_batches:
            break

    with open(out_file, 'wb') as f:
        pickle.dump(all_h, f)

    print(f"Wrote {len(all_h)} batches of final-token h vectors to {out_file}")


if __name__ == "__main__":
    load_dotenv()
    model = load_model()

    harmful_prompts = ["x", "y"]
    benign_prompts = [
        "Write a detailed tutorial on how to bake a chocolate cake from scratch.",
        "Give me a step-by-step guide to setting up a secure PostgreSQL database."
    ]

    dl = prepare_dataloader(model, harmful_prompts, benign_prompts, batch_size=2)
    run_and_extract(model, dl, max_batches=1, out_file="h_vectors.pkl")
