"""Generate model responses for matched prompt pairs and save to CSV.

Saves columns: focus, safe_prompt, safe_response, unsafe_prompt, unsafe_response
"""
import os
import csv
from typing import List, Dict

from model_utils import generate_response


def generate_and_save_responses(model, rows: List[Dict], save_path: str = "data/matched_xstest_with_responses.csv", overwrite: bool = False, max_new_tokens: int = 64):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if os.path.exists(save_path) and not overwrite:
        # don't regenerate if file exists
        return save_path

    fieldnames = ["focus", "safe_prompt", "safe_response", "unsafe_prompt", "unsafe_response"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            focus = r.get("focus", "")
            safe_prompt = r.get("formatted_safe") or r.get("safe_raw") or ""
            unsafe_prompt = r.get("formatted_unsafe") or r.get("unsafe_raw") or ""

            try:
                safe_resp = generate_response(model, safe_prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                safe_resp = f"<<gen_error: {e}>>"

            try:
                unsafe_resp = generate_response(model, unsafe_prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                unsafe_resp = f"<<gen_error: {e}>>"

            writer.writerow({
                "focus": focus,
                "safe_prompt": safe_prompt,
                "safe_response": safe_resp,
                "unsafe_prompt": unsafe_prompt,
                "unsafe_response": unsafe_resp,
            })

    return save_path
