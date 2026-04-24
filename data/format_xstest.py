"""XSTest dataset loading, formatting, and saving utilities.

Two modes:
  Full dataset  — load_and_save_full_dataset() / load_full_dataset()
    Preserves every row individually with a `formatted_prompt` column appended.
    Schema: focus, label ("safe"/"unsafe"), prompt, formatted_prompt
    This is the primary format used by run_steering_extraction.py.

  Matched pairs — format_and_save_matched_pairs() / load_formatted_pairs()
    Legacy: pairs each unsafe prompt with a safe prompt of the same focus.
    Kept for backward compatibility with older test scripts.
"""
import os
import csv
from typing import List, Dict, Optional

from datasets import load_dataset, Dataset, DatasetDict


def load_and_match_xstest(dataset_name: str = "walledai/XSTest", split: str | None = None) -> List[Dict]:
    # Load dataset and select an appropriate split. Some datasets (like XSTest)
    # only provide a 'test' split; fall back gracefully when 'train' is not present.
    ds_all = load_dataset(dataset_name)

    if isinstance(ds_all, DatasetDict):
        chosen_split = None
        if split and split in ds_all:
            chosen_split = split
        elif "train" in ds_all:
            chosen_split = "train"
        elif "validation" in ds_all:
            chosen_split = "validation"
        elif "test" in ds_all:
            chosen_split = "test"
        else:
            # fallback to first available
            chosen_split = list(ds_all.keys())[0]

        print(f"Selected split '{chosen_split}' from dataset '{dataset_name}'")
        ds = ds_all[chosen_split]
    else:
        # Single-dataset returned
        ds = ds_all

    # Collect safe prompts by focus and unsafe prompts separately
    safe_by_focus: Dict[str, List[str]] = {}
    unsafe_list: List[Dict] = []

    for item in ds:
        # robust accessors
        focus = item.get("focus") or item.get("Focus")
        prompt = item.get("prompt") or item.get("text")
        label = item.get("label")

        if focus is None or prompt is None or label is None:
            continue

        # normalize label to boolean is_safe
        is_safe = None
        if isinstance(label, str):
            is_safe = label.lower() == "safe"
        else:
            try:
                # assume 0 == safe, 1 == unsafe if numeric
                is_safe = int(label) == 0
            except Exception:
                continue

        if is_safe:
            safe_by_focus.setdefault(focus, []).append(prompt)
        else:
            unsafe_list.append({"focus": focus, "prompt": prompt})

    matched: List[Dict] = []
    for u in unsafe_list:
        candidates = safe_by_focus.get(u["focus"])
        if candidates and len(candidates) > 0:
            matched.append({"focus": u["focus"], "unsafe": u["prompt"], "safe": candidates[0]})

    return matched


def format_and_save_matched_pairs(model, save_path: str = "data/matched_xstest_prompts.csv", overwrite: bool = False, dataset_name: str = "walledai/XSTest", split: str = "train") -> List[Dict]:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if os.path.exists(save_path) and not overwrite:
        # load and return existing rows
        with open(save_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    matched = load_and_match_xstest(dataset_name=dataset_name, split=split)

    tokenizer = getattr(model, "tokenizer", None)
    rows: List[Dict] = []
    for pair in matched:
        unsafe_raw = pair["unsafe"]
        safe_raw = pair["safe"]
        formatted_unsafe = unsafe_raw
        formatted_safe = safe_raw
        try:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                formatted_unsafe = tokenizer.apply_chat_template([{"role": "user", "content": unsafe_raw}], tokenize=False, add_generation_prompt=True)
                formatted_safe = tokenizer.apply_chat_template([{"role": "user", "content": safe_raw}], tokenize=False, add_generation_prompt=True)
        except Exception:
            # fallback to raw strings
            formatted_unsafe = unsafe_raw
            formatted_safe = safe_raw

        rows.append({
            "focus": pair["focus"],
            "unsafe_raw": unsafe_raw,
            "safe_raw": safe_raw,
            "formatted_unsafe": formatted_unsafe,
            "formatted_safe": formatted_safe,
        })

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["focus", "unsafe_raw", "safe_raw", "formatted_unsafe", "formatted_safe"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_formatted_pairs(save_path: str = "data/matched_xstest_prompts.csv", limit: Optional[int] = None) -> List[Dict]:
    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)
    with open(save_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return rows


# ── Full-dataset API ───────────────────────────────────────────────────────────

def load_and_save_full_dataset(
    model=None,
    save_path: str = "data/xstest_full.csv",
    overwrite: bool = False,
    dataset_name: str = "walledai/XSTest",
) -> List[Dict]:
    """Load the complete XSTest dataset and save with a formatted_prompt column.

    Every row from the original dataset is preserved individually — no pairing.
    The `label` column is normalised to the strings "safe" or "unsafe".

    Columns: focus, label, prompt, formatted_prompt
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if os.path.exists(save_path) and not overwrite:
        return load_full_dataset(save_path)

    ds_all = load_dataset(dataset_name)
    if isinstance(ds_all, DatasetDict):
        # XSTest ships with only a test split; fall back to first available
        chosen = "test" if "test" in ds_all else list(ds_all.keys())[0]
        print(f"Selected split '{chosen}' from dataset '{dataset_name}'")
        ds = ds_all[chosen]
    else:
        ds = ds_all

    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    rows: List[Dict] = []

    for item in ds:
        focus = item.get("focus") or item.get("Focus")
        prompt = item.get("prompt") or item.get("text")
        label_raw = item.get("label")

        if focus is None or prompt is None or label_raw is None:
            continue

        if isinstance(label_raw, str):
            label = "safe" if label_raw.lower() == "safe" else "unsafe"
        else:
            try:
                label = "safe" if int(label_raw) == 0 else "unsafe"
            except Exception:
                continue

        formatted_prompt = prompt
        try:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass

        rows.append({
            "focus": focus,
            "label": label,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
        })

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["focus", "label", "prompt", "formatted_prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} prompts ({sum(1 for r in rows if r['label']=='unsafe')} unsafe, "
          f"{sum(1 for r in rows if r['label']=='safe')} safe) to {save_path}")
    return rows


def load_full_dataset(
    save_path: str = "data/xstest_full.csv",
    limit: Optional[int] = None,
) -> List[Dict]:
    """Load the full-dataset CSV produced by load_and_save_full_dataset."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)
    with open(save_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit is not None else rows
