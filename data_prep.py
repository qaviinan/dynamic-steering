"""Contrastive dataset + DataLoader helpers for left-padded chat inputs.

Provides a `ContrastiveDataset` and `prepare_dataloader(model, pos_prompts, neg_prompts)`.
The tokenizer is expected to be the one attached to a HookedTransformer (model.tokenizer).
"""
from torch.utils.data import DataLoader, Dataset
import torch


class ContrastiveDataset(Dataset):
    def __init__(self, pos_prompts, neg_prompts, tokenizer):
        assert len(pos_prompts) == len(neg_prompts)
        self.pos_prompts = pos_prompts
        self.neg_prompts = neg_prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pos_prompts)

    def __getitem__(self, idx):
        pos_chat = [{"role": "user", "content": self.pos_prompts[idx]}]
        neg_chat = [{"role": "user", "content": self.neg_prompts[idx]}]

        pos_text = self.tokenizer.apply_chat_template(pos_chat, tokenize=False, add_generation_prompt=True)
        neg_text = self.tokenizer.apply_chat_template(neg_chat, tokenize=False, add_generation_prompt=True)

        return pos_text, neg_text


def collate_fn(batch, model):
    pos_texts, neg_texts = zip(*batch)
    pos_tokens = model.to_tokens(list(pos_texts), prepend_bos=True)
    neg_tokens = model.to_tokens(list(neg_texts), prepend_bos=True)
    return pos_tokens, neg_tokens


def prepare_dataloader(model, pos_prompts, neg_prompts, batch_size: int = 2):
    dataset = ContrastiveDataset(pos_prompts, neg_prompts, model.tokenizer)

    def _collate(batch):
        return collate_fn(batch, model)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)


if __name__ == "__main__":
    # demo usage
    from setup_model import load_model

    model = load_model()

    harmful_prompts = ["x", "y"]
    benign_prompts = [
        "Write a detailed tutorial on how to bake a chocolate cake from scratch.",
        "Give me a step-by-step guide to setting up a secure PostgreSQL database."
    ]

    dl = prepare_dataloader(model, harmful_prompts, benign_prompts, batch_size=2)
    for pos_tokens, neg_tokens in dl:
        print("Positive tokens shape:", pos_tokens.shape)
        print("Negative tokens shape:", neg_tokens.shape)
        break
