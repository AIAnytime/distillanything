"""Tokenization for SFT/KD: prompt tokens are masked (-100), response tokens supervised."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from distillanything.data.formats import render_pair

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """Pre-tokenizes records into {input_ids, labels} tensors."""

    def __init__(self, records: list[dict], tokenizer, max_seq_len: int = 512):
        self.examples: list[dict] = []
        eos = tokenizer.eos_token or ""
        for record in records:
            prompt_text, response_text = render_pair(record, tokenizer)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
            response_ids = tokenizer(response_text + eos, add_special_tokens=False)["input_ids"]

            input_ids = (prompt_ids + response_ids)[:max_seq_len]
            labels = ([IGNORE_INDEX] * len(prompt_ids) + list(response_ids))[:max_seq_len]

            # Need at least one supervised position after the shift.
            if sum(1 for label in labels[1:] if label != IGNORE_INDEX) == 0:
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def pad_collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        ids, labs = item["input_ids"], item["labels"]
        pad = max_len - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([labs, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        attention_mask.append(
            torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
