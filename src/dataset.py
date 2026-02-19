"""
PyTorch Dataset for (diff, message) pairs.
Loads JSONL with keys: diff, message; optional: intent.
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .preprocess import preprocess_pair
from .tokenizer import DiffTokenizer


class CommitDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: DiffTokenizer,
        max_diff_tokens: int = 512,
        max_msg_tokens: int = 20,
        normalize_literals: bool = True,
        lowercase_message: bool = True,
        max_msg_words: Optional[int] = 15,
        intent_aware: bool = False,
        limit: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_diff_tokens = max_diff_tokens
        self.max_msg_tokens = max_msg_tokens
        self.normalize_literals = normalize_literals
        self.lowercase_message = lowercase_message
        self.max_msg_words = max_msg_words
        self.intent_aware = intent_aware
        self.samples: List[Dict[str, Any]] = []
        self._load(data_path, limit)

    def _load(self, data_path: str, limit: Optional[int]) -> None:
        if not os.path.isfile(data_path):
            return
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    diff = obj.get("diff", "")
                    message = obj.get("message", "")
                    if not diff or not message:
                        continue
                    intent = obj.get("intent") if self.intent_aware else None
                    self.samples.append(
                        {"diff": diff, "message": message, "intent": intent}
                    )
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw = self.samples[idx]
        diff, msg = preprocess_pair(
            raw["diff"],
            raw["message"],
            normalize_literals=self.normalize_literals,
            lowercase_message=self.lowercase_message,
            max_msg_words=self.max_msg_words,
        )
        diff_ids = self.tokenizer.encode(
            diff,
            add_bos=False,
            add_eos=True,
            max_len=self.max_diff_tokens,
        )
        msg_ids = self.tokenizer.encode(
            msg,
            add_bos=True,
            add_eos=True,
            max_len=self.max_msg_tokens,
        )
        diff_tensor = torch.tensor(diff_ids, dtype=torch.long)
        msg_tensor = torch.tensor(msg_ids, dtype=torch.long)
        out = {"diff_ids": diff_tensor, "msg_ids": msg_tensor, "diff_len": len(diff_ids)}
        if self.intent_aware and raw.get("intent") is not None:
            out["intent"] = torch.tensor(raw["intent"], dtype=torch.long)
        return out


def collate_commit(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Pad sequences to max length in batch."""
    diff_lens = [b["diff_len"] for b in batch]
    max_diff = max(diff_lens)
    max_msg = max(b["msg_ids"].size(0) for b in batch)
    diff_ids = torch.full(
        (len(batch), max_diff),
        pad_id,
        dtype=torch.long,
    )
    msg_ids = torch.full(
        (len(batch), max_msg),
        pad_id,
        dtype=torch.long,
    )
    for i, b in enumerate(batch):
        d = b["diff_ids"]
        m = b["msg_ids"]
        diff_ids[i, : d.size(0)] = d
        msg_ids[i, : m.size(0)] = m
    diff_mask = (diff_ids != pad_id).long()
    msg_mask = (msg_ids != pad_id).long()
    out = {
        "diff_ids": diff_ids,
        "msg_ids": msg_ids,
        "diff_mask": diff_mask,
        "msg_mask": msg_mask,
        "diff_lens": torch.tensor(diff_lens, dtype=torch.long),
    }
    if "intent" in batch[0]:
        out["intent"] = torch.stack([b["intent"] for b in batch])
    return out
