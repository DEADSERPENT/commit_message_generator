"""
ML intent classifier: EmbeddingBag over BPE diff token IDs → 4-class linear head.

Trains in seconds on CPU using silver labels produced by the heuristic (or gold
labels when the JSONL data already has an 'intent' field).  Once trained, it
replaces classify_intent_heuristic() at inference time.

Model architecture
------------------
  token IDs (B, T)
      │
  EmbeddingBag (mean pooling, ignores PAD)
      │  (B, embed_dim)
  Linear → 4 logits  [fix=0, feat=1, refactor=2, docs=3]

Load / classify helpers are kept here so both generate.py and server/api.py
can import them without touching intent.py's torch-free heuristic path.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class IntentClassifier(nn.Module):
    """Lightweight bag-of-embeddings intent classifier."""

    NUM_CLASSES = 4  # fix=0, feat=1, refactor=2, docs=3

    def __init__(self, vocab_size: int, embed_dim: int = 64, pad_id: int = 0):
        super().__init__()
        self.embed = nn.EmbeddingBag(
            vocab_size, embed_dim, padding_idx=pad_id, mode="mean"
        )
        self.fc = nn.Linear(embed_dim, self.NUM_CLASSES)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) — returns logits (B, 4)."""
        return self.fc(self.embed(token_ids))


def load_intent_classifier(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[IntentClassifier], Optional[torch.device]]:
    """
    Load a trained IntentClassifier from *ckpt_path*.
    Returns (None, None) when the file does not exist so callers can fall back
    gracefully to the heuristic without raising an exception.
    """
    if not os.path.isfile(ckpt_path):
        return None, None
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = IntentClassifier(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt.get("embed_dim", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def classify_intent_ml(
    diff_ids: List[int],
    model: IntentClassifier,
    device: torch.device,
) -> int:
    """
    Classify using the trained model.
    *diff_ids* is a plain list[int] of BPE token IDs (no BOS/EOS needed).
    Returns one of {0=fix, 1=feat, 2=refactor, 3=docs}.
    """
    if not diff_ids:
        return 1  # default: feat
    x = torch.tensor([diff_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    return int(logits.argmax(dim=-1).item())
