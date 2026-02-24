"""
Train the ML intent classifier on silver labels derived from the heuristic.

For each diff in train.jsonl the label is determined by:
  1. The JSONL 'intent' field (int 0-3) if present  → gold label.
  2. classify_intent_heuristic()                     → silver label.

The trained model is saved to runs/intent_classifier.pt and is automatically
picked up by src/generate.py and server/api.py at inference time.

Usage:
    python -m src.train_intent_classifier
    python -m src.train_intent_classifier --config configs/default.yaml --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent import classify_intent_heuristic
from src.intent_classifier import IntentClassifier
from src.preprocess import normalize_diff
from src.tokenizer import DiffTokenizer


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_tensors(
    data_path: str,
    tokenizer: DiffTokenizer,
    max_diff_tokens: int,
    normalize_literals: bool,
):
    """
    Read JSONL, auto-label each diff, tokenize, and return (X, y) tensors.
    Returns (None, None) when no usable samples are found.
    """
    ids_list: list[list[int]] = []
    labels: list[int] = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            diff = obj.get("diff", "")
            if not diff:
                continue

            # Gold label takes priority; otherwise fall back to heuristic.
            raw_label = obj.get("intent")
            if isinstance(raw_label, int) and 0 <= raw_label < 4:
                label = raw_label
            else:
                label = classify_intent_heuristic(diff)

            norm = normalize_diff(diff, normalize_literals=normalize_literals)
            token_ids = tokenizer.encode(
                norm, add_bos=False, add_eos=False, max_len=max_diff_tokens
            )
            if not token_ids:
                continue

            ids_list.append(token_ids)
            labels.append(label)

    if not ids_list:
        return None, None

    max_len = max(len(x) for x in ids_list)
    X = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        X[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train ML intent classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="Embedding dimension for the intent classifier (default: 64)",
    )
    parser.add_argument(
        "--out",
        default="runs/intent_classifier.pt",
        help="Output checkpoint path (default: runs/intent_classifier.pt)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    cfg = _load_config(str(base_dir / args.config))

    sp_prefix = cfg["tokenizer"].get("model_prefix", "data/sp_model")
    sp_path = base_dir / (sp_prefix + ".model")
    tokenizer = DiffTokenizer(
        model_path=str(sp_path) if sp_path.is_file() else None,
        model_prefix=os.path.basename(sp_prefix),
        data_dir=str(base_dir / "data"),
    )
    tokenizer.load()

    data_path = str(base_dir / cfg["data"]["train_path"])
    max_diff_tokens = cfg["data"].get("max_diff_tokens", 512)
    normalize_literals = cfg["data"].get("normalize_literals", True)

    print(f"Building intent dataset from {data_path} ...")
    X, y = _build_tensors(data_path, tokenizer, max_diff_tokens, normalize_literals)
    if X is None:
        print("No training data found. Check your train.jsonl.")
        return

    label_names = ["fix", "feat", "refactor", "docs"]
    dist = {label_names[i]: int((y == i).sum()) for i in range(4)}
    print(f"  {len(y)} samples  |  label distribution: {dist}")

    loader = DataLoader(
        TensorDataset(X, y), batch_size=args.batch_size, shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size_actual,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} for {args.epochs} epochs ...")
    for ep in range(args.epochs):
        model.train()
        total_loss = correct = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        print(
            f"  Epoch {ep + 1:>2}/{args.epochs}  "
            f"loss={total_loss / n:.4f}  acc={correct / n:.3f}"
        )

    out_path = base_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": tokenizer.vocab_size_actual,
            "embed_dim": args.embed_dim,
        },
        str(out_path),
    )
    print(f"Intent classifier saved → {out_path}")
    print(
        "To use it at inference, pass --intent_model runs/intent_classifier.pt "
        "to src/generate.py, or it will be loaded automatically by server/api.py."
    )


if __name__ == "__main__":
    main()
