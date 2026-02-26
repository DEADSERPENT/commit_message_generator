"""
Training script for commit message generator.
Usage: python -m src.train --config configs/default.yaml
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import CommitDataset, collate_commit
from src.tokenizer import DiffTokenizer
from src.models import Seq2SeqCommit, TransformerCommit


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model(cfg: dict, vocab_size: int, pad_id: int = 0):
    m_cfg = cfg["model"]
    if m_cfg["type"] == "seq2seq":
        return Seq2SeqCommit(
            vocab_size=vocab_size,
            embed_dim=m_cfg.get("embed_dim", 256),
            hidden_dim=m_cfg.get("hidden_dim", 512),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=m_cfg.get("dropout_seq2seq", 0.3),
            pad_id=pad_id,
        )
    return TransformerCommit(
        vocab_size=vocab_size,
        d_model=m_cfg.get("d_model", 256),
        nhead=m_cfg.get("nhead", 8),
        num_encoder_layers=m_cfg.get("num_encoder_layers", 4),
        num_decoder_layers=m_cfg.get("num_decoder_layers", 4),
        dim_feedforward=m_cfg.get("dim_feedforward", 1024),
        dropout=m_cfg.get("dropout_transformer", 0.1),
        pad_id=pad_id,
        max_diff_len=cfg["data"].get("max_diff_tokens", 512),
        max_msg_len=cfg["data"].get("max_msg_tokens", 20),
    )



@torch.no_grad()
def eval_epoch(model, loader, criterion, device, pad_id=0):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        diff_ids = batch["diff_ids"].to(device)
        msg_ids = batch["msg_ids"].to(device)
        diff_mask = batch["diff_mask"].to(device)
        msg_mask = batch["msg_mask"].to(device)
        logits = model(
            diff_ids=diff_ids,
            msg_ids=msg_ids,
            diff_mask=diff_mask,
            msg_mask=msg_mask,
        )
        target = msg_ids[:, 1:]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
        ignore = (target == pad_id)
        loss = loss.masked_fill(ignore.reshape(-1), 0).sum() / (
            (~ignore).sum().clamp(min=1).float()
        )
        total_loss += loss.item()
        n += 1
    return total_loss / n if n else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=500)
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    seed = cfg["project"].get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    tok_cfg = cfg["tokenizer"]
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / data_cfg["train_path"]
    val_path = base_dir / data_cfg["val_path"]
    sp_path = base_dir / tok_cfg["model_prefix"]
    sp_path = str(sp_path) + ".model"

    tokenizer = DiffTokenizer(
        model_path=sp_path if os.path.isfile(sp_path) else None,
        model_prefix=tok_cfg["model_prefix"],
        vocab_size=tok_cfg["vocab_size"],
        data_dir=str(base_dir / "data"),
    )
    if not tokenizer.model_path or not os.path.isfile(tokenizer.model_path):
        print("Tokenizer not found. Run prepare_data.py first to train tokenizer and create sample data.")
        return
    tokenizer.load()
    print("Tokenizer loaded.", flush=True)

    train_ds = CommitDataset(
        str(train_path),
        tokenizer,
        max_diff_tokens=data_cfg["max_diff_tokens"],
        max_msg_tokens=data_cfg["max_msg_tokens"],
        normalize_literals=data_cfg.get("normalize_literals", True),
        lowercase_message=data_cfg.get("lowercase_message", True),
        max_msg_words=data_cfg.get("max_msg_words"),
        intent_aware=cfg.get("intent_aware", False),
        limit=args.train_limit,
    )
    print(f"Dataset loaded: {len(train_ds)} train samples.", flush=True)
    if len(train_ds) == 0:
        print("No training data. Add data/train.jsonl or run prepare_data.py")
        return
    val_ds = CommitDataset(
        str(val_path),
        tokenizer,
        max_diff_tokens=data_cfg["max_diff_tokens"],
        max_msg_tokens=data_cfg["max_msg_tokens"],
        normalize_literals=data_cfg.get("normalize_literals", True),
        lowercase_message=data_cfg.get("lowercase_message", True),
        max_msg_words=data_cfg.get("max_msg_words"),
        intent_aware=cfg.get("intent_aware", False),
        limit=args.val_limit,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_commit,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_commit,
        num_workers=0,
    ) if len(val_ds) > 0 else None

    vocab_size = tokenizer.vocab_size_actual
    model = get_model(cfg, vocab_size, pad_id=0)
    model = model.to(device)
    print(f"Model: {cfg['model']['type']} | Device: {device} | Vocab: {vocab_size}", flush=True)
    criterion = nn.CrossEntropyLoss(reduction="none")
    lr = cfg["training"]["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    # Linear warmup + cosine decay scheduler
    epochs = cfg["training"]["epochs"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    total_steps = epochs * len(train_loader)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    clip = cfg["training"].get("gradient_clip", 1.0)
    best_val = float("inf")
    ckpt_dir = base_dir / cfg["output"]["checkpoint_dir"]

    print(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch, {total_steps} total steps", flush=True)
    print("(CUDA warm-up on first batch may take ~30s — this is normal)", flush=True)
    global_step = 0
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            diff_ids = batch["diff_ids"].to(device)
            msg_ids = batch["msg_ids"].to(device)
            diff_mask = batch["diff_mask"].to(device)
            msg_mask = batch["msg_mask"].to(device)
            logits = model(
                diff_ids=diff_ids,
                msg_ids=msg_ids,
                diff_mask=diff_mask,
                msg_mask=msg_mask,
            )
            target = msg_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            pad_mask = (target == model.pad_id)
            loss = loss.masked_fill(pad_mask.reshape(-1), 0).sum() / (
                (~pad_mask).sum().clamp(min=1).float()
            )
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n += 1
            global_step += 1

            if n % 100 == 0:
                print(
                    f"  Epoch {ep+1}/{epochs}  step {n}/{len(train_loader)}"
                    f"  loss={total_loss/n:.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        train_loss = total_loss / n if n else 0.0
        print(f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loader:
            val_loss = eval_epoch(model, val_loader, criterion, device)
            print(f"  val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": cfg, "epoch": ep},
                    ckpt_dir / "best.pt",
                )

        if (ep + 1) % cfg["training"].get("save_every", 1) == 0:
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "epoch": ep},
                ckpt_dir / f"epoch_{ep+1}.pt",
            )

    # If no validation data, promote the last epoch checkpoint to best.pt
    if not val_loader:
        last_ckpt = ckpt_dir / f"epoch_{epochs}.pt"
        if last_ckpt.is_file():
            import shutil
            shutil.copy(str(last_ckpt), str(ckpt_dir / "best.pt"))

    print("Done. Best checkpoint saved to", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
