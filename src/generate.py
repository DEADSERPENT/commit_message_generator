"""
Generate commit message from a diff (file or last commit in a repo).
Usage:
  python -m src.generate --diff path/to/diff.txt --checkpoint runs/best.pt
  python -m src.generate --repo path/to/repo --checkpoint runs/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import normalize_diff
from src.tokenizer import DiffTokenizer
from src.models import Seq2SeqCommit, TransformerCommit
from src.intent import classify_intent, intent_to_style_prefix
from src.intent_classifier import load_intent_classifier


def load_config_from_checkpoint(ckpt_path: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config")
    if not cfg:
        raise ValueError("Checkpoint must contain 'config'")
    return ckpt, cfg


def get_model(cfg, vocab_size, device):
    m_cfg = cfg["model"]
    if m_cfg["type"] == "seq2seq":
        model = Seq2SeqCommit(
            vocab_size=vocab_size,
            embed_dim=m_cfg.get("embed_dim", 256),
            hidden_dim=m_cfg.get("hidden_dim", 512),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=0.0,  # inference: no dropout
            pad_id=0,
        )
    else:
        model = TransformerCommit(
            vocab_size=vocab_size,
            d_model=m_cfg.get("d_model", 256),
            nhead=m_cfg.get("nhead", 8),
            num_encoder_layers=m_cfg.get("num_encoder_layers", 4),
            num_decoder_layers=m_cfg.get("num_decoder_layers", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 1024),
            dropout=0.0,  # inference: no dropout
            pad_id=0,
            max_diff_len=cfg["data"].get("max_diff_tokens", 512),
            max_msg_len=cfg["data"].get("max_msg_tokens", 20),
        )
    return model.to(device)


def get_diff_from_repo(repo_path: str, staged_only: bool = False) -> str:
    try:
        import git
    except ImportError:
        raise ImportError("Install GitPython: pip install GitPython")
    repo = git.Repo(repo_path)
    if staged_only:
        return repo.git.diff("--cached") or ""
    if repo.is_dirty(untracked_files=False):
        diff = repo.head.commit.diff(None, create_patch=True)
        out = []
        for d in diff:
            try:
                out.append(d.diff.decode("utf-8") if hasattr(d.diff, "decode") else str(d.diff))
            except Exception:
                out.append(str(d))
        return "\n".join(out) if out else ""
    # Last commit diff
    if len(repo.head.log()) == 0:
        return ""
    commit = repo.head.commit
    if commit.parents:
        return repo.git.diff(commit.parents[0].hexsha, commit.hexsha)
    return repo.git.show(commit.hexsha, format="%B", patch=True).split("\n", 1)[-1] if repo.git.show(commit.hexsha) else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff", type=str, help="Path to file containing raw diff")
    parser.add_argument("--repo", type=str, help="Path to git repo (current or last commit diff)")
    parser.add_argument("--stdin", action="store_true", help="Read diff from stdin (e.g. git diff --staged | ...)")
    parser.add_argument("--staged", action="store_true", help="With --repo: use only staged changes (what you are about to commit)")
    parser.add_argument("--checkpoint", type=str, default="runs/best.pt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument(
        "--beam_size", type=int, default=1,
        help="Beam size for decoding (default: 1 = greedy). "
             "Values > 1 enable beam search for higher-quality output.",
    )
    parser.add_argument("--intent", action="store_true", help="Prepend intent-style prefix (fix/feat/refactor/docs)")
    parser.add_argument(
        "--intent_model",
        type=str,
        default="runs/intent_classifier.pt",
        help="Path to trained ML intent classifier checkpoint (default: runs/intent_classifier.pt). "
             "Falls back to the heuristic if the file does not exist.",
    )
    args = parser.parse_args()

    if args.stdin:
        raw_diff = sys.stdin.read()
        if not raw_diff.strip():
            print("No diff on stdin. Exiting.")
            return
    elif args.diff:
        with open(args.diff, "r", encoding="utf-8") as f:
            raw_diff = f.read()
    elif args.repo:
        raw_diff = get_diff_from_repo(args.repo, staged_only=args.staged)
        if not raw_diff:
            print("No diff (nothing staged / clean tree?). Exiting.")
            return
    else:
        print("Provide --diff <file>, --repo <path>, or --stdin (e.g. git diff --staged | python -m src.generate --stdin)")
        return

    base_dir = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / args.checkpoint
    if not ckpt_path.is_file():
        ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print("Checkpoint not found:", args.checkpoint)
        return

    ckpt, cfg = load_config_from_checkpoint(str(ckpt_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = base_dir / "data"
    sp_prefix = cfg["tokenizer"].get("model_prefix", "data/sp_model")
    sp_path = base_dir / (sp_prefix + ".model")
    tokenizer = DiffTokenizer(
        model_path=str(sp_path) if sp_path.is_file() else None,
        model_prefix=os.path.basename(sp_prefix),
        data_dir=str(data_dir),
    )
    tokenizer.load()
    vocab_size = tokenizer.vocab_size_actual

    model = get_model(cfg, vocab_size, device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    norm_diff = normalize_diff(raw_diff, normalize_literals=True)
    max_diff_tokens = cfg["data"].get("max_diff_tokens", 512)
    full_ids = tokenizer.encode(norm_diff, add_bos=False, add_eos=True)
    if len(full_ids) > max_diff_tokens:
        print(
            f"Warning: diff is {len(full_ids)} tokens but the model accepts at most "
            f"{max_diff_tokens}. Only the first {max_diff_tokens} tokens will be used; "
            f"changes beyond that point are invisible to the model.",
            file=sys.stderr,
        )
    diff_ids = full_ids[:max_diff_tokens]
    diff_tensor = torch.tensor([diff_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        if args.beam_size > 1:
            gen = model.generate_beam(
                diff_tensor,
                beam_size=args.beam_size,
                max_len=args.max_len,
                eos_id=3,
            )
        else:
            gen = model.generate(
                diff_tensor,
                max_len=args.max_len,
                eos_id=3,
                temperature=args.temperature,
            )
    msg = tokenizer.decode(gen[0].tolist(), skip_special=True)
    msg = msg.strip() or "(empty)"
    if args.intent:
        intent_ckpt = base_dir / args.intent_model
        ml_model, ml_device = load_intent_classifier(str(intent_ckpt), device)
        if ml_model is not None:
            # Encode without EOS for classification (matches training)
            clf_ids = tokenizer.encode(
                norm_diff, add_bos=False, add_eos=False,
                max_len=cfg["data"].get("max_diff_tokens", 512),
            )
        else:
            clf_ids = None
        intent_id = classify_intent(raw_diff, diff_ids=clf_ids, ml_model=ml_model, ml_device=ml_device)
        prefix = intent_to_style_prefix(intent_id)
        if not msg.lower().startswith(prefix.rstrip().lower()):
            msg = prefix + msg
    print(msg)


if __name__ == "__main__":
    main()
