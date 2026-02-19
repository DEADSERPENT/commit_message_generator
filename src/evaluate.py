"""
Evaluation: BLEU, ROUGE-L, METEOR.
Usage: python -m src.evaluate --checkpoint runs/best.pt --data data/val.jsonl
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import DiffTokenizer
from src.dataset import CommitDataset, collate_commit
from src.models import Seq2SeqCommit, TransformerCommit
from torch.utils.data import DataLoader


def load_model_and_tokenizer(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    base_dir = Path(__file__).resolve().parent.parent
    sp_path = base_dir / (cfg["tokenizer"].get("model_prefix", "data/sp_model") + ".model")
    tokenizer = DiffTokenizer(
        model_path=str(sp_path) if sp_path.is_file() else None,
        model_prefix=cfg["tokenizer"].get("model_prefix", "data/sp_model"),
        data_dir=str(base_dir / "data"),
    )
    tokenizer.load()
    vocab_size = tokenizer.vocab_size_actual
    m_cfg = cfg["model"]
    if m_cfg["type"] == "seq2seq":
        model = Seq2SeqCommit(
            vocab_size=vocab_size,
            embed_dim=m_cfg.get("embed_dim", 256),
            hidden_dim=m_cfg.get("hidden_dim", 512),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=0.0,
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
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer, cfg


def generate_predictions(
    model,
    tokenizer,
    cfg,
    data_path: str,
    device: torch.device,
    batch_size: int = 16,
    max_len: int = 20,
) -> List[tuple]:
    data_cfg = cfg["data"]
    base_dir = Path(__file__).resolve().parent.parent
    ds = CommitDataset(
        data_path,
        tokenizer,
        max_diff_tokens=data_cfg["max_diff_tokens"],
        max_msg_tokens=data_cfg["max_msg_tokens"],
        normalize_literals=data_cfg.get("normalize_literals", True),
        lowercase_message=data_cfg.get("lowercase_message", True),
        max_msg_words=data_cfg.get("max_msg_words"),
    )
    if len(ds) == 0:
        return []
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_commit,
    )
    refs = []
    hyps = []
    for batch in loader:
        diff_ids = batch["diff_ids"].to(device)
        msg_ids = batch["msg_ids"].to(device)
        diff_mask = batch["diff_mask"].to(device)
        with torch.no_grad():
            gen = model.generate(
                diff_ids,
                diff_mask=diff_mask,
                max_len=max_len,
                eos_id=3,
                temperature=1.0,
            )
        for i in range(diff_ids.size(0)):
            ref = tokenizer.decode(msg_ids[i].tolist(), skip_special=True)
            hyp = tokenizer.decode(gen[i].tolist(), skip_special=True)
            refs.append(ref.strip())
            hyps.append(hyp.strip())
    return list(zip(refs, hyps))


def bleu(refs_hyps: List[tuple]) -> float:
    try:
        import sacrebleu
    except ImportError:
        try:
            from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
            refs = [[r.split()] for r, _ in refs_hyps]
            hyps = [h.split() for _, h in refs_hyps]
            smooth = SmoothingFunction().method1
            return corpus_bleu(refs, hyps, smoothing_function=smooth) * 100
        except Exception:
            return 0.0
    refs = [r for r, _ in refs_hyps]
    hyps = [h for _, h in refs_hyps]
    bleu = sacrebleu.corpus_bleu(refs, hyps)
    return bleu.score


def rouge_l(refs_hyps: List[tuple]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return _rouge_l_fallback(refs_hyps)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, hyp in refs_hyps:
        s = scorer.score(ref, hyp)
        scores.append(s["rougeL"].fmeasure)
    return sum(scores) / len(scores) * 100 if scores else 0.0


def _rouge_l_fallback(refs_hyps: List[tuple]) -> float:
    def lcs(a: List, b: List) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    scores = []
    for ref, hyp in refs_hyps:
        r, h = ref.split(), hyp.split()
        if not r or not h:
            scores.append(0.0)
            continue
        l = lcs(r, h)
        p = l / len(h) if h else 0
        rec = l / len(r) if r else 0
        f = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        scores.append(f)
    return sum(scores) / len(scores) * 100 if scores else 0.0


def meteor(refs_hyps: List[tuple]) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
    except ImportError:
        return 0.0
    scores = []
    for ref, hyp in refs_hyps:
        r_tok = ref.split()
        h_tok = hyp.split()
        if not r_tok:
            scores.append(0.0)
            continue
        m = meteor_score([r_tok], h_tok)
        scores.append(m)
    return sum(scores) / len(scores) * 100 if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/best.pt")
    parser.add_argument("--data", default="data/val.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / args.checkpoint
    if not ckpt_path.is_file():
        ckpt_path = Path(args.checkpoint)
    data_path = base_dir / args.data
    if not data_path.is_file():
        print("Data file not found:", args.data)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_model_and_tokenizer(str(ckpt_path), device)
    refs_hyps = generate_predictions(
        model, tokenizer, cfg, str(data_path), device, batch_size=args.batch_size
    )
    if not refs_hyps:
        print("No samples to evaluate.")
        return
    b = bleu(refs_hyps)
    r = rouge_l(refs_hyps)
    m = meteor(refs_hyps)
    print("BLEU:    {:.2f}".format(b))
    print("ROUGE-L: {:.2f}".format(r))
    print("METEOR:  {:.2f}".format(m))
