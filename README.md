# Git Commit Message Generator

Generate short, descriptive commit messages from your code diff. Use it **every day**: stage your changes, run the tool, copy the suggested message into `git commit -m "..."`.

## Use in daily work

**One-time setup** (from this project folder):

```bash
pip install -r requirements.txt
# Optional: train on real data for better messages (see "Dataset" below)
python -m src.prepare_data
python -m src.train --config configs/default.yaml
```

**Then, from any Git repo** when you want a commit message:

- **Staged changes only** (what you’re about to commit):
  ```powershell
  # Windows (from the repo you're working in)
  D:\hub\commit_message_generator\scripts\suggest.ps1
  ```
  Or pipe a diff (from the commit_message_generator folder, or with `PYTHONPATH` set to it):
  ```powershell
  cd D:\hub\commit_message_generator
  $env:PYTHONPATH = ".;.\packages"
  git -C C:\path\to\your\repo diff --staged | python -m src.generate --stdin --checkpoint runs/best.pt --intent
  ```

- **From a specific repo path:**
  ```bash
  python -m src.generate --repo C:\path\to\your\repo --staged --checkpoint runs/best.pt --intent
  ```
  `--intent` adds a prefix like `fix:` or `feat:` when it fits.

**Suggested workflow:** Stage files → run the suggest script or command → use the printed line as your commit message (edit if you want).

**Optional – alias** so you can type one command from any repo. In PowerShell profile (`$PROFILE`):
```powershell
function suggest-commit { & "D:\hub\commit_message_generator\scripts\suggest.ps1" (Get-Location) }
```
Then from any repo: `suggest-commit` prints a suggested message for your staged changes.

---

## Problem

Given a **Git diff** (added `+` and removed `-` lines), the system generates a **human-readable commit message** that explains *what changed* and *why* (abstractive, not extractive).

## System Architecture

```
Git Repo → Diff Extractor → Code Preprocessor → Tokenizer → Seq2Seq/Transformer → Commit Message
```

## Features

- **Diff preprocessing**: Clean headers, normalize code (literals → placeholders), trim messages
- **Models**: Encoder–Decoder with attention (baseline) and Transformer-based generator
- **Intent-aware** (optional): Classify diff as bugfix/feature/refactor/docs; use `--intent` when generating to prepend style prefix
- **Evaluation**: BLEU, ROUGE-L, METEOR

## Setup

```bash
cd commit_message_generator
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Quick Start (real data: CommitBench)

```bash
# 1) Download CommitBench and convert to JSONL (optional: --max_train 50000 --max_val 5000)
pip install datasets
python scripts/download_commitbench.py --max_train 50000 --max_val 5000

# 2) Train tokenizer on train.jsonl (writes data/sp_model.model)
python -m src.prepare_data

# 3) Train model
python -m src.train --config configs/default.yaml
```

Then generate or evaluate:

```bash
python -m src.generate --diff path/to/diff.txt --checkpoint runs/best.pt
python -m src.evaluate --checkpoint runs/best.pt --data data/val.jsonl
```

- **No validation set?** Training still saves `runs/best.pt` (last epoch is copied so generate/evaluate work).

**After training** (compare Seq2Seq vs Transformer, document): see [AFTER_TRAINING.md](AFTER_TRAINING.md). Full design and NLP details: [DOCUMENTATION.md](DOCUMENTATION.md). Architecture and viva notes: [ARCHITECTURE.md](ARCHITECTURE.md).

## Dataset

We use **CommitBench** (Hugging Face: [Maxscha/commitbench](https://huggingface.co/datasets/Maxscha/commitbench)): 1.66M (diff, commit message) pairs from GitHub across six languages (Java, Python, Go, JavaScript, PHP, Ruby), with train/validation/test splits.

- **Download and convert to JSONL** (run once; optionally cap size for faster runs):
  ```bash
  pip install datasets
  python scripts/download_commitbench.py --max_train 50000 --max_val 5000 --max_test 5000
  ```
  Writes `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`. Each line is one JSON object: `{"diff": "...", "message": "..."}`.

- **Source**: CommitBench (CC BY-NC 4.0). Repository selection based on CodeSearchNet; quality-focused filtering (e.g. no bot commits). See `data/README.md` for format details.

- **Alternative**: Mine your own repo: `python scripts/extract_commits.py --repo /path/to/repo --output data/commits.jsonl --max 5000`, then split into train/val manually.

## Project Structure

```
commit_message_generator/
├── configs/           # YAML configs (model type, dropout_seq2seq / dropout_transformer, warmup_steps)
├── src/
│   ├── preprocess.py  # Diff + message normalization
│   ├── dataset.py    # PyTorch Dataset
│   ├── tokenizer.py  # SentencePiece wrapper
│   ├── models/       # Seq2Seq, Transformer
│   ├── train.py      # Training (warmup + cosine LR, best.pt even without val)
│   ├── generate.py   # Inference (--diff, --repo, --stdin, --staged)
│   ├── evaluate.py   # BLEU, ROUGE-L, METEOR
│   └── intent.py     # Intent heuristic (fix/feat/refactor/docs)
├── scripts/          # suggest.ps1, suggest.sh, download_commitbench.py, extract_commits.py
├── data/              # Train/val/test JSONL, tokenizer; see data/README.md
├── runs/              # Checkpoints (best.pt, epoch_*.pt)
└── requirements.txt
```

## Evaluation Metrics

- **BLEU**: n-gram overlap with reference
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Semantic alignment (synonyms, stemming)

## License

MIT (academic use). CommitBench data: CC BY-NC 4.0.