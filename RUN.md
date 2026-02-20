# RUN.md — Complete Execution Guide

Everything you need to go from a fresh clone to a working commit-message generator, step by step.

---

## Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.9+ | Training & inference server |
| Go | 1.21+ | Native developer CLI (optional) |
| Git | any | Diff extraction |
| Docker | any | Server deployment (optional) |

---

## Part 1 — One-Time Setup (ML / DevOps Engineer)

### 1. Create the virtual environment

```bash
python -m venv .venv

# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate — bash / Linux / macOS
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
# PyTorch CPU build (works everywhere, no CUDA required)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install -r requirements.txt

# FastAPI server (only needed when deploying the API)
pip install -r server/requirements.txt
```

### 3. Download NLTK data (one-time)

```bash
python -c "
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

---

## Part 2 — Training the Model

### Step 1 — Get training data

**Option A: CommitBench (recommended — 1.66M real commits)**

```bash
# Full training set — takes ~30 min to download
python scripts/download_commitbench.py --max_train 50000 --max_val 5000

# Quick demo (10k train, 1k val — what this guide uses)
python scripts/download_commitbench.py --max_train 10000 --max_val 1000
```

**Option B: Mine your own repository**

```bash
python scripts/extract_commits.py --repo /path/to/your/repo --output data/train.jsonl --max 5000
```

Both options produce `data/train.jsonl` and `data/val.jsonl` with lines like:
```json
{"diff": "- old line\n+ new line", "message": "fix: handle null input"}
```

### Step 2 — Train the SentencePiece tokenizer

```bash
python -m src.prepare_data
```

Output: `data/sp_model.model` and `data/sp_model.vocab`

### Step 3 — Train the model

**Quick demo (5 min on CPU, good for testing the pipeline):**

```bash
python -m src.train --config configs/demo.yaml --train_limit 10000 --val_limit 500
```

**Full training (production quality):**

```bash
python -m src.train --config configs/default.yaml
```

**What the config files control:**

| Setting | `demo.yaml` | `default.yaml` | Effect |
|---------|-------------|----------------|--------|
| `model.d_model` | 128 | 256 | Model width |
| `model.num_encoder_layers` | 2 | 4 | Depth |
| `training.epochs` | 10 | 20 | Training rounds |
| `data.max_diff_tokens` | 256 | 512 | Max diff length |

Training output (watch for decreasing loss):
```
Epoch 1/10  train_loss=8.45  lr=2.82e-04
  val_loss=7.53
Epoch 2/10  train_loss=6.97  lr=2.60e-04
  val_loss=6.79
...
Done. Best checkpoint saved to runs/best.pt
```

### Step 4 — Evaluate (optional)

```bash
python -m src.evaluate --checkpoint runs/best.pt --data data/val.jsonl
```

Outputs BLEU, ROUGE-L, and METEOR scores.

---

## Part 3 — Running the Inference API Server

The API server wraps the trained model as a REST endpoint.
Developers call it from their machines — **no Python on their end**.

### Option A: Run locally (development)

```bash
# From project root, with venv active
uvicorn server.api:app --host 0.0.0.0 --port 8000

# With a specific checkpoint
CHECKPOINT=runs/best.pt uvicorn server.api:app --host 0.0.0.0 --port 8000
```

### Option B: Docker (production)

```bash
# Build the image
docker build -t commit-suggest-api .

# Run — mount your checkpoint and tokenizer
docker run -p 8000:8000 \
  -v $(pwd)/runs:/app/runs:ro \
  -v $(pwd)/data:/app/data:ro \
  commit-suggest-api
```

### API endpoints

```
GET  /health     → {"status": "ok", "model_loaded": true}

POST /generate
  Body:  {"diff": "<raw git diff>", "intent": true, "temperature": 0.8}
  Reply: {"message": "feat: add rate limiting middleware"}
```

**Quick test:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"diff": "+ def hash_password(pw):\n+     return hashlib.sha256(pw.encode()).hexdigest()", "intent": true}'
# → {"message": "feat: add password hashing utility"}
```

### Securing the API (optional)

```bash
# Start server with a bearer token
API_KEY=my-secret-token uvicorn server.api:app --host 0.0.0.0 --port 8000

# Clients must pass the token
curl -H "Authorization: Bearer my-secret-token" ...
```

---

## Part 4 — Developer Daily Workflow (Native Client)

Developers **do not need Python**. They use the native `commit-suggest` binary.

### Build the Go binary

```bash
cd client
go build -o commit-suggest .         # Linux / macOS
go build -o commit-suggest.exe .     # Windows
```

Distribute this single file to your team. No install, no runtime, no dependencies.

### First-time setup (per developer)

```bash
# Point the binary at your company's server and save to ~/.commit-suggest.json
commit-suggest --api http://commit-api.internal.company.com --init
```

Config file (`~/.commit-suggest.json`):
```json
{
  "api_url": "http://commit-api.internal.company.com",
  "api_key": "",
  "intent": true,
  "temperature": 0.8
}
```

Override with env vars: `COMMIT_SUGGEST_API_URL`, `COMMIT_SUGGEST_API_KEY`

### Daily usage

```bash
# 1. Do your work and stage files
git add src/auth.go tests/auth_test.go

# 2. Get a commit message suggestion
commit-suggest
# → feat: add JWT token validation to auth handler

# 3. Commit with the suggestion
git commit -m "feat: add JWT token validation to auth handler"
```

**All CLI options:**

```
commit-suggest                       Stage diff → API → print message
commit-suggest --diff patch.txt      Use a specific diff file
git diff --staged | commit-suggest --stdin   Pipe mode
commit-suggest --no-intent           Skip the conventional prefix
commit-suggest --temp 1.2            Higher temperature = more creative
commit-suggest --init                Save current flags to config
commit-suggest --version             Print version
```

### Shell alias (optional convenience)

**Linux / macOS** (`~/.bashrc` or `~/.zshrc`):
```bash
alias suggest-commit='/path/to/commit-suggest'
```

**Windows PowerShell** (`$PROFILE`):
```powershell
function suggest-commit { & "C:\tools\commit-suggest.exe" @args }
```

---

## Part 5 — Generate Without the API (Python CLI)

If you have the venv active and don't want to run the server:

```bash
# From staged diff in the current repo
python -m src.generate --repo . --staged --checkpoint runs/best.pt --intent

# From a diff file
python -m src.generate --diff data/sample.diff --checkpoint runs/best.pt --intent

# Pipe from git
git diff --staged | python -m src.generate --stdin --checkpoint runs/best.pt --intent
```

---

## Full Pipeline at a Glance

```
[Data]                    [Train]                  [Serve]             [Use]
CommitBench dataset  →  prepare_data.py       →  server/api.py   →  commit-suggest
  10k–50k JSONL           (SentencePiece)         (FastAPI)          (Go binary)
                        src/train.py                Docker             No Python
                           runs/best.pt        Port 8000
```

---

## File Structure

```
commit_message_generator/
├── configs/
│   ├── default.yaml      Full-scale training config
│   └── demo.yaml         Fast CPU demo config (d_model=128, 10 epochs)
├── data/
│   ├── train.jsonl       Training data (generated by download script)
│   ├── val.jsonl         Validation data
│   ├── sp_model.model    Trained SentencePiece tokenizer
│   └── sample.diff       Bundled example diff
├── runs/
│   ├── best.pt           Best checkpoint (by val loss)
│   └── epoch_N.pt        Per-epoch checkpoints
├── src/
│   ├── prepare_data.py   Build corpus + train tokenizer
│   ├── train.py          Training loop
│   ├── generate.py       CLI inference
│   ├── evaluate.py       BLEU / ROUGE-L / METEOR
│   ├── preprocess.py     Diff normalisation
│   ├── tokenizer.py      SentencePiece wrapper
│   ├── intent.py         Conventional-commit prefix classifier
│   └── models/
│       ├── seq2seq.py           BiLSTM + Bahdanau attention (baseline)
│       └── transformer_model.py Transformer encoder-decoder (default)
├── server/
│   ├── api.py            FastAPI inference server
│   └── requirements.txt  fastapi + uvicorn
├── client/
│   ├── main.go           Native Go CLI client
│   └── go.mod            Go module (stdlib only, zero deps)
├── scripts/
│   ├── download_commitbench.py  Download HuggingFace dataset
│   ├── extract_commits.py       Mine commits from a local repo
│   ├── suggest.sh               Legacy bash wrapper (Python)
│   └── suggest.ps1              Legacy PowerShell wrapper (Python)
├── Dockerfile            Server container (project root)
├── requirements.txt      Python dependencies
└── RUN.md                This file
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Checkpoint not found` | Run `src.train` first; check `runs/best.pt` exists |
| `Tokenizer not found` | Run `python -m src.prepare_data` first |
| `No diff on stdin / nothing staged` | Run `git add <files>` before calling the tool |
| `Connection refused` at API | Start `uvicorn server.api:app --port 8000` |
| `Model not loaded` (API 503) | Check server logs; verify `CHECKPOINT` env var |
| PyTorch nested tensor warnings | Harmless — suppress with `python -W ignore` |
| Go binary: `git not found` | Install Git and add it to PATH |
