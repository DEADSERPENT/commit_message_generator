# Project documentation: From idea to daily use

This document covers the **project idea**, **design**, **NLP foundations**, **implementation**, **industry alignment**, **novelty**, and **day-to-day use** as a developer.

---

## 1. Idea and motivation

**Core idea:** Automatically suggest a short, readable Git commit message from the code diff (the lines you added and removed). You stage your changes, run the tool, and get a one-line message you can use or edit.

**Why it matters in practice:**
- Many developers write vague messages (“fix”, “update”, “WIP”) or skip them under time pressure.
- Good messages make code review, blame, and history search much easier.
- Tools that generate messages from diffs fit directly into existing Git workflows (no new UI; just a suggested line of text).

**What we build:** A small pipeline: **raw diff → preprocessing → tokenization → neural model → commit message**. The model is **abstractive** (generates new text), not copy-paste from the diff.

---

## 2. Problem definition

**Input:** A Git diff: added lines (prefixed with `+`) and removed lines (prefixed with `-`), plus optional headers (file names, hunk markers).

**Output:** A single-line commit message in natural language that describes what changed.

**Constraints:**
- The message should be concise (e.g. under 15 words in training).
- It should reflect the *content* of the change (semantics), not just keywords.
- The system should work with **code** (mixed languages, symbols, structure), not just plain English.

This is a **sequence-to-sequence (seq2seq)** problem: one sequence (diff) in, one sequence (message) out, with **abstractive text generation** — classic NLP.

---

## 3. NLP in this project

The project is built around standard **NLP and neural generation** concepts. Below is how each appears in the codebase.

### 3.1 Text as sequences

- **Diff** and **message** are treated as **token sequences**.
- We limit length (e.g. 512 tokens for diff, 20 for message) and use **padding** for batching.
- **Special tokens:** PAD (0), UNK (1), BOS (2), EOS (3) for training and decoding.

**Where:** `src/tokenizer.py` (encode/decode with BOS/EOS), `src/dataset.py` (length limits, padding), `src/models/*` (embedding and generation with EOS).

### 3.2 Subword tokenization (BPE)

- **SentencePiece (BPE)** is used so we can handle:
  - Mixed **code and natural language** (identifiers, symbols, English words).
  - **Out-of-vocabulary** tokens via subwords.
- Placeholders like `<STRING>`, `<NUM>` are kept as **user-defined symbols** so the tokenizer doesn’t split them.

**Where:** `src/tokenizer.py` (SentencePiece train/load, `user_defined_symbols` in `prepare_data.py`), `src/preprocess.py` (literal → placeholder before tokenization).

### 3.3 Vocabulary and normalization

- **Vocabulary size** is controlled (e.g. 8k) to keep the model tractable.
- **Code normalization** (string/number/path → placeholders) reduces vocabulary explosion and helps the model generalize across different literals.

**Where:** `src/preprocess.py` (`normalize_code_line`, `STRING_PLACEHOLDER`, `NUM_PLACEHOLDER`, `PATH_PLACEHOLDER`), `configs/default.yaml` (`tokenizer.vocab_size`).

### 3.4 Encoder–decoder and attention

**Baseline (Seq2Seq):**
- **Encoder:** Bidirectional LSTM over the diff token sequence → a sequence of hidden states.
- **Decoder:** LSTM that generates the message token-by-token.
- **Bahdanau (additive) attention:** Decoder attends over encoder outputs at each step; the context vector is concatenated with the decoder input.
- This is the standard **attention-based seq2seq** from NLP.

**Where:** `src/models/seq2seq.py` (`Attention` class, `Seq2SeqCommit` encode/decode/generate).

**Transformer:**
- **Encoder:** Stack of Transformer encoder layers (self-attention + FFN) over the diff.
- **Decoder:** Stack of Transformer decoder layers (masked self-attention over the message so far, cross-attention to encoder, FFN).
- **Positional encoding:** Sinusoidal so the model knows token order.
- **Causal mask** in the decoder so each position only sees past tokens (autoregressive generation).

**Where:** `src/models/transformer_model.py` (`PositionalEncoding`, `TransformerCommit` encode/decode/generate, `_make_tgt_mask` for causality).

### 3.5 Training objective

- **Teacher forcing:** During training, the decoder is fed the **gold** previous token (not the model’s own prediction) at each step.
- **Loss:** Cross-entropy on the **next token**; padding positions are masked out so they don’t contribute to the loss.

**Where:** `src/train.py` (forward pass with `msg_ids`, target `msg_ids[:, 1:]`, mask for `pad_id`).

### 3.6 Inference (decoding)

- **Autoregressive decoding:** Start with BOS; at each step feed the current sequence to the decoder and take argmax (or sample) for the next token until EOS or max length.
- **Temperature** is used to control randomness (e.g. 0.8 in generation).

**Where:** `src/models/seq2seq.py` (`generate`), `src/models/transformer_model.py` (`generate`), `src/generate.py` (calls `model.generate` with `temperature`).

### 3.7 Evaluation metrics (NLP standard)

- **BLEU:** N-gram overlap with reference (sacrebleu or NLTK).
- **ROUGE-L:** Longest common subsequence overlap.
- **METEOR:** Alignments with synonyms/stemming.

**Where:** `src/evaluate.py` (references and hypotheses, then `bleu`, `rouge_l`, `meteor`).

So: **seq2seq, attention, subword tokenization, teacher forcing, autoregressive generation, and standard NLG metrics** are all used in the implementation — the project is clearly **NLP-based**.

---

## 4. System architecture and data flow

```
[Developer stages changes]
         │
         ▼
   Git diff (raw)  ──►  Diff file / stdin / --repo (staged or last commit)
         │
         ▼
   Preprocessing   ──►  clean_diff(), normalize_code_line()  (src/preprocess.py)
         │               • Drop headers, keep +/- lines
         │               • Replace literals with <STRING>, <NUM>, <PATH>
         ▼
   Tokenization    ──►  SentencePiece encode (src/tokenizer.py)
         │               • Same vocab for diff and message; BOS/EOS for message
         ▼
   Model           ──►  Seq2SeqCommit or TransformerCommit (src/models/)
         │               • Encode diff → context
         │               • Decode → message token ids
         ▼
   Decode + post   ──►  tokenizer.decode(); optional intent prefix (src/intent.py)
         │
         ▼
   Commit message (one line)
```

**Training pipeline:** JSONL (diff, message) → `CommitDataset` (preprocess + tokenize) → DataLoader → train loop with **linear warmup + cosine decay** LR scheduler (`warmup_steps` in config), cross-entropy loss, gradient clip, checkpointing. Best model saved to `runs/best.pt`; if there is no validation set, the last epoch checkpoint is copied to `best.pt` so generate/evaluate still work. Reproducibility: `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` from config `project.seed`.

**Inference pipeline:** Diff (file / stdin / repo with `--staged`) → same preprocess + tokenize → `model.generate()` → decode → optional `fix:`/`feat:`/… prefix → print.

---

## 5. Implementation map (where is what)

| Component | Purpose | File(s) / location |
|-----------|---------|---------------------|
| Diff cleaning | Keep only +/- lines, drop metadata | `src/preprocess.py`: `clean_diff()` |
| Code normalization | Literals → placeholders | `src/preprocess.py`: `normalize_code_line()`, `normalize_diff()` |
| Message normalization | Lowercase, trim length, remove emoji | `src/preprocess.py`: `normalize_commit_message()`, `preprocess_pair()` |
| Subword tokenizer | BPE, shared vocab, special tokens | `src/tokenizer.py`: `DiffTokenizer` (SentencePiece) |
| Tokenizer training | Build vocab from corpus (diffs + messages) | `src/preprocess.py` + `prepare_data.py` (corpus from train.jsonl) |
| Dataset | Load JSONL, preprocess, tokenize, pad | `src/dataset.py`: `CommitDataset`, `collate_commit()` |
| Seq2Seq model | BiLSTM encoder, LSTM decoder, Bahdanau attention | `src/models/seq2seq.py`: `Attention`, `Seq2SeqCommit` |
| Transformer model | Encoder–decoder, positional encoding | `src/models/transformer_model.py`: `PositionalEncoding`, `TransformerCommit` |
| Training loop | Loss, AdamW, warmup + cosine LR, gradient clip, save best / last→best | `src/train.py` |
| Config (model) | `model.type`, `dropout_seq2seq`, `dropout_transformer`, `warmup_steps` | `configs/default.yaml` |
| Inference | Load checkpoint, run generate, decode | `src/generate.py` (--diff, --repo, --stdin, --staged) |
| Evaluation | BLEU, ROUGE-L, METEOR | `src/evaluate.py` |
| Intent (conventional style) | Heuristic: fix/feat/refactor/docs → prefix | `src/intent.py`: `classify_intent_heuristic()`, `intent_to_style_prefix()` |
| Daily-use scripts | Suggest message for staged changes | `scripts/suggest.ps1`, `scripts/suggest.sh` |
| Public data | CommitBench → JSONL | `scripts/download_commitbench.py` |

Everything from “idea” (diff in → message out) to “daily use” (suggest from staged changes) is implemented in the repo.

---

## 6. Industry alignment

**Real-world problem:** Commit message quality and consistency matter for teams and tooling (code review, CI, changelogs). The tool targets exactly that.

**Production-style pipeline:**
- **Config-driven:** Model type, sizes, dropout per model (`dropout_seq2seq`, `dropout_transformer`), `warmup_steps`, paths in `configs/default.yaml`.
- **Reproducible:** Fixed seed (CPU + CUDA), public dataset (CommitBench), fixed train/val split, documented format in `data/README.md`.
- **Standard metrics:** BLEU, ROUGE-L, METEOR (same as in industry and research).
- **Conventional commits:** Optional intent-based prefix (`fix:`, `feat:`, etc.) matches common team conventions.
- **Integration:** Can be run from any repo (script or pipe), no lock-in to a specific IDE or platform.

**Deployment-style use:** One-time training produces a checkpoint; inference is a single Python call (or script) reading diff from file, stdin, or Git. Suitable for local use, CI, or internal tools.

---

## 7. Novelty in the implementation

**1. Code-aware preprocessing**
- Input is **code diffs**, not plain text. Preprocessing is tailored to that:
  - Strip Git metadata but **keep** `+`/`-` so the model sees add/remove structure.
  - **Literal normalization** (`<STRING>`, `<NUM>`, `<PATH>`) to control vocabulary and generalize across different constants.
- Implemented in `src/preprocess.py`; no generic “text summarization” pipeline.

**2. Single tokenizer for diff and message**
- One SentencePiece model is trained on **both** diff text and commit messages, with **user_defined_symbols** for `+`, `-`, and placeholders.
- The same tokenizer is used for encoding the diff and decoding the message, which is a deliberate design for this task (code + short natural language).

**3. Two model families**
- **Seq2Seq (LSTM + Bahdanau attention)** as a strong baseline.
- **Transformer** encoder–decoder for longer context and parallel encoding.
- You can switch via config and compare (e.g. BLEU/ROUGE-L/METEOR) — standard in industry and research.

**4. Intent-aware conventional style**
- A separate **intent** step (heuristic in `src/intent.py`) classifies the diff (fix / feat / refactor / docs) and prepends a conventional-commits-style prefix to the generated message.
- This combines **generation** (NLP) with **style/format** (developer convention), which is a small but clear novelty for commit messages.

**5. Staged-only and stdin**
- **`--staged`** uses only the diff that will be committed (`git diff --cached`), matching how developers actually commit.
- **`--stdin`** allows piping (e.g. `git diff --staged | …`) for scripting and automation.
- These choices are about **workflow fit**, not only model design.

So the project is not “generic summarization”; it is **commit-message generation from code diffs** with code-specific preprocessing, shared tokenization, two model types, and optional intent-based styling — and it’s implemented end-to-end.

---

## 8. Does the implementation match “industry + NLP + novelty”?

**Industry:** Yes. Real problem, config-driven training, standard metrics, conventional-commits option, and a path to daily use (scripts, stdin, staged diff).

**NLP:** Yes. The pipeline uses:
- Sequence-to-sequence (encoder–decoder).
- Attention (Bahdanau and transformer self/cross-attention).
- Subword tokenization (BPE via SentencePiece).
- Abstractive generation with teacher forcing and autoregressive decoding.
- Standard NLG metrics (BLEU, ROUGE-L, METEOR).

**Novelty:** Yes, in the sense above: code-diff-specific preprocessing, unified tokenizer for diff+message, two models, intent-based prefix, and workflow-oriented options (staged, stdin). It’s not a copy of a generic tutorial; it’s tailored to commit messages from diffs.

**Gaps you could extend later (optional):**
- Beam search instead of greedy/argmax decoding.
- Trained intent classifier instead of (or in addition to) heuristics.
- Hierarchical or chunk-level attention over diff hunks.
- Larger or domain-specific datasets and pre-trained code models (e.g. CodeBERT) for even stronger industry-grade results.

---

## 9. Day-to-day use as a developer

**One-time setup (from the project folder):**
1. Install dependencies: `pip install -r requirements.txt` (and optionally `datasets` for CommitBench).
2. (Optional but recommended) Get real data and train:
   - `python scripts/download_commitbench.py --max_train 50000 --max_val 5000`
   - `python -m src.prepare_data`
   - `python -m src.train --config configs/default.yaml`
3. You should have `runs/best.pt` and `data/sp_model.model`.

**Daily workflow:**
1. Work in your repo as usual; stage the changes you want to commit (`git add ...`).
2. Get a suggested message:
   - **Option A (script):** From your repo directory run the project’s suggest script, e.g.  
     `D:\hub\commit_message_generator\scripts\suggest.ps1`  
     (or `.\scripts\suggest.ps1` if you’re inside the commit_message_generator repo).
   - **Option B (pipe):** From the commit_message_generator folder, with `PYTHONPATH` set:
     ```bash
     git -C C:\path\to\your\repo diff --staged | python -m src.generate --stdin --checkpoint runs/best.pt --intent
     ```
   - **Option C (explicit repo):**
     ```bash
     python -m src.generate --repo C:\path\to\your\repo --staged --checkpoint runs/best.pt --intent
     ```
3. Use the printed line as your commit message (e.g. `git commit -m "feat: add password hashing utility"`) or edit it.

**Optional:** Add a PowerShell alias (e.g. `suggest-commit`) that runs the suggest script so you can get a suggestion from any repo with one command. See README for the exact alias.

**Summary:** The project is designed so that from **idea** (suggest messages from diffs) to **NLP** (seq2seq, attention, tokenization, metrics) to **implementation** (preprocess, train, evaluate, generate) to **industry-style use** (config, metrics, conventional commits) and **novelty** (code-aware pipeline, intent, staged/stdin), everything is present in the codebase and documented here. Training once and then using the script or pipe from any repo gives you a concrete, day-to-day developer tool.

---

## 10. Implementation checklist (idea → daily use)

Use this to verify that the project matches the full story (idea, NLP, industry, novelty, daily use).

| # | Requirement | Implemented? | Where |
|---|-------------|--------------|--------|
| 1 | Idea: generate commit message from diff | Yes | End-to-end pipeline: preprocess → tokenize → model → decode |
| 2 | Input: raw Git diff (+, − lines) | Yes | `preprocess.clean_diff`, `generate.py` (--diff, --repo, --stdin) |
| 3 | Output: single-line natural language message | Yes | `generate.py` prints one line; dataset `max_msg_words` |
| 4 | NLP: sequence-to-sequence | Yes | `Seq2SeqCommit`, `TransformerCommit` (encoder + decoder) |
| 5 | NLP: attention (decoder over encoder) | Yes | `seq2seq.Attention` (Bahdanau), Transformer cross-attention |
| 6 | NLP: subword tokenization | Yes | `tokenizer.DiffTokenizer` (SentencePiece BPE) |
| 7 | NLP: abstractive generation (not extractive) | Yes | Decoder generates tokens; no copy mechanism from diff |
| 8 | NLP: standard metrics (BLEU, ROUGE, METEOR) | Yes | `evaluate.py`: `bleu()`, `rouge_l()`, `meteor()` |
| 9 | Code-specific preprocessing | Yes | `preprocess`: literal normalization, keep +/−, drop headers |
| 10 | Two models (baseline + advanced) | Yes | `seq2seq.py`, `transformer_model.py`; config `model.type` |
| 11 | Intent / conventional commits style | Yes | `intent.py`: `classify_intent_heuristic`, `intent_to_style_prefix`; `--intent` in generate |
| 12 | Industry: config-driven training | Yes | `configs/default.yaml`, loaded in `train.py` |
| 13 | Industry: public dataset (CommitBench) | Yes | `scripts/download_commitbench.py` |
| 14 | Daily use: suggest for staged changes | Yes | `--staged`, `scripts/suggest.ps1`, `suggest.sh` |
| 15 | Daily use: pipe diff from Git | Yes | `--stdin` in `generate.py` |
| 16 | Reproducible data format | Yes | JSONL `{"diff","message"}`, `data/README.md` |
| 17 | Training: LR warmup + cosine decay | Yes | `train.py`: `warmup_steps` from config, `LambdaLR` scheduler |
| 18 | Training: best.pt when no val set | Yes | `train.py`: last epoch copied to `best.pt` |
| 19 | Reproducibility: CPU + CUDA seed | Yes | `train.py`: `torch.manual_seed`, `torch.cuda.manual_seed_all` |

All items are implemented in the current codebase and align with the narrative from idea through NLP, industry relevance, novelty, and day-to-day use.
