# Architecture & Viva Notes

## Project Title

**"Automated Semantic-Aware Git Commit Message Generation from Code Diffs using Deep Learning"**

---

## 1. Problem Definition

- **Real-world problem**: Developers write vague commit messages ("fix bug", "update code"), causing poor code review, bug tracking, and maintenance.
- **Formal statement**: Given a **Git diff** (added `+` and removed `-` lines), generate a **concise, meaningful commit message** that explains *what* and *why* (abstractive generation, not extractive).

---

## 2. Input–Output

| Input | Output |
|-------|--------|
| Raw Git diff (e.g. `+ function validateEmail(...)`) | Short commit message (e.g. "Add basic email validation utility") |

---

## 3. System Architecture (Describe in Viva)

```
Git Repo
   ↓
Diff Extractor (GitPython / raw diff file)
   ↓
Code Preprocessor (clean headers, normalize literals → <STRING>/<NUM>)
   ↓
Tokenizer (SentencePiece BPE)
   ↓
Seq2Seq / Transformer Model (Encoder–Decoder)
   ↓
Commit Message Generator (decode + optional intent prefix)
```

---

## 4. Data Pipeline

1. **Diff cleaning**: Remove `diff --git`, `index`, `---`/`+++`, keep only `+`/`-` lines.
2. **Code normalization**: Replace string/numeric literals with `<STRING>`, `<NUM>` to reduce vocabulary.
3. **Message normalization**: Lowercase, remove emojis, trim to ≤15 words.
4. **Dataset**: JSONL with `{"diff": "...", "message": "..."}`; 20k–50k pairs recommended.

---

## 5. Models

- **Baseline**: Encoder–Decoder with **Bahdanau attention** (BiLSTM encoder, LSTM decoder). Config: `model.type: seq2seq`, `dropout_seq2seq: 0.3`.
- **Advanced**: **Transformer** encoder–decoder; handles long diffs and global context. Config: `model.type: transformer`, `dropout_transformer: 0.1`.

---

## 6. Evaluation Metrics

- **BLEU**: n-gram overlap with reference.
- **ROUGE-L**: Longest common subsequence.
- **METEOR**: Semantic alignment (synonyms, stemming).
- **Human eval** (optional): Relevance, readability, correctness.

---

## 7. Innovation Hooks

- **Intent-aware generation**: Classify diff as fix/feat/refactor/docs (heuristic or small classifier), then prepend style prefix (e.g. "fix: ", "feat: ") for conventional commits.
- **Diff chunk attention** (future): Group by file/function, hierarchical attention.
- **Explainability** (future): Output top influencing code lines.

---

## 8. Implementation Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.8+ |
| Framework | PyTorch |
| Tokenizer | SentencePiece (BPE) |
| Dataset parsing | GitPython, JSONL |
| Training | AdamW, linear warmup + cosine LR decay, gradient clip; best.pt saved (or last epoch if no val) |
| Reproducibility | `project.seed` (CPU + CUDA) in config |
| Evaluation | NLTK, SacreBLEU, rouge-score |

---

## 9. Expected Outcomes

- Automatically generated commit messages from diffs.
- BLEU/ROUGE-L/METEOR scores on a held-out set.
- Demo on real Git repos via `generate.py --repo <path>`.

---

## 10. Error Analysis (For Viva)

- **Multi-purpose commits**: One diff doing many things → model may pick one theme.
- **Refactor-only**: Few semantic clues → generic message.
- **Very small diffs**: Little context → short/generic output.
- **Mitigation**: Intent classification, hierarchical attention, larger training data.
