# After training: what to do next

Once you have trained on real data (e.g. CommitBench), follow these steps. No demo script; focus on comparison and documentation.

**Note:** If you train without a validation set, `runs/best.pt` is still created (the last epoch checkpoint is copied), so `generate.py` and `evaluate.py` work as usual.

---

## 1. Compare baseline vs Transformer

You have two model types in `configs/default.yaml`: `model.type: seq2seq` (uses `dropout_seq2seq`) and `model.type: transformer` (uses `dropout_transformer`).

- Train **Seq2Seq** (baseline): set `model.type: seq2seq`, run training, save checkpoint as e.g. `runs/best_seq2seq.pt`.
- Train **Transformer**: set `model.type: transformer`, run training, save as e.g. `runs/best_transformer.pt`.
- **Evaluate both** on the same val/test set:
  ```bash
  python -m src.evaluate --checkpoint runs/best_seq2seq.pt --data data/val.jsonl
  python -m src.evaluate --checkpoint runs/best_transformer.pt --data data/val.jsonl
  ```
- Record **BLEU, ROUGE-L, METEOR** for each. Optionally run on `data/test.jsonl` for final numbers.
- In your report/slides: one table (Seq2Seq vs Transformer) and 2–3 qualitative examples where one model is better or both fail.

This gives you a clear **model comparison** section and viva answers.

---

## 2. Document your setup

- **README.md** — Dataset section already describes CommitBench and how to download/split. Add one line with your actual choice, e.g. “We use 50k train / 5k val from CommitBench.”
- **requirements.txt** — All packages are listed with versions; keep them so the project is reproducible.
- **data/README.md** — Already describes JSONL format and source; no change needed unless you add another dataset.

This makes the project reproducible and easier to defend in viva.

---

## 3. Viva / report

- Use **ARCHITECTURE.md** for: problem definition, system architecture, data pipeline, models, metrics, innovation (intent-aware).
- **Failure cases**: Prepare 2–3 examples where the model fails (e.g. very small diff, multi-purpose commit, refactor-only). Show the bad output and say: “Future work: hierarchical attention / more data / intent conditioning.”
- **One slide/section**: “What we did: data (CommitBench) → preprocessing → tokenizer → Seq2Seq and Transformer → evaluation (BLEU, ROUGE, METEOR) and optional intent prefix.”

No demo script required; focus on results, comparison, and limitations.
