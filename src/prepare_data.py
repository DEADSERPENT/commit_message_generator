"""
Prepare data and train tokenizer.
1) Creates sample train/val JSONL if data/train.jsonl does not exist.
2) Trains SentencePiece on diffs + messages and saves to data/sp_model.model.

For real data: place data/train.jsonl and data/val.jsonl with lines:
  {"diff": "<raw diff text>", "message": "commit message"}
Optional: {"diff": "...", "message": "...", "intent": 0} for intent-aware (0=fix,1=feat,2=refactor,3=docs)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import preprocess_pair


def sample_diffs_and_messages():
    """Return list of (diff, message) for training tokenizer and as minimal dataset."""
    return [
        (
            """
diff --git a/user.js b/user.js
+ function validateEmail(email) {
+   return email.includes("@");
+ }
""",
            "Add basic email validation utility",
        ),
        (
            """
diff --git a/api.py b/api.py
- def get_user(id):
-     return db.query(User).get(id)
+ def get_user(user_id):
+     return db.query(User).filter_by(id=user_id).first()
""",
            "Refactor get_user to use filter_by and rename parameter",
        ),
        (
            """
diff --git a/config.py b/config.py
+ DEBUG = True
+ SECRET_KEY = "dev-secret"
""",
            "Add debug and secret key config for development",
        ),
        (
            """
diff --git a/utils.py b/utils.py
- import os
+ import os
+ import hashlib
+ def hash_password(pwd):
+     return hashlib.sha256(pwd.encode()).hexdigest()
""",
            "Add password hashing utility using sha256",
        ),
        (
            """
diff --git a/README.md b/README.md
+ ## Installation
+ pip install -r requirements.txt
""",
            "Add installation section to readme",
        ),
        (
            """
diff --git a/test_user.py b/test_user.py
+ def test_validate_email():
+     assert validate_email("a@b.com") == True
+     assert validateEmail("invalid") == False
""",
            "Add unit tests for email validation",
        ),
        (
            """
diff --git a/app.py b/app.py
- app.run(host='0.0.0.0')
+ app.run(host='0.0.0.0', port=5000)
""",
            "Fix app run to specify port 5000",
        ),
        (
            """
diff --git a/models.py b/models.py
- class User:
-     name = Column(String)
- class User(Base):
-     __tablename__ = 'users'
-     name = Column(String(128))
+ class User(Base):
+     __tablename__ = 'users'
+     name = Column(String(128))
+     email = Column(String(256), unique=True)
""",
            "Add email column to User model",
        ),
    ]


def write_jsonl(path: str, pairs: list, normalize: bool = True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for diff, msg in pairs:
            if normalize:
                from src.preprocess import preprocess_pair
                diff, msg = preprocess_pair(diff, msg)
            f.write(json.dumps({"diff": diff, "message": msg}) + "\n")


def build_corpus_for_sp(train_path: str, out_corpus_path: str):
    """Build a one-sentence-per-line corpus (diffs and messages) for SentencePiece."""
    lines = []
    if os.path.isfile(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    d = obj.get("diff", "").replace("\n", " ")
                    m = obj.get("message", "")
                    if d:
                        lines.append(d)
                    if m:
                        lines.append(m)
                except json.JSONDecodeError:
                    pass
    if not lines:
        return False
    os.makedirs(os.path.dirname(out_corpus_path) or ".", exist_ok=True)
    with open(out_corpus_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.strip() + "\n")
    return True


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.is_file():
        samples = sample_diffs_and_messages()
        # Write raw diffs and messages (preprocess in Dataset)
        with open(train_path, "w", encoding="utf-8") as f:
            for diff, msg in samples[: max(1, len(samples) - 2)]:
                f.write(json.dumps({"diff": diff, "message": msg}) + "\n")
        with open(val_path, "w", encoding="utf-8") as f:
            for diff, msg in samples[-2:]:
                f.write(json.dumps({"diff": diff, "message": msg}) + "\n")
        print("Created sample train/val at", train_path, val_path)

    corpus_path = data_dir / "sp_corpus.txt"
    if not build_corpus_for_sp(str(train_path), str(corpus_path)):
        print("No data to build tokenizer corpus.")
        return
    print("Built tokenizer corpus at", corpus_path)

    from src.tokenizer import DiffTokenizer
    import yaml
    config_path = base_dir / "configs" / "default.yaml"
    cfg = yaml.safe_load(open(config_path)) if config_path.is_file() else {}
    tok_cfg = cfg.get("tokenizer", {})
    prefix = "sp_model"
    tokenizer = DiffTokenizer(
        model_prefix=prefix,
        vocab_size=tok_cfg.get("vocab_size", 8000),
        data_dir=str(data_dir),
    )
    # With tiny corpora, SentencePiece requires vocab_size <= ~num_pieces; cap at 500 for sample data
    vocab_size = tok_cfg.get("vocab_size", 8000)
    with open(corpus_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    if num_lines < 100:
        vocab_size = min(vocab_size, 500)
    model_path = tokenizer.train(
        str(corpus_path),
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=tok_cfg.get("character_coverage", 0.9999),
    )
    print("Tokenizer saved to", model_path)


if __name__ == "__main__":
    main()
