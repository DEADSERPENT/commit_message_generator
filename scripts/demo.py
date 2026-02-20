"""
Developer demo — run the trained model against 5 realistic diffs.
Usage: python scripts/demo.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generate import load_config_from_checkpoint, get_model
from src.tokenizer import DiffTokenizer
from src.preprocess import normalize_diff
from src.intent import classify_intent_heuristic, intent_to_style_prefix

# ── Load model ────────────────────────────────────────────────────────────────
ckpt, cfg = load_config_from_checkpoint(str(ROOT / "runs" / "best.pt"))
device = torch.device("cpu")
sp_path = ROOT / (cfg["tokenizer"].get("model_prefix", "data/sp_model") + ".model")
tok = DiffTokenizer(model_path=str(sp_path), data_dir=str(ROOT / "data"))
tok.load()
model = get_model(cfg, tok.vocab_size_actual, device)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()


def suggest(diff: str) -> str:
    norm = normalize_diff(diff, normalize_literals=True)
    ids = tok.encode(norm, add_bos=False, add_eos=True, max_len=256)
    src = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        gen = model.generate(src, max_len=20, eos_id=3, temperature=0.8)
    msg = tok.decode(gen[0].tolist(), skip_special=True).strip() or "(empty)"
    prefix = intent_to_style_prefix(classify_intent_heuristic(diff))
    if not msg.lower().startswith(prefix.strip().lower()):
        msg = prefix + msg
    return msg


# ── Demo diffs ────────────────────────────────────────────────────────────────
DIFFS = [
    (
        "Bug fix   | null guard added",
        """\
- def get_user(uid):
-     return db.query(User).filter_by(id=uid).first()
+ def get_user(uid):
+     if uid is None:
+         return None
+     return db.query(User).filter_by(id=uid).first()""",
    ),
    (
        "Feature   | new rate-limiter class",
        """\
+class RateLimiter:
+    def __init__(self, max_req, window):
+        self.max_req = max_req
+        self.window  = window
+        self.log     = {}
+    def is_allowed(self, ip):
+        import time; now = time.time()
+        hits = [t for t in self.log.get(ip, []) if now - t < self.window]
+        self.log[ip] = hits
+        if len(hits) >= self.max_req:
+            return False
+        self.log[ip].append(now)
+        return True""",
    ),
    (
        "Refactor  | extract order helpers",
        """\
- def process_order(order):
-     total = sum(i['price']*i['qty'] for i in order['items'])
-     tax = total * 0.08
-     return total + tax
+def _total(items):
+    return sum(i['price'] * i['qty'] for i in items)
+def _tax(t, rate=0.08):
+    return t * rate
+def process_order(order):
+    t = _total(order['items'])
+    return t + _tax(t)""",
    ),
    (
        "Docs      | add install section",
        """\
-## Install
-Run pip.
+## Install
+
+    pip install -r requirements.txt
+    python -m src.prepare_data
+    python -m src.train
+
+Requires Python 3.9+ and PyTorch 2.0.""",
    ),
    (
        "Auth      | password hashing",
        (ROOT / "data" / "sample.diff").read_text(),
    ),
]

# ── Print results ─────────────────────────────────────────────────────────────
W = 64
print()
print("=" * W)
print("   COMMIT MESSAGE GENERATOR  -  DEVELOPER DEMO")
print(f"   Model : Transformer d_model=128  |  10 epochs  |  10k samples")
print(f"   Val loss: {ckpt.get('epoch', '?')+1}/10 epochs, best checkpoint")
print("=" * W)

for label, diff in DIFFS:
    msg = suggest(diff)
    first_line = diff.strip().splitlines()[0]
    snippet = first_line[:56] + ("..." if len(first_line) > 56 else "")
    print()
    print(f"  Scenario : {label}")
    print(f"  Diff     : {snippet}")
    print(f"  Suggested: {msg}")

print()
print("  How it works:")
print("    git add <files>")
print("    git diff --staged | python -m src.generate --stdin \\")
print("        --checkpoint runs/best.pt --intent")
print("=" * W)
