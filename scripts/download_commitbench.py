"""
Download CommitBench from Hugging Face and convert to project JSONL format.

CommitBench: https://huggingface.co/datasets/Maxscha/commitbench
Fields: Diff, Commit Message, Hash, Project, Split (train/validation/test).

Usage:
  python scripts/download_commitbench.py [--max_train 50000] [--max_val 5000] [--max_test 5000]

Output: data/train.jsonl, data/val.jsonl, data/test.jsonl (each line: {"diff": "...", "message": "..."}).
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Download CommitBench and convert to JSONL")
    parser.add_argument("--max_train", type=int, default=None, help="Cap train samples (default: all)")
    parser.add_argument("--max_val", type=int, default=5000, help="Cap validation samples")
    parser.add_argument("--max_test", type=int, default=5000, help="Cap test samples")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: project data/)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install Hugging Face datasets: pip install datasets")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CommitBench from Hugging Face (Maxscha/commitbench)...")
    ds = load_dataset("Maxscha/commitbench", trust_remote_code=True)

    # Resolve column names (CommitBench: Diff, Commit Message, Hash, Project, Split)
    first_split = list(ds.keys())[0]
    first_row = ds[first_split][0]
    diff_key = next((k for k in first_row if k.lower() == "diff"), "Diff")
    msg_key = next(
        (k for k in first_row if "message" in k.lower() or k == "Commit Message"),
        "Commit Message",
    )

    def write_split(split_name, hf_split, max_rows, out_path):
        if hf_split not in ds:
            print(f"  Skip {split_name}: split '{hf_split}' not in dataset")
            return 0
        subset = ds[hf_split]
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(subset.num_rows):
                if max_rows is not None and n >= max_rows:
                    break
                row = subset[i]
                diff = row.get(diff_key) or row.get("Diff") or ""
                msg = row.get(msg_key) or row.get("Commit Message") or ""
                if not diff or not msg:
                    continue
                try:
                    f.write(json.dumps({"diff": diff, "message": msg}) + "\n")
                    n += 1
                except (TypeError, UnicodeEncodeError):
                    continue
        print(f"  Wrote {n} rows to {out_path}")
        return n

    write_split("train", "train", args.max_train, out_dir / "train.jsonl")
    write_split("val", "validation", args.max_val, out_dir / "val.jsonl")
    if "test" in ds:
        write_split("test", "test", args.max_test, out_dir / "test.jsonl")
    print("Done. Next: run prepare_data to train tokenizer, then train.")


if __name__ == "__main__":
    main()
