"""
Extract (diff, message) pairs from a Git repository into JSONL for training.
Usage: python scripts/extract_commits.py --repo path/to/repo --output data/train.jsonl --max 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import git
except ImportError:
    print("Install GitPython: pip install GitPython")
    sys.exit(1)


def get_commit_diff(repo: git.Repo, commit: git.Commit) -> str:
    if not commit.parents:
        return repo.git.show(commit.hexsha, format="%B", patch=True).split("\n", 1)[-1] if repo.git.show(commit.hexsha) else ""
    return repo.git.diff(commit.parents[0].hexsha, commit.hexsha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to Git repository")
    parser.add_argument("--output", default="data/commits.jsonl")
    parser.add_argument("--max", type=int, default=5000)
    parser.add_argument("--min-diff-lines", type=int, default=2)
    parser.add_argument("--min-msg-words", type=int, default=2)
    args = parser.parse_args()
    repo = git.Repo(args.repo)
    base_dir = Path(__file__).resolve().parent.parent
    out_path = base_dir / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for commit in repo.iter_commits("HEAD", max_count=args.max):
            try:
                diff = get_commit_diff(repo, commit)
                msg = (commit.message or "").strip().split("\n")[0]
            except Exception:
                continue
            if not diff or diff.count("\n") < args.min_diff_lines:
                continue
            if not msg or len(msg.split()) < args.min_msg_words:
                continue
            try:
                line = json.dumps({"diff": diff, "message": msg}) + "\n"
                f.write(line)
                n += 1
            except (UnicodeDecodeError, TypeError):
                continue
    print(f"Wrote {n} commit pairs to {out_path}")


if __name__ == "__main__":
    main()
