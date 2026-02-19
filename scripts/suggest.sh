#!/usr/bin/env bash
# Suggest a commit message for staged changes. Run from any repo:
#   /path/to/commit_message_generator/scripts/suggest.sh
#   or: /path/to/commit_message_generator/scripts/suggest.sh /path/to/repo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_PATH="${1:-.}"
CHECKPOINT="$PROJECT_ROOT/runs/best.pt"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found. Train first: python -m src.train --config configs/default.yaml"
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/packages"
exec python -m src.generate --repo "$REPO_PATH" --staged --checkpoint "$CHECKPOINT" --intent
