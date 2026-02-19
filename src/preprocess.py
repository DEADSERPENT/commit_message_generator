"""
Data preprocessing for Git commit message generation.

- Diff cleaning: remove file headers, index lines; keep + / - lines
- Code normalization: replace string/numeric literals to reduce vocabulary
- Commit message normalization: lowercase, trim length, remove emojis
"""

import re
from typing import Optional, Tuple


# Literal placeholders (reduces vocabulary explosion)
STRING_PLACEHOLDER = "<STRING>"
NUM_PLACEHOLDER = "<NUM>"
PATH_PLACEHOLDER = "<PATH>"


def clean_diff(raw_diff: str) -> str:
    """
    Remove diff metadata; keep only added (+) and removed (-) lines.
    Preserves line type prefix for model to learn add/remove semantics.
    """
    if not raw_diff or not raw_diff.strip():
        return ""
    lines = []
    for line in raw_diff.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        # Skip diff header lines (file paths, index, etc.)
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@") and "@@" in line[2:]:
            # Optional: keep @@ as context marker (e.g. @@ -10,5 +10,6 @@)
            # lines.append(line[: min(20, len(line))])  # truncate long hunks
            continue
        if line.startswith("+") or line.startswith("-"):
            lines.append(line)
    return "\n".join(lines)


def normalize_code_line(line: str, normalize_literals: bool = True) -> str:
    """
    Normalize a single line of code for vocabulary reduction.
    - Optionally replace string literals, numbers, paths with placeholders.
    - Collapse repeated whitespace.
    """
    prefix = ""
    if line.startswith("+") or line.startswith("-"):
        prefix = line[0]
        line = line[1:]
    line = " ".join(line.split())
    if not normalize_literals:
        return (prefix + " " + line).strip() if prefix else line
    # String literals (single and double quoted)
    line = re.sub(r'"[^"]*"', STRING_PLACEHOLDER, line)
    line = re.sub(r"'[^']*'", STRING_PLACEHOLDER, line)
    line = re.sub(r"`[^`]*`", STRING_PLACEHOLDER, line)
    # Numbers (int/float/hex)
    line = re.sub(r"\b\d+\.?\d*\b", NUM_PLACEHOLDER, line)
    line = re.sub(r"\b0x[0-9a-fA-F]+\b", NUM_PLACEHOLDER, line)
    # Simple path-like tokens (optional, avoid over-normalizing)
    line = re.sub(r"\b[\w\-]+/[\w\-./]+\b", PATH_PLACEHOLDER, line)
    return (prefix + " " + line).strip() if prefix else line


def normalize_diff(diff: str, normalize_literals: bool = True) -> str:
    """Clean diff and normalize each line."""
    diff = clean_diff(diff)
    lines = [
        normalize_code_line(ln, normalize_literals)
        for ln in diff.split("\n")
        if ln.strip()
    ]
    return "\n".join(lines)


def normalize_commit_message(
    message: str,
    lowercase: bool = True,
    max_words: Optional[int] = 15,
    remove_emoji: bool = True,
) -> str:
    """
    Normalize commit message for training target.
    - Lowercase, remove emojis, trim to max_words.
    """
    if not message or not isinstance(message, str):
        return ""
    msg = message.strip()
    if remove_emoji:
        # Remove common emoji and symbols used as bullets
        msg = re.sub(r"[^\w\s.,;:!?\-'\"()]", " ", msg)
        msg = " ".join(msg.split())
    if lowercase:
        msg = msg.lower()
    if max_words is not None and max_words > 0:
        words = msg.split()
        msg = " ".join(words[: max_words])
    return msg.strip()


def preprocess_pair(
    diff: str,
    message: str,
    normalize_literals: bool = True,
    lowercase_message: bool = True,
    max_msg_words: Optional[int] = 15,
) -> Tuple[str, str]:
    """Preprocess a single (diff, message) pair. Returns (normalized_diff, normalized_message)."""
    norm_diff = normalize_diff(diff, normalize_literals=normalize_literals)
    norm_msg = normalize_commit_message(
        message,
        lowercase=lowercase_message,
        max_words=max_msg_words,
    )
    return norm_diff, norm_msg
