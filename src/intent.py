"""
Intent-Aware Commit Generation (Innovation Hook).

Classify diff into: bugfix (0), feature (1), refactor (2), docs (3).
Can be used to condition the message generator for style (e.g. "Fix ..." vs "Add ...").
"""

import re
from typing import Literal

IntentLabel = Literal["fix", "feat", "refactor", "docs"]
INTENT_MAP = {"fix": 0, "feat": 1, "refactor": 2, "docs": 3}
INTENT_NAMES = ["fix", "feat", "refactor", "docs"]


def classify_intent_heuristic(diff: str) -> int:
    """
    Rule-based intent classification from diff content.
    Use this when no trained classifier is available; otherwise train a small classifier.
    """
    diff_lower = diff.lower()
    lines = diff_lower.split("\n")
    added = " ".join(l for l in lines if l.strip().startswith("+"))
    removed = " ".join(l for l in lines if l.strip().startswith("-"))

    # Docs: README, .md, comments, docstring
    if any(x in diff_lower for x in ["readme", ".md", "docstring", "'''", '"""', "# "]):
        if sum(1 for l in lines if l.strip().startswith("+")) > 2:
            return INTENT_MAP["docs"]

    # Bugfix: fix, bug, error, exception, correct, remove broken
    if re.search(r"\b(fix|bug|error|exception|correct|wrong|broken)\b", added + removed):
        return INTENT_MAP["fix"]

    # Refactor: rename, extract, move, simplify, cleanup
    if re.search(r"\b(rename|extract|move|simplify|cleanup|reformat)\b", added + removed):
        return INTENT_MAP["refactor"]
    if len(removed.split()) > 5 and len(added.split()) > 5 and "test" not in diff_lower:
        if 0.5 < len(added) / max(len(removed), 1) < 2.0:
            return INTENT_MAP["refactor"]

    # Feature: add, new, implement, support
    if re.search(r"\b(add|new|implement|support|introduce)\b", added):
        return INTENT_MAP["feat"]

    # Default: feature for net additions, refactor for balanced
    if len(added) > len(removed) * 1.5:
        return INTENT_MAP["feat"]
    return INTENT_MAP["refactor"]


def intent_to_style_prefix(intent_id: int) -> str:
    """Suggested prefix for commit message by intent (conventional commits style)."""
    return INTENT_NAMES[intent_id] + ": "
