"""
Subword tokenization for diffs and commit messages.
Uses SentencePiece to preserve symbols and handle code + natural language.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class DiffTokenizer:
    """
    Wrapper around SentencePiece for encoding/decoding diff and message sequences.
    Uses special tokens: [PAD], [UNK], [BOS], [EOS].
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_prefix: str = "sp_model",
        vocab_size: int = 8000,
        data_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.data_dir = data_dir or "data"
        self._sp: Optional[object] = None
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def load(self, path: Optional[str] = None) -> "DiffTokenizer":
        path = path or self.model_path
        if not path and self.model_prefix:
            path = os.path.join(self.data_dir, self.model_prefix + ".model")
        if path and os.path.isfile(path):
            if spm is None:
                raise RuntimeError("sentencepiece is not installed")
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(path)
            self.model_path = path
        return self

    @property
    def sp(self):
        if self._sp is None:
            self.load()
        return self._sp

    @property
    def vocab_size_actual(self) -> int:
        if self._sp is None:
            return self.vocab_size
        return len(self._sp)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
    ) -> List[int]:
        if not text.strip():
            ids = []
        else:
            ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        if not ids:
            return ""
        return self.sp.DecodeIds(ids)

    def train(
        self,
        corpus_path: str,
        model_prefix: Optional[str] = None,
        vocab_size: Optional[int] = None,
        character_coverage: float = 0.9999,
    ) -> str:
        """Train SentencePiece model on a text corpus (one sentence per line)."""
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        prefix = model_prefix or self.model_prefix
        vs = vocab_size or self.vocab_size
        out_dir = str(Path(corpus_path).parent)
        out_path = os.path.join(out_dir, prefix)
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=out_path,
            vocab_size=vs,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            model_type="bpe",
            user_defined_symbols=["+", "-", "<STRING>", "<NUM>", "<PATH>"],
        )
        self.model_path = out_path + ".model"
        self.load(self.model_path)
        return self.model_path
