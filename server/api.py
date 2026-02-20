"""
FastAPI inference server for the commit message generator.

Start with:
    uvicorn server.api:app --host 0.0.0.0 --port 8000

Environment variables:
    CHECKPOINT   Path to runs/best.pt (default: <project_root>/runs/best.pt)
    DEVICE       torch device string, e.g. "cpu" or "cuda" (default: "cpu")
    API_KEY      Optional bearer token required on every request
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Make src/ importable whether the server is run from the project root
# or from inside the server/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intent import classify_intent_heuristic, intent_to_style_prefix  # noqa: E402
from src.models import Seq2SeqCommit, TransformerCommit  # noqa: E402
from src.preprocess import normalize_diff  # noqa: E402
from src.tokenizer import DiffTokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Commit Message Generator",
    description="ML-powered commit message suggestions from git diffs.",
    version="1.0.0",
)

# --------------------------------------------------------------------------- #
# Auth                                                                          #
# --------------------------------------------------------------------------- #

_REQUIRED_KEY = os.environ.get("API_KEY", "")
_bearer = HTTPBearer(auto_error=False)


def _check_auth(creds: HTTPAuthorizationCredentials | None = Security(_bearer)):
    if not _REQUIRED_KEY:
        return  # Auth disabled
    if creds is None or creds.credentials != _REQUIRED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# --------------------------------------------------------------------------- #
# Global model state (loaded once at startup)                                  #
# --------------------------------------------------------------------------- #

_model: Seq2SeqCommit | TransformerCommit | None = None
_tokenizer: DiffTokenizer | None = None
_cfg: dict = {}
_device: torch.device = torch.device("cpu")


@app.on_event("startup")
async def _startup() -> None:
    global _model, _tokenizer, _cfg, _device

    checkpoint_path = os.environ.get("CHECKPOINT", str(ROOT / "runs" / "best.pt"))
    device_str = os.environ.get("DEVICE", "cpu")
    _device = torch.device(device_str)

    if not Path(checkpoint_path).is_file():
        log.error("Checkpoint not found: %s", checkpoint_path)
        log.error("Train the model first: python -m src.train --config configs/default.yaml")
        return

    log.info("Loading checkpoint from %s on %s", checkpoint_path, _device)
    ckpt = torch.load(checkpoint_path, map_location=_device)

    _cfg = ckpt.get("config")
    if not _cfg:
        log.error("Checkpoint is missing 'config' key — was it saved by src/train.py?")
        return

    # Load tokenizer
    sp_prefix = _cfg["tokenizer"].get("model_prefix", "data/sp_model")
    sp_path = ROOT / (sp_prefix + ".model")
    _tokenizer = DiffTokenizer(
        model_path=str(sp_path) if sp_path.is_file() else None,
        model_prefix=os.path.basename(sp_prefix),
        data_dir=str(ROOT / "data"),
    )
    _tokenizer.load()
    vocab_size = _tokenizer.vocab_size_actual
    log.info("Tokenizer loaded. vocab_size=%d", vocab_size)

    # Instantiate model
    m_cfg = _cfg["model"]
    if m_cfg["type"] == "seq2seq":
        _model = Seq2SeqCommit(
            vocab_size=vocab_size,
            embed_dim=m_cfg.get("embed_dim", 256),
            hidden_dim=m_cfg.get("hidden_dim", 512),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=0.0,
            pad_id=0,
        )
    else:
        _model = TransformerCommit(
            vocab_size=vocab_size,
            d_model=m_cfg.get("d_model", 256),
            nhead=m_cfg.get("nhead", 8),
            num_encoder_layers=m_cfg.get("num_encoder_layers", 4),
            num_decoder_layers=m_cfg.get("num_decoder_layers", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 1024),
            dropout=0.0,
            pad_id=0,
            max_diff_len=_cfg["data"].get("max_diff_tokens", 512),
            max_msg_len=_cfg["data"].get("max_msg_tokens", 20),
        )

    _model.load_state_dict(ckpt["model_state"], strict=True)
    _model.to(_device)
    _model.eval()
    log.info("Model ready (%s).", m_cfg["type"])


# --------------------------------------------------------------------------- #
# Request / Response schemas                                                    #
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
    diff: str = Field(..., description="Raw git diff text (output of git diff --staged).")
    intent: bool = Field(True, description="Prepend conventional commit prefix (fix:/feat:/etc.).")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature.")
    max_len: int = Field(20, ge=1, le=64, description="Max tokens to generate.")


class GenerateResponse(BaseModel):
    message: str = Field(..., description="Suggested commit message.")


# --------------------------------------------------------------------------- #
# Endpoints                                                                     #
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest, _: None = Depends(_check_auth)):
    if _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs.",
        )

    raw_diff = req.diff.strip()
    if not raw_diff:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="diff must not be empty.",
        )

    norm = normalize_diff(raw_diff, normalize_literals=True)
    max_diff_tokens = _cfg["data"].get("max_diff_tokens", 512)
    ids = _tokenizer.encode(norm, add_bos=False, add_eos=True, max_len=max_diff_tokens)

    src = torch.tensor([ids], dtype=torch.long, device=_device)
    with torch.no_grad():
        gen = _model.generate(src, max_len=req.max_len, eos_id=3, temperature=req.temperature)

    msg = _tokenizer.decode(gen[0].tolist(), skip_special=True).strip() or "(empty)"

    if req.intent:
        intent_id = classify_intent_heuristic(raw_diff)
        prefix = intent_to_style_prefix(intent_id)
        if not msg.lower().startswith(prefix.rstrip().lower()):
            msg = prefix + msg

    return GenerateResponse(message=msg)
