"""
Transformer Encoder-Decoder for commit message generation.
Handles long diffs and global context via self-attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerCommit(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_diff_len: int = 512,
        max_msg_len: int = 64,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(max_diff_len, max_msg_len), dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_key_padding_mask(self, diff_ids: torch.Tensor) -> torch.Tensor:
        return (diff_ids == self.pad_id)

    def _make_tgt_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1
        )

    def encode(
        self,
        diff_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_key_padding = self._make_src_key_padding_mask(diff_ids)
        x = self.pos_enc(self.embed(diff_ids) * math.sqrt(self.d_model))
        memory = self.encoder(x, src_key_padding_mask=src_key_padding)
        return memory

    def forward(
        self,
        diff_ids: torch.Tensor,
        msg_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
        msg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Teacher forcing: decoder input is msg_ids shifted right (BOS ... EOS-1)
        tgt = msg_ids[:, :-1]
        tgt_key_padding = (tgt == self.pad_id)
        tgt_mask = self._make_tgt_mask(tgt.size(1), tgt.device)
        memory = self.encode(diff_ids, diff_mask)
        src_key_padding = self._make_src_key_padding_mask(diff_ids)
        y = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))
        out = self.decoder(
            y,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
        )
        return self.fc(out)

    def generate(
        self,
        diff_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
        max_len: int = 20,
        eos_id: int = 3,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        memory = self.encode(diff_ids, diff_mask)
        B = diff_ids.size(0)
        bos_id = 2
        generated = torch.full(
            (B, 1),
            bos_id,
            dtype=torch.long,
            device=diff_ids.device,
        )
        # Compute once — diff_ids does not change during generation
        src_key_padding = self._make_src_key_padding_mask(diff_ids)
        for _ in range(max_len - 1):
            tgt_key_padding = (generated == self.pad_id)
            tgt_mask = self._make_tgt_mask(generated.size(1), generated.device)
            y = self.pos_enc(self.embed(generated) * math.sqrt(self.d_model))
            out = self.decoder(
                y,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding,
                memory_key_padding_mask=src_key_padding,
            )
            logit = self.fc(out[:, -1])
            if temperature != 1.0:
                logit = logit / temperature
            next_id = logit.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return generated
