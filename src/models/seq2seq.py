"""
Encoder-Decoder with attention (baseline) for commit message generation.
Encoder: BiLSTM over diff tokens; Decoder: LSTM with Bahdanau attention.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Bahdanau attention over encoder outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: (B, hidden*2) for BiLSTM
        # encoder_outputs: (B, T_enc, hidden*2)
        B, T, H = encoder_outputs.size()
        dec = decoder_hidden.unsqueeze(1).expand(-1, T, -1)
        energy = self.v(torch.tanh(self.w(encoder_outputs + dec))).squeeze(-1)
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask == 0, -1e9)
        attn = F.softmax(energy, dim=1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn


class Seq2SeqCommit(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = Attention(hidden_dim)
        self.enc_to_dec = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        diff_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # diff_ids: (B, T_enc)
        x = self.dropout(self.embed(diff_ids))
        enc_out, (h, c) = self.encoder(x)
        # Use last layer hidden from both directions; project to decoder hidden_dim
        h = h.view(self.encoder.num_layers, 2, -1, self.hidden_dim)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)
        c = c.view(self.encoder.num_layers, 2, -1, self.hidden_dim)
        c = c[-1]
        c = torch.cat([c[0], c[1]], dim=1)
        h_dec = self.enc_to_dec(h).unsqueeze(0)
        c_dec = self.enc_to_dec(c).unsqueeze(0)
        dec_init = (h_dec, c_dec)
        return enc_out, dec_init

    def forward(
        self,
        diff_ids: torch.Tensor,
        msg_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
        msg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out, (h, c) = self.encode(diff_ids, diff_mask)
        h, c = h.squeeze(0), c.squeeze(0)
        T_dec = msg_ids.size(1)
        msg_emb = self.dropout(self.embed(msg_ids[:, :-1]))
        logits_list = []
        for t in range(T_dec - 1):
            dec_hidden = torch.cat([h, c], dim=1)
            context, _ = self.attention(dec_hidden, enc_out, diff_mask)
            inp = torch.cat([msg_emb[:, t], context], dim=1).unsqueeze(1)
            out, (h, c) = self.decoder_lstm(inp, (h.unsqueeze(0), c.unsqueeze(0)))
            h, c = h.squeeze(0), c.squeeze(0)
            logit = self.fc(torch.cat([out.squeeze(1), context], dim=1))
            logits_list.append(logit)
        logits = torch.stack(logits_list, dim=1)
        return logits

    def generate(
        self,
        diff_ids: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
        max_len: int = 20,
        eos_id: int = 3,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        enc_out, (h, c) = self.encode(diff_ids, diff_mask)
        h, c = h.squeeze(0), c.squeeze(0)
        B = diff_ids.size(0)
        bos_id = 2
        ids = [bos_id] * B
        generated = torch.tensor([ids], device=diff_ids.device).t()
        for _ in range(max_len - 1):
            dec_hidden = torch.cat([h, c], dim=1)
            context, _ = self.attention(dec_hidden, enc_out, diff_mask)
            inp = torch.cat(
                [self.embed(generated[:, -1]), context], dim=1
            ).unsqueeze(1)
            out, (h, c) = self.decoder_lstm(inp, (h.unsqueeze(0), c.unsqueeze(0)))
            h, c = h.squeeze(0), c.squeeze(0)
            logit = self.fc(torch.cat([out.squeeze(1), context], dim=1))
            if temperature != 1.0:
                logit = logit / temperature
            next_id = logit.argmax(dim=-1)
            generated = torch.cat(
                [generated, next_id.unsqueeze(1)], dim=1
            )
            if (next_id == eos_id).all():
                break
        return generated
