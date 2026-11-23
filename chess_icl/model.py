"""Small GPT-style transformer for chess position evaluation (W/D/L)."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    max_seq_len: int = 4096
    dropout: float = 0.1
    n_classes: int = 3  # Win, Draw, Loss


class CausalSelfAttention(nn.Module):
    """Causal self-attention using PyTorch scaled_dot_product_attention (FlashAttention)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Uses FlashAttention / Memory-Efficient attention automatically
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ChessTransformer(nn.Module):
    """Decoder-only transformer for W/D/L classification.

    Configs:
        tiny:   d=128, h=4, L=4,  ff=512  → ~1.0M params
        small:  d=256, h=8, L=8,  ff=1024 → ~6.4M params
        medium: d=384, h=8, L=12, ff=1536 → ~21M params
    """

    def __init__(self, config: ModelConfig, pad_token_id: int = 0):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            input_ids: (B, T) token indices
            attention_mask: optional (B, T) boolean or float mask

        Returns:
            logits: (B, 3) classification logits for [WIN, DRAW, LOSS]
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Classification pooling: gather at the last non-pad token
        non_pad_mask = (input_ids != self.pad_token_id)
        lengths = non_pad_mask.sum(dim=-1) - 1
        lengths = lengths.clamp(min=0)

        batch_indices = torch.arange(B, device=device)
        last_hidden = x[batch_indices, lengths]  # (B, d_model)

        logits = self.head(last_hidden)  # (B, 3)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Union[str, Path]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else self.config,
                "pad_token_id": self.pad_token_id,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "ChessTransformer":
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=map_location)
        cfg_data = ckpt["config"]
        if isinstance(cfg_data, dict):
            config = ModelConfig(**cfg_data)
        else:
            config = cfg_data
        pad_token_id = ckpt.get("pad_token_id", 0)
        model = cls(config, pad_token_id=pad_token_id)
        model.load_state_dict(ckpt["state_dict"])
        return model


# Preset configs
TINY = ModelConfig(d_model=128, n_heads=4, n_layers=4, d_ff=512)
SMALL = ModelConfig(d_model=256, n_heads=8, n_layers=8, d_ff=1024)
MEDIUM = ModelConfig(d_model=384, n_heads=8, n_layers=12, d_ff=1536)
