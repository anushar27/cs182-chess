"""Small GPT-style transformer for chess position evaluation (W/D/L) with Maximal Update Parametrization (muP)."""

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

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
    # Maximal Update Parametrization (muP) fields
    use_mup: bool = False
    d_base: int = 128      # Base model width (TINY is 128)
    d_base_head: int = 32  # Base head dimension (128 / 4 = 32)


class CausalSelfAttention(nn.Module):
    """Causal self-attention using PyTorch scaled_dot_product_attention (FlashAttention).

    Supports muP attention scaling: logit scale factor = sqrt(d_base_head) / d_head
    instead of 1 / sqrt(d_head), preserving attention entropy across head dimensions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout_p = config.dropout

        if config.use_mup:
            self.attn_scale = (float(config.d_base_head) ** 0.5) / float(self.head_dim)
        else:
            self.attn_scale = 1.0 / math.sqrt(float(self.head_dim))

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
            scale=self.attn_scale,
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
        tiny:   d=128,  h=4,  L=4,  ff=512  → ~1.0M params
        small:  d=256,  h=8,  L=8,  ff=1024 → ~6.4M params
        medium: d=384,  h=8,  L=12, ff=1536 → ~21M params
        large:  d=1024, h=16, L=16, ff=4096 → ~205M params
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

        # Initialize weights with muP or standard scaling
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.use_mup:
                r = float(self.config.d_model) / float(self.config.d_base)
                # Hidden weights scaled by 1 / sqrt(r)
                std = 0.02 / math.sqrt(r)
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

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
        if self.config.use_mup:
            # Readout multiplier in muP: d_base / d_model = 1 / r
            logits = logits * (float(self.config.d_base) / float(self.config.d_model))
        return logits

    def get_optimizer(self, base_lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
        """Construct AdamW optimizer with muP parameter groups or standard groups."""
        if not self.config.use_mup:
            decay_params = []
            nodecay_params = []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    nodecay_params.append(p)
            return torch.optim.AdamW(
                [
                    {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
                    {"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0},
                ]
            )

        # muP AdamW:
        # width_mult r = d_model / d_base
        r = float(self.config.d_model) / float(self.config.d_base)
        hidden_decay = []
        hidden_nodecay = []
        embedding_params = []
        head_params = []
        ln_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "token_emb" in name or "pos_emb" in name:
                embedding_params.append(param)
            elif "head" in name:
                head_params.append(param)
            elif "ln" in name:
                ln_params.append(param)
            else:
                # Hidden transformer weights (c_attn, c_proj, ff)
                if param.dim() >= 2:
                    hidden_decay.append(param)
                else:
                    hidden_nodecay.append(param)

        param_groups = [
            {"params": embedding_params, "lr": base_lr, "weight_decay": 0.0},
            {"params": hidden_decay, "lr": base_lr / r, "weight_decay": weight_decay},
            {"params": hidden_nodecay, "lr": base_lr / r, "weight_decay": 0.0},
            {"params": ln_params, "lr": base_lr, "weight_decay": 0.0},
            {"params": head_params, "lr": base_lr, "weight_decay": weight_decay if any(p.dim() >= 2 for p in head_params) else 0.0},
        ]
        return torch.optim.AdamW(param_groups)

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
            # Safe init filtering out any unknown kwargs while setting defaults
            valid_keys = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            filtered_cfg = {k: v for k, v in cfg_data.items() if k in valid_keys}
            config = ModelConfig(**filtered_cfg)
        else:
            config = cfg_data
        pad_token_id = ckpt.get("pad_token_id", 0)
        model = cls(config, pad_token_id=pad_token_id)
        model.load_state_dict(ckpt["state_dict"])
        return model


# Preset configs
TINY = ModelConfig(d_model=128, n_heads=4, n_layers=4, d_ff=512, d_base=128)
SMALL = ModelConfig(d_model=256, n_heads=8, n_layers=8, d_ff=1024, d_base=128)
MEDIUM = ModelConfig(d_model=384, n_heads=8, n_layers=12, d_ff=1536, d_base=128)
LARGE = ModelConfig(d_model=1024, n_heads=16, n_layers=16, d_ff=4096, d_base=128)
XLARGE = ModelConfig(d_model=2048, n_heads=16, n_layers=16, d_ff=6144, d_base=128)

