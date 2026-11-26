"""Track 2 from-scratch Chess Transformer v2 with RoPE, muP, and segment-bidirectional attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

AttentionMode = Literal["causal", "segment_bidirectional"]


@dataclass
class ModelConfigV2:
    """Configuration for ChessTransformerV2."""

    vocab_size: int = 5473  # Matches ChessV2Tokenizer().vocab_size
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    max_seq_len: int = 8192
    dropout: float = 0.1
    n_classes: int = 3  # [Win, Draw, Loss]
    rope_theta: float = 10000.0
    attention_mode: AttentionMode = "causal"
    # Maximal Update Parametrization (muP)
    use_mup: bool = True
    d_base: int = 128
    d_base_head: int = 32


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) with dynamic sequence caching."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.theta = float(theta)

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, head_dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, T, head_dim)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.cos_cached.numel() == 0
            or self.cos_cached.shape[2] < seq_len
            or self.cos_cached.device != device
        ):
            self.inv_freq = self.inv_freq.to(device)
            self._build_cache(max(seq_len, 1))
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors of shape (B, n_heads, T, head_dim)."""
    orig_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    q_rot = (qf * cos) + (_rotate_half(qf) * sin)
    k_rot = (kf * cos) + (_rotate_half(kf) * sin)
    return q_rot.to(orig_dtype), k_rot.to(orig_dtype)


class SelfAttentionV2(nn.Module):
    """Multi-head self-attention with RoPE, muP scaling, and causal/segment masks."""

    def __init__(self, config: ModelConfigV2) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
            )
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        if config.use_mup:
            self.attn_scale = math.sqrt(float(config.d_base_head)) / float(self.head_dim)
        else:
            self.attn_scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=self.attn_scale,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlockV2(nn.Module):
    """Pre-norm Transformer block with RoPE attention and GELU MLP."""

    def __init__(self, config: ModelConfigV2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttentionV2(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ff(self.ln2(x))
        return x


class ChessTransformerV2(nn.Module):
    """Decoder-only Transformer v2 for W/D/L soft-target evaluation with RoPE and muP."""

    def __init__(self, config: ModelConfigV2, pad_token_id: int = 0) -> None:
        super().__init__()
        if config.attention_mode not in ("causal", "segment_bidirectional"):
            raise ValueError(f"Unsupported attention_mode: {config.attention_mode!r}")

        self.config = config
        self.pad_token_id = pad_token_id

        head_dim = config.d_model // config.n_heads
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlockV2(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_classes)

        self._init_all_weights()

    def _init_all_weights(self) -> None:
        r = float(self.config.d_model) / float(self.config.d_base)
        hidden_std = (0.02 / math.sqrt(r)) if self.config.use_mup else 0.02

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], bool, torch.Tensor]:
        """Construct (attn_mask, is_causal, valid_mask) for the configured attention mode."""
        B, T = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            valid_mask = input_ids != self.pad_token_id
        else:
            valid_mask = attention_mask.bool()

        mode = self.config.attention_mode
        if mode == "causal":
            if valid_mask.all():
                return None, True, valid_mask

            causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
            allow = causal.unsqueeze(0).unsqueeze(0) & valid_mask[:, None, None, :]
            diag = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
            query_pad = (~valid_mask)[:, None, :, None]
            allow = allow | (query_pad & diag)
            return allow, False, valid_mask

        # segment_bidirectional
        if segment_ids is None:
            raise ValueError(
                "segment_ids must be provided when attention_mode='segment_bidirectional'"
            )
        if segment_ids.shape != (B, T):
            raise ValueError(
                f"segment_ids shape {tuple(segment_ids.shape)} must match input_ids {(B, T)}"
            )

        seg_q = segment_ids[:, None, :, None]
        seg_k = segment_ids[:, None, None, :]
        pos = torch.arange(T, device=device)
        strict_causal_pos = (pos[None, :] < pos[:, None]).unsqueeze(0).unsqueeze(0)

        same_seg = seg_k == seg_q
        earlier_seg = (seg_k < seg_q) & strict_causal_pos
        allow = (same_seg | earlier_seg) & valid_mask[:, None, None, :]

        diag = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
        query_pad = (~valid_mask)[:, None, :, None]
        allow = allow | (query_pad & diag)
        return allow, False, valid_mask

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run transformer backbone and return (hidden_states, valid_mask)."""
        _, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        attn_mask, is_causal, valid_mask = self._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        cos, sin = self.rope(T, input_ids.device)

        x = self.drop(self.token_emb(input_ids))
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask=attn_mask, is_causal=is_causal)
        x = self.ln_f(x)
        return x, valid_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, 3) classification logits read out at the last non-pad token."""
        B, T = input_ids.shape
        device = input_ids.device

        x, valid_mask = self.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        last_idx = positions.masked_fill(~valid_mask, 0).max(dim=-1).values
        batch_idx = torch.arange(B, device=device)
        last_hidden = x[batch_idx, last_idx]  # (B, d_model)

        logits = self.head(last_hidden)  # (B, 3)
        if self.config.use_mup:
            logits = logits * (float(self.config.d_base) / float(self.config.d_model))
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, 3) predicted W/D/L probabilities."""
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        return F.softmax(logits, dim=-1)

    @staticmethod
    def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """Soft-target cross-entropy: -(target_probs * log_softmax(logits)).sum(-1).mean()."""
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_probs).sum(dim=-1).mean()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft-target CE for (B, 3) target_probs or hard CE for (B,) indices."""
        if targets.ndim == 2:
            return self.soft_cross_entropy(logits, targets)
        return F.cross_entropy(logits, targets)

    def get_optimizer(self, base_lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
        """Construct AdamW with exact muP LR scaling (lr/r for hidden, lr for emb/ln/head)."""
        if not self.config.use_mup:
            decay_params = []
            nodecay_params = []
            for _, p in self.named_parameters():
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

        r = float(self.config.d_model) / float(self.config.d_base)
        embedding_params = []
        hidden_decay = []
        hidden_nodecay = []
        ln_params = []
        head_decay = []
        head_nodecay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "token_emb" in name:
                embedding_params.append(param)
            elif "head" in name:
                if param.dim() >= 2:
                    head_decay.append(param)
                else:
                    head_nodecay.append(param)
            elif "ln" in name:
                ln_params.append(param)
            else:
                if param.dim() >= 2:
                    hidden_decay.append(param)
                else:
                    hidden_nodecay.append(param)

        param_groups = [
            {"params": embedding_params, "lr": base_lr, "weight_decay": 0.0},
            {"params": hidden_decay, "lr": base_lr / r, "weight_decay": weight_decay},
            {"params": hidden_nodecay, "lr": base_lr / r, "weight_decay": 0.0},
            {"params": ln_params, "lr": base_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0},
        ]
        # Filter empty groups cleanly while keeping deterministic index structure when non-empty
        param_groups = [g for g in param_groups if g["params"]]
        return torch.optim.AdamW(param_groups)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "pad_token_id": self.pad_token_id,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "ChessTransformerV2":
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=map_location)
        cfg_data = ckpt["config"]
        if isinstance(cfg_data, dict):
            valid_keys = {f.name for f in ModelConfigV2.__dataclass_fields__.values()}
            filtered = {k: v for k, v in cfg_data.items() if k in valid_keys}
            config = ModelConfigV2(**filtered)
        else:
            config = cfg_data
        pad_token_id = ckpt.get("pad_token_id", 0)
        model = cls(config, pad_token_id=pad_token_id)
        model.load_state_dict(ckpt["state_dict"])
        return model


TINY = ModelConfigV2(
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    d_base=128,
    d_base_head=32,
    max_seq_len=8192,
)
SMALL = ModelConfigV2(
    d_model=256,
    n_heads=8,
    n_layers=8,
    d_ff=1024,
    d_base=128,
    d_base_head=32,
    max_seq_len=8192,
)
MEDIUM = ModelConfigV2(
    d_model=384,
    n_heads=8,
    n_layers=12,
    d_ff=1536,
    d_base=128,
    d_base_head=32,
    max_seq_len=8192,
)
LARGE = ModelConfigV2(
    d_model=1024,
    n_heads=16,
    n_layers=16,
    d_ff=4096,
    d_base=128,
    d_base_head=32,
    max_seq_len=8192,
)

__all__ = [
    "AttentionMode",
    "ModelConfigV2",
    "RotaryEmbedding",
    "SelfAttentionV2",
    "TransformerBlockV2",
    "ChessTransformerV2",
    "apply_rotary_pos_emb",
    "TINY",
    "SMALL",
    "MEDIUM",
    "LARGE",
]
