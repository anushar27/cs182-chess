"""ChessBot with Dual Policy-Value Heads and AlphaZero-style MSE Loss."""

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_icl.model import ModelConfig, TransformerBlock


class ChessBotPolicyValue(nn.Module):
    """Causal Transformer with dual heads:
    1. Policy Head: Next-token prediction over chess moves / vocabulary.
    2. Value Head: Continuous position evaluation v in [-1.0, 1.0] (AlphaZero formulation).
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

        # 1. Next-token policy head
        self.policy_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights with token embedding if standard practice, or separate head
        self.policy_head.weight = self.token_emb.weight

        # 2. AlphaZero continuous value head: outputs v in [-1.0, 1.0]
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, max(64, config.d_model // 2)),
            nn.GELU(),
            nn.Linear(max(64, config.d_model // 2), 1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.use_mup:
                r = float(self.config.d_model) / float(self.config.d_base)
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
        return_policy: bool = True,
        return_value: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Args:
            input_ids: (B, T) token indices
        Returns:
            policy_logits: (B, T, vocab_size) if return_policy else None
            value: (B, 1) scalar in [-1, 1] gathered at last non-pad token if return_value else None
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        policy_logits = None
        if return_policy:
            policy_logits = self.policy_head(x)

        value = None
        if return_value:
            # Find last non-pad token per sequence for value prediction
            is_pad = (input_ids == self.pad_token_id)
            lengths = (~is_pad).long().sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            last_tokens = x[torch.arange(B, device=device), lengths]
            value = self.value_head(last_tokens)

        return policy_logits, value

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_value: torch.Tensor,
        value_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AlphaZero joint loss: Policy CrossEntropy + Value MSE.

        Loss = Policy_CE + value_weight * MSE(v, z)
        """
        policy_logits, value = self.forward(input_ids, return_policy=True, return_value=True)

        # 1. Policy next-token prediction loss
        B, T, V = policy_logits.shape
        # Flatten for cross-entropy
        shift_logits = policy_logits.view(-1, V)
        shift_targets = target_token_ids.view(-1)
        policy_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)

        # 2. AlphaZero value MSE loss
        value_mse = F.mse_loss(value.squeeze(-1), target_value.float())

        total_loss = policy_loss + value_weight * value_mse
        return total_loss, policy_loss, value_mse

    def get_optimizer(self, base_lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        r = float(self.config.d_model) / float(self.config.d_base) if self.config.use_mup else 1.0
        decay_params = []
        no_decay_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "value_head" in name:
                head_params.append(param)
            elif param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "lr": base_lr / r, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": base_lr / r, "weight_decay": 0.0},
            {"params": head_params, "lr": base_lr, "weight_decay": 0.0},
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
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "ChessBotPolicyValue":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg_data = ckpt["config"]
        if isinstance(cfg_data, dict):
            valid_keys = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            filtered_cfg = {k: v for k, v in cfg_data.items() if k in valid_keys}
            config = ModelConfig(**filtered_cfg)
        else:
            config = cfg_data
        pad_token_id = ckpt.get("pad_token_id", 0)
        model = cls(config, pad_token_id=pad_token_id)
        model.load_state_dict(ckpt["state_dict"])
        return model
