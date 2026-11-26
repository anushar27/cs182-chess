"""From-scratch 494M Qwen2-style causal Transformer + 3-way probability head."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model

from .qwen_model import (
    QwenClassifierOutput,
    extract_last_non_pad_hidden,
    soft_cross_entropy,
)

SCRATCH_QWEN2_VOCAB_SIZE = 152064
SCRATCH_QWEN2_HIDDEN_SIZE = 896
SCRATCH_QWEN2_INTERMEDIATE_SIZE = 4864
SCRATCH_QWEN2_NUM_LAYERS = 24
SCRATCH_QWEN2_NUM_ATTENTION_HEADS = 14
SCRATCH_QWEN2_NUM_KEY_VALUE_HEADS = 2
SCRATCH_QWEN2_MAX_POSITION_EMBEDDINGS = 32768
SCRATCH_QWEN2_RMS_NORM_EPS = 1e-6
SCRATCH_QWEN2_ROPE_THETA = 1000000.0


def create_scratch_qwen2_config(
    vocab_size: int = SCRATCH_QWEN2_VOCAB_SIZE,
    hidden_size: int = SCRATCH_QWEN2_HIDDEN_SIZE,
    intermediate_size: int = SCRATCH_QWEN2_INTERMEDIATE_SIZE,
    num_hidden_layers: int = SCRATCH_QWEN2_NUM_LAYERS,
    num_attention_heads: int = SCRATCH_QWEN2_NUM_ATTENTION_HEADS,
    num_key_value_heads: int = SCRATCH_QWEN2_NUM_KEY_VALUE_HEADS,
    max_position_embeddings: int = SCRATCH_QWEN2_MAX_POSITION_EMBEDDINGS,
    rms_norm_eps: float = SCRATCH_QWEN2_RMS_NORM_EPS,
    rope_theta: float = SCRATCH_QWEN2_ROPE_THETA,
    hidden_act: str = "silu",
    attention_dropout: float = 0.0,
    use_cache: bool = False,
    attn_implementation: str = "sdpa",
    initializer_range: float = 0.02,
    tie_word_embeddings: bool = False,
) -> Qwen2Config:
    """Build the Qwen2Config for the 494M from-scratch causal Transformer (no LM head)."""
    return Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act=hidden_act,
        attention_dropout=attention_dropout,
        use_cache=use_cache,
        attn_implementation=attn_implementation,
        initializer_range=initializer_range,
        tie_word_embeddings=tie_word_embeddings,
    )


class ScratchQwen2Classifier(nn.Module):
    """Randomly initialized Qwen2Model backbone (no lm_head) + FP32 3-way W/D/L head.

    All master network parameters reside in FP32 (``torch.float32``). When executing on
    CUDA, the transformer forward pass runs under ``torch.amp.autocast("cuda", dtype=torch.bfloat16)``
    while the 3-way classification head and soft cross-entropy loss stay in FP32.
    """

    WEIGHTS_FILENAME = "scratch_qwen2_weights.pt"
    CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        config: Optional[Qwen2Config] = None,
        tokenizer: Any = None,
        torch_dtype: torch.dtype = torch.float32,
        head_dtype: torch.dtype = torch.float32,
        gradient_checkpointing: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.config: Qwen2Config = config if config is not None else create_scratch_qwen2_config()
        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "sdpa"
        self.config.use_cache = False

        self.tokenizer = tokenizer
        self.torch_dtype: torch.dtype = torch_dtype
        self.head_dtype: torch.dtype = head_dtype
        self.hidden_size: int = int(self.config.hidden_size)
        self.max_position_embeddings: int = int(self.config.max_position_embeddings)
        self.num_hidden_layers: int = int(self.config.num_hidden_layers)

        dev_ctx = torch.device(device) if device is not None else nullcontext()
        with dev_ctx:
            self.backbone = Qwen2Model(self.config)
            if torch_dtype != torch.float32:
                self.backbone = self.backbone.to(dtype=torch_dtype)
            self.classifier = nn.Linear(self.hidden_size, 3, dtype=head_dtype)

        # Ensure all master weights are trainable and in expected dtypes.
        for param in self.backbone.parameters():
            if param.dtype != torch_dtype:
                param.data = param.data.to(dtype=torch_dtype)
            param.requires_grad_(True)
        for param in self.classifier.parameters():
            if param.dtype != head_dtype:
                param.data = param.data.to(dtype=head_dtype)
            param.requires_grad_(True)

        self._gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self) -> None:
        """Enable non-reentrant activation checkpointing for long-context training."""
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
        self._gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        self._gradient_checkpointing_enabled = False

    def parameter_counts(self) -> dict[str, int]:
        """Return exact counts of embedding, non-embedding/head, and total parameters."""
        embedding_params = sum(
            p.numel() for n, p in self.backbone.named_parameters() if "embed_tokens" in n
        )
        backbone_other_params = sum(
            p.numel() for n, p in self.backbone.named_parameters() if "embed_tokens" not in n
        )
        head_params = sum(p.numel() for p in self.classifier.parameters())
        other_params = backbone_other_params + head_params
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "embedding_params": embedding_params,
            "backbone_other_params": backbone_other_params,
            "head_params": head_params,
            "other_params": other_params,
            "total_params": total_params,
            "total_trainable": total_trainable,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_probs: Optional[torch.Tensor] = None,
    ) -> QwenClassifierOutput:
        """Run Qwen2Model backbone and classify last non-pad hidden state.

        Passes ``attention_mask=None`` directly when unpadded so SDPA uses the native
        causal kernel without materializing a dense quadratic mask.
        """
        use_cuda_autocast = input_ids.is_cuda and torch.cuda.is_available()
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda_autocast
            else nullcontext()
        )
        with autocast_ctx:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            last_hidden = extract_last_non_pad_hidden(outputs.last_hidden_state, attention_mask)

        head_autocast_off = (
            torch.amp.autocast("cuda", enabled=False)
            if use_cuda_autocast and self.classifier.weight.dtype == torch.float32
            else nullcontext()
        )
        with head_autocast_off:
            logits = self.classifier(last_hidden.to(dtype=self.classifier.weight.dtype))
            probs = F.softmax(logits.float(), dim=-1)
            loss = soft_cross_entropy(logits, target_probs) if target_probs is not None else None
        return QwenClassifierOutput(logits=logits, probs=probs, loss=loss)

    @torch.no_grad()
    def predict_proba(
        self,
        prompts: Sequence[str],
        batch_size: int = 4,
        max_length: Optional[int] = None,
        tokenizer: Any = None,
    ) -> np.ndarray:
        """Compute (N, 3) float32 outcome probabilities for formatted prompt strings."""
        if len(prompts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        tok = tokenizer if tokenizer is not None else self.tokenizer
        if tok is None:
            raise ValueError("predict_proba requires a tokenizer on the model or passed explicitly")

        eff_max_len = self.max_position_embeddings if max_length is None else max_length
        device = self.classifier.weight.device
        was_training = self.training
        self.eval()
        all_probs: list[np.ndarray] = []
        try:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = list(prompts[start : start + batch_size])
                encoded = tok(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                seq_len = int(encoded["input_ids"].shape[1])
                if seq_len > eff_max_len:
                    raise ValueError(
                        f"Batch sequence length {seq_len} exceeds max_length={eff_max_len}; "
                        "refusing to truncate prompt."
                    )
                input_ids = encoded["input_ids"].to(device)
                # Unpadded batch_size=1 fast path: attention_mask=None uses SDPA causal kernel.
                attention_mask = (
                    None
                    if len(batch_prompts) == 1
                    else encoded["attention_mask"].to(device)
                )
                out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
                all_probs.append(out.probs.detach().cpu().numpy().astype(np.float32))
        finally:
            if was_training:
                self.train()

        return np.concatenate(all_probs, axis=0)

    @classmethod
    def _resolve_weights_file(cls, path: Union[str, Path]) -> Path:
        p = Path(path)
        if p.is_dir() or p.suffix not in (".pt", ".bin"):
            return p / cls.WEIGHTS_FILENAME
        return p

    def save_weights(self, save_path: Union[str, Path]) -> Path:
        """Save FP32 model weights and Qwen2Config to a directory or checkpoint file."""
        p = Path(save_path)
        is_file_target = p.suffix in (".pt", ".bin")
        weights_file = p if is_file_target else (p / self.WEIGHTS_FILENAME)
        weights_file.parent.mkdir(parents=True, exist_ok=True)

        cfg_dict = self.config.to_dict()
        if not is_file_target:
            (p / self.CONFIG_FILENAME).write_text(
                json.dumps(cfg_dict, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        payload = {
            "model_type": "scratch_qwen2",
            "config": cfg_dict,
            "torch_dtype": str(self.torch_dtype),
            "head_dtype": str(self.head_dtype),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "parameter_counts": self.parameter_counts(),
        }
        torch.save(payload, weights_file)
        return p

    def load_weights(self, load_path: Union[str, Path]) -> None:
        """Load saved weights into this ScratchQwen2Classifier instance."""
        weights_file = self._resolve_weights_file(load_path)
        if not weights_file.exists():
            raise FileNotFoundError(f"ScratchQwen2 weights file not found: {weights_file}")
        ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
        state_dict: Mapping[str, torch.Tensor] = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=True)
        for param in self.backbone.parameters():
            if param.dtype != self.torch_dtype:
                param.data = param.data.to(dtype=self.torch_dtype)
        for param in self.classifier.parameters():
            if param.dtype != self.head_dtype:
                param.data = param.data.to(dtype=self.head_dtype)

    @classmethod
    def from_weights(
        cls,
        load_path: Union[str, Path],
        tokenizer: Any = None,
        gradient_checkpointing: bool = False,
        is_trainable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "ScratchQwen2Classifier":
        """Instantiate and load a ScratchQwen2Classifier from saved weights."""
        weights_file = cls._resolve_weights_file(load_path)
        if not weights_file.exists():
            raise FileNotFoundError(f"ScratchQwen2 weights file not found: {weights_file}")
        ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)

        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            cfg_file = weights_file.parent / cls.CONFIG_FILENAME
            if not cfg_file.exists():
                raise ValueError(f"Missing Qwen2Config in {weights_file} and {cfg_file}")
            cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))

        config = Qwen2Config.from_dict(cfg_dict)
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "sdpa"
        config.use_cache = False

        torch_dtype_str = ckpt.get("torch_dtype", "torch.float32")
        head_dtype_str = ckpt.get("head_dtype", "torch.float32")
        torch_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), torch.float32)
        head_dtype = getattr(torch, head_dtype_str.replace("torch.", ""), torch.float32)

        instance = cls(
            config=config,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            head_dtype=head_dtype,
            gradient_checkpointing=gradient_checkpointing,
        )
        state_dict: Mapping[str, torch.Tensor] = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        instance.load_state_dict(state_dict, strict=True)
        if device is not None:
            instance = instance.to(device=device)
        if not is_trainable:
            instance.eval()
            for param in instance.parameters():
                param.requires_grad_(False)
        return instance


__all__ = [
    "SCRATCH_QWEN2_HIDDEN_SIZE",
    "SCRATCH_QWEN2_INTERMEDIATE_SIZE",
    "SCRATCH_QWEN2_MAX_POSITION_EMBEDDINGS",
    "SCRATCH_QWEN2_NUM_ATTENTION_HEADS",
    "SCRATCH_QWEN2_NUM_KEY_VALUE_HEADS",
    "SCRATCH_QWEN2_NUM_LAYERS",
    "SCRATCH_QWEN2_RMS_NORM_EPS",
    "SCRATCH_QWEN2_ROPE_THETA",
    "SCRATCH_QWEN2_VOCAB_SIZE",
    "ScratchQwen2Classifier",
    "create_scratch_qwen2_config",
]
