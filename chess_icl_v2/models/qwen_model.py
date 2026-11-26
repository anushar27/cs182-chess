"""Track 1: Qwen2.5-7B-Instruct LoRA + 3-way probability head for Chess-Variant ICL v2."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file as load_safetensors_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LOCAL_SNAPSHOT_PATH = Path(
    "/data/backups/rganapa/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

STANDARD_INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUTCOME_LABELS: tuple[str, str, str] = ("win", "draw", "loss")
LORA_HANDOFF_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VALID_PIECE_CHARS = frozenset("PNBRQKpnbrqk")
FILES = "abcdefgh"

LoraMode = Literal["handoff", "full_lora"]
Notation = Literal["uci", "san"]


def resolve_qwen_model_path(model_path: Optional[Union[str, Path]] = None) -> str:
    """Resolve the Qwen2.5-7B-Instruct model path, preferring the pinned local snapshot."""
    if model_path is not None:
        return str(model_path)
    if LOCAL_SNAPSHOT_PATH.exists():
        return str(LOCAL_SNAPSHOT_PATH)
    return MODEL_ID


def get_preferred_attn_implementation() -> str:
    """Return flash_attention_2 if installed, otherwise sdpa."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_qwen_tokenizer(
    model_path: Optional[Union[str, Path]] = None,
    local_files_only: bool = True,
) -> Any:
    """Load the Qwen2.5 tokenizer locally without network access."""
    resolved = resolve_qwen_model_path(model_path)
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if not Path(resolved).exists():
        kwargs["revision"] = REVISION
    tokenizer = AutoTokenizer.from_pretrained(resolved, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


@dataclass(frozen=True)
class DemoGame:
    """One full demonstration game (no variant name or rule fields)."""

    moves: Sequence[str]
    outcome: str
    initial_fen: str = STANDARD_INITIAL_FEN
    blocked_squares: Optional[Sequence[Union[int, str]]] = None


@dataclass(frozen=True)
class QueryPosition:
    """Query board state for outcome prediction (no ground-truth label)."""

    fen: str
    blocked_squares: Optional[Sequence[Union[int, str]]] = None


@dataclass(frozen=True)
class FormattedPrompt:
    """Formatted ICL prompt along with trimming/token accounting metadata."""

    text: str
    num_demos_total: int
    num_demos_kept: int
    num_demos_dropped: int
    num_tokens: Optional[int] = None


@dataclass
class QwenClassifierOutput:
    """Output container for QwenICLClassifier forward pass."""

    logits: torch.Tensor
    probs: torch.Tensor
    loss: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def normalize_outcome(outcome: str) -> str:
    """Normalize White-perspective realized game outcome to 'win', 'draw', or 'loss'."""
    key = str(outcome).strip().lower()
    mapping = {
        "win": "win",
        "w": "win",
        "1-0": "win",
        "white": "win",
        "draw": "draw",
        "d": "draw",
        "1/2-1/2": "draw",
        "loss": "loss",
        "l": "loss",
        "0-1": "loss",
        "black": "loss",
    }
    if key not in mapping:
        raise ValueError(f"Unrecognized game outcome: {outcome!r}")
    return mapping[key]


def parse_blocked_squares(
    blocked_squares: Optional[Sequence[Union[int, str]]],
) -> set[tuple[int, int]]:
    """Parse blocked squares (0..63 ints or algebraic 'a3'..'h6') into (rank, file) pairs."""
    coords: set[tuple[int, int]] = set()
    if not blocked_squares:
        return coords
    for sq in blocked_squares:
        if isinstance(sq, int):
            if not (0 <= sq < 64):
                raise ValueError(f"Blocked square index out of range [0, 63]: {sq}")
            coords.add((sq // 8, sq % 8))
        elif isinstance(sq, str):
            s = sq.strip().lower()
            if len(s) != 2 or s[0] not in FILES or s[1] not in "12345678":
                raise ValueError(f"Invalid algebraic square: {sq!r}")
            coords.add((int(s[1]) - 1, FILES.index(s[0])))
        else:
            raise TypeError(f"Unsupported blocked square type: {type(sq).__name__}")
    return coords


def format_board_state(
    fen: str,
    blocked_squares: Optional[Sequence[Union[int, str]]] = None,
) -> str:
    """Render a FEN + optional blocked squares into an explicit 8x8 ASCII grid + metadata."""
    parts = fen.strip().split()
    if not parts:
        raise ValueError("Empty FEN string")

    placement = parts[0]
    rank_strs = placement.split("/")
    if len(rank_strs) != 8:
        raise ValueError(f"FEN placement must have 8 ranks, got {len(rank_strs)}: {fen!r}")

    blocked_coords = parse_blocked_squares(blocked_squares)
    grid: list[list[str]] = []

    for row_idx, rank_str in enumerate(rank_strs):
        rank_num = 7 - row_idx  # FEN starts at rank 8 (index 7) down to rank 1 (index 0)
        row: list[str] = []
        col_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                count = int(ch)
                for _ in range(count):
                    cell = "X" if (rank_num, col_idx) in blocked_coords else "."
                    row.append(cell)
                    col_idx += 1
            elif ch in ("X", "x"):
                row.append("X")
                col_idx += 1
            elif ch in VALID_PIECE_CHARS:
                cell = "X" if (rank_num, col_idx) in blocked_coords else ch
                row.append(cell)
                col_idx += 1
            else:
                raise ValueError(f"Unexpected character {ch!r} in FEN placement: {fen!r}")
        if col_idx != 8:
            raise ValueError(f"FEN rank {rank_num + 1} has {col_idx} files instead of 8: {fen!r}")
        grid.append(row)

    side_char = parts[1].lower() if len(parts) > 1 else "w"
    side_to_move = "White" if side_char == "w" else "Black" if side_char == "b" else side_char
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"

    lines = [f"{8 - idx} | {' '.join(row)}" for idx, row in enumerate(grid)]
    lines.append("    a b c d e f g h")
    lines.append(f"Side to move: {side_to_move}")
    lines.append(f"Castling: {castling}")
    lines.append(f"En passant: {ep}")
    return "\n".join(lines)


def _extract_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class QwenPromptFormatter:
    """Zero-leakage, query-preserving prompt formatter for Chess-Variant ICL v2.

    Formats K complete demonstration games (initial board, move sequence in UCI or SAN,
    and White-perspective realized outcome) followed by the query board state.
    Never emits variant names, rule descriptions, or query labels.
    If the full sequence exceeds ``max_length``, oldest demonstrations are trimmed
    first while guaranteeing the query block is never truncated or dropped.
    """

    DEFAULT_HEADER = (
        "You are observing a two-player chess-like board game with unknown rules.\n"
        "Each demonstration game shows the initial board, the full move sequence, "
        "and the final outcome from White's perspective (win, draw, or loss).\n"
        "Based solely on the demonstrations, predict the outcome distribution "
        "(win, draw, loss) for the side to move in the query position."
    )

    def __init__(
        self,
        tokenizer: Any = None,
        notation: Notation = "uci",
        max_length: Optional[int] = 32768,
        header: Optional[str] = None,
        strict_no_trim: bool = False,
    ) -> None:
        if notation not in ("uci", "san"):
            raise ValueError(f"notation must be 'uci' or 'san', got {notation!r}")
        self.tokenizer = tokenizer
        self.notation: Notation = notation
        self.max_length = max_length
        self.header = self.DEFAULT_HEADER if header is None else header
        self.strict_no_trim = bool(strict_no_trim)

    def _get_demo_moves(self, demo: Any) -> Sequence[str]:
        preferred_key = f"moves_{self.notation}"
        moves = _extract_field(demo, preferred_key, None)
        if moves is None:
            moves = _extract_field(demo, "moves", None)
        if moves is None:
            raise ValueError("Demo object must provide 'moves' (or 'moves_uci'/'moves_san')")
        return moves

    def format_demo(self, demo: Union[DemoGame, Mapping[str, Any], Any]) -> str:
        """Format a single full-game demonstration without any variant/rule metadata."""
        initial_fen = _extract_field(demo, "initial_fen", None)
        if initial_fen is None:
            initial_fen = _extract_field(demo, "fen", STANDARD_INITIAL_FEN)
        blocked = _extract_field(demo, "blocked_squares", None)
        moves = self._get_demo_moves(demo)
        raw_outcome = _extract_field(demo, "outcome", None)
        if raw_outcome is None:
            raw_outcome = _extract_field(demo, "result", None)
        if raw_outcome is None:
            raise ValueError("Demo object must provide 'outcome' (or 'result')")
        outcome = normalize_outcome(raw_outcome)

        moves_str = " ".join(str(m) for m in moves) if moves else "(none)"
        board_str = format_board_state(initial_fen, blocked_squares=blocked)
        return (
            "[Demo]\n"
            f"Initial:\n{board_str}\n"
            f"Moves ({self.notation}): {moves_str}\n"
            f"Outcome (White): {outcome}"
        )

    def format_query(self, query: Union[QueryPosition, Mapping[str, Any], str, Any]) -> str:
        """Format the query board state without ever reading or emitting any label."""
        if isinstance(query, str):
            fen = query
            blocked = None
        else:
            fen = _extract_field(query, "fen", None)
            if fen is None:
                raise ValueError("Query object must provide 'fen'")
            blocked = _extract_field(query, "blocked_squares", None)

        board_str = format_board_state(fen, blocked_squares=blocked)
        return (
            "[Query]\n"
            f"{board_str}\n"
            "Predict outcome for side to move (win, draw, loss):"
        )

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        encoded = self.tokenizer(text, add_special_tokens=True)
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        return len(input_ids)

    @staticmethod
    def _assemble_prompt(
        header: Optional[str],
        demo_blocks: Sequence[str],
        query_block: str,
    ) -> str:
        sections: list[str] = []
        if header:
            sections.append(header)
        sections.extend(demo_blocks)
        sections.append(query_block)
        return "\n\n".join(sections)

    def format_prompt_with_info(
        self,
        demos: Sequence[Any],
        query: Any,
        max_length: Optional[int] = None,
        strict_no_trim: Optional[bool] = None,
    ) -> FormattedPrompt:
        """Format prompt and trim oldest demos if needed, never truncating the query.

        When ``strict_no_trim`` is True, never drop demos or trim; raise ``ValueError``
        immediately if the full prompt exceeds ``max_length``.
        """
        eff_max_length = self.max_length if max_length is None else max_length
        eff_strict = self.strict_no_trim if strict_no_trim is None else bool(strict_no_trim)
        demo_blocks = [self.format_demo(d) for d in demos]
        query_block = self.format_query(query)
        total_demos = len(demo_blocks)

        if self.tokenizer is None or eff_max_length is None:
            text = self._assemble_prompt(self.header, demo_blocks, query_block)
            return FormattedPrompt(
                text=text,
                num_demos_total=total_demos,
                num_demos_kept=total_demos,
                num_demos_dropped=0,
                num_tokens=None,
            )

        # Fast path: all demos fit.
        full_text = self._assemble_prompt(self.header, demo_blocks, query_block)
        full_tokens = self._count_tokens(full_text)
        if full_tokens <= eff_max_length:
            return FormattedPrompt(
                text=full_text,
                num_demos_total=total_demos,
                num_demos_kept=total_demos,
                num_demos_dropped=0,
                num_tokens=full_tokens,
            )

        if eff_strict:
            raise ValueError(
                f"Formatted prompt requires {full_tokens} tokens, exceeding "
                f"max_length={eff_max_length} with strict_no_trim=True; "
                "refusing to drop demonstrations or trim query/games."
            )

        # Binary search for minimum number of oldest demos to drop.
        if total_demos > 0:
            lo, hi = 1, total_demos
            best_start = total_demos
            best_text = self._assemble_prompt(self.header, [], query_block)
            best_tokens = self._count_tokens(best_text)

            while lo <= hi:
                mid = (lo + hi) // 2
                cand_text = self._assemble_prompt(self.header, demo_blocks[mid:], query_block)
                cand_tokens = self._count_tokens(cand_text)
                if cand_tokens <= eff_max_length:
                    best_start = mid
                    best_text = cand_text
                    best_tokens = cand_tokens
                    hi = mid - 1
                else:
                    lo = mid + 1

            # Guard against +/-1 BPE merge non-monotonicity near the boundary.
            while best_start > 0:
                prev_text = self._assemble_prompt(
                    self.header, demo_blocks[best_start - 1 :], query_block
                )
                prev_tokens = self._count_tokens(prev_text)
                if prev_tokens <= eff_max_length:
                    best_start -= 1
                    best_text = prev_text
                    best_tokens = prev_tokens
                else:
                    break

            if best_tokens <= eff_max_length:
                return FormattedPrompt(
                    text=best_text,
                    num_demos_total=total_demos,
                    num_demos_kept=total_demos - best_start,
                    num_demos_dropped=best_start,
                    num_tokens=best_tokens,
                )

        # Zero demos still exceeded max_length -> drop header before ever touching query.
        query_only_tokens = self._count_tokens(query_block)
        if query_only_tokens <= eff_max_length:
            return FormattedPrompt(
                text=query_block,
                num_demos_total=total_demos,
                num_demos_kept=0,
                num_demos_dropped=total_demos,
                num_tokens=query_only_tokens,
            )

        raise ValueError(
            f"Query block alone requires {query_only_tokens} tokens, exceeding "
            f"max_length={eff_max_length}; refusing to truncate query."
        )

    def format_prompt(
        self,
        demos: Sequence[Any],
        query: Any,
        max_length: Optional[int] = None,
    ) -> str:
        """Format prompt string, trimming oldest demos if sequence exceeds max_length."""
        return self.format_prompt_with_info(demos=demos, query=query, max_length=max_length).text


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Compute soft-target cross-entropy: -(p * log_softmax(logits)).sum(-1).mean()."""
    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError(f"Expected logits of shape (B, 3), got {tuple(logits.shape)}")
    if target_probs.shape != logits.shape:
        raise ValueError(
            f"target_probs shape {tuple(target_probs.shape)} != logits shape {tuple(logits.shape)}"
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(target_probs.float() * log_probs).sum(dim=-1).mean()


def build_lora_config(
    lora_mode: LoraMode = "handoff",
    num_hidden_layers: int = 28,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.0,
) -> LoraConfig:
    """Create the PEFT LoraConfig for Mode A ('handoff') or Mode B ('full_lora')."""
    if lora_mode == "handoff":
        r = 8 if lora_r is None else lora_r
        alpha = 16 if lora_alpha is None else lora_alpha
        start_layer = max(0, num_hidden_layers - 4)
        layers_to_transform: Optional[list[int]] = list(range(start_layer, num_hidden_layers))
        target_modules = list(LORA_HANDOFF_MODULES)
    elif lora_mode == "full_lora":
        r = 16 if lora_r is None else lora_r
        alpha = 32 if lora_alpha is None else lora_alpha
        layers_to_transform = None
        target_modules = list(LORA_TARGET_MODULES)
    else:
        raise ValueError(f"Unsupported lora_mode: {lora_mode!r}; expected 'handoff' or 'full_lora'")

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def extract_last_non_pad_hidden(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather the hidden state at the last non-pad token position for each batch item.

    Works identically for right-padded, left-padded, and unpadded sequences.
    """
    batch_size, seq_len, _ = hidden_states.shape
    if attention_mask is None:
        return hidden_states[:, -1, :]

    positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    valid_positions = torch.where(attention_mask > 0, positions, -1)
    last_indices = valid_positions.max(dim=1).values.clamp(min=0)
    batch_indices = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


class QwenICLClassifier(nn.Module):
    """Qwen2.5-7B-Instruct LoRA classifier with a 3-way W/D/L head on the last token."""

    HEAD_FILENAME = "classifier_head.pt"

    def __init__(
        self,
        base_model: Optional[Qwen2ForCausalLM] = None,
        tokenizer: Any = None,
        model_path: Optional[Union[str, Path]] = None,
        lora_mode: LoraMode = "handoff",
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        head_dtype: torch.dtype = torch.float32,
        adapter_dtype: torch.dtype = torch.float32,
        attn_implementation: Optional[str] = None,
        local_files_only: bool = True,
        gradient_checkpointing: bool = False,
        _peft_wrapped: bool = False,
    ) -> None:
        super().__init__()
        self.lora_mode: LoraMode = lora_mode
        self.tokenizer = tokenizer
        self.model_path = resolve_qwen_model_path(model_path)
        self.head_dtype: torch.dtype = head_dtype
        self.adapter_dtype: torch.dtype = adapter_dtype

        if base_model is None:
            eff_attn = attn_implementation or get_preferred_attn_implementation()
            load_kwargs: dict[str, Any] = {
                "local_files_only": local_files_only,
                "torch_dtype": torch_dtype,
                "attn_implementation": eff_attn,
            }
            if not Path(self.model_path).exists():
                load_kwargs["revision"] = REVISION
            raw_model = Qwen2ForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        else:
            raw_model = base_model

        self.config = raw_model.config
        self.hidden_size: int = int(self.config.hidden_size)
        self.max_position_embeddings: int = int(self.config.max_position_embeddings)
        self.num_hidden_layers: int = int(self.config.num_hidden_layers)

        if _peft_wrapped:
            self.backbone: PeftModel = raw_model  # type: ignore[assignment]
        else:
            self.lora_config = build_lora_config(
                lora_mode=lora_mode,
                num_hidden_layers=self.num_hidden_layers,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.backbone = get_peft_model(raw_model, self.lora_config)

        # Ensure base weights (including lm_head & embeddings) remain frozen,
        # and trainable LoRA adapter weights are stored in FP32 (adapter_dtype).
        self._cast_and_freeze_backbone(preserve_lora_requires_grad=_peft_wrapped)

        model_device = next(self.backbone.parameters()).device
        self.classifier = nn.Linear(self.hidden_size, 3, dtype=head_dtype, device=model_device)
        self._gradient_checkpointing_enabled = False

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _cast_and_freeze_backbone(self, preserve_lora_requires_grad: bool = False) -> None:
        """Freeze non-LoRA backbone weights and ensure LoRA weights use ``self.adapter_dtype``."""
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                if param.dtype != self.adapter_dtype:
                    param.data = param.data.to(dtype=self.adapter_dtype)
                if not preserve_lora_requires_grad:
                    param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def _get_transformer_backbone(self) -> nn.Module:
        """Return Qwen2Model (bypassing the 152k-vocab lm_head)."""
        base = (
            self.backbone.get_base_model()
            if hasattr(self.backbone, "get_base_model")
            else self.backbone
        )
        return base.model if hasattr(base, "model") else base

    def gradient_checkpointing_enable(self) -> None:
        """Enable activation/gradient checkpointing for long-context training."""
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        self._gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        self._gradient_checkpointing_enabled = False

    def parameter_counts(self) -> dict[str, int]:
        """Return counts of trainable LoRA, trainable head, total trainable, and total params."""
        lora_trainable = sum(
            p.numel()
            for n, p in self.backbone.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        head_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "lora_trainable": lora_trainable,
            "head_trainable": head_trainable,
            "total_trainable": total_trainable,
            "total_params": total_params,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_probs: Optional[torch.Tensor] = None,
    ) -> QwenClassifierOutput:
        """Run Qwen2Model backbone (without LM head) and classify last non-pad hidden state."""
        transformer = self._get_transformer_backbone()
        use_cuda_autocast = input_ids.is_cuda and torch.cuda.is_available()
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda_autocast
            else nullcontext()
        )
        with autocast_ctx:
            outputs = transformer(
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
                        "format prompts with QwenPromptFormatter to trim demos without truncating query."
                    )
                input_ids = encoded["input_ids"].to(device)
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

    def save_adapter_and_head(self, save_dir: Union[str, Path]) -> Path:
        """Deterministically save the LoRA adapter and 3-way classification head."""
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(out_dir))
        head_payload = {
            "head_state_dict": {
                k: v.detach().cpu() for k, v in self.classifier.state_dict().items()
            },
            "lora_mode": self.lora_mode,
            "hidden_size": self.hidden_size,
            "head_dtype": str(self.classifier.weight.dtype),
            "adapter_dtype": str(self.adapter_dtype),
            "model_id": MODEL_ID,
            "revision": REVISION,
        }
        torch.save(head_payload, out_dir / self.HEAD_FILENAME)
        return out_dir

    def load_adapter_and_head(self, save_dir: Union[str, Path]) -> None:
        """Load saved LoRA adapter and classification head weights into this instance."""
        in_dir = Path(save_dir)
        safetensors_path = in_dir / "adapter_model.safetensors"
        bin_path = in_dir / "adapter_model.bin"
        if safetensors_path.exists():
            adapter_sd = load_safetensors_file(str(safetensors_path), device="cpu")
        elif bin_path.exists():
            adapter_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {in_dir}")

        set_peft_model_state_dict(self.backbone, adapter_sd)
        self._cast_and_freeze_backbone(preserve_lora_requires_grad=True)

        head_ckpt = torch.load(in_dir / self.HEAD_FILENAME, map_location="cpu", weights_only=True)
        self.classifier.load_state_dict(head_ckpt["head_state_dict"])
        self.classifier.to(dtype=self.head_dtype)

    @classmethod
    def from_adapter_and_head(
        cls,
        save_dir: Union[str, Path],
        base_model: Optional[Qwen2ForCausalLM] = None,
        tokenizer: Any = None,
        model_path: Optional[Union[str, Path]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        adapter_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        local_files_only: bool = True,
        is_trainable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "QwenICLClassifier":
        """Reload a fresh QwenICLClassifier from a saved adapter + classification head directory."""
        in_dir = Path(save_dir)
        head_ckpt = torch.load(in_dir / cls.HEAD_FILENAME, map_location="cpu", weights_only=True)
        adapter_cfg = json.loads((in_dir / "adapter_config.json").read_text())

        if base_model is None:
            resolved = resolve_qwen_model_path(model_path)
            eff_attn = attn_implementation or get_preferred_attn_implementation()
            load_kwargs: dict[str, Any] = {
                "local_files_only": local_files_only,
                "torch_dtype": torch_dtype,
                "attn_implementation": eff_attn,
            }
            if not Path(resolved).exists():
                load_kwargs["revision"] = REVISION
            raw_model = Qwen2ForCausalLM.from_pretrained(resolved, **load_kwargs)
        else:
            raw_model = base_model

        if device is not None:
            raw_model = raw_model.to(device)

        peft_model = PeftModel.from_pretrained(raw_model, str(in_dir), is_trainable=is_trainable)
        head_dtype_str = head_ckpt.get("head_dtype", "torch.float32")
        head_dtype = getattr(torch, head_dtype_str.replace("torch.", ""), torch.float32)
        if adapter_dtype is None:
            adapter_dtype_str = head_ckpt.get("adapter_dtype", "torch.float32")
            eff_adapter_dtype = getattr(torch, adapter_dtype_str.replace("torch.", ""), torch.float32)
        else:
            eff_adapter_dtype = adapter_dtype

        instance = cls(
            base_model=peft_model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            model_path=model_path,
            lora_mode=head_ckpt.get("lora_mode", "handoff"),
            lora_r=adapter_cfg.get("r"),
            lora_alpha=adapter_cfg.get("lora_alpha"),
            head_dtype=head_dtype,
            adapter_dtype=eff_adapter_dtype,
            _peft_wrapped=True,
        )
        instance.classifier.load_state_dict(head_ckpt["head_state_dict"])
        instance.classifier.to(
            device=next(peft_model.parameters()).device,
            dtype=head_dtype,
        )
        if not is_trainable:
            instance.eval()
        return instance


def is_no_weight_decay_param(name: str, param: torch.Tensor) -> bool:
    """Return True if a parameter (1D/scalar, norm, or bias) must be excluded from weight decay."""
    lower = name.lower()
    return param.ndim < 2 or "norm" in lower or "ln" in lower or "bias" in lower


def build_long_study_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW with betas=(0.9, 0.95) and weight-decay exclusions for norms & biases."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_weight_decay_param(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if not param_groups:
        raise ValueError("build_long_study_optimizer received a model with no trainable parameters")

    return torch.optim.AdamW(
        param_groups,
        lr=float(lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
    )


def build_cosine_warmup_to_min_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: Optional[int] = None,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_steps`` (default 5%) then cosine decay to ``min_lr_ratio * peak_lr``."""
    eff_total = max(1, int(total_steps))
    if warmup_steps is None:
        eff_warmup = max(1, int(round(0.05 * eff_total))) if eff_total > 1 else 0
    else:
        eff_warmup = max(0, min(int(warmup_steps), eff_total))

    min_ratio = float(min_lr_ratio)
    if not (0.0 <= min_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0.0, 1.0], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        if eff_warmup > 0 and step < eff_warmup:
            return float(step + 1) / float(eff_warmup)
        if eff_total <= eff_warmup:
            return 1.0
        progress = float(step - eff_warmup) / float(max(1, eff_total - eff_warmup))
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
