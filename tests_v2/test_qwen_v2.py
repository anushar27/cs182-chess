"""Sandboxed CPU unit tests for chess_icl_v2.models.qwen_model (no HF cache access)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from chess_icl_v2.models.qwen_model import (
    MODEL_ID,
    REVISION,
    STANDARD_INITIAL_FEN,
    DemoGame,
    QueryPosition,
    QwenICLClassifier,
    QwenPromptFormatter,
    build_lora_config,
    extract_last_non_pad_hidden,
    format_board_state,
    normalize_outcome,
    resolve_qwen_model_path,
    soft_cross_entropy,
)


class MockWhitespaceTokenizer:
    """Deterministic lightweight tokenizer for sandboxed unit tests."""

    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {self.pad_token: 0, self.eos_token: 1}

    def _tok_id(self, tok: str) -> int:
        if tok not in self._vocab:
            # Keep IDs within tiny Qwen2 test vocab [0, 255] via deterministic hash mod
            self._vocab[tok] = 2 + (sum(ord(c) for c in tok) % 250)
        return self._vocab[tok]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self._tok_id(tok) for tok in text.split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)
        encoded = [self.encode(t, add_special_tokens=add_special_tokens) for t in batch]
        if is_single and return_tensors is None:
            return {"input_ids": encoded[0], "attention_mask": [1] * len(encoded[0])}

        max_len = max(len(seq) for seq in encoded) if padding else len(encoded[0])
        padded_ids = []
        attn_masks = []
        for seq in encoded:
            pad_n = max_len - len(seq)
            padded_ids.append(seq + [self.pad_token_id] * pad_n)
            attn_masks.append([1] * len(seq) + [0] * pad_n)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": attn_masks}


def _make_tiny_qwen(num_hidden_layers: int = 6, hidden_size: int = 32) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_cache=False,
    )
    torch.manual_seed(1234)
    return Qwen2ForCausalLM(cfg)


def test_constants_and_path_resolution(tmp_path: Path) -> None:
    assert MODEL_ID == "Qwen/Qwen2.5-7B-Instruct"
    assert REVISION == "a09a35458c702b33eeacc393d103063234e8bc28"
    custom = tmp_path / "my_qwen"
    assert resolve_qwen_model_path(custom) == str(custom)


def test_board_rendering_with_blocked_squares() -> None:
    # Block d3 (square 19: rank=2, file=3) and e5 (algebraic)
    rendered = format_board_state(STANDARD_INITIAL_FEN, blocked_squares=[19, "e5"])
    lines = rendered.splitlines()
    assert lines[0] == "8 | r n b q k b n r"
    assert lines[3] == "5 | . . . . X . . ."
    assert lines[5] == "3 | . . . X . . . ."
    assert "Side to move: White" in rendered
    assert "Castling: KQkq" in rendered
    assert "En passant: -" in rendered

    # Also support 'X' directly inside FEN placement
    fen_with_x = "rnbqkbnr/pppppppp/8/4X3/8/3X4/PPPPPPPP/RNBQKBNR b - e3 0 1"
    rendered_x = format_board_state(fen_with_x)
    assert "5 | . . . . X . . ." in rendered_x
    assert "3 | . . . X . . . ." in rendered_x
    assert "Side to move: Black" in rendered_x
    assert "En passant: e3" in rendered_x


def test_outcome_normalization() -> None:
    assert normalize_outcome("win") == "win"
    assert normalize_outcome("1-0") == "win"
    assert normalize_outcome("Draw") == "draw"
    assert normalize_outcome("1/2-1/2") == "draw"
    assert normalize_outcome("LOSS") == "loss"
    assert normalize_outcome("0-1") == "loss"
    with pytest.raises(ValueError):
        normalize_outcome("unknown")


def test_prompt_formatter_zero_leakage_and_notations() -> None:
    tok = MockWhitespaceTokenizer()
    demos = [
        {
            "initial_fen": STANDARD_INITIAL_FEN,
            "moves": ["e2e4", "e7e5", "g1f3"],
            "moves_san": ["e4", "e5", "Nf3"],
            "outcome": "win",
            "variant": "diagonal_rook_SECRET",
            "rule": "Rooks move diagonally only!",
            "description": "LEAK_TEST_DESCRIPTION",
        },
        SimpleNamespace(
            initial_fen=STANDARD_INITIAL_FEN,
            moves=["d2d4", "d7d5"],
            moves_san=["d4", "d5"],
            outcome="draw",
            variant="antichess_SECRET",
            rule="Lose all pieces to win",
        ),
    ]
    query = {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "blocked_squares": ["c4"],
        "variant": "blocked_squares_SECRET",
        "label": "QUERY_LABEL_LEAK",
        "target_probs": [0.9, 0.05, 0.05],
    }

    for notation, expected_moves in [("uci", "e2e4 e7e5 g1f3"), ("san", "e4 e5 Nf3")]:
        fmt = QwenPromptFormatter(tokenizer=tok, notation=notation, max_length=2048)
        prompt = fmt.format_prompt(demos, query)
        assert f"Moves ({notation}): {expected_moves}" in prompt
        assert "Outcome (White): win" in prompt
        assert "Outcome (White): draw" in prompt
        assert "4 | . . X . P . . ." in prompt
        assert "En passant: e6" in prompt
        # Strictly zero leakage
        for forbidden in [
            "diagonal_rook",
            "antichess",
            "blocked_squares",
            "SECRET",
            "Rooks move diagonally",
            "Lose all pieces",
            "LEAK_TEST",
            "QUERY_LABEL_LEAK",
        ]:
            assert forbidden not in prompt


def test_prompt_formatter_never_truncates_query_and_trims_oldest_demos() -> None:
    tok = MockWhitespaceTokenizer()
    fmt = QwenPromptFormatter(tokenizer=tok, notation="uci", max_length=512)

    demos = [
        DemoGame(moves=[f"a2a{i}", f"b7b{i}"] * 8, outcome="win")
        for i in range(1, 7)
    ]
    # Unique marker moves per demo to verify oldest are dropped and newest kept
    marked_demos = [
        DemoGame(moves=[f"TAGOLD{i}"] + ["e2e4", "e7e5"] * 6, outcome="draw")
        for i in range(6)
    ]
    query = QueryPosition(fen=STANDARD_INITIAL_FEN, blocked_squares=["d4"])
    query_str = fmt.format_query(query)

    q_only_len = len(tok.encode(query_str))
    header_plus_q_len = len(tok.encode(fmt.header + "\n\n" + query_str))
    one_demo_len = len(tok.encode(fmt.format_demo(marked_demos[-1])))

    # Cap just big enough for header + 2 newest demos + query
    cap = header_plus_q_len + 2 * one_demo_len + 5
    info = fmt.format_prompt_with_info(marked_demos, query, max_length=cap)
    assert info.num_tokens is not None and info.num_tokens <= cap
    assert info.num_demos_dropped >= 3
    assert info.num_demos_kept >= 1
    assert info.text.endswith(query_str)
    # Oldest demo (TAGOLD0) must be dropped, newest demo (TAGOLD5) must be kept
    assert "TAGOLD0" not in info.text
    assert "TAGOLD5" in info.text

    # Tight cap that fits query only (drops all demos + header, still preserves full query)
    info_q_only = fmt.format_prompt_with_info(demos, query, max_length=q_only_len + 1)
    assert info_q_only.num_demos_kept == 0
    assert info_q_only.num_demos_dropped == len(demos)
    assert info_q_only.text == query_str

    # Cap smaller than query alone must raise rather than silently truncating query
    with pytest.raises(ValueError, match="refusing to truncate query"):
        fmt.format_prompt_with_info(demos, query, max_length=q_only_len - 1)


def test_soft_cross_entropy_math() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]], dtype=torch.float32)
    loss = soft_cross_entropy(logits, targets)

    expected_rows = []
    for row_l, row_t in zip(logits, targets):
        lse = math.log(sum(math.exp(v.item()) for v in row_l))
        row_loss = -sum(t.item() * (l.item() - lse) for t, l in zip(row_t, row_l))
        expected_rows.append(row_loss)
    expected = sum(expected_rows) / 2.0
    assert abs(loss.item() - expected) < 1e-6


def test_extract_last_non_pad_hidden_right_and_left_padding() -> None:
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    # Right-padded: lengths 3 and 5
    mask_right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out_r = extract_last_non_pad_hidden(hidden, mask_right)
    assert torch.equal(out_r[0], hidden[0, 2])
    assert torch.equal(out_r[1], hidden[1, 4])

    # Left-padded: real tokens end at index 4 for both rows
    mask_left = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    out_l = extract_last_non_pad_hidden(hidden, mask_left)
    assert torch.equal(out_l[0], hidden[0, 4])
    assert torch.equal(out_l[1], hidden[1, 4])


def test_lora_config_modes() -> None:
    cfg_a = build_lora_config("handoff", num_hidden_layers=28)
    assert cfg_a.r == 8
    assert cfg_a.lora_alpha == 16
    assert list(cfg_a.layers_to_transform) == [24, 25, 26, 27]

    cfg_b = build_lora_config("full_lora", num_hidden_layers=28)
    assert cfg_b.r == 16
    assert cfg_b.lora_alpha == 32
    assert cfg_b.layers_to_transform is None


def test_mock_classifier_forward_backward_grad_ckpt_and_save_reload(tmp_path: Path) -> None:
    tok = MockWhitespaceTokenizer()
    base = _make_tiny_qwen(num_hidden_layers=6, hidden_size=32)
    base_snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}

    # Poison lm_head to verify forward pass bypasses 152k LM head completely
    def _fail_lm_head(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("lm_head should not be called by QwenICLClassifier")

    base.lm_head.forward = _fail_lm_head  # type: ignore[method-assign]

    model = QwenICLClassifier(
        base_model=base,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
        gradient_checkpointing=True,
    )

    # Verify only last 4 layers (layers 2..5) have LoRA weights & base is frozen
    lora_names = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert lora_names, "Expected trainable LoRA parameters"
    assert all("lora_" in n for n in lora_names)
    assert all(
        any(f".layers.{idx}." in n for idx in (2, 3, 4, 5))
        for n in lora_names
    )

    counts = model.parameter_counts()
    assert counts["head_trainable"] == 32 * 3 + 3
    assert counts["total_trainable"] == counts["lora_trainable"] + counts["head_trainable"]

    prompts = [
        "board state alpha e2e4 e7e5",
        "board state beta d2d4 d7d5 c2c4",
    ]
    encoded = tok(prompts, return_tensors="pt", padding=True)
    targets = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    # Run 2 steps so both lora_B (step 1) and lora_A (step 2) accumulate nonzero grads
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            target_probs=targets,
        )
        assert out.loss is not None and torch.isfinite(out.loss)
        out.loss.backward()
        assert model.classifier.weight.grad is not None
        assert model.classifier.weight.grad.abs().sum().item() > 0
        optimizer.step()

    probs_before = model.predict_proba(prompts, batch_size=1)
    assert probs_before.shape == (2, 3)
    assert np.allclose(probs_before.sum(axis=-1), 1.0, atol=1e-5)

    ckpt_dir = tmp_path / "ckpt"
    model.save_adapter_and_head(ckpt_dir)

    # Reload via classmethod on a fresh base model
    fresh_base = _make_tiny_qwen(num_hidden_layers=6, hidden_size=32)
    fresh_base.load_state_dict(base_snapshot)
    reloaded = QwenICLClassifier.from_adapter_and_head(
        ckpt_dir,
        base_model=fresh_base,
        tokenizer=tok,
        is_trainable=False,
    )
    probs_reloaded = reloaded.predict_proba(prompts, batch_size=2)
    assert np.max(np.abs(probs_before - probs_reloaded)) < 1e-6

    # Also verify instance-level load_adapter_and_head
    fresh_base_2 = _make_tiny_qwen(num_hidden_layers=6, hidden_size=32)
    fresh_base_2.load_state_dict(base_snapshot)
    fresh_clf = QwenICLClassifier(
        base_model=fresh_base_2,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
    )
    fresh_clf.load_adapter_and_head(ckpt_dir)
    probs_reloaded_2 = fresh_clf.predict_proba(prompts, batch_size=2)
    assert np.max(np.abs(probs_before - probs_reloaded_2)) < 1e-6
