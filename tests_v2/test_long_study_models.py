"""Unit tests for Qwen handoff update, Scratch 494M Qwen2, optimizer/LR, strict no-trim, and deterministic snapshot sampler."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from chess_icl_v2.data.long_study_sampler import (
    DeterministicEpisodeSampler,
    build_full_val_episodes,
    load_val_panel_episodes,
    verify_snapshot_manifest,
)
from chess_icl_v2.models import (
    ScratchQwen2Classifier,
    create_scratch_qwen2_config,
)
from chess_icl_v2.models.qwen_model import (
    LORA_HANDOFF_MODULES,
    LORA_TARGET_MODULES,
    STANDARD_INITIAL_FEN,
    DemoGame,
    QueryPosition,
    QwenICLClassifier,
    QwenPromptFormatter,
    build_cosine_warmup_to_min_scheduler,
    build_long_study_optimizer,
    build_lora_config,
)


class MockLongStudyTokenizer:
    """Fast deterministic tokenizer bounded inside tiny vocab for CPU/GPU unit tests."""

    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2 + (sum(ord(c) for c in tok) % 250) for tok in text.split()]
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


def _tiny_qwen_causal(
    num_hidden_layers: int = 6,
    hidden_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_cache=False,
        attn_implementation="sdpa",
    )
    torch.manual_seed(1234)
    model = Qwen2ForCausalLM(cfg)
    return model.to(dtype=dtype)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(raw.splitlines()) if raw else 0,
        "byte_size": len(raw),
    }


def _build_snapshot(
    snapshot_dir: Path,
    variants: tuple[str, ...] = ("diagonal_rook", "straight_bishop"),
    num_train_demos: int = 6,
    num_val_demos: int = 5,
    num_train_pos: int = 4,
    num_val_pos: int = 4,
) -> dict[str, Any]:
    files_manifest: dict[str, dict[str, Any]] = {}
    sample_san_moves = [
        ["e4", "e5", "Nf3", "Nc6"],
        ["d4", "d5", "c4", "e6"],
        ["c4", "e5", "Nc3", "Nf6"],
        ["Nf3", "d5", "d4", "Nf6"],
        ["e4", "c5", "Nf3", "d6"],
        ["d4", "Nf6", "c4", "g6"],
    ]
    outcomes = ["win", "draw", "loss"]
    fens = [
        STANDARD_INITIAL_FEN,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d6 0 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ]

    for v in variants:
        v_dir = snapshot_dir / v

        train_demos = []
        for i in range(num_train_demos):
            train_demos.append(
                {
                    "game_id": f"{v}_tr_demo_{i}",
                    "family_id": f"{v}_fam_demo_{i}",
                    "variant": v,
                    "moves": sample_san_moves[i % len(sample_san_moves)],
                    "moves_san": sample_san_moves[i % len(sample_san_moves)],
                    "outcome": outcomes[i % 3],
                    "initial_fen": STANDARD_INITIAL_FEN,
                    "blocked_squares": None,
                }
            )
        files_manifest[f"{v}/games_demo_train.jsonl"] = _write_jsonl(
            v_dir / "games_demo_train.jsonl", train_demos
        )

        val_demos = []
        for i in range(num_val_demos):
            val_demos.append(
                {
                    "game_id": f"{v}_va_demo_{i}",
                    "family_id": f"{v}_fam_val_demo_{i}",
                    "variant": v,
                    "moves": sample_san_moves[(i + 1) % len(sample_san_moves)],
                    "moves_san": sample_san_moves[(i + 1) % len(sample_san_moves)],
                    "outcome": outcomes[(i + 1) % 3],
                    "initial_fen": STANDARD_INITIAL_FEN,
                    "blocked_squares": None,
                }
            )
        files_manifest[f"{v}/games_demo_val.jsonl"] = _write_jsonl(
            v_dir / "games_demo_val.jsonl", val_demos
        )

        train_pos = []
        for i in range(num_train_pos):
            train_pos.append(
                {
                    "root_id": f"{v}_tr_root_{i}",
                    "family_id": f"{v}_fam_demo_{i % max(1, num_train_demos)}"
                    if i == 0
                    else f"{v}_fam_tr_{i}",
                    "source_game_id": f"{v}_src_tr_{i}",
                    "variant": v,
                    "fen": fens[i % len(fens)],
                    "blocked_squares": None,
                    "side_to_move": "w" if i % 2 == 0 else "b",
                    "label_probs": [0.5, 0.3, 0.2] if i % 2 == 0 else [0.2, 0.3, 0.5],
                    "n_unresolved": 0,
                    "equal_material": True,
                }
            )
        files_manifest[f"{v}/positions_train.jsonl"] = _write_jsonl(
            v_dir / "positions_train.jsonl", train_pos
        )

        val_pos = []
        for i in range(num_val_pos):
            val_pos.append(
                {
                    "root_id": f"{v}_va_root_{i}",
                    "family_id": f"{v}_fam_va_{i}",
                    "source_game_id": f"{v}_src_va_{i}",
                    "variant": v,
                    "fen": fens[i % len(fens)],
                    "blocked_squares": None,
                    "side_to_move": "w" if i % 2 == 0 else "b",
                    "label_probs": [0.6, 0.2, 0.2],
                    "n_unresolved": 0,
                    "equal_material": False,
                }
            )
        files_manifest[f"{v}/positions_val.jsonl"] = _write_jsonl(
            v_dir / "positions_val.jsonl", val_pos
        )

    manifest = {
        "schema_version": 1,
        "exposed_variants": list(variants),
        "max_k": 4,
        "n_unresolved": 0,
        "files": files_manifest,
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def test_scratch_494m_exact_parameter_counts_on_meta_and_proxy_train(tmp_path: Path) -> None:
    cfg = create_scratch_qwen2_config()
    assert cfg.num_hidden_layers == 24
    assert cfg.hidden_size == 896
    assert cfg.intermediate_size == 4864
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.vocab_size == 152064
    assert cfg.max_position_embeddings == 32768
    assert cfg.rms_norm_eps == 1e-6
    rope_theta = getattr(cfg, "rope_theta", None)
    if rope_theta is None:
        rope_theta = cfg.rope_parameters["rope_theta"]
    assert rope_theta == 1000000.0
    assert cfg.hidden_act == "silu"
    assert cfg.attention_dropout == 0.0
    assert cfg.use_cache is False
    assert cfg._attn_implementation == "sdpa"

    meta_model = ScratchQwen2Classifier(config=cfg, device="meta")
    counts = meta_model.parameter_counts()
    assert counts["embedding_params"] == 136_249_344
    assert counts["other_params"] == 357_900_803
    assert counts["total_trainable"] == 494_150_147
    assert counts["total_params"] == 494_150_147
    assert not hasattr(meta_model.backbone, "lm_head")

    # Small proxy model for real forward/backward, gradient checkpointing, FP32 master weights, save/reload
    proxy_cfg = create_scratch_qwen2_config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tok = MockLongStudyTokenizer()
    torch.manual_seed(11)
    proxy = ScratchQwen2Classifier(
        config=proxy_cfg,
        tokenizer=tok,
        gradient_checkpointing=True,
    )
    assert all(p.dtype == torch.float32 and p.requires_grad for p in proxy.parameters())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    proxy = proxy.to(device)
    input_ids = torch.randint(0, 256, (1, 24), device=device)
    targets = torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float32, device=device)

    optimizer = build_long_study_optimizer(proxy, lr=1e-3, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    # Test attention_mask=None fast path
    out = proxy(input_ids=input_ids, attention_mask=None, target_probs=targets)
    assert out.logits.dtype == torch.float32
    assert out.probs.dtype == torch.float32
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()

    for name, p in proxy.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert p.grad.dtype == torch.float32
        assert torch.isfinite(p.grad).all()
    optimizer.step()

    probs_before = proxy.predict_proba(["e4 e5 Nf3 Nc6", "d4 d5 c4"], batch_size=1)
    assert probs_before.shape == (2, 3)
    assert np.allclose(probs_before.sum(axis=-1), 1.0, atol=1e-5)

    # Verify save_weights / from_weights / load_weights roundtrip
    save_dir = tmp_path / "scratch_ckpt"
    proxy.save_weights(save_dir)
    reloaded = ScratchQwen2Classifier.from_weights(save_dir, tokenizer=tok, device=device)
    probs_after = reloaded.predict_proba(["e4 e5 Nf3 Nc6", "d4 d5 c4"], batch_size=2)
    assert np.max(np.abs(probs_before - probs_after)) < 1e-5

    fresh = ScratchQwen2Classifier(config=proxy_cfg, tokenizer=tok).to(device)
    fresh.load_weights(save_dir)
    probs_after_load = fresh.predict_proba(["e4 e5 Nf3 Nc6", "d4 d5 c4"], batch_size=2)
    assert np.max(np.abs(probs_before - probs_after_load)) < 1e-5


def test_qwen_7b_handoff_exact_parameter_counts_on_meta() -> None:
    assert LORA_HANDOFF_MODULES == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert len(LORA_TARGET_MODULES) == 7

    cfg_handoff = build_lora_config("handoff", num_hidden_layers=28)
    assert cfg_handoff.r == 8
    assert cfg_handoff.lora_alpha == 16
    assert list(cfg_handoff.layers_to_transform) == [24, 25, 26, 27]
    assert set(cfg_handoff.target_modules) == set(LORA_HANDOFF_MODULES)

    cfg_full = build_lora_config("full_lora", num_hidden_layers=28)
    assert set(cfg_full.target_modules) == set(LORA_TARGET_MODULES)

    cfg_7b = Qwen2Config(
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        use_cache=False,
    )
    with torch.device("meta"):
        meta_base = Qwen2ForCausalLM(cfg_7b).to(dtype=torch.bfloat16)
        clf = QwenICLClassifier(
            base_model=meta_base,
            lora_mode="handoff",
            torch_dtype=torch.bfloat16,
            head_dtype=torch.float32,
            adapter_dtype=torch.float32,
        )

    counts = clf.parameter_counts()
    assert counts["lora_trainable"] == 720_896
    assert counts["head_trainable"] == 10_755
    assert counts["total_trainable"] == 731_651

    # Verify trainable LoRA/head dtypes are FP32 while frozen base stays BF16
    for n, p in clf.backbone.named_parameters():
        if "lora_" in n:
            assert p.requires_grad is True
            assert p.dtype == torch.float32
        else:
            assert p.requires_grad is False
            assert p.dtype == torch.bfloat16
    for p in clf.classifier.parameters():
        assert p.requires_grad is True
        assert p.dtype == torch.float32


def test_qwen_fp32_lora_and_head_on_bf16_base_forward_backward(tmp_path: Path) -> None:
    tok = MockLongStudyTokenizer()
    base = _tiny_qwen_causal(num_hidden_layers=6, hidden_size=32, dtype=torch.bfloat16)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    base = base.to(device)

    clf = QwenICLClassifier(
        base_model=base,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.bfloat16,
        head_dtype=torch.float32,
        adapter_dtype=torch.float32,
        gradient_checkpointing=True,
    )

    for name, p in clf.backbone.named_parameters():
        if "lora_" in name:
            assert p.requires_grad
            assert p.dtype == torch.float32
        else:
            assert not p.requires_grad
            assert p.dtype == torch.bfloat16
    for p in clf.classifier.parameters():
        assert p.requires_grad
        assert p.dtype == torch.float32

    prompts = ["[Demo] e4 e5 [Query] start", "[Demo] d4 d5 c4 [Query] mid"]
    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    targets = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]], dtype=torch.float32, device=device)

    opt = build_long_study_optimizer(clf, lr=1e-2, weight_decay=0.05)

    # Run 2 steps so both lora_B and lora_A receive nonzero FP32 gradients
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = clf(input_ids=input_ids, attention_mask=attn_mask, target_probs=targets)
        else:
            out = clf(input_ids=input_ids, attention_mask=attn_mask, target_probs=targets)

        assert out.logits.dtype == torch.float32
        assert out.probs.dtype == torch.float32
        assert out.loss is not None and torch.isfinite(out.loss)
        out.loss.backward()

        for name, p in clf.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad on {name}"
                assert p.grad.dtype == torch.float32
                assert torch.isfinite(p.grad).all()
            else:
                assert p.grad is None
        opt.step()

    # Check reload preserves FP32 trainable adapter & head on BF16 frozen base
    ckpt = tmp_path / "qwen_ckpt"
    clf.save_adapter_and_head(ckpt)
    fresh_base = _tiny_qwen_causal(num_hidden_layers=6, hidden_size=32, dtype=torch.bfloat16).to(device)
    reloaded = QwenICLClassifier.from_adapter_and_head(
        ckpt,
        base_model=fresh_base,
        tokenizer=tok,
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    )
    for name, p in reloaded.backbone.named_parameters():
        if "lora_" in name:
            assert p.requires_grad and p.dtype == torch.float32
        else:
            assert not p.requires_grad and p.dtype == torch.bfloat16


def test_long_study_optimizer_and_cosine_scheduler() -> None:
    class TinyNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(16, 8)
            self.linear = nn.Linear(8, 8, bias=True)
            self.rms_norm = nn.LayerNorm(8)
            self.ln_final = nn.LayerNorm(8)
            self.head = nn.Linear(8, 3, bias=True)
            self.scale = nn.Parameter(torch.ones(8))  # ndim < 2

    m = TinyNormModel()
    opt = build_long_study_optimizer(m, lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2
    assert opt.defaults["betas"] == (0.9, 0.95)
    for pg in opt.param_groups:
        assert pg["betas"] == (0.9, 0.95)

    decay_group = next(pg for pg in opt.param_groups if pg["weight_decay"] > 0)
    no_decay_group = next(pg for pg in opt.param_groups if pg["weight_decay"] == 0.0)
    assert math.isclose(decay_group["weight_decay"], 0.1)

    decay_ids = {id(p) for p in decay_group["params"]}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}

    assert id(m.emb.weight) in decay_ids
    assert id(m.linear.weight) in decay_ids
    assert id(m.head.weight) in decay_ids

    assert id(m.linear.bias) in no_decay_ids
    assert id(m.head.bias) in no_decay_ids
    assert id(m.rms_norm.weight) in no_decay_ids
    assert id(m.rms_norm.bias) in no_decay_ids
    assert id(m.ln_final.weight) in no_decay_ids
    assert id(m.ln_final.bias) in no_decay_ids
    assert id(m.scale) in no_decay_ids

    # Scheduler: 5% warmup -> cosine decay to 0.1 * peak_lr
    total_steps = 100
    warmup_steps = 5  # 5% of 100
    peak_lr = 1e-3
    sched = build_cosine_warmup_to_min_scheduler(
        opt, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=0.1
    )
    observed_lrs: list[float] = []
    for _ in range(total_steps):
        observed_lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    final_lr = opt.param_groups[0]["lr"]

    # Linear warmup: 0.2, 0.4, 0.6, 0.8, 1.0 * peak_lr
    assert math.isclose(observed_lrs[0], 0.2 * peak_lr, rel_tol=1e-6)
    assert math.isclose(observed_lrs[2], 0.6 * peak_lr, rel_tol=1e-6)
    assert math.isclose(observed_lrs[4], 1.0 * peak_lr, rel_tol=1e-6)
    # Cosine midpoint & end at 0.1 * peak_lr
    assert observed_lrs[50] < peak_lr and observed_lrs[50] > 0.1 * peak_lr
    assert math.isclose(final_lr, 0.1 * peak_lr, rel_tol=1e-6)


def test_strict_no_trim_formatter_raises_on_overflow() -> None:
    tok = MockLongStudyTokenizer()
    demos = [DemoGame(moves=["e4", "e5", "Nf3", "Nc6"] * 5, outcome="win") for _ in range(4)]
    query = QueryPosition(fen=STANDARD_INITIAL_FEN)

    # Non-strict trims oldest demos cleanly
    fmt_trim = QwenPromptFormatter(tokenizer=tok, notation="san", max_length=150, strict_no_trim=False)
    trimmed = fmt_trim.format_prompt_with_info(demos, query)
    assert trimmed.num_demos_dropped > 0

    # Strict never drops demos or trims; raises ValueError immediately
    fmt_strict = QwenPromptFormatter(tokenizer=tok, notation="san", max_length=150, strict_no_trim=True)
    with pytest.raises(ValueError, match="strict_no_trim=True"):
        fmt_strict.format_prompt_with_info(demos, query)

    # When sequence fits, strict_no_trim succeeds with 0 dropped
    fmt_ok = QwenPromptFormatter(tokenizer=tok, notation="san", max_length=4096, strict_no_trim=True)
    ok_res = fmt_ok.format_prompt_with_info(demos, query)
    assert ok_res.num_demos_dropped == 0
    assert ok_res.num_demos_kept == 4


def test_verify_snapshot_manifest_tamper_and_completeness_checks(tmp_path: Path) -> None:
    snap = tmp_path / "snap_ok"
    _build_snapshot(snap)
    report = verify_snapshot_manifest(snap)
    assert report["exposed_variants"] == ["diagonal_rook", "straight_bishop"]

    # 1. Missing manifest
    snap_missing = tmp_path / "snap_missing"
    snap_missing.mkdir()
    with pytest.raises(FileNotFoundError, match="snapshot_manifest.json"):
        verify_snapshot_manifest(snap_missing)

    # 2. Tampered file -> SHA-256 mismatch
    snap_tamper = tmp_path / "snap_tamper"
    _build_snapshot(snap_tamper)
    p_file = snap_tamper / "diagonal_rook" / "positions_train.jsonl"
    content = p_file.read_text(encoding="utf-8")
    # Modify 1 byte keeping same line count
    tampered = content.replace("diagonal_rook_tr_root_0", "diagonal_rook_tr_root_X", 1)
    p_file.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshot_manifest(snap_tamper)

    # 3. Incomplete labels: label_probs is None
    snap_none_lbl = tmp_path / "snap_none_lbl"
    _build_snapshot(snap_none_lbl)
    pos_path = snap_none_lbl / "straight_bishop" / "positions_train.jsonl"
    rows = [json.loads(ln) for ln in pos_path.read_text().splitlines() if ln.strip()]
    rows[0]["label_probs"] = None
    manifest_none = json.loads((snap_none_lbl / "snapshot_manifest.json").read_text())
    manifest_none["files"]["straight_bishop/positions_train.jsonl"] = _write_jsonl(pos_path, rows)
    (snap_none_lbl / "snapshot_manifest.json").write_text(json.dumps(manifest_none))
    with pytest.raises(ValueError, match="label_probs=None"):
        verify_snapshot_manifest(snap_none_lbl)

    # 4. Incomplete labels: n_unresolved > 0
    snap_unres = tmp_path / "snap_unres"
    _build_snapshot(snap_unres)
    pos_path_u = snap_unres / "diagonal_rook" / "positions_val.jsonl"
    rows_u = [json.loads(ln) for ln in pos_path_u.read_text().splitlines() if ln.strip()]
    rows_u[1]["n_unresolved"] = 3
    manifest_u = json.loads((snap_unres / "snapshot_manifest.json").read_text())
    manifest_u["files"]["diagonal_rook/positions_val.jsonl"] = _write_jsonl(pos_path_u, rows_u)
    (snap_unres / "snapshot_manifest.json").write_text(json.dumps(manifest_u))
    with pytest.raises(ValueError, match="n_unresolved"):
        verify_snapshot_manifest(snap_unres)

    # 5. Insufficient demos for max_k
    snap_few = tmp_path / "snap_few"
    _build_snapshot(snap_few, num_train_demos=3)
    with pytest.raises(ValueError, match="max_k=4"):
        verify_snapshot_manifest(snap_few, max_k=4)


def test_deterministic_episode_sampler_resume_and_val_builders(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _build_snapshot(snap)
    tok = MockLongStudyTokenizer()

    sampler_a = DeterministicEpisodeSampler(
        snapshot_dir=snap,
        tokenizer=tok,
        exposed_variants=["diagonal_rook", "straight_bishop"],
        k_values=(0, 1, 2, 4),
        seed=2026,
        notation="san",
        max_length=32768,
        strict_no_trim=True,
    )
    sampler_b = DeterministicEpisodeSampler(
        snapshot_dir=snap,
        tokenizer=tok,
        exposed_variants=["diagonal_rook", "straight_bishop"],
        k_values=(0, 1, 2, 4),
        seed=2026,
        notation="san",
        max_length=32768,
        strict_no_trim=True,
    )

    # Mutate global PyTorch & NumPy RNGs on sampler_b's side to ensure complete independence
    torch.manual_seed(99999)
    np.random.seed(88888)
    _ = np.random.rand(10)

    # Draw 5 episodes and compare hashes, plus check family exclusion & zero leakage
    for step_idx in range(5):
        sampler_a.set_step(step_idx)
        sampler_b.set_step(step_idx)
        pa, ta, ma = sampler_a.sample_episode()
        pb, tb, mb = sampler_b.sample_episode()
        assert pa == pb
        assert np.array_equal(ta, tb)
        assert ma["input_sha256"] == mb["input_sha256"]
        assert ma["demo_ids"] == mb["demo_ids"]
        assert ma["draw_counter"] == step_idx
        assert ma["step_counter"] == step_idx
        assert len(ma["demo_ids"]) == ma["k"]
        assert "diagonal_rook" not in pa
        assert "straight_bishop" not in pa

        # Verify excluded family demo is never chosen
        if ma["root_id"].endswith("_tr_root_0"):
            forbidden_demo_id = f"{ma['variant']}_tr_demo_0"
            assert forbidden_demo_id not in ma["demo_ids"]

    # Save state_dict mid-run, advance sampler_a, then resume sampler_c from saved state_dict
    saved_state = json.loads(json.dumps(sampler_a.state_dict()))
    expected_seq = [sampler_a.sample_episode() for _ in range(4)]

    sampler_c = DeterministicEpisodeSampler(
        snapshot_dir=snap,
        tokenizer=tok,
        exposed_variants=["diagonal_rook", "straight_bishop"],
        k_values=(0, 1, 2, 4),
        seed=1,  # Deliberately different initial seed before load_state_dict
    )
    sampler_c.load_state_dict(saved_state)
    resumed_seq = [sampler_c.sample_episode() for _ in range(4)]

    for (p1, t1, m1), (p2, t2, m2) in zip(expected_seq, resumed_seq):
        assert p1 == p2
        assert np.array_equal(t1, t2)
        assert m1 == m2

    # Verify val panel and full val builders read games_demo_val.jsonl deterministically
    val_panel_1 = load_val_panel_episodes(snap, tokenizer=tok, roots_per_variant=2, seed=7)
    val_panel_2 = load_val_panel_episodes(snap, tokenizer=tok, roots_per_variant=2, seed=7)
    assert len(val_panel_1) == 2 * 2 * 4  # 2 variants * 2 roots * 4 K values
    assert [e["input_sha256"] for e in val_panel_1] == [e["input_sha256"] for e in val_panel_2]
    assert all("_va_demo_" in did for ep in val_panel_1 for did in ep["demo_ids"])

    full_val = build_full_val_episodes(snap, tokenizer=tok, seed=7)
    assert len(full_val) == 2 * 4 * 4  # 2 variants * 4 val roots * 4 K values
    assert all("_va_demo_" in did for ep in full_val for did in ep["demo_ids"])

    # Verify val_panel_manifest.json schema (with selected_root_ids dicts) overrides random sampling
    (snap / "val_panel_manifest.json").write_text(
        json.dumps(
            {
                "variants": {
                    "diagonal_rook": {"selected_root_ids": ["diagonal_rook_va_root_1"]},
                    "straight_bishop": {"selected_root_ids": ["straight_bishop_va_root_2"]},
                }
            }
        ),
        encoding="utf-8",
    )
    pinned_panel = load_val_panel_episodes(snap, tokenizer=tok, roots_per_variant=3, seed=7)
    assert len(pinned_panel) == 2 * 1 * 4
    assert {e["root_id"] for e in pinned_panel} == {
        "diagonal_rook_va_root_1",
        "straight_bishop_va_root_2",
    }
