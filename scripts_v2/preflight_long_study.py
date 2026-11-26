#!/usr/bin/env python3
"""Preflight verification and profiling suite for the Chess-Variant ICL v2 long study.

Implements all 5 preflight checks from Section 6:
  1. Scratch 494M & Qwen 7B parameter counts, trainable/frozen splits, all-param gradients,
     and exposure lineage verification.
  2. Loss, padding/attention invariance (``attention_mask=None`` vs padded), complete-state
     input reconstruction, strict no-trim overflow handling, and dynamic context sampling.
  3. Atomic save/resume equivalence across an optimizer boundary (step 3 -> 5 continuous
     vs resume from step 3 -> 5: identical sampled ``input_sha256``, losses, and weights).
  4. Mixed-K profiling (50-100 updates, default 60, including a longest supported K=4 input)
     per architecture; discards profile weights afterwards.
  5. Records tokens/s, peak allocated/reserved VRAM, optimizer-update time, model-loading
     time, validation overhead, checkpoint size, and 1k-warmup + 10k-main GPU-hour/disk
     estimates into ``preflight_report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.long_study_sampler import (  # noqa: E402
    DeterministicEpisodeSampler,
    load_val_panel_episodes,
    verify_snapshot_manifest,
)
from chess_icl_v2.models.qwen_model import (  # noqa: E402
    STANDARD_INITIAL_FEN,
    DemoGame,
    QueryPosition,
    QwenICLClassifier,
    QwenPromptFormatter,
    build_long_study_optimizer,
    get_preferred_attn_implementation,
    load_qwen_tokenizer,
    resolve_qwen_model_path,
)
from chess_icl_v2.models.scratch_qwen2 import (  # noqa: E402
    ScratchQwen2Classifier,
    create_scratch_qwen2_config,
)
from scripts_v2.train_long_study import (  # noqa: E402
    ARCH_DEFAULTS,
    _directory_size_bytes,
    _resolve_lineage,
    atomic_write_json,
    compute_trainable_weights_sha256,
    evaluate_val_panel,
    resolve_device,
    resolve_exposure_variants,
    run_training,
    save_model_weights,
    save_rolling_checkpoint,
    set_global_seed,
    utc_now_iso,
)


class _MockPreflightTokenizer:
    """Fast deterministic tokenizer for ``--mock-small`` CPU/unit preflight runs."""

    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"
    vocab_size = 256

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [2 + (sum(ord(c) for c in tok) % 250) for tok in text.split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def get_vocab(self) -> dict[str, int]:
        return {self.pad_token: 0, self.eos_token: 1}

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
        padded_ids: list[list[int]] = []
        attn_masks: list[list[int]] = []
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(raw.splitlines()) if raw else 0,
        "byte_size": len(raw),
    }


def build_synthetic_preflight_snapshot(snapshot_dir: Path) -> Path:
    """Create a tiny checksum-verified snapshot for ``--mock-small`` preflight runs."""
    variants = (
        "standard",
        "diagonal_rook",
        "straight_bishop",
        "weakened_queen",
        "extended_pawn",
    )
    sample_moves = [
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
    files_manifest: dict[str, dict[str, Any]] = {}
    for v in variants:
        v_dir = snapshot_dir / v
        tr_demos = [
            {
                "game_id": f"{v}_tr_demo_{i}",
                "family_id": f"{v}_fam_demo_{i}",
                "variant": v,
                "moves": sample_moves[i % len(sample_moves)],
                "moves_san": sample_moves[i % len(sample_moves)],
                "outcome": outcomes[i % 3],
                "initial_fen": STANDARD_INITIAL_FEN,
                "blocked_squares": None,
            }
            for i in range(6)
        ]
        va_demos = [
            {
                "game_id": f"{v}_va_demo_{i}",
                "family_id": f"{v}_fam_va_demo_{i}",
                "variant": v,
                "moves": sample_moves[(i + 1) % len(sample_moves)],
                "moves_san": sample_moves[(i + 1) % len(sample_moves)],
                "outcome": outcomes[(i + 1) % 3],
                "initial_fen": STANDARD_INITIAL_FEN,
                "blocked_squares": None,
            }
            for i in range(5)
        ]
        tr_pos = [
            {
                "root_id": f"{v}_tr_root_{i}",
                "family_id": f"{v}_fam_demo_0" if i == 0 else f"{v}_fam_tr_{i}",
                "source_game_id": f"{v}_src_tr_{i}",
                "variant": v,
                "fen": fens[i % len(fens)],
                "blocked_squares": None,
                "side_to_move": "w" if i % 2 == 0 else "b",
                "label_probs": [0.5, 0.3, 0.2] if i % 2 == 0 else [0.2, 0.3, 0.5],
                "n_unresolved": 0,
            }
            for i in range(4)
        ]
        va_pos = [
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
            }
            for i in range(4)
        ]
        files_manifest[f"{v}/games_demo_train.jsonl"] = _write_jsonl(
            v_dir / "games_demo_train.jsonl", tr_demos
        )
        files_manifest[f"{v}/games_demo_val.jsonl"] = _write_jsonl(
            v_dir / "games_demo_val.jsonl", va_demos
        )
        files_manifest[f"{v}/positions_train.jsonl"] = _write_jsonl(
            v_dir / "positions_train.jsonl", tr_pos
        )
        files_manifest[f"{v}/positions_val.jsonl"] = _write_jsonl(
            v_dir / "positions_val.jsonl", va_pos
        )

    manifest = {
        "schema_version": 1,
        "exposed_variants": list(variants),
        "max_k": 4,
        "n_unresolved": 0,
        "files": files_manifest,
    }
    atomic_write_json(snapshot_dir / "snapshot_manifest.json", manifest)
    return snapshot_dir


def _build_proxy_qwen_model(
    tokenizer: Any,
    device: torch.device,
    seed: int = 1234,
    gradient_checkpointing: bool = False,
) -> QwenICLClassifier:
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        use_cache=False,
        attn_implementation="sdpa",
    )
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(cfg)
    model = QwenICLClassifier(
        base_model=base,
        tokenizer=tokenizer,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
        adapter_dtype=torch.float32,
        gradient_checkpointing=gradient_checkpointing,
    )
    return model.to(device)


def _build_proxy_scratch_model(
    tokenizer: Any,
    device: torch.device,
    seed: int = 1234,
    gradient_checkpointing: bool = False,
) -> ScratchQwen2Classifier:
    cfg = create_scratch_qwen2_config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    torch.manual_seed(seed)
    model = ScratchQwen2Classifier(
        config=cfg,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
        gradient_checkpointing=gradient_checkpointing,
        device=device,
    )
    return model


def _make_model_factory(
    *,
    arch: str,
    mock_small: bool,
    tokenizer: Any,
    device: torch.device,
    model_path: Optional[str],
    seed: int = 1234,
) -> Callable[[], nn.Module]:
    if mock_small:
        if arch == "qwen":
            return lambda: _build_proxy_qwen_model(
                tokenizer=tokenizer, device=device, seed=seed, gradient_checkpointing=False
            )
        return lambda: _build_proxy_scratch_model(
            tokenizer=tokenizer, device=device, seed=seed, gradient_checkpointing=False
        )

    if arch == "qwen":
        resolved = resolve_qwen_model_path(model_path)
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        attn_impl = get_preferred_attn_implementation()

        def _build_qwen() -> QwenICLClassifier:
            set_global_seed(seed)
            return QwenICLClassifier(
                tokenizer=tokenizer,
                model_path=resolved,
                lora_mode="handoff",
                torch_dtype=torch_dtype,
                head_dtype=torch.float32,
                adapter_dtype=torch.float32,
                attn_implementation=attn_impl,
                local_files_only=True,
                gradient_checkpointing=True,
            ).to(device)

        return _build_qwen

    def _build_scratch() -> ScratchQwen2Classifier:
        set_global_seed(seed)
        return ScratchQwen2Classifier(
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            head_dtype=torch.float32,
            gradient_checkpointing=True,
            device=device,
        )

    return _build_scratch


def run_check1_params_and_grads(
    *,
    arches: Sequence[str],
    mock_small: bool,
    tokenizer: Any,
    device: torch.device,
    model_path: Optional[str],
) -> dict[str, Any]:
    """Check 1: Exact param counts, frozen/trainable split, all-param grads, exposure lineage."""
    results: dict[str, Any] = {}

    if "scratch" in arches:
        meta_cfg = create_scratch_qwen2_config()
        meta_scratch = ScratchQwen2Classifier(config=meta_cfg, device="meta")
        meta_counts = meta_scratch.parameter_counts()
        assert meta_counts["embedding_params"] == 136_249_344, meta_counts
        assert meta_counts["other_params"] == 357_900_803, meta_counts
        assert meta_counts["total_params"] == 494_150_147, meta_counts
        assert meta_counts["total_trainable"] == 494_150_147, meta_counts

        factory = _make_model_factory(
            arch="scratch",
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
        )
        m = factory()
        m.train()
        counts = m.parameter_counts()  # type: ignore[operator]
        if not mock_small:
            assert counts["embedding_params"] == 136_249_344, counts
            assert counts["other_params"] == 357_900_803, counts
            assert counts["total_trainable"] == 494_150_147, counts

        for name, p in m.named_parameters():
            assert p.requires_grad, f"Scratch param {name} must be trainable"
            assert p.dtype == torch.float32, f"Scratch param {name} expected FP32, got {p.dtype}"

        inp = torch.randint(2, 100, (1, 16), device=device)
        tgt = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32, device=device)
        m.zero_grad(set_to_none=True)
        out = m(input_ids=inp, attention_mask=None, target_probs=tgt)
        assert out.loss is not None and torch.isfinite(out.loss)
        out.loss.backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        non_finite = [n for n, p in m.named_parameters() if not torch.isfinite(p.grad).all()]
        assert not missing, f"Scratch missing grads: {missing[:5]}"
        assert not non_finite, f"Scratch non-finite grads: {non_finite[:5]}"

        results["scratch"] = {
            "passed": True,
            "meta_494m_counts": meta_counts,
            "runnable_counts": counts,
            "all_params_fp32": True,
            "all_params_have_finite_grad": True,
        }
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "qwen" in arches:
        factory = _make_model_factory(
            arch="qwen",
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
        )
        m = factory()
        m.train()
        counts = m.parameter_counts()  # type: ignore[operator]
        if not mock_small:
            assert counts["total_trainable"] == 731_651, f"Expected 731,651, got {counts}"

        assert counts["lora_trainable"] + counts["head_trainable"] == counts["total_trainable"]
        for n, p in m.backbone.named_parameters():  # type: ignore[union-attr]
            if "lora_" in n:
                assert p.requires_grad, f"LoRA param {n} not trainable"
                assert p.dtype == torch.float32, f"LoRA param {n} expected FP32, got {p.dtype}"
            else:
                assert not p.requires_grad, f"Base param {n} unexpectedly trainable"

        inp = torch.randint(2, 100, (1, 16), device=device)
        tgt = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32, device=device)
        m.zero_grad(set_to_none=True)
        out = m(input_ids=inp, attention_mask=None, target_probs=tgt)
        assert out.loss is not None and torch.isfinite(out.loss)
        out.loss.backward()
        trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        non_finite = [
            n
            for n, p in m.named_parameters()
            if p.requires_grad and not torch.isfinite(p.grad).all()
        ]
        assert trainable_names and not missing and not non_finite

        results["qwen"] = {
            "passed": True,
            "counts": counts,
            "trainable_param_tensors": len(trainable_names),
            "all_trainable_fp32_with_finite_grad": True,
        }
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Exposure lineage sanity check
    lineage = _resolve_lineage(
        init_weights_dir=Path("/tmp/warmup_parent/best_weights"),
        phase="main",
        exposure="familiar",
        exposed_variants=resolve_exposure_variants("familiar"),
        seed=42,
        arch=arches[0],
    )
    assert len(lineage) >= 2 and lineage[-1]["exposure"] == "familiar"
    results["exposure_lineage_check"] = {"passed": True, "sample_lineage": lineage}
    return results


def run_check2_loss_padding_prompt_sampling(
    *,
    arches: Sequence[str],
    mock_small: bool,
    snapshot_dir: Path,
    tokenizer: Any,
    device: torch.device,
    model_path: Optional[str],
    max_seq_len: int,
) -> dict[str, Any]:
    """Check 2: Attention mask invariance, complete-state reconstruction, overflow, dynamic sampling."""
    # 2a. Complete-state prompt reconstruction & strict no-trim overflow
    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation="san",
        max_length=max_seq_len,
        strict_no_trim=True,
    )
    demos = [
        DemoGame(moves=["e4", "e5", "Nf3", "Nc6"], outcome="win"),
        DemoGame(moves=["d4", "d5", "c4", "e6"], outcome="draw"),
    ]
    query = QueryPosition(fen=STANDARD_INITIAL_FEN)
    formatted = formatter.format_prompt_with_info(demos, query)
    assert "a b c d e f g h" in formatted.text
    assert "Side to move: White" in formatted.text
    assert "Castling: KQkq" in formatted.text
    assert "Moves (san): e4 e5 Nf3 Nc6" in formatted.text
    assert "Outcome (White): win" in formatted.text
    assert "Predict outcome for side to move (win, draw, loss):" in formatted.text
    assert formatted.num_demos_dropped == 0 and formatted.num_demos_kept == 2

    overflow_raised = False
    try:
        formatter.format_prompt_with_info(demos, query, max_length=24, strict_no_trim=True)
    except ValueError as exc:
        overflow_raised = "strict_no_trim=True" in str(exc)
    assert overflow_raised, "strict_no_trim=True failed to raise ValueError on overflow"

    # 2b. Dynamic context sampling checks
    snap_report = verify_snapshot_manifest(snapshot_dir)
    avail_variants = list(snap_report["exposed_variants"])
    sampler = DeterministicEpisodeSampler(
        snapshot_dir=snapshot_dir,
        tokenizer=tokenizer,
        exposed_variants=avail_variants[: min(4, len(avail_variants))],
        k_values=(0, 1, 2, 4),
        seed=42,
        notation="san",
        max_length=max_seq_len,
        strict_no_trim=True,
        verify_manifest=False,
    )
    seen_ks: set[int] = set()
    seen_vars: set[str] = set()
    sample_prompt = ""
    for i in range(16):
        sampler.set_step(i)
        prompt_text, target_probs, meta = sampler.sample_episode()
        assert target_probs.shape == (3,) and np.isfinite(target_probs).all()
        assert abs(float(target_probs.sum()) - 1.0) < 1e-4
        assert len(meta["demo_ids"]) == meta["k"]
        assert all("_va_demo_" not in d for d in meta["demo_ids"])
        seen_ks.add(int(meta["k"]))
        seen_vars.add(str(meta["variant"]))
        if not sample_prompt:
            sample_prompt = prompt_text

    # 2c. Attention mask invariance: unpadded (None) vs ones vs right-padded batch
    mask_invariance: dict[str, Any] = {}
    enc = tokenizer(sample_prompt, return_tensors="pt", padding=False, truncation=False)
    ids_unpadded = enc["input_ids"].to(device)
    seq_len = int(ids_unpadded.shape[1])
    pad_len = 8
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    ids_padded = torch.cat(
        [
            ids_unpadded,
            torch.full((1, pad_len), pad_id, dtype=ids_unpadded.dtype, device=device),
        ],
        dim=1,
    )
    mask_ones = torch.ones_like(ids_unpadded)
    mask_padded = torch.cat(
        [
            torch.ones((1, seq_len), dtype=torch.long, device=device),
            torch.zeros((1, pad_len), dtype=torch.long, device=device),
        ],
        dim=1,
    )
    tgt = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32, device=device)
    tol_ones = 1e-4
    # In BF16 across 28 Qwen2.5-7B layers, adding trailing pad tokens switches SDPA
    # from the is_causal=True kernel to the additive 4D-mask kernel (~1 BF16 ULP shift,
    # verified 0.0 exact invariance in FP32); microbatch=1 training/eval never pads.
    tol_pad = 5e-2 if (not mock_small and device.type == "cuda") else 1e-4

    for arch in arches:
        m = _make_model_factory(
            arch=arch,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
        )()
        m.eval()
        with torch.no_grad():
            o_none = m(input_ids=ids_unpadded, attention_mask=None, target_probs=tgt)
            o_ones = m(input_ids=ids_unpadded, attention_mask=mask_ones, target_probs=tgt)
            o_pad = m(input_ids=ids_padded, attention_mask=mask_padded, target_probs=tgt)
        diff_ones = float((o_none.probs - o_ones.probs).abs().max().item())
        diff_pad = float((o_none.probs - o_pad.probs).abs().max().item())
        loss_diff = float(abs(o_none.loss.item() - o_pad.loss.item()))  # type: ignore[union-attr]
        assert diff_ones <= tol_ones, f"{arch} None vs ones prob diff {diff_ones} > {tol_ones}"
        assert diff_pad <= tol_pad, f"{arch} None vs padded prob diff {diff_pad} > {tol_pad}"
        assert loss_diff <= tol_pad, f"{arch} None vs padded loss diff {loss_diff} > {tol_pad}"
        mask_invariance[arch] = {
            "passed": True,
            "max_prob_diff_none_vs_ones": diff_ones,
            "max_prob_diff_none_vs_padded": diff_pad,
            "loss_diff_none_vs_padded": loss_diff,
            "tol_ones": tol_ones,
            "tol_pad": tol_pad,
        }
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "passed": True,
        "complete_state_prompt_reconstruction": True,
        "strict_no_trim_overflow_error": True,
        "dynamic_sampling_ks_seen": sorted(seen_ks),
        "dynamic_sampling_variants_seen": sorted(seen_vars),
        "attention_mask_invariance": mask_invariance,
    }


def run_check3_save_resume_equivalence(
    *,
    arches: Sequence[str],
    mock_small: bool,
    snapshot_dir: Path,
    tokenizer: Any,
    device: torch.device,
    model_path: Optional[str],
    grad_accum: int,
    max_seq_len: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Check 3: Exact save/resume equivalence across an optimizer boundary (steps 3 -> 5)."""
    variants = list(verify_snapshot_manifest(snapshot_dir)["exposed_variants"])
    exposure = "familiar" if set(["diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"]).issubset(variants) else "standard"
    explicit_variants = None if exposure == "familiar" else [variants[0]]

    results: dict[str, Any] = {}
    for arch in arches:
        arch_dir = work_dir / f"check3_{arch}"
        cont_dir = arch_dir / "continuous"
        res_dir = arch_dir / "resumed"

        factory_cont = _make_model_factory(
            arch=arch,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            seed=777,
        )
        m_cont = factory_cont()
        cont_summary = run_training(
            arch=arch,
            phase="main",
            exposure=exposure,
            explicit_variants=explicit_variants,
            seed=42,
            snapshot_dir=snapshot_dir,
            output_dir=cont_dir,
            max_steps=5,
            eval_interval=100,
            ckpt_interval=100,
            milestones=[],
            batch_size=1,
            grad_accum=grad_accum,
            k_values=(0, 1, 2, 4),
            val_roots_per_variant=2,
            max_seq_len=max_seq_len,
            verify_manifest=False,
            heartbeat_interval=10,
            model=m_cont,
            tokenizer=tokenizer,
            device=device,
        )
        del m_cont
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Run 3 steps and save atomic full-state checkpoint at step 3
        m_res1 = _make_model_factory(
            arch=arch,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            seed=777,
        )()
        run_training(
            arch=arch,
            phase="main",
            exposure=exposure,
            explicit_variants=explicit_variants,
            seed=42,
            snapshot_dir=snapshot_dir,
            output_dir=res_dir,
            max_steps=5,
            stop_at_step=3,
            eval_interval=100,
            ckpt_interval=3,
            milestones=[],
            batch_size=1,
            grad_accum=grad_accum,
            k_values=(0, 1, 2, 4),
            val_roots_per_variant=2,
            max_seq_len=max_seq_len,
            verify_manifest=False,
            heartbeat_interval=10,
            model=m_res1,
            tokenizer=tokenizer,
            device=device,
        )
        del m_res1
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Resume from step 3 in a fresh model instance (perturb trainable weights first) and run steps 4-5
        m_res2 = _make_model_factory(
            arch=arch,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            seed=777 if arch == "qwen" else 9999,
        )()
        with torch.no_grad():
            for p in m_res2.parameters():
                if p.requires_grad:
                    p.add_(0.12345)
        res_summary = run_training(
            arch=arch,
            phase="main",
            exposure=exposure,
            explicit_variants=explicit_variants,
            seed=42,
            snapshot_dir=snapshot_dir,
            output_dir=res_dir,
            resume=True,
            max_steps=5,
            eval_interval=100,
            ckpt_interval=5,
            milestones=[],
            batch_size=1,
            grad_accum=grad_accum,
            k_values=(0, 1, 2, 4),
            val_roots_per_variant=2,
            max_seq_len=max_seq_len,
            verify_manifest=False,
            heartbeat_interval=10,
            model=m_res2,
            tokenizer=tokenizer,
            device=device,
        )
        del m_res2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cont_trace = [
            json.loads(ln)
            for ln in (cont_dir / "train_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        res_trace = [
            json.loads(ln)
            for ln in (res_dir / "train_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(cont_trace) == 5 and len(res_trace) == 5

        cont_hashes_45 = [r["input_hashes"] for r in cont_trace[3:5]]
        res_hashes_45 = [r["input_hashes"] for r in res_trace[3:5]]
        assert cont_hashes_45 == res_hashes_45, (
            f"{arch} sampled input_sha256 mismatch on resume: {cont_hashes_45} != {res_hashes_45}"
        )

        cont_losses_45 = [float(r["train_loss"]) for r in cont_trace[3:5]]
        res_losses_45 = [float(r["train_loss"]) for r in res_trace[3:5]]
        # CUDA SDPA backward + cuBLASLt BF16 atomic reductions exhibit ~3e-3 run-to-run variance
        # even without checkpointing (measured 2.98e-3 at step 2), while un-restored weights
        # diverge by >0.2; exact input_sha256 and exact CPU/FP32 determinism remain enforced.
        loss_tol = 1.5e-2 if (not mock_small and device.type == "cuda") else 1e-6
        max_loss_diff = max(abs(a - b) for a, b in zip(cont_losses_45, res_losses_45))
        max_rel_diff = max(
            abs(a - b) / max(1.0, abs(a)) for a, b in zip(cont_losses_45, res_losses_45)
        )
        assert max_rel_diff <= loss_tol, (
            f"{arch} loss mismatch on resume: {cont_losses_45} vs {res_losses_45} "
            f"(max_abs={max_loss_diff:.6f}, max_rel={max_rel_diff:.6f} > tol={loss_tol})"
        )

        weights_exact = cont_summary["final_weights_sha256"] == res_summary["final_weights_sha256"]
        if device.type == "cpu" or mock_small:
            assert weights_exact, f"{arch} final weights SHA mismatch on CPU resume"

        results[arch] = {
            "passed": True,
            "identical_input_sha256_steps_4_5": True,
            "max_loss_diff_steps_4_5": max_loss_diff,
            "final_weights_sha256_exact_match": weights_exact,
        }
    return results


def _find_or_build_longest_k4_episode(
    *,
    snapshot_dir: Path,
    tokenizer: Any,
    variants: Sequence[str],
    max_seq_len: int,
    mock_small: bool,
) -> tuple[str, np.ndarray, int]:
    """Find the longest K=4 training episode in ``snapshot_dir`` (or construct one in mock mode)."""
    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation="san",
        max_length=max_seq_len,
        strict_no_trim=True,
    )
    sampler = DeterministicEpisodeSampler(
        snapshot_dir=snapshot_dir,
        tokenizer=tokenizer,
        exposed_variants=variants,
        k_values=(4,),
        seed=999,
        notation="san",
        max_length=max_seq_len,
        strict_no_trim=True,
        verify_manifest=False,
    )
    best_prompt = ""
    best_target = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    best_len = -1

    # Pick longest demos per variant directly for worst-case K=4 probe, plus sampled episodes
    for v in variants:
        demos_sorted = sorted(
            sampler.demos_by_variant[v],
            key=lambda d: len(d.get("moves_san") or d.get("moves") or []),
            reverse=True,
        )
        top_demos = demos_sorted[:4]
        pos = sampler.positions_by_variant[v][0]
        try:
            res = formatter.format_prompt_with_info(
                top_demos,
                {"fen": pos.get("fen") or pos.get("query_fen"), "blocked_squares": pos.get("blocked_squares")},
            )
            tok_len = int(res.num_tokens or len(tokenizer.encode(res.text)))
            if tok_len > best_len and tok_len <= max_seq_len:
                best_len = tok_len
                best_prompt = res.text
                best_target = np.asarray(pos["label_probs"], dtype=np.float32)
        except ValueError:
            pass

    if mock_small:
        # Construct an explicit longer K=4 sequence bounded by proxy max_position_embeddings
        long_moves = ["e4", "e5", "Nf3", "Nc6"] * 20
        long_demos = [DemoGame(moves=long_moves, outcome="win") for _ in range(4)]
        long_prompt = formatter.format_prompt_with_info(
            long_demos,
            QueryPosition(fen=STANDARD_INITIAL_FEN),
            max_length=min(int(max_seq_len), 1024),
            strict_no_trim=True,
        )
        best_prompt = long_prompt.text
        best_len = int(long_prompt.num_tokens or len(tokenizer.encode(best_prompt)))

    return best_prompt, best_target, best_len


def run_check4_and_5_profile_and_project(
    *,
    arches: Sequence[str],
    mock_small: bool,
    snapshot_dir: Path,
    tokenizer: Any,
    device: torch.device,
    model_path: Optional[str],
    profile_steps: int,
    grad_accum: int,
    max_seq_len: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Checks 4 & 5: Profile mixed-K updates (+ longest K=4), measure memory/disk/val, project."""
    avail_variants = list(verify_snapshot_manifest(snapshot_dir)["exposed_variants"])
    if set(["diagonal_rook", "straight_bishop", "weakened_queen", "extended_pawn"]).issubset(avail_variants):
        prof_variants = resolve_exposure_variants("familiar")
    else:
        prof_variants = avail_variants

    longest_prompt, longest_target, longest_k4_tokens = _find_or_build_longest_k4_episode(
        snapshot_dir=snapshot_dir,
        tokenizer=tokenizer,
        variants=prof_variants,
        max_seq_len=max_seq_len,
        mock_small=mock_small,
    )

    val_episodes = load_val_panel_episodes(
        snapshot_dir=snapshot_dir,
        tokenizer=tokenizer,
        variants=prof_variants,
        k_values=(0, 1, 2, 4),
        roots_per_variant=2 if mock_small else 32,
        seed=0,
        notation="san",
        max_length=max_seq_len,
        strict_no_trim=True,
        verify_manifest=False,
    )

    arch_profiles: dict[str, Any] = {}
    for arch in arches:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t_load0 = time.perf_counter()
        model = _make_model_factory(
            arch=arch,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            seed=42,
        )()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - t_load0

        model.train()
        defaults = ARCH_DEFAULTS[arch]
        optimizer = build_long_study_optimizer(
            model,
            lr=defaults["lr"],
            weight_decay=defaults["weight_decay"],
            betas=(0.9, 0.95),
        )
        from chess_icl_v2.models.qwen_model import build_cosine_warmup_to_min_scheduler

        scheduler = build_cosine_warmup_to_min_scheduler(
            optimizer, total_steps=max(profile_steps, 10)
        )
        sampler = DeterministicEpisodeSampler(
            snapshot_dir=snapshot_dir,
            tokenizer=tokenizer,
            exposed_variants=prof_variants,
            k_values=(0, 1, 2, 4),
            seed=42,
            notation="san",
            max_length=max_seq_len,
            strict_no_trim=True,
            verify_manifest=False,
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_tokens = 0
        opt_step_times: list[float] = []
        step_times: list[float] = []

        t_prof0 = time.perf_counter()
        for step_idx in range(profile_steps):
            s_t0 = time.perf_counter()
            sampler.set_step(step_idx)
            optimizer.zero_grad(set_to_none=True)

            for micro in range(grad_accum):
                if step_idx == 0 and micro == 0:
                    prompt_text = longest_prompt
                    target_probs = longest_target
                else:
                    prompt_text, target_probs, _ = sampler.sample_episode()

                encoded = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].to(device)
                seq_len = int(input_ids.shape[1])
                total_tokens += seq_len
                targets = torch.as_tensor(
                    target_probs, dtype=torch.float32, device=device
                ).unsqueeze(0)

                out = model(input_ids=input_ids, attention_mask=None, target_probs=targets)
                assert out.loss is not None and torch.isfinite(out.loss)
                (out.loss / float(grad_accum)).backward()

            opt_t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            opt_step_times.append(time.perf_counter() - opt_t0)
            step_times.append(time.perf_counter() - s_t0)

        total_profile_seconds = time.perf_counter() - t_prof0

        peak_alloc_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        peak_res_mb = (
            float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        # Measure validation overhead
        val_res = evaluate_val_panel(
            model=model,
            val_episodes=val_episodes,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=max_seq_len,
        )
        val_panel_seconds = float(val_res["eval_seconds"])

        # Measure checkpoint & milestone sizes in temporary directory and discard
        prof_tmp_dir = work_dir / f"profile_tmp_{arch}"
        shutil.rmtree(prof_tmp_dir, ignore_errors=True)
        prof_tmp_dir.mkdir(parents=True, exist_ok=True)
        t_ckpt0 = time.perf_counter()
        ckpt_path = save_rolling_checkpoint(
            output_dir=prof_tmp_dir,
            arch=arch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            step=profile_steps,
            total_tokens_processed=total_tokens,
            total_episodes_processed=profile_steps * grad_accum,
            unique_roots_seen={"probe"},
            unique_families_seen={"probe"},
            gpu_process_seconds=total_profile_seconds,
            wall_seconds=total_profile_seconds,
            best_val_nll=float(val_res["nll"]),
            best_val_step=profile_steps,
            exposure_lineage=[],
            hashes={
                "snapshot_sha256": "profile",
                "model_config_sha256": "profile",
                "tokenizer_sha256": "profile",
                "initial_weights_sha256": "profile",
            },
            run_config={"profile": True},
        )
        ckpt_save_seconds = time.perf_counter() - t_ckpt0
        ckpt_size_mb = float(_directory_size_bytes(ckpt_path)) / (1024.0 * 1024.0)

        milestone_path = save_model_weights(model, arch, prof_tmp_dir / "milestone_probe")
        milestone_size_mb = float(_directory_size_bytes(milestone_path)) / (1024.0 * 1024.0)

        # Discard profile weights
        shutil.rmtree(prof_tmp_dir, ignore_errors=True)
        del optimizer, scheduler, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        avg_step_s = float(sum(step_times) / max(1, len(step_times)))
        avg_opt_s = float(sum(opt_step_times) / max(1, len(opt_step_times)))
        toks_per_sec = float(total_tokens / max(1e-9, total_profile_seconds))

        # Project 1,000-step warmup (eval every 100, ckpt every 250) + 10,000-step main (eval every 500, ckpt every 250)
        warmup_sec = (
            model_load_seconds
            + 1000.0 * avg_step_s
            + 10.0 * val_panel_seconds
            + 4.0 * ckpt_save_seconds
        )
        main_sec = (
            model_load_seconds
            + 10000.0 * avg_step_s
            + 20.0 * val_panel_seconds
            + 40.0 * ckpt_save_seconds
        )
        # Disk per main run: 2 rolling ckpts + 6 milestones + 20 candidates + best_weights
        main_disk_mb = 2.0 * ckpt_size_mb + 27.0 * milestone_size_mb
        warmup_disk_mb = 2.0 * ckpt_size_mb + 14.0 * milestone_size_mb

        arch_profiles[arch] = {
            "passed": True,
            "profile_steps": int(profile_steps),
            "grad_accum": int(grad_accum),
            "longest_k4_tokens_tested": int(longest_k4_tokens),
            "total_tokens_profiled": int(total_tokens),
            "tokens_per_second": toks_per_sec,
            "model_load_seconds": float(model_load_seconds),
            "total_profile_seconds": float(total_profile_seconds),
            "avg_step_seconds": avg_step_s,
            "avg_optimizer_step_seconds": avg_opt_s,
            "peak_allocated_mb": peak_alloc_mb,
            "peak_reserved_mb": peak_res_mb,
            "val_panel_episodes": len(val_episodes),
            "val_panel_seconds": val_panel_seconds,
            "checkpoint_save_seconds": float(ckpt_save_seconds),
            "checkpoint_size_mb": ckpt_size_mb,
            "milestone_weights_size_mb": milestone_size_mb,
            "projections": {
                "warmup_1k_gpu_hours": float(warmup_sec / 3600.0),
                "main_10k_gpu_hours": float(main_sec / 3600.0),
                "warmup_plus_main_gpu_hours": float((warmup_sec + main_sec) / 3600.0),
                "warmup_disk_gb": float(warmup_disk_mb / 1024.0),
                "main_disk_gb": float(main_disk_mb / 1024.0),
            },
        }

    return arch_profiles


def run_preflight(
    *,
    snapshot_dir: Optional[Path] = None,
    output_report: Path,
    profile_steps: int = 60,
    grad_accum: int = 8,
    max_seq_len: int = 32768,
    gpu: int = 0,
    arch: str = "all",
    mock_small: bool = False,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run all 5 preflight checks and write ``preflight_report.json``."""
    arches = ["qwen", "scratch"] if arch == "all" else [arch]
    device = torch.device("cpu") if mock_small else resolve_device(gpu)
    set_global_seed(42)

    tmp_root = Path(tempfile.mkdtemp(prefix="long_study_preflight_"))
    try:
        if mock_small:
            tokenizer: Any = _MockPreflightTokenizer()
            eff_snapshot = (
                snapshot_dir
                if snapshot_dir is not None
                else build_synthetic_preflight_snapshot(tmp_root / "synthetic_snapshot")
            )
            eff_max_seq_len = min(max_seq_len, 1024)
        else:
            if snapshot_dir is None:
                raise ValueError("--snapshot-dir is required unless --mock-small is set")
            resolved_qwen = resolve_qwen_model_path(model_path)
            tokenizer = load_qwen_tokenizer(resolved_qwen, local_files_only=True)
            eff_snapshot = Path(snapshot_dir)
            eff_max_seq_len = max_seq_len

        check1 = run_check1_params_and_grads(
            arches=arches,
            mock_small=mock_small,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
        )
        check2 = run_check2_loss_padding_prompt_sampling(
            arches=arches,
            mock_small=mock_small,
            snapshot_dir=eff_snapshot,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            max_seq_len=eff_max_seq_len,
        )
        check3 = run_check3_save_resume_equivalence(
            arches=arches,
            mock_small=mock_small,
            snapshot_dir=eff_snapshot,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            grad_accum=min(grad_accum, 2) if mock_small else grad_accum,
            max_seq_len=eff_max_seq_len,
            work_dir=tmp_root,
        )
        check4_5 = run_check4_and_5_profile_and_project(
            arches=arches,
            mock_small=mock_small,
            snapshot_dir=eff_snapshot,
            tokenizer=tokenizer,
            device=device,
            model_path=model_path,
            profile_steps=profile_steps,
            grad_accum=min(grad_accum, 2) if mock_small else grad_accum,
            max_seq_len=eff_max_seq_len,
            work_dir=tmp_root,
        )

        # Study-wide aggregate projections (6 warmups + 8 mains)
        total_study_gpu_hours = 0.0
        total_study_disk_gb = 0.0
        if "qwen" in check4_5:
            qp = check4_5["qwen"]["projections"]
            total_study_gpu_hours += 3.0 * qp["warmup_1k_gpu_hours"] + 4.0 * qp["main_10k_gpu_hours"]
            total_study_disk_gb += 3.0 * qp["warmup_disk_gb"] + 4.0 * qp["main_disk_gb"]
        if "scratch" in check4_5:
            sp = check4_5["scratch"]["projections"]
            total_study_gpu_hours += 3.0 * sp["warmup_1k_gpu_hours"] + 4.0 * sp["main_10k_gpu_hours"]
            total_study_disk_gb += 3.0 * sp["warmup_disk_gb"] + 4.0 * sp["main_disk_gb"]

        report = {
            "status": "passed",
            "timestamp": utc_now_iso(),
            "mock_small": bool(mock_small),
            "device": str(device),
            "arches": arches,
            "snapshot_dir": str(eff_snapshot),
            "checks": {
                "check1_params_grads_lineage": check1,
                "check2_loss_padding_prompt_sampling": check2,
                "check3_save_resume_equivalence": check3,
                "check4_5_profile_and_projections": check4_5,
            },
            "study_projections": {
                "num_warmup_jobs": 6,
                "num_main_jobs": 8,
                "total_gpu_hours_all_14_jobs": float(total_study_gpu_hours),
                "total_disk_gb_all_14_jobs": float(total_study_disk_gb),
            },
        }
        atomic_write_json(output_report, report)
        return report
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preflight verification & profiling for long study.")
    p.add_argument("--snapshot-dir", type=Path, default=None)
    p.add_argument(
        "--output-report",
        type=Path,
        default=Path("runs_v2/long_study/preflight_report.json"),
    )
    p.add_argument("--profile-steps", type=int, default=60)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--arch", choices=["all", "qwen", "scratch"], default="all")
    p.add_argument("--mock-small", action="store_true")
    p.add_argument("--model-path", type=str, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_preflight(
        snapshot_dir=args.snapshot_dir,
        output_report=args.output_report,
        profile_steps=args.profile_steps,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        gpu=args.gpu,
        arch=args.arch,
        mock_small=args.mock_small,
        model_path=args.model_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_report": str(args.output_report),
                "study_projections": report["study_projections"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
