"""Sandboxed end-to-end pipeline test for scripts_v2/train_qwen.py and eval_qwen.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from chess_icl_v2.models.qwen_model import STANDARD_INITIAL_FEN, QwenICLClassifier
from scripts_v2.eval_qwen import parse_args as parse_eval_args
from scripts_v2.eval_qwen import run_evaluation
from scripts_v2.train_qwen import parse_args as parse_train_args
from scripts_v2.train_qwen import resolve_variants, run_training


class MockPipelineTokenizer:
    """Fast deterministic tokenizer bounded inside tiny Qwen2Config(vocab_size=256)."""

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


def _tiny_qwen_cfg() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        use_cache=False,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_synthetic_dataset(data_dir: Path) -> None:
    """Create minimal v2-contract dataset for standard + 2 variants."""
    sample_games = [
        (["e2e4", "e7e5", "g1f3", "b8c6"], ["e4", "e5", "Nf3", "Nc6"], "win"),
        (["d2d4", "d7d5", "c2c4", "e7e6"], ["d4", "d5", "c4", "e6"], "draw"),
        (["c2c4", "e7e5", "b1c3", "g8f6"], ["c4", "e5", "Nc3", "Nf6"], "loss"),
        (["g1f3", "d7d5", "d2d4", "g8f6"], ["Nf3", "d5", "d4", "Nf6"], "win"),
    ]
    sample_fens = [
        (STANDARD_INITIAL_FEN, "w", [0.4, 0.3, 0.3], 0, True),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "w",
            [0.6, 0.2, 0.2],
            0,
            True,
        ),
        (
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d6 0 2",
            "b",
            [0.2, 0.3, 0.5],
            1,
            False,
        ),
        (
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "w",
            [0.7, 0.2, 0.1],
            0,
            True,
        ),
    ]

    for variant_idx, variant in enumerate(["standard", "diagonal_rook", "straight_bishop"]):
        v_dir = data_dir / variant
        demos = []
        for g_idx, (uci_mvs, san_mvs, res) in enumerate(sample_games):
            rec: dict[str, Any] = {
                "game_id": f"{variant}_g{g_idx}",
                "variant": variant,
                "moves": list(uci_mvs),
                "result": res,
                "init_fen": STANDARD_INITIAL_FEN,
                "blocked_squares": None,
                "policy_id": "uniform_legal_v1",
                "move_count": len(uci_mvs),
            }
            # Omit moves_san on straight_bishop to verify UCI->SAN conversion via VariantBoard
            if variant != "straight_bishop":
                rec["moves_san"] = list(san_mvs)
            demos.append(rec)
        _write_jsonl(v_dir / "games_demo.jsonl", demos)

        for split in ("train", "val", "test"):
            pos_rows = []
            for p_idx, (fen, side, probs, mdiff, eq_mat) in enumerate(sample_fens):
                pos_rows.append(
                    {
                        "root_id": f"{variant}_{split}_r{p_idx}",
                        "variant": variant,
                        "fen": fen,
                        "blocked_squares": None,
                        "side_to_move": side,
                        "label_probs": probs,
                        "source_game_id": f"{variant}_g{p_idx % 2}",
                        "ply": p_idx + variant_idx,
                        "material_diff_white": mdiff,
                        "equal_material": eq_mat,
                    }
                )
            _write_jsonl(v_dir / f"positions_{split}.jsonl", pos_rows)


def test_cli_parsers_and_variant_presets(tmp_path: Path) -> None:
    t_args = parse_train_args(
        [
            "--mode",
            "continue",
            "--lora-mode",
            "full_lora",
            "--notation",
            "san",
            "--variants",
            "move_variants",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert t_args.mode == "continue"
    assert t_args.lora_mode == "full_lora"
    assert resolve_variants(t_args.variants, mode="continue") == [
        "diagonal_rook",
        "straight_bishop",
        "weakened_queen",
        "extended_pawn",
    ]

    e_args = parse_eval_args(
        [
            "--zero-shot-base",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "eval"),
            "--split",
            "test",
        ]
    )
    assert e_args.zero_shot_base is True
    assert e_args.split == "test"


def test_warmup_continue_and_eval_pipeline_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _build_synthetic_dataset(data_dir)

    tok = MockPipelineTokenizer()
    cfg = _tiny_qwen_cfg()
    torch.manual_seed(7)
    base_init = Qwen2ForCausalLM(cfg)
    base_snapshot = {k: v.detach().clone() for k, v in base_init.state_dict().items()}

    # 1. Warmup training (1 step on standard)
    warmup_out = tmp_path / "runs" / "warmup"
    warmup_model = QwenICLClassifier(
        base_model=base_init,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
        gradient_checkpointing=True,
    )
    warmup_res = run_training(
        mode="warmup",
        lora_mode="handoff",
        notation="san",
        variants=["standard"],
        data_dir=data_dir,
        output_dir=warmup_out,
        seed=42,
        lr=1e-2,
        batch_size=2,
        grad_accum=1,
        max_steps=1,
        eval_interval=1,
        k_values=[0, 2],
        val_k_values=[0, 2],
        val_max_roots=4,
        max_seq_len=1024,
        gpu=-1,
        model=warmup_model,
        tokenizer=tok,
    )

    warmup_best = warmup_out / "best_adapter"
    assert (warmup_out / "history.json").exists()
    assert (warmup_out / "train_config.json").exists()
    assert (warmup_best / "adapter_config.json").exists()
    assert (warmup_best / "adapter_model.safetensors").exists()
    assert (warmup_best / "classifier_head.pt").exists()
    warmup_hist = json.loads((warmup_out / "history.json").read_text())
    assert len(warmup_hist) == 1
    assert np.isfinite(warmup_res["best_val_nll"])

    # 2. Continue-training cloned from warmup best_adapter
    continue_out = tmp_path / "runs" / "continue"
    fresh_base_cont = Qwen2ForCausalLM(cfg)
    fresh_base_cont.load_state_dict(base_snapshot)
    cont_model = QwenICLClassifier(
        base_model=fresh_base_cont,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
        gradient_checkpointing=True,
    )
    cont_res = run_training(
        mode="continue",
        lora_mode="handoff",
        notation="san",
        variants=["diagonal_rook", "straight_bishop"],
        data_dir=data_dir,
        output_dir=continue_out,
        init_adapter_dir=warmup_best,
        seed=43,
        lr=1e-2,
        batch_size=2,
        grad_accum=1,
        max_steps=2,
        eval_interval=1,
        k_values=[0, 2],
        val_k_values=[0, 2],
        val_max_roots=4,
        max_seq_len=1024,
        gpu=-1,
        model=cont_model,
        tokenizer=tok,
    )

    cont_hist = json.loads((continue_out / "history.json").read_text())
    assert len(cont_hist) == 2
    assert cont_hist[0]["step"] == 1 and cont_hist[1]["step"] == 2
    assert cont_res["best_step"] in (1, 2)
    assert (continue_out / "best_adapter" / "classifier_head.pt").exists()

    # 3. Reload saved best_adapter into fresh instance and evaluate with all 3 controls
    fresh_base_eval = Qwen2ForCausalLM(cfg)
    fresh_base_eval.load_state_dict(base_snapshot)
    eval_model = QwenICLClassifier.from_adapter_and_head(
        continue_out / "best_adapter",
        base_model=fresh_base_eval,
        tokenizer=tok,
        is_trainable=False,
    )
    eval_out = tmp_path / "eval_out"
    eval_payload = run_evaluation(
        data_dir=data_dir,
        output_dir=eval_out,
        adapter_dir=continue_out / "best_adapter",
        split="test",
        variants=["diagonal_rook", "straight_bishop"],
        k_values=[0, 2],
        num_draws=2,
        controls=["mismatched_context", "corrupted_outcomes", "corrupted_trajectories"],
        notation="san",
        max_seq_len=1024,
        batch_size=2,
        max_roots=4,
        n_boot=20,
        seed=42,
        gpu=-1,
        model=eval_model,
        tokenizer=tok,
    )

    eval_json_path = eval_out / "eval_test.json"
    assert eval_json_path.exists()
    loaded_eval = json.loads(eval_json_path.read_text())
    summary = loaded_eval["summary"]
    assert "0" in summary["clean"] and "2" in summary["clean"]
    assert "context_gain" in summary["clean"]["2"]
    assert summary["equal_material"] is not None
    assert "2" in summary["equal_material"]
    for ctrl in ("mismatched_context", "corrupted_outcomes", "corrupted_trajectories"):
        assert ctrl in summary["controls"]
        assert "2" in summary["controls"][ctrl]
        assert "delta_nll_vs_clean" in summary["controls"][ctrl]["2"]
    assert len(eval_payload["per_root_predictions"]) > 0

    # 4. Verify zero-shot base evaluation path also runs cleanly
    zs_base = Qwen2ForCausalLM(cfg)
    zs_base.load_state_dict(base_snapshot)
    zs_model = QwenICLClassifier(
        base_model=zs_base,
        tokenizer=tok,
        lora_mode="handoff",
        torch_dtype=torch.float32,
        head_dtype=torch.float32,
    )
    zs_out = tmp_path / "eval_zs"
    zs_payload = run_evaluation(
        data_dir=data_dir,
        output_dir=zs_out,
        zero_shot_base=True,
        split="test",
        variants=["diagonal_rook"],
        k_values=[0, 2],
        num_draws=1,
        controls=["none"],
        notation="uci",
        max_seq_len=1024,
        batch_size=2,
        max_roots=2,
        n_boot=10,
        seed=42,
        gpu=-1,
        model=zs_model,
        tokenizer=tok,
    )
    assert zs_payload["metadata"]["zero_shot_base"] is True
    assert (zs_out / "eval_test.json").exists()
