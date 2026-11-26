#!/usr/bin/env python3
"""Frozen post-training evaluation suite for Chess-Variant ICL v2 long study.

Subcommands:
  1. select-checkpoint  Score 500-step candidate checkpoints on full validation fold
                        (canonical unpadded batch_size=1) and write frozen
                        `val_selection_manifest.json`.
  2. evaluate           Evaluate a frozen checkpoint manifest on test (or val),
                        verifying weight freeze, uncached repeat determinism (<=1e-5),
                        model/token-ID prediction cache, feasible vs common cohorts,
                        10% token-matched foreign & corruption controls, separate
                        familiar / 9-rule transfer / standard reporting, and
                        optional R256 test subset re-estimation.
  3. select-r256-subset Select M test families per variant using an outcome-independent
                        `sha256(family_id)` rule fixed prior to reading model scores.
  4. run-baselines      Evaluate all 7 declared non-neural baselines fitted strictly
                        on allowed training data over the frozen test episodes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

import chess
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.dataset_v2 import (  # noqa: E402
    _stable_seed,
    load_jsonl,
    normalize_demo_record,
)
from chess_icl_v2.eval.baselines import (  # noqa: E402
    CONTEXT_INVARIANT_BASELINE_NAMES,
    DECLARED_BASELINE_NAMES,
    build_declared_baselines,
    compute_standard_material_diff,
)
from chess_icl_v2.eval.controls import (  # noqa: E402
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_mismatched_context_episodes,
    make_token_matched_foreign_episodes,
)
from chess_icl_v2.eval.metrics import (  # noqa: E402
    compute_argmax_accuracy,
    compute_brier,
    compute_brier_per_sample,
    compute_ece,
    compute_mc_noise_floor,
    compute_nll,
    compute_side_calibration,
)
from chess_icl_v2.eval.uncertainty import aggregate_evaluation_records  # noqa: E402
from chess_icl_v2.models.qwen_model import (  # noqa: E402
    QwenICLClassifier,
    QwenPromptFormatter,
    get_preferred_attn_implementation,
    load_qwen_tokenizer,
    resolve_qwen_model_path,
)
from chess_icl_v2.models.scratch_qwen2 import ScratchQwen2Classifier  # noqa: E402
from chess_icl_v2.variants import (  # noqa: E402
    COMPOSITIONAL_VARIANTS,
    SINGLE_VARIANTS,
    TEST_VARIANTS,
    TRAIN_VARIANTS,
)
from scripts_v2.train_qwen import (  # noqa: E402
    prepare_demo_for_notation,
    prepare_episodes_for_notation,
    resolve_device,
    set_global_seed,
)

DEFAULT_K_VALUES: tuple[int, ...] = (0, 1, 2, 4)
DEFAULT_CONTROLS: tuple[str, ...] = (
    "token_matched_foreign",
    "corrupted_outcomes",
    "corrupted_trajectories",
)
VALID_CONTROLS: frozenset[str] = frozenset(
    {
        "token_matched_foreign",
        "foreign_matched",
        "corrupted_outcomes",
        "corrupted_outcomes_inverted",
        "corrupted_outcomes_scrambled",
        "corrupted_outcomes_all_draw",
        "corrupted_trajectories",
        "shuffled_trajectories",
        "mismatched_context",
    }
)
STEP_DIR_RE = re.compile(r"^(?:step|checkpoint)[_-]?(\d+)$")


# ---------------------------------------------------------------------------
# Hashing, Snapshot & Manifest Utilities
# ---------------------------------------------------------------------------


def compute_checkpoint_sha256(checkpoint_path: Path) -> str:
    """Compute a deterministic SHA-256 hash over checkpoint weight/config files."""
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    h = hashlib.sha256()
    if p.is_file():
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
        return h.hexdigest()

    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    selected_files = sorted(
        f
        for f in p.rglob("*")
        if f.is_file()
        and (f.suffix in weight_suffixes or f.name in {"adapter_config.json", "config.json"})
    )
    if not selected_files:
        selected_files = sorted(f for f in p.rglob("*") if f.is_file())
    if not selected_files:
        raise ValueError(f"Checkpoint directory is empty: {p}")

    for f in selected_files:
        rel = f.relative_to(p).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(f.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def compute_model_state_sha256(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 hash over an in-memory PyTorch model state_dict."""
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state.keys()):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        # Convert bf16/fp16 safely via uint16 view or float32 bytes
        if tensor.dtype == torch.bfloat16:
            raw = tensor.view(torch.uint16).numpy().tobytes()
        else:
            raw = tensor.numpy().tobytes()
        h.update(raw)
    return h.hexdigest()


def compute_snapshot_sha256(
    data_dir: Path,
    variants: Optional[Sequence[str]] = None,
) -> str:
    """Compute deterministic SHA-256 over the frozen split JSONL files under `data_dir`."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    if variants is not None:
        variant_dirs = [root / v for v in sorted(set(variants)) if (root / v).is_dir()]
    else:
        variant_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    target_names = (
        "games_demo.jsonl",
        "games_demo_train.jsonl",
        "games_demo_val.jsonl",
        "games_demo_test.jsonl",
        "positions_train.jsonl",
        "positions_val.jsonl",
        "positions_test.jsonl",
    )
    h = hashlib.sha256()
    hashed_any = False
    for v_dir in variant_dirs:
        for fname in target_names:
            fpath = v_dir / fname
            if fpath.exists():
                hashed_any = True
                rel = fpath.relative_to(root).as_posix()
                h.update(rel.encode("utf-8"))
                h.update(b"\x00")
                h.update(fpath.read_bytes())
                h.update(b"\x00")
    if not hashed_any:
        raise ValueError(f"No split JSONL files found under {root}")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Split-Aware Data & Episode Loading
# ---------------------------------------------------------------------------


def discover_variants(data_dir: Path, split: str = "train") -> list[str]:
    """Discover variant subdirectories containing ``positions_<split>.jsonl`` and split/shared demos."""
    root = Path(data_dir)
    if not root.exists():
        return []
    found: list[str] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        has_pos = (sub / f"positions_{split}.jsonl").exists()
        has_demo = (sub / f"games_demo_{split}.jsonl").exists() or (sub / "games_demo.jsonl").exists()
        if has_pos and has_demo:
            found.append(sub.name)
    return found


def load_split_demos(data_dir: Path, variant: str, split: str) -> list[dict[str, Any]]:
    """Load split-specific `games_demo_<split>.jsonl` if present, else `games_demo.jsonl`."""
    v_dir = Path(data_dir) / variant
    split_path = v_dir / f"games_demo_{split}.jsonl"
    fallback_path = v_dir / "games_demo.jsonl"
    target = split_path if split_path.exists() else fallback_path
    if not target.exists():
        raise FileNotFoundError(f"Demo file not found for {variant!r} ({split}): {target}")
    return [normalize_demo_record(d) for d in load_jsonl(target)]


def load_split_positions(
    data_dir: Path,
    variant: str,
    split: str,
    positions_filename: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Load position records for `(variant, split)`."""
    fname = positions_filename if positions_filename is not None else f"positions_{split}.jsonl"
    path = Path(data_dir) / variant / fname
    return load_jsonl(path)


def _extract_y_true(pos: Mapping[str, Any]) -> list[float]:
    for key in ("y_true", "label_probs", "target_probs", "target", "probs"):
        val = pos.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape == (3,):
                s = float(np.sum(arr))
                if s > 0:
                    return (arr / s).tolist()
    return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def _extract_side(pos: Mapping[str, Any], fen: str) -> str:
    raw = pos.get("side_to_move")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in {"w", "white"}:
            return "w"
        if s in {"b", "black"}:
            return "b"
    parts = fen.strip().split()
    if len(parts) >= 2 and parts[1].lower() in {"w", "b"}:
        return parts[1].lower()
    return "w"


def _extract_family_id(pos: Mapping[str, Any], fallback: Any) -> str:
    for key in ("family_id", "source_game_id", "source_game", "game_id"):
        val = pos.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return str(fallback)


def build_long_study_episodes(
    data_dir: Path,
    variant: str,
    split: str,
    k: int,
    draw_seed: int = 0,
    max_roots: Optional[int] = None,
    notation: Literal["uci", "san"] = "san",
    positions_filename: Optional[str] = None,
    allowed_root_ids: Optional[set[str]] = None,
    excluded_root_ids: Optional[set[str]] = None,
    demos_data_dir: Optional[Path] = None,
    allowed_family_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Deterministically build normalized evaluation episodes for `(variant, split, k, draw_seed)`."""
    positions = load_split_positions(
        data_dir, variant=variant, split=split, positions_filename=positions_filename
    )
    if allowed_root_ids is not None:
        positions = [
            p
            for idx, p in enumerate(positions)
            if str(p.get("root_id") or p.get("id") or f"{variant}_{split}_{idx}")
            in allowed_root_ids
        ]
    if allowed_family_ids is not None:
        positions = [
            p
            for idx, p in enumerate(positions)
            if _extract_family_id(
                p, fallback=str(p.get("root_id") or p.get("id") or f"{variant}_{split}_{idx}")
            )
            in allowed_family_ids
        ]
    if excluded_root_ids:
        positions = [
            p
            for idx, p in enumerate(positions)
            if str(p.get("root_id") or p.get("id") or f"{variant}_{split}_{idx}")
            not in excluded_root_ids
        ]
    if max_roots is not None:
        positions = positions[:max_roots]

    demo_pool = load_split_demos(demos_data_dir if demos_data_dir is not None else data_dir, variant=variant, split=split)
    rng = np.random.default_rng(_stable_seed(variant, split, int(k), int(draw_seed)))

    episodes: list[dict[str, Any]] = []
    for idx, pos in enumerate(positions):
        root_id = str(pos.get("root_id") or pos.get("id") or f"{variant}_{split}_{idx}")
        family_id = _extract_family_id(pos, fallback=root_id)
        fen = str(pos.get("query_fen") or pos.get("fen") or chess.STARTING_FEN)
        side = _extract_side(pos, fen)
        y_true = _extract_y_true(pos)
        r_val = int(pos.get("R") or pos.get("n_rollouts") or pos.get("num_rollouts") or 20)

        if "equal_material" in pos and pos["equal_material"] is not None:
            eq_mat = bool(pos["equal_material"])
        else:
            eq_mat = abs(compute_standard_material_diff(fen)) <= 2

        k_eff = min(int(k), len(demo_pool))
        if k_eff > 0:
            cand = [
                i
                for i, d in enumerate(demo_pool)
                if d.get("game_id") is None or str(d.get("game_id")) != family_id
            ]
            if len(cand) < k_eff:
                cand = list(range(len(demo_pool)))
            chosen_idx = rng.choice(cand, size=k_eff, replace=False)
            demos = [dict(demo_pool[int(i)]) for i in chosen_idx]
        else:
            demos = []

        episodes.append(
            {
                "root_id": root_id,
                "family_id": family_id,
                "source_game_id": family_id,
                "variant": str(pos.get("variant", variant)),
                "split": split,
                "k": int(k),
                "draw_seed": int(draw_seed),
                "demos": demos,
                "query_fen": fen,
                "blocked_squares": pos.get("blocked_squares"),
                "side_to_move": side,
                "y_true": y_true,
                "R": r_val,
                "ply": int(pos.get("ply", 0)),
                "equal_material": eq_mat,
            }
        )

    return prepare_episodes_for_notation(episodes, notation=notation)


def build_training_episodes_strict(
    data_dir: Path,
    k_values: Sequence[int],
    seed: int = 42,
    variants: Optional[Sequence[str]] = None,
    max_per_variant: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Build baseline fitting episodes strictly from `positions_train.jsonl` + train demos."""
    train_variants = (
        list(variants) if variants is not None else discover_variants(data_dir, split="train")
    )
    if not train_variants:
        raise FileNotFoundError(
            f"No variants with positions_train.jsonl + games_demo*.jsonl under {data_dir}"
        )

    rng = np.random.default_rng(seed + 99991)
    train_episodes: list[dict[str, Any]] = []
    for variant in train_variants:
        demos = load_split_demos(data_dir, variant=variant, split="train")
        raw_pos = load_split_positions(data_dir, variant=variant, split="train")
        if max_per_variant is not None:
            raw_pos = raw_pos[:max_per_variant]

        for idx, pos in enumerate(raw_pos):
            root_id = str(pos.get("root_id") or pos.get("id") or f"{variant}_train_{idx}")
            family_id = _extract_family_id(pos, fallback=root_id)
            fen = str(pos.get("query_fen") or pos.get("fen") or chess.STARTING_FEN)
            side = _extract_side(pos, fen)
            y_true = _extract_y_true(pos)

            valid_ks = [int(k) for k in k_values if int(k) <= len(demos)] or [0]
            k_pick = int(valid_ks[int(rng.integers(0, len(valid_ks)))])
            if k_pick > 0:
                chosen = rng.choice(len(demos), size=k_pick, replace=False)
                sampled_demos = [dict(demos[int(i)]) for i in chosen]
            else:
                sampled_demos = []

            train_episodes.append(
                {
                    "root_id": root_id,
                    "family_id": family_id,
                    "source_game_id": family_id,
                    "variant": variant,
                    "query_fen": fen,
                    "blocked_squares": pos.get("blocked_squares"),
                    "side_to_move": side,
                    "y_true": y_true,
                    "demos": sampled_demos,
                }
            )
    return train_episodes


def resolve_variants_for_regime(
    data_dir: Path,
    split: str,
    regime: str = "all",
    explicit_variants: Optional[Sequence[str]] = None,
) -> list[str]:
    """Resolve variant list for selection or evaluation while keeping exclusions clear."""
    available = discover_variants(data_dir, split=split)
    if explicit_variants:
        return [v for v in explicit_variants if v in available] or list(explicit_variants)

    if regime in {"familiar", "train_variants"}:
        return [v for v in TRAIN_VARIANTS if v in available]
    if regime in {"warmup", "standard"}:
        return ["standard"] if "standard" in available else []
    if regime in {"heldout", "test_variants"}:
        return [v for v in TEST_VARIANTS if v in available]
    if regime in {"all_single", "all_variants"}:
        return [v for v in SINGLE_VARIANTS if v in available]
    if regime == "all":
        return available
    raise ValueError(f"Unsupported regime {regime!r}")


# ---------------------------------------------------------------------------
# Canonical Unpadded Single-Episode Predictor & Integrity Checks
# ---------------------------------------------------------------------------


def verify_model_frozen(model: nn.Module) -> None:
    """Verify `model` is in eval() mode and all parameters have `requires_grad == False`."""
    if model.training:
        raise RuntimeError("Model must be in eval() mode before evaluation")
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"Model weights are not frozen (found {len(trainable)} params with requires_grad=True, "
            f"e.g. {trainable[:3]})"
        )


def freeze_model_in_place(model: nn.Module) -> None:
    """Set eval() mode and disable gradients on all parameters."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    verify_model_frozen(model)


class CanonicalQwenPredictor:
    """Unpadded single-episode (batch_size=1) predictor with model-bound token-ID cache."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        model_sha256: Optional[str] = None,
        notation: Literal["uci", "san"] = "san",
        max_seq_len: int = 32768,
        device: Optional[torch.device] = None,
    ) -> None:
        freeze_model_in_place(model)
        self.model = model
        self.tokenizer = tokenizer
        self.notation: Literal["uci", "san"] = notation
        self.max_seq_len = int(max_seq_len)
        try:
            param_device = next(model.parameters()).device
        except StopIteration:
            param_device = None
        if param_device is not None:
            self.device = param_device
        elif device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.model_sha256 = (
            model_sha256 if model_sha256 is not None else compute_model_state_sha256(model)
        )
        self.formatter = QwenPromptFormatter(
            tokenizer=tokenizer,
            notation=notation,
            max_length=self.max_seq_len,
        )
        # Keyed strictly by (model_sha256, input_ids_sha256)
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def _encode_prompt_unpadded(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        if hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            except TypeError:
                ids = self.tokenizer.encode(prompt)
        else:
            enc = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            ids_t = enc["input_ids"] if isinstance(enc, Mapping) else enc.input_ids
            ids = [int(x) for x in ids_t[0].tolist()]

        if len(ids) > self.max_seq_len:
            raise ValueError(
                f"Unpadded prompt length {len(ids)} exceeds max_seq_len={self.max_seq_len}"
            )
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_sha256 = hashlib.sha256(ids_np.tobytes()).hexdigest()
        input_ids = torch.tensor(ids_np, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        return input_ids, attention_mask, ids_sha256

    @torch.no_grad()
    def predict_single_uncached(self, prompt: str) -> np.ndarray:
        """Run canonical unpadded single-episode inference (batch_size=1) bypassing cache."""
        verify_model_frozen(self.model)
        input_ids, attention_mask, _ = self._encode_prompt_unpadded(prompt)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = out.probs if hasattr(out, "probs") else out["probs"]
        arr = probs[0].detach().float().cpu().numpy().astype(np.float64)
        arr = np.clip(arr, 0.0, 1.0)
        s = float(np.sum(arr))
        return arr / s if s > 0 else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    def verify_uncached_repeat_determinism(
        self,
        prompts: Sequence[str],
        tol: float = 1e-5,
        max_samples: int = 3,
    ) -> dict[str, Any]:
        """Verify two uncached batch_size=1 passes agree within `tol`."""
        sample = list(prompts[:max_samples])
        if not sample:
            return {"checked_samples": 0, "max_abs_diff": 0.0, "tol": float(tol), "passed": True}

        max_diff = 0.0
        for p in sample:
            p1 = self.predict_single_uncached(p)
            p2 = self.predict_single_uncached(p)
            diff = float(np.max(np.abs(p1 - p2)))
            max_diff = max(max_diff, diff)
            if diff > tol:
                raise RuntimeError(
                    f"Uncached repeat determinism check failed: max_abs_diff={diff:.3e} > tol={tol:.3e}"
                )

        return {
            "checked_samples": len(sample),
            "max_abs_diff": float(max_diff),
            "tol": float(tol),
            "passed": True,
        }

    @torch.no_grad()
    def predict_episode(self, ep: Mapping[str, Any]) -> dict[str, Any]:
        """Format and score one episode using unpadded batch_size=1 + model/token-ID cache."""
        verify_model_frozen(self.model)
        query_obj = {
            "fen": ep["query_fen"],
            "blocked_squares": ep.get("blocked_squares"),
        }
        info = self.formatter.format_prompt_with_info(ep.get("demos", []), query_obj)
        input_ids, attention_mask, ids_sha256 = self._encode_prompt_unpadded(info.text)
        cache_key = (self.model_sha256, ids_sha256)

        if cache_key in self._cache:
            self.cache_hits += 1
            probs = self._cache[cache_key].copy()
            hit = True
        else:
            self.cache_misses += 1
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs_t = out.probs if hasattr(out, "probs") else out["probs"]
            arr = probs_t[0].detach().float().cpu().numpy().astype(np.float64)
            arr = np.clip(arr, 0.0, 1.0)
            s = float(np.sum(arr))
            probs = (
                arr / s
                if s > 0
                else np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
            )
            self._cache[cache_key] = probs.copy()
            hit = False

        return {
            "p_pred": probs,
            "num_tokens": int(info.num_tokens if info.num_tokens is not None else input_ids.shape[1]),
            "num_demos_total": int(info.num_demos_total),
            "num_demos_kept": int(info.num_demos_kept),
            "num_demos_dropped": int(info.num_demos_dropped),
            "input_ids_sha256": ids_sha256,
            "cache_hit": hit,
        }


# ---------------------------------------------------------------------------
# Metrics, Calibration, Cohorts & Regime Reporting
# ---------------------------------------------------------------------------


def compute_categorical_brier_per_sample(p_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute multiclass Brier score against hard argmax one-hot target: sum (p - e_{y*})^2."""
    if len(p_pred) == 0:
        return np.empty((0,), dtype=np.float64)
    true_cls = np.argmax(y_true, axis=-1)
    y_onehot = np.zeros_like(y_true, dtype=np.float64)
    y_onehot[np.arange(len(y_true)), true_cls] = 1.0
    return compute_brier_per_sample(p_pred, y_onehot)


def _calibration_with_counts(
    p_mat: np.ndarray,
    y_mat: np.ndarray,
    sides: Sequence[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute ECE, classwise ECE, bin sample counts, and per-side calibration."""
    n = len(p_mat)
    ece_stats = compute_ece(p_mat, y_mat, n_bins=n_bins)
    side_cal = compute_side_calibration(p_mat, y_mat, sides, n_bins=n_bins)

    if n > 0:
        pred_cls = np.argmax(p_mat, axis=-1)
        top_conf = p_mat[np.arange(n), pred_cls]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_counts: list[int] = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (top_conf >= lo) & (top_conf <= hi) if b == 0 else (top_conf > lo) & (top_conf <= hi)
            bin_counts.append(int(np.sum(mask)))

        for side_key, target_side in (("white", "w"), ("black", "b")):
            mask = np.array([str(s).lower().startswith(target_side) for s in sides], dtype=bool)
            if np.any(mask):
                p_s = p_mat[mask]
                y_s = y_mat[mask]
                side_cal[side_key]["brier_vector"] = float(compute_brier(p_s, y_s))
                side_cal[side_key]["brier_categorical"] = float(
                    np.mean(compute_categorical_brier_per_sample(p_s, y_s))
                )
                side_cal[side_key]["accuracy"] = float(
                    compute_argmax_accuracy(p_s, y_s)["accuracy"]
                )
            else:
                side_cal[side_key]["brier_vector"] = None
                side_cal[side_key]["brier_categorical"] = None
                side_cal[side_key]["accuracy"] = None
    else:
        bin_counts = [0] * n_bins

    return {
        "n_samples": int(n),
        "ece": float(ece_stats["ece"]),
        "classwise_ece": float(ece_stats["classwise_ece"]),
        "bin_counts": bin_counts,
        "side_calibration": side_cal,
    }


def _partition_variants_by_regime(variants: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Partition variants into strictly separated evaluation regimes (no leakage of labels)."""
    v_set = list(dict.fromkeys(variants))
    regimes: dict[str, dict[str, Any]] = {}

    familiar = [v for v in TRAIN_VARIANTS if v in v_set]
    if familiar:
        regimes["familiar_adaptation"] = {
            "variants": familiar,
            "role": "familiar_rule_adaptation_in_distribution",
            "is_transfer": False,
        }

    heldout = [v for v in TEST_VARIANTS if v in v_set]
    if heldout:
        regimes["heldout_transfer"] = {
            "variants": heldout,
            "role": "gradient_held_out_transfer",
            "is_transfer": True,
        }

    if "standard" in v_set:
        regimes["standard_control"] = {
            "variants": ["standard"],
            "role": "standard_chess_control",
            "is_transfer": False,
        }

    compositional = [v for v in COMPOSITIONAL_VARIANTS if v in v_set]
    if compositional:
        regimes["compositional_transfer"] = {
            "variants": compositional,
            "role": "compositional_transfer",
            "is_transfer": True,
        }

    known = set(TRAIN_VARIANTS) | set(TEST_VARIANTS) | {"standard"} | set(COMPOSITIONAL_VARIANTS)
    custom = [v for v in v_set if v not in known]
    if custom:
        regimes["custom"] = {
            "variants": custom,
            "role": "custom_variants",
            "is_transfer": False,
        }
    return regimes


def _uncertainty_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(r.get("model_seed", 0)) for r in records})
    n_seeds = len(seeds)
    return {
        "cluster_unit": "family_id",
        "n_seeds": int(n_seeds),
        "small_seed_caveat": (
            f"Small seed count (S={n_seeds} <= 3): across-seed SE is noisy and should be "
            "interpreted cautiously alongside family/game cluster bootstrap CIs."
            if n_seeds <= 3
            else None
        ),
        "mc_label_uncertainty_kept_separate": True,
    }


def _summarize_record_group(
    records: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    k0_lookup: Optional[Mapping[tuple[str, int, str], float]] = None,
    clean_lookup: Optional[Mapping[tuple[str, int, int, str], float]] = None,
) -> Optional[dict[str, Any]]:
    """Summarize one cohort/regime/k slice with hierarchical cluster CIs & calibration."""
    if not records:
        return None

    nll_agg = aggregate_evaluation_records(
        records, value_key="nll", game_key="family_id", n_boot=n_boot, seed=seed
    )
    brier_vec_agg = aggregate_evaluation_records(
        records, value_key="brier_vector", game_key="family_id", n_boot=n_boot, seed=seed + 11
    )
    brier_cat_agg = aggregate_evaluation_records(
        records, value_key="brier_categorical", game_key="family_id", n_boot=n_boot, seed=seed + 17
    )
    acc_agg = aggregate_evaluation_records(
        records, value_key="correct", game_key="family_id", n_boot=n_boot, seed=seed + 23
    )

    p_mat = np.asarray([r["p_pred"] for r in records], dtype=np.float64)
    y_mat = np.asarray([r["y_true"] for r in records], dtype=np.float64)
    sides = [str(r.get("side_to_move", "w")) for r in records]
    calibration = _calibration_with_counts(p_mat, y_mat, sides=sides, n_bins=10)

    # Separate MC noise floor per variant (averaged across unique roots)
    mc_by_variant: dict[str, Any] = {}
    by_var_unique_roots: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in records:
        by_var_unique_roots[str(r["variant"])][str(r["root_id"])] = r
    for v, root_map in sorted(by_var_unique_roots.items()):
        ordered_roots = [root_map[rid] for rid in sorted(root_map.keys())]
        yv = np.asarray([rr["y_true"] for rr in ordered_roots], dtype=np.float64)
        r_eff = max(1, int(round(float(np.mean([rr.get("R", 20) for rr in ordered_roots])))))
        mc_by_variant[v] = compute_mc_noise_floor(yv, R=r_eff)

    mc_macro = {
        key: float(np.mean([m[key] for m in mc_by_variant.values()]))
        for key in (
            "entropy_floor",
            "debiased_entropy_floor",
            "mc_nll_bias",
            "mc_nll_variance_per_root",
            "mc_brier_floor",
            "mc_brier_variance",
        )
    }

    # Equal-material slice
    eq_recs = [r for r in records if bool(r.get("equal_material", False))]
    if eq_recs:
        eq_p = np.asarray([r["p_pred"] for r in eq_recs], dtype=np.float64)
        eq_y = np.asarray([r["y_true"] for r in eq_recs], dtype=np.float64)
        eq_sides = [str(r.get("side_to_move", "w")) for r in eq_recs]
        eq_summary: dict[str, Any] = {
            "n_records": len(eq_recs),
            "n_roots": len({(str(r["variant"]), str(r["root_id"])) for r in eq_recs}),
            "nll": aggregate_evaluation_records(
                eq_recs, value_key="nll", game_key="family_id", n_boot=n_boot, seed=seed + 31
            ),
            "brier_vector": aggregate_evaluation_records(
                eq_recs,
                value_key="brier_vector",
                game_key="family_id",
                n_boot=n_boot,
                seed=seed + 37,
            ),
            "brier_categorical": aggregate_evaluation_records(
                eq_recs,
                value_key="brier_categorical",
                game_key="family_id",
                n_boot=n_boot,
                seed=seed + 41,
            ),
            "accuracy": aggregate_evaluation_records(
                eq_recs, value_key="correct", game_key="family_id", n_boot=n_boot, seed=seed + 43
            ),
            "calibration": _calibration_with_counts(eq_p, eq_y, sides=eq_sides, n_bins=10),
        }
    else:
        eq_summary = {"n_records": 0, "n_roots": 0}

    # Per-variant summary (calibration, side, equal-material, mc_noise_floor)
    per_variant_extra: dict[str, Any] = {}
    for v in sorted(by_var_unique_roots.keys()):
        v_recs = [r for r in records if str(r["variant"]) == v]
        vp = np.asarray([r["p_pred"] for r in v_recs], dtype=np.float64)
        vy = np.asarray([r["y_true"] for r in v_recs], dtype=np.float64)
        vsides = [str(r.get("side_to_move", "w")) for r in v_recs]
        per_variant_extra[v] = {
            "n_records": len(v_recs),
            "n_roots": len(by_var_unique_roots[v]),
            "nll": nll_agg["per_variant"][v],
            "brier_vector": brier_vec_agg["per_variant"][v],
            "brier_categorical": brier_cat_agg["per_variant"][v],
            "accuracy": acc_agg["per_variant"][v],
            "calibration": _calibration_with_counts(vp, vy, sides=vsides, n_bins=10),
            "mc_noise_floor": mc_by_variant[v],
        }

    out: dict[str, Any] = {
        "n_records": len(records),
        "n_roots": len({(str(r["variant"]), str(r["root_id"])) for r in records}),
        "nll": nll_agg,
        "brier_vector": brier_vec_agg,
        "brier_categorical": brier_cat_agg,
        "accuracy": acc_agg,
        "calibration": calibration,
        "equal_material": eq_summary,
        "per_variant": per_variant_extra,
        "mc_noise_floor": {
            "macro": mc_macro,
            "per_variant": mc_by_variant,
        },
        "uncertainty_metadata": _uncertainty_metadata(records),
    }

    if k0_lookup is not None:
        gain_records = [
            {
                **r,
                "delta_nll": float(
                    k0_lookup[(str(r["variant"]), int(r["model_seed"]), str(r["root_id"]))]
                    - float(r["nll"])
                ),
            }
            for r in records
            if (str(r["variant"]), int(r["model_seed"]), str(r["root_id"])) in k0_lookup
        ]
        if gain_records:
            out["context_gain"] = aggregate_evaluation_records(
                gain_records,
                value_key="delta_nll",
                game_key="family_id",
                n_boot=n_boot,
                seed=seed + 53,
            )
            if eq_summary["n_records"] > 0:
                eq_gains = [r for r in gain_records if bool(r.get("equal_material", False))]
                if eq_gains:
                    eq_summary["context_gain"] = aggregate_evaluation_records(
                        eq_gains,
                        value_key="delta_nll",
                        game_key="family_id",
                        n_boot=n_boot,
                        seed=seed + 59,
                    )

    if clean_lookup is not None:
        rel_records = [
            {
                **r,
                "nll_diff_ctrl_minus_clean": float(
                    float(r["nll"])
                    - clean_lookup[
                        (
                            str(r["variant"]),
                            int(r["model_seed"]),
                            int(r["demo_draw"]),
                            str(r["root_id"]),
                        )
                    ]
                ),
            }
            for r in records
            if (
                str(r["variant"]),
                int(r["model_seed"]),
                int(r["demo_draw"]),
                str(r["root_id"]),
            )
            in clean_lookup
        ]
        if rel_records:
            out["relevance_comparison_vs_clean"] = {
                "nll_diff_ctrl_minus_clean": aggregate_evaluation_records(
                    rel_records,
                    value_key="nll_diff_ctrl_minus_clean",
                    game_key="family_id",
                    n_boot=n_boot,
                    seed=seed + 67,
                ),
                "n_paired_records": len(rel_records),
            }

    return out


def _make_record(
    *,
    ep: Mapping[str, Any],
    p_pred: np.ndarray,
    condition: str,
    k: int,
    demo_draw: int,
    model_seed: int,
    pred_meta: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    y_true = np.asarray(ep["y_true"], dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    _, nll_arr = compute_nll(p[None, :], y_true[None, :])
    brier_vec_arr = compute_brier_per_sample(p[None, :], y_true[None, :])
    brier_cat_arr = compute_categorical_brier_per_sample(p[None, :], y_true[None, :])
    correct = float(1.0 if int(np.argmax(p)) == int(np.argmax(y_true)) else 0.0)

    rec: dict[str, Any] = {
        "condition": condition,
        "k": int(k),
        "demo_draw": int(demo_draw),
        "model_seed": int(model_seed),
        "variant": str(ep["variant"]),
        "root_id": str(ep["root_id"]),
        "family_id": str(ep.get("family_id", ep.get("source_game_id", ep["root_id"]))),
        "source_game_id": str(ep.get("source_game_id", ep["root_id"])),
        "side_to_move": str(ep.get("side_to_move", "w")),
        "equal_material": bool(ep.get("equal_material", False)),
        "R": int(ep.get("R", 20)),
        "y_true": [float(v) for v in y_true],
        "p_pred": [float(v) for v in p],
        "nll": float(nll_arr[0]),
        "brier": float(brier_vec_arr[0]),
        "brier_vector": float(brier_vec_arr[0]),
        "brier_categorical": float(brier_cat_arr[0]),
        "correct": correct,
        "match_found": bool(ep.get("match_found", True)),
    }
    if pred_meta is not None:
        rec["num_tokens"] = int(pred_meta["num_tokens"])
        rec["num_demos_kept"] = int(pred_meta["num_demos_kept"])
        rec["num_demos_dropped"] = int(pred_meta["num_demos_dropped"])
        rec["input_ids_sha256"] = str(pred_meta["input_ids_sha256"])
        rec["feasible"] = bool(rec["match_found"] and int(pred_meta["num_demos_dropped"]) == 0)
    else:
        rec["feasible"] = bool(rec["match_found"])
    return rec


def _compute_cohort_summaries(
    records: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str],
    k_values: Sequence[int],
    num_draws: int,
    active_controls: Sequence[str],
    total_roots_by_variant: Mapping[str, int],
    n_boot: int,
    seed: int,
    require_zero_dropped_demos: bool = True,
) -> dict[str, Any]:
    """Build feasible-cohort, common-cohort, and coverage summaries split by regime."""
    sorted_ks = sorted(int(k) for k in k_values)
    # Index records by (condition, k, variant, root_id, demo_draw)
    by_cond_k_root_draws: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for r in records:
        by_cond_k_root_draws[
            (str(r["condition"]), int(r["k"]), str(r["variant"]), str(r["root_id"]))
        ].append(r)

    def _is_rec_feasible(r: Mapping[str, Any]) -> bool:
        if not bool(r.get("match_found", True)):
            return False
        if require_zero_dropped_demos and int(r.get("num_demos_dropped", 0)) > 0:
            return False
        return bool(r.get("feasible", True))

    # Feasible roots per (condition, k): root must be feasible on ALL num_draws draws
    feasible_roots: dict[tuple[str, int], set[tuple[str, str]]] = {}
    all_conditions = ["clean"] + list(active_controls)
    for cond in all_conditions:
        for k in sorted_ks:
            if cond != "clean" and k == 0:
                continue
            fset: set[tuple[str, str]] = set()
            for (c, kk, v, rid), grp in by_cond_k_root_draws.items():
                if c != cond or kk != k:
                    continue
                draws_present = {int(r["demo_draw"]) for r in grp}
                if len(draws_present) >= num_draws and all(_is_rec_feasible(r) for r in grp):
                    fset.add((v, rid))
            feasible_roots[(cond, k)] = fset

    # Common cohort across ALL evaluated K (clean) and ALL active controls (K > 0)
    common_roots: Optional[set[tuple[str, str]]] = None
    for cond in all_conditions:
        for k in sorted_ks:
            if cond != "clean" and k == 0:
                continue
            fset = feasible_roots[(cond, k)]
            common_roots = set(fset) if common_roots is None else (common_roots & fset)
    if common_roots is None:
        common_roots = set()

    total_roots_all = int(sum(total_roots_by_variant.values()))
    coverage_report: dict[str, Any] = {}
    for cond in all_conditions:
        cond_cov: dict[str, Any] = {}
        for k in sorted_ks:
            if cond != "clean" and k == 0:
                continue
            fset = feasible_roots[(cond, k)]
            cond_cov[str(k)] = {
                "n_total_roots": total_roots_all,
                "n_feasible_roots": len(fset),
                "feasible_coverage_rate": (
                    float(len(fset) / total_roots_all) if total_roots_all > 0 else 0.0
                ),
                "n_common_roots": len(common_roots),
                "common_coverage_rate": (
                    float(len(common_roots) / total_roots_all) if total_roots_all > 0 else 0.0
                ),
                "all_roots_feasible": bool(len(fset) == total_roots_all),
            }
        coverage_report[cond] = cond_cov

    regimes = _partition_variants_by_regime(variants)

    def _summarize_cohort(cohort_mode: Literal["feasible", "common"]) -> dict[str, Any]:
        cohort_out: dict[str, Any] = {}
        for regime_name, regime_info in regimes.items():
            r_vars = set(regime_info["variants"])
            clean_by_k: dict[str, Any] = {}
            k0_lookup: Optional[dict[tuple[str, int, str], float]] = None
            clean_lookup_by_k: dict[int, dict[tuple[str, int, int, str], float]] = {}

            for k in sorted_ks:
                allowed = (
                    feasible_roots[("clean", k)]
                    if cohort_mode == "feasible"
                    else common_roots
                )
                k_recs = [
                    r
                    for r in records
                    if str(r["condition"]) == "clean"
                    and int(r["k"]) == k
                    and str(r["variant"]) in r_vars
                    and (str(r["variant"]), str(r["root_id"])) in allowed
                ]
                if k == 0 and k_recs:
                    k0_lookup = {
                        (str(r["variant"]), int(r["model_seed"]), str(r["root_id"])): float(
                            r["nll"]
                        )
                        for r in k_recs
                        if int(r["demo_draw"]) == 0
                    }
                clean_lookup_by_k[k] = {
                    (
                        str(r["variant"]),
                        int(r["model_seed"]),
                        int(r["demo_draw"]),
                        str(r["root_id"]),
                    ): float(r["nll"])
                    for r in k_recs
                }
                clean_by_k[str(k)] = _summarize_record_group(
                    k_recs,
                    n_boot=n_boot,
                    seed=seed + k * 100,
                    k0_lookup=k0_lookup if k > 0 else None,
                )

            controls_out: dict[str, Any] = {}
            for ctrl in active_controls:
                ctrl_by_k: dict[str, Any] = {}
                for k in sorted_ks:
                    if k == 0:
                        continue
                    allowed = (
                        feasible_roots[(ctrl, k)]
                        if cohort_mode == "feasible"
                        else common_roots
                    )
                    c_recs = [
                        r
                        for r in records
                        if str(r["condition"]) == ctrl
                        and int(r["k"]) == k
                        and str(r["variant"]) in r_vars
                        and (str(r["variant"]), str(r["root_id"])) in allowed
                    ]
                    ctrl_by_k[str(k)] = _summarize_record_group(
                        c_recs,
                        n_boot=n_boot,
                        seed=seed + 1000 + k * 100,
                        k0_lookup=k0_lookup,
                        clean_lookup=clean_lookup_by_k.get(k),
                    )
                controls_out[ctrl] = ctrl_by_k

            cohort_out[regime_name] = {
                "role": regime_info["role"],
                "is_transfer": regime_info["is_transfer"],
                "variants": regime_info["variants"],
                "clean": clean_by_k,
                "controls": controls_out,
            }
        return cohort_out

    return {
        "coverage": coverage_report,
        "all_root_clean_coverage": coverage_report.get("clean", {}),
        "feasible_cohort": _summarize_cohort("feasible"),
        "common_cohort": _summarize_cohort("common"),
    }


# ---------------------------------------------------------------------------
# Checkpoint Loading Helper
# ---------------------------------------------------------------------------


def discover_candidate_checkpoints(checkpoints_dir: Path) -> list[tuple[Optional[int], Path]]:
    """Discover candidate checkpoint directories sorted numerically by step if available."""
    root = Path(checkpoints_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {root}")

    candidates: list[tuple[int, Optional[int], Path]] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        has_ckpt = (
            (sub / "adapter_config.json").exists()
            or (sub / "classifier_head.pt").exists()
            or (sub / ScratchQwen2Classifier.WEIGHTS_FILENAME).exists()
        )
        if not has_ckpt:
            continue
        m = STEP_DIR_RE.match(sub.name)
        if m:
            step = int(m.group(1))
            candidates.append((0, step, sub))
        else:
            candidates.append((1, None, sub))

    candidates.sort(key=lambda item: (item[0], item[1] if item[1] is not None else 0, item[2].name))
    return [(step, path) for _, step, path in candidates]


def _resolve_checkpoint_arch(checkpoint_path: Path, arch: str = "auto") -> Literal["qwen", "scratch"]:
    if arch in ("qwen", "scratch"):
        return arch  # type: ignore[return-value]
    p = Path(checkpoint_path)
    if p.is_file() or (p / ScratchQwen2Classifier.WEIGHTS_FILENAME).exists():
        return "scratch"
    if (p / "adapter_config.json").exists() or (p / "classifier_head.pt").exists():
        return "qwen"
    raise ValueError(
        f"Cannot auto-detect model arch for checkpoint {p}; pass --arch qwen or --arch scratch"
    )


def _default_model_loader(
    checkpoint_path: Path,
    *,
    tokenizer: Any,
    model_path: Optional[str],
    device: torch.device,
    arch: str = "auto",
) -> nn.Module:
    resolved_arch = _resolve_checkpoint_arch(checkpoint_path, arch=arch)
    if resolved_arch == "scratch":
        model = ScratchQwen2Classifier.from_weights(
            checkpoint_path,
            tokenizer=tokenizer,
            gradient_checkpointing=False,
            is_trainable=False,
            device=device,
        )
        freeze_model_in_place(model)
        return model

    attn_impl = get_preferred_attn_implementation()
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = QwenICLClassifier.from_adapter_and_head(
        checkpoint_path,
        tokenizer=tokenizer,
        model_path=model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        local_files_only=True,
        is_trainable=False,
        device=device,
    )
    freeze_model_in_place(model)
    return model


_default_qwen_model_loader = _default_model_loader


# ---------------------------------------------------------------------------
# Subcommand 1: select-checkpoint
# ---------------------------------------------------------------------------


def select_checkpoint(
    *,
    data_dir: Path,
    output_manifest: Path,
    checkpoints_dir: Optional[Path] = None,
    checkpoint_paths: Optional[Sequence[Path]] = None,
    selection_regime: str = "familiar",
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    num_draws: int = 5,
    controls: Sequence[str] = DEFAULT_CONTROLS,
    notation: Literal["uci", "san"] = "san",
    max_seq_len: int = 32768,
    max_roots: Optional[int] = None,
    seed: int = 42,
    gpu: int = 0,
    model_path: Optional[str] = None,
    arch: Literal["auto", "qwen", "scratch"] = "auto",
    tokenizer: Any = None,
    model_loader: Optional[Callable[[Path], nn.Module]] = None,
) -> dict[str, Any]:
    """Score candidate checkpoints on full validation fold and freeze selection manifest."""
    set_global_seed(seed)
    device = resolve_device(gpu)

    if checkpoint_paths:
        cand_list: list[tuple[Optional[int], Path]] = []
        for p in checkpoint_paths:
            m = STEP_DIR_RE.match(p.name)
            cand_list.append((int(m.group(1)) if m else None, p))
    elif checkpoints_dir is not None:
        cand_list = discover_candidate_checkpoints(checkpoints_dir)
    else:
        raise ValueError("Provide either checkpoints_dir or checkpoint_paths")

    if not cand_list:
        raise ValueError("No candidate checkpoints found for selection")

    sel_variants = resolve_variants_for_regime(
        data_dir, split="val", regime=selection_regime, explicit_variants=variants
    )
    if not sel_variants:
        raise ValueError(f"No validation variants found for regime={selection_regime!r}")

    if tokenizer is None:
        resolved_path = resolve_qwen_model_path(model_path)
        tokenizer = load_qwen_tokenizer(resolved_path, local_files_only=True)

    sorted_ks = sorted(int(k) for k in k_values)
    # Build deterministic validation episodes once so every candidate sees identical inputs
    val_episodes_by_vkd: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for v in sel_variants:
        for k in sorted_ks:
            n_eff = 1 if k == 0 else max(1, int(num_draws))
            for d_idx in range(n_eff):
                val_episodes_by_vkd[(v, k, d_idx)] = build_long_study_episodes(
                    data_dir=data_dir,
                    variant=v,
                    split="val",
                    k=k,
                    draw_seed=d_idx,
                    max_roots=max_roots,
                    notation=notation,
                )

    candidate_results: list[dict[str, Any]] = []
    for step, ckpt_path in cand_list:
        ckpt_sha = compute_checkpoint_sha256(ckpt_path)
        if model_loader is not None:
            model = model_loader(ckpt_path)
        else:
            model = _default_model_loader(
                ckpt_path,
                tokenizer=tokenizer,
                model_path=model_path,
                device=device,
                arch=arch,
            )

        predictor = CanonicalQwenPredictor(
            model=model,
            tokenizer=tokenizer,
            model_sha256=ckpt_sha,
            notation=notation,
            max_seq_len=max_seq_len,
            device=device,
        )

        # Verify uncached repeat determinism on first validation episode
        first_eps = val_episodes_by_vkd[(sel_variants[0], sorted_ks[0], 0)]
        sample_prompts = [
            predictor.formatter.format_prompt(
                ep["demos"],
                {"fen": ep["query_fen"], "blocked_squares": ep.get("blocked_squares")},
            )
            for ep in first_eps[:2]
        ]
        det_check = predictor.verify_uncached_repeat_determinism(sample_prompts, tol=1e-5)

        per_variant_k_nll: dict[str, dict[str, float]] = {}
        per_variant_k_bvec: dict[str, dict[str, float]] = {}
        per_variant_k_bcat: dict[str, dict[str, float]] = {}
        per_k_summary: dict[str, Any] = {}

        for v in sel_variants:
            per_variant_k_nll[v] = {}
            per_variant_k_bvec[v] = {}
            per_variant_k_bcat[v] = {}
            for k in sorted_ks:
                n_eff = 1 if k == 0 else max(1, int(num_draws))
                nll_vals: list[float] = []
                bvec_vals: list[float] = []
                bcat_vals: list[float] = []
                for d_idx in range(n_eff):
                    for ep in val_episodes_by_vkd[(v, k, d_idx)]:
                        pred_out = predictor.predict_episode(ep)
                        if pred_out["num_demos_dropped"] > 0:
                            continue
                        y = np.asarray(ep["y_true"], dtype=np.float64)[None, :]
                        p = np.asarray(pred_out["p_pred"], dtype=np.float64)[None, :]
                        mean_nll, _ = compute_nll(p, y)
                        nll_vals.append(float(mean_nll))
                        bvec_vals.append(float(compute_brier(p, y)))
                        bcat_vals.append(float(compute_categorical_brier_per_sample(p, y)[0]))
                if not nll_vals:
                    raise RuntimeError(
                        f"All validation episodes for ({v}, K={k}) exceeded max_seq_len={max_seq_len}"
                    )
                per_variant_k_nll[v][str(k)] = float(np.mean(nll_vals))
                per_variant_k_bvec[v][str(k)] = float(np.mean(bvec_vals))
                per_variant_k_bcat[v][str(k)] = float(np.mean(bcat_vals))

        for k in sorted_ks:
            ks = str(k)
            per_k_summary[ks] = {
                "macro_nll": float(np.mean([per_variant_k_nll[v][ks] for v in sel_variants])),
                "macro_brier_vector": float(
                    np.mean([per_variant_k_bvec[v][ks] for v in sel_variants])
                ),
                "macro_brier_categorical": float(
                    np.mean([per_variant_k_bcat[v][ks] for v in sel_variants])
                ),
            }

        macro_nll = float(np.mean([per_k_summary[str(k)]["macro_nll"] for k in sorted_ks]))
        macro_bvec = float(
            np.mean([per_k_summary[str(k)]["macro_brier_vector"] for k in sorted_ks])
        )
        macro_bcat = float(
            np.mean([per_k_summary[str(k)]["macro_brier_categorical"] for k in sorted_ks])
        )

        candidate_results.append(
            {
                "name": ckpt_path.name,
                "step": step,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "macro_nll": macro_nll,
                "macro_brier_vector": macro_bvec,
                "macro_brier_categorical": macro_bcat,
                "per_k": per_k_summary,
                "per_variant_nll": per_variant_k_nll,
                "determinism_check": det_check,
                "cache_stats": {
                    "hits": predictor.cache_hits,
                    "misses": predictor.cache_misses,
                },
            }
        )

    best = min(
        candidate_results,
        key=lambda c: (
            c["macro_nll"],
            c["macro_brier_vector"],
            c["step"] if c["step"] is not None else 10**9,
        ),
    )
    snapshot_sha256 = compute_snapshot_sha256(data_dir)
    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "arch": _resolve_checkpoint_arch(Path(best["checkpoint_path"]), arch=arch) if model_loader is None else arch,
        "selection_regime": selection_regime,
        "selection_variants": sel_variants,
        "k_values": sorted_ks,
        "num_draws": int(num_draws),
        "controls": list(controls),
        "exclusions": {
            "max_seq_len": int(max_seq_len),
            "require_zero_dropped_demos": True,
            "excluded_variants": [],
            "excluded_root_ids": [],
        },
        "notation": notation,
        "data_dir": str(data_dir),
        "snapshot_sha256": snapshot_sha256,
        "selected_checkpoint_path": best["checkpoint_path"],
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selected_checkpoint": {
            "name": best["name"],
            "step": best["step"],
            "checkpoint_path": best["checkpoint_path"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "macro_nll": best["macro_nll"],
            "macro_brier_vector": best["macro_brier_vector"],
        },
        "candidates": candidate_results,
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Subcommand 2: evaluate
# ---------------------------------------------------------------------------


def _validate_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("frozen") is False:
        raise ValueError("Refusing to evaluate with unfrozen manifest (frozen=False)")
    required_keys = (
        "selected_checkpoint_sha256",
        "snapshot_sha256",
        "k_values",
        "num_draws",
        "controls",
        "exclusions",
    )
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        raise ValueError(f"Frozen manifest missing required keys: {missing}")


def _normalize_control_name(ctrl: str) -> str:
    if ctrl == "foreign_matched":
        return "token_matched_foreign"
    if ctrl == "shuffled_trajectories":
        return "corrupted_trajectories"
    return ctrl


def _build_control_episodes(
    ctrl: str,
    episodes: Sequence[dict[str, Any]],
    *,
    foreign_demos_by_variant: Mapping[str, Sequence[dict[str, Any]]],
    tokenizer: Any,
    notation: Literal["uci", "san"],
    seed: int,
    token_match_tolerance: float = 0.10,
    max_search_attempts: int = 200,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """Dispatch control construction uniformly and return `(ctrl_episodes, search_report)`."""
    norm_ctrl = _normalize_control_name(ctrl)
    if norm_ctrl == "token_matched_foreign":
        return make_token_matched_foreign_episodes(
            episodes=episodes,
            foreign_demos_by_variant=foreign_demos_by_variant,
            tokenizer=tokenizer,
            notation=notation,
            tolerance=token_match_tolerance,
            max_search_attempts=max_search_attempts,
            seed=seed,
        )
    if norm_ctrl == "corrupted_outcomes":
        return make_corrupted_outcome_episodes(episodes, mode="inverted", seed=seed), None
    if norm_ctrl == "corrupted_outcomes_inverted":
        return make_corrupted_outcome_episodes(episodes, mode="inverted", seed=seed), None
    if norm_ctrl == "corrupted_outcomes_scrambled":
        return make_corrupted_outcome_episodes(episodes, mode="scrambled", seed=seed), None
    if norm_ctrl == "corrupted_outcomes_all_draw":
        return make_corrupted_outcome_episodes(episodes, mode="all_draw", seed=seed), None
    if norm_ctrl == "corrupted_trajectories":
        return make_corrupted_trajectory_episodes(episodes, mode="shuffled", seed=seed), None
    if norm_ctrl == "mismatched_context":
        return (
            make_mismatched_context_episodes(
                episodes, foreign_demos_by_variant, seed=seed
            ),
            None,
        )
    raise ValueError(f"Unsupported control {ctrl!r}")


def evaluate_from_manifest(
    *,
    manifest_path: Path,
    data_dir: Path,
    output_file: Path,
    split: Literal["val", "test"] = "test",
    variants: Optional[Sequence[str]] = None,
    checkpoint_path: Optional[Path] = None,
    notation: Optional[Literal["uci", "san"]] = None,
    max_roots: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 42,
    model_seed: int = 0,
    gpu: int = 0,
    model_path: Optional[str] = None,
    token_match_tolerance: float = 0.10,
    max_search_attempts: int = 200,
    r256_subset_manifest: Optional[Path] = None,
    r256_positions_filename: Optional[str] = None,
    r256_data_dir: Optional[Path] = None,
    arch: Literal["auto", "qwen", "scratch"] = "auto",
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Evaluate a frozen checkpoint manifest with canonical batch_size=1 inference."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_frozen_manifest(manifest)

    actual_snapshot_sha = compute_snapshot_sha256(data_dir)
    if actual_snapshot_sha != manifest["snapshot_sha256"]:
        raise ValueError(
            f"Data snapshot SHA-256 mismatch: expected {manifest['snapshot_sha256']}, "
            f"got {actual_snapshot_sha}"
        )

    exclusions = manifest["exclusions"]
    excluded_variants = set(exclusions.get("excluded_variants", []))
    excluded_root_ids = set(str(x) for x in exclusions.get("excluded_root_ids", []))
    require_zero_dropped = bool(exclusions.get("require_zero_dropped_demos", True))
    max_seq_len = int(exclusions.get("max_seq_len", manifest.get("max_seq_len", 32768)))
    eff_notation: Literal["uci", "san"] = notation or manifest.get("notation", "san")
    k_values = sorted(int(k) for k in manifest["k_values"])
    num_draws = max(1, int(manifest["num_draws"]))
    active_controls = [str(c) for c in manifest["controls"] if c and str(c).lower() != "none"]
    for c in active_controls:
        if c not in VALID_CONTROLS:
            raise ValueError(f"Unsupported control in manifest: {c!r}")

    ckpt_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (
            Path(manifest["selected_checkpoint_path"])
            if manifest.get("selected_checkpoint_path")
            else None
        )
    )

    set_global_seed(seed)
    device = resolve_device(gpu)

    if tokenizer is None:
        resolved_path = resolve_qwen_model_path(model_path)
        tokenizer = load_qwen_tokenizer(resolved_path, local_files_only=True)

    if model is None:
        if ckpt_path is None:
            raise ValueError("Manifest missing selected_checkpoint_path and no model provided")
        actual_ckpt_sha = compute_checkpoint_sha256(ckpt_path)
        if actual_ckpt_sha != manifest["selected_checkpoint_sha256"]:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch: expected {manifest['selected_checkpoint_sha256']}, "
                f"got {actual_ckpt_sha}"
            )
        eff_arch = arch if arch != "auto" else str(manifest.get("arch", "auto"))
        model = _default_model_loader(
            ckpt_path,
            tokenizer=tokenizer,
            model_path=model_path,
            device=device,
            arch=eff_arch,
        )
        model_sha = actual_ckpt_sha
    else:
        freeze_model_in_place(model)
        if ckpt_path is not None and ckpt_path.exists():
            model_sha = compute_checkpoint_sha256(ckpt_path)
        else:
            model_sha = compute_model_state_sha256(model)
        if model_sha != manifest["selected_checkpoint_sha256"]:
            raise ValueError(
                f"Model SHA-256 mismatch: expected {manifest['selected_checkpoint_sha256']}, "
                f"got {model_sha}"
            )

    predictor = CanonicalQwenPredictor(
        model=model,
        tokenizer=tokenizer,
        model_sha256=model_sha,
        notation=eff_notation,
        max_seq_len=max_seq_len,
        device=device,
    )

    available_variants = discover_variants(data_dir, split=split)
    eval_variants = (
        [v for v in variants if v not in excluded_variants]
        if variants
        else [v for v in available_variants if v not in excluded_variants]
    )
    if not eval_variants:
        raise ValueError(f"No evaluation variants found for split={split!r} in {data_dir}")

    # Load foreign demo pools from all available variants for foreign-matched controls
    foreign_pools: dict[str, list[dict[str, Any]]] = {}
    if any(
        _normalize_control_name(c) in {"token_matched_foreign", "mismatched_context"}
        for c in active_controls
    ):
        for pv in available_variants:
            demos = load_split_demos(data_dir, variant=pv, split=split)
            foreign_pools[pv] = [
                prepare_demo_for_notation(d, variant=pv, notation=eff_notation) for d in demos
            ]

    all_records: list[dict[str, Any]] = []
    foreign_search_reports: dict[str, Any] = {}
    total_roots_by_variant: dict[str, int] = {}
    determinism_checked = False
    determinism_report: Optional[dict[str, Any]] = None

    for k in k_values:
        if k == 0:
            k0_episodes: list[dict[str, Any]] = []
            for v in eval_variants:
                eps = build_long_study_episodes(
                    data_dir=data_dir,
                    variant=v,
                    split=split,
                    k=0,
                    draw_seed=0,
                    max_roots=max_roots,
                    notation=eff_notation,
                    excluded_root_ids=excluded_root_ids,
                )
                total_roots_by_variant[v] = len(eps)
                k0_episodes.extend(eps)

            if not determinism_checked and k0_episodes:
                sample_prompts = [
                    predictor.formatter.format_prompt(
                        ep["demos"],
                        {"fen": ep["query_fen"], "blocked_squares": ep.get("blocked_squares")},
                    )
                    for ep in k0_episodes[:3]
                ]
                determinism_report = predictor.verify_uncached_repeat_determinism(
                    sample_prompts, tol=1e-5
                )
                determinism_checked = True

            # Score K=0 once, replicate across draws (cache also guarantees identical input_ids -> identical result)
            base_recs = []
            for ep in k0_episodes:
                pred_out = predictor.predict_episode(ep)
                base_recs.append(
                    _make_record(
                        ep=ep,
                        p_pred=pred_out["p_pred"],
                        condition="clean",
                        k=0,
                        demo_draw=0,
                        model_seed=model_seed,
                        pred_meta=pred_out,
                    )
                )
            for d_idx in range(num_draws):
                for rec in base_recs:
                    rc = dict(rec)
                    rc["demo_draw"] = int(d_idx)
                    all_records.append(rc)
            continue

        for d_idx in range(num_draws):
            draw_episodes: list[dict[str, Any]] = []
            for v in eval_variants:
                eps = build_long_study_episodes(
                    data_dir=data_dir,
                    variant=v,
                    split=split,
                    k=k,
                    draw_seed=d_idx,
                    max_roots=max_roots,
                    notation=eff_notation,
                    excluded_root_ids=excluded_root_ids,
                )
                if v not in total_roots_by_variant:
                    total_roots_by_variant[v] = len(eps)
                draw_episodes.extend(eps)

            if not determinism_checked and draw_episodes:
                sample_prompts = [
                    predictor.formatter.format_prompt(
                        ep["demos"],
                        {"fen": ep["query_fen"], "blocked_squares": ep.get("blocked_squares")},
                    )
                    for ep in draw_episodes[:3]
                ]
                determinism_report = predictor.verify_uncached_repeat_determinism(
                    sample_prompts, tol=1e-5
                )
                determinism_checked = True

            for ep in draw_episodes:
                pred_out = predictor.predict_episode(ep)
                all_records.append(
                    _make_record(
                        ep=ep,
                        p_pred=pred_out["p_pred"],
                        condition="clean",
                        k=k,
                        demo_draw=d_idx,
                        model_seed=model_seed,
                        pred_meta=pred_out,
                    )
                )

            ctrl_seed = seed + k * 1000 + d_idx
            for ctrl in active_controls:
                ctrl_eps, search_rep = _build_control_episodes(
                    ctrl,
                    draw_episodes,
                    foreign_demos_by_variant=foreign_pools,
                    tokenizer=tokenizer,
                    notation=eff_notation,
                    seed=ctrl_seed,
                    token_match_tolerance=token_match_tolerance,
                    max_search_attempts=max_search_attempts,
                )
                if search_rep is not None:
                    foreign_search_reports[f"{ctrl}|K={k}|draw={d_idx}"] = search_rep

                for ep in ctrl_eps:
                    pred_out = predictor.predict_episode(ep)
                    all_records.append(
                        _make_record(
                            ep=ep,
                            p_pred=pred_out["p_pred"],
                            condition=ctrl,
                            k=k,
                            demo_draw=d_idx,
                            model_seed=model_seed,
                            pred_meta=pred_out,
                        )
                    )

    cohorts_summary = _compute_cohort_summaries(
        all_records,
        variants=eval_variants,
        k_values=k_values,
        num_draws=num_draws,
        active_controls=active_controls,
        total_roots_by_variant=total_roots_by_variant,
        n_boot=n_boot,
        seed=seed,
        require_zero_dropped_demos=require_zero_dropped,
    )

    r256_report: Optional[dict[str, Any]] = None
    if r256_subset_manifest is not None:
        r256_report = _evaluate_r256_subset(
            predictor=predictor,
            data_dir=data_dir,
            r256_manifest_path=r256_subset_manifest,
            positions_filename=r256_positions_filename,
            r256_data_dir=r256_data_dir,
            split=split,
            k_values=k_values,
            num_draws=num_draws,
            notation=eff_notation,
            n_boot=n_boot,
            seed=seed,
            model_seed=model_seed,
        )

    payload: dict[str, Any] = {
        "metadata": {
            "manifest_path": str(manifest_path),
            "selected_checkpoint_path": str(ckpt_path) if ckpt_path else None,
            "selected_checkpoint_sha256": model_sha,
            "snapshot_sha256": actual_snapshot_sha,
            "split": split,
            "notation": eff_notation,
            "variants": eval_variants,
            "k_values": k_values,
            "num_draws": num_draws,
            "controls": active_controls,
            "exclusions": exclusions,
            "inference_mode": "canonical_unpadded_batch_size_1",
            "determinism_check": determinism_report,
            "cache_stats": {
                "key_schema": "(model_sha256, input_ids_sha256)",
                "model_sha256": model_sha,
                "hits": predictor.cache_hits,
                "misses": predictor.cache_misses,
                "unique_cached_inputs": len(predictor._cache),
            },
            "model_seed": int(model_seed),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
        "foreign_control_search_reports": foreign_search_reports,
        "summary": cohorts_summary,
        "r256_reestimation": r256_report,
        "per_root_predictions": all_records,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _evaluate_r256_subset(
    *,
    predictor: CanonicalQwenPredictor,
    data_dir: Path,
    r256_manifest_path: Path,
    positions_filename: Optional[str],
    r256_data_dir: Optional[Path] = None,
    split: str,
    k_values: Sequence[int],
    num_draws: int,
    notation: Literal["uci", "san"],
    n_boot: int,
    seed: int,
    model_seed: int,
) -> dict[str, Any]:
    """Score the R256 subset manifest using cached/canonical predictions & updated R256 targets."""
    r256_man = json.loads(Path(r256_manifest_path).read_text(encoding="utf-8"))
    r256_variants = sorted(r256_man["variants"].keys())
    pos_dir = Path(r256_data_dir) if r256_data_dir is not None else Path(data_dir)
    records: list[dict[str, Any]] = []
    total_by_var: dict[str, int] = {}

    for k in k_values:
        n_eff = 1 if k == 0 else num_draws
        for d_idx in range(n_eff):
            for v in r256_variants:
                v_meta = r256_man["variants"][v]
                raw_roots = v_meta.get("selected_root_ids")
                raw_fams = v_meta.get("selected_family_ids") or v_meta.get("selected_families")
                allowed_roots = set(str(r) for r in raw_roots) if raw_roots is not None else None
                allowed_fams = (
                    set(str(f) for f in raw_fams)
                    if allowed_roots is None and raw_fams is not None
                    else None
                )
                eps = build_long_study_episodes(
                    data_dir=pos_dir,
                    variant=v,
                    split=split,
                    k=k,
                    draw_seed=d_idx,
                    notation=notation,
                    positions_filename=positions_filename,
                    allowed_root_ids=allowed_roots,
                    allowed_family_ids=allowed_fams,
                    demos_data_dir=data_dir,
                )
                total_by_var[v] = len(eps)
                for ep in eps:
                    pred_out = predictor.predict_episode(ep)
                    rec = _make_record(
                        ep=ep,
                        p_pred=pred_out["p_pred"],
                        condition="clean",
                        k=k,
                        demo_draw=d_idx,
                        model_seed=model_seed,
                        pred_meta=pred_out,
                    )
                    if k == 0:
                        for replicate_d in range(num_draws):
                            rc = dict(rec)
                            rc["demo_draw"] = int(replicate_d)
                            records.append(rc)
                    else:
                        records.append(rec)

    summary = _compute_cohort_summaries(
        records,
        variants=r256_variants,
        k_values=k_values,
        num_draws=num_draws,
        active_controls=[],
        total_roots_by_variant=total_by_var,
        n_boot=n_boot,
        seed=seed + 777,
    )
    return {
        "r256_manifest_path": str(r256_manifest_path),
        "positions_filename": positions_filename or f"positions_{split}.jsonl",
        "selection_rule": r256_man.get("selection_rule"),
        "outcome_independent": bool(r256_man.get("outcome_independent", True)),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Subcommand 3: select-r256-subset
# ---------------------------------------------------------------------------


def select_r256_subset(
    *,
    data_dir: Path,
    output_manifest: Path,
    families_per_variant: int = 16,
    variants: Optional[Sequence[str]] = None,
    split: Literal["val", "test"] = "test",
    hash_salt: str = "",
) -> dict[str, Any]:
    """Select M families per variant using an outcome-independent `sha256(family_id)` rule.

    Strictly never reads model scores, predictions, labels, or rollout targets.
    """
    if families_per_variant < 1:
        raise ValueError(f"families_per_variant must be >= 1, got {families_per_variant}")

    avail = discover_variants(data_dir, split=split)
    sel_variants = [v for v in variants if v in avail] if variants else avail
    if not sel_variants:
        raise ValueError(f"No variants found for split={split!r} under {data_dir}")

    per_variant: dict[str, Any] = {}
    total_fams = 0
    total_roots = 0

    for v in sel_variants:
        positions = load_split_positions(data_dir, variant=v, split=split)
        roots_by_family: dict[str, list[str]] = defaultdict(list)
        for idx, pos in enumerate(positions):
            root_id = str(pos.get("root_id") or pos.get("id") or f"{v}_{split}_{idx}")
            fam_id = _extract_family_id(pos, fallback=root_id)
            roots_by_family[fam_id].append(root_id)

        sorted_families = sorted(
            roots_by_family.keys(),
            key=lambda fid: (
                hashlib.sha256(f"{hash_salt}{fid}".encode("utf-8")).hexdigest(),
                fid,
            ),
        )
        chosen_families = sorted_families[:families_per_variant]
        chosen_roots = [rid for fid in chosen_families for rid in roots_by_family[fid]]

        per_variant[v] = {
            "n_total_families": len(sorted_families),
            "n_selected_families": len(chosen_families),
            "selected_family_ids": chosen_families,
            "selected_root_ids": chosen_roots,
            "n_selected_roots": len(chosen_roots),
        }
        total_fams += len(chosen_families)
        total_roots += len(chosen_roots)

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "sha256(family_id)_ascending",
        "outcome_independent": True,
        "hash_salt": hash_salt,
        "split": split,
        "families_per_variant": int(families_per_variant),
        "data_dir": str(data_dir),
        "snapshot_sha256": compute_snapshot_sha256(data_dir),
        "total_selected_families": int(total_fams),
        "total_selected_roots": int(total_roots),
        "variants": per_variant,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Subcommand 4: run-baselines
# ---------------------------------------------------------------------------


def run_declared_baselines(
    *,
    data_dir: Path,
    output_file: Path,
    split: Literal["val", "test"] = "test",
    manifest_path: Optional[Path] = None,
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    num_draws: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    knn_top_k: int = 10,
    knn_metric: Literal["cosine", "hamming"] = "cosine",
    max_train_per_variant: Optional[int] = None,
    max_roots: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate all 7 declared non-neural baselines fitted strictly on training data."""
    excluded_variants: set[str] = set()
    excluded_root_ids: set[str] = set()
    if manifest_path is not None:
        man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        _validate_frozen_manifest(man)
        actual_snap = compute_snapshot_sha256(data_dir)
        if actual_snap != man["snapshot_sha256"]:
            raise ValueError(
                f"Snapshot SHA-256 mismatch vs manifest: {actual_snap} != {man['snapshot_sha256']}"
            )
        k_values = [int(k) for k in man["k_values"]]
        num_draws = int(man["num_draws"])
        excluded_variants = set(man["exclusions"].get("excluded_variants", []))
        excluded_root_ids = set(str(x) for x in man["exclusions"].get("excluded_root_ids", []))

    sorted_ks = sorted(int(k) for k in k_values)
    n_draws = max(1, int(num_draws))

    # Strictly fit parametric baselines ONLY on training split
    train_episodes = build_training_episodes_strict(
        data_dir=data_dir,
        k_values=sorted_ks,
        seed=seed,
        max_per_variant=max_train_per_variant,
    )
    baselines = build_declared_baselines(
        train_episodes,
        knn_top_k=knn_top_k,
        knn_metric=knn_metric,
    )
    assert tuple(baselines.keys()) == DECLARED_BASELINE_NAMES

    avail = discover_variants(data_dir, split=split)
    eval_variants = (
        [v for v in variants if v not in excluded_variants]
        if variants
        else [v for v in avail if v not in excluded_variants]
    )
    if not eval_variants:
        raise ValueError(f"No evaluation variants found for split={split!r} in {data_dir}")

    # Prebuild identical frozen episodes across (variant, k, draw)
    episodes_by_vkd: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    total_roots_by_variant: dict[str, int] = {}
    for v in eval_variants:
        for k in sorted_ks:
            n_eff = 1 if k == 0 else n_draws
            for d_idx in range(n_eff):
                eps = build_long_study_episodes(
                    data_dir=data_dir,
                    variant=v,
                    split=split,
                    k=k,
                    draw_seed=d_idx,
                    max_roots=max_roots,
                    notation="uci",
                    excluded_root_ids=excluded_root_ids,
                )
                episodes_by_vkd[(v, k, d_idx)] = eps
                total_roots_by_variant[v] = len(eps)

    results_by_baseline: dict[str, Any] = {}
    for b_name, predictor in baselines.items():
        records: list[dict[str, Any]] = []
        is_ctx_inv = b_name in CONTEXT_INVARIANT_BASELINE_NAMES
        inv_preds_by_v: dict[str, np.ndarray] = {}
        if is_ctx_inv:
            k_ref = sorted_ks[0]
            for v in eval_variants:
                inv_preds_by_v[v] = predictor.predict_proba(episodes_by_vkd[(v, k_ref, 0)])

        for k in sorted_ks:
            n_eff = 1 if (k == 0 or is_ctx_inv) else n_draws
            for v in eval_variants:
                base_probs_by_draw: list[np.ndarray] = []
                for d_idx in range(n_eff):
                    eps = episodes_by_vkd[(v, k, 0 if k == 0 else d_idx)]
                    probs = inv_preds_by_v[v] if is_ctx_inv else predictor.predict_proba(eps)
                    base_probs_by_draw.append(probs)

                for d_idx in range(n_draws):
                    eff_d = 0 if (k == 0 or is_ctx_inv) else d_idx
                    eps = episodes_by_vkd[(v, k, 0 if k == 0 else d_idx)]
                    probs = base_probs_by_draw[eff_d]
                    for ep, p_row in zip(eps, probs):
                        records.append(
                            _make_record(
                                ep=ep,
                                p_pred=p_row,
                                condition="clean",
                                k=k,
                                demo_draw=d_idx,
                                model_seed=0,
                                pred_meta=None,
                            )
                        )

        results_by_baseline[b_name] = _compute_cohort_summaries(
            records,
            variants=eval_variants,
            k_values=sorted_ks,
            num_draws=n_draws,
            active_controls=[],
            total_roots_by_variant=total_roots_by_variant,
            n_boot=n_boot,
            seed=seed,
            require_zero_dropped_demos=False,
        )

    payload = {
        "metadata": {
            "split": split,
            "variants": eval_variants,
            "k_values": sorted_ks,
            "num_draws": n_draws,
            "declared_baselines": list(DECLARED_BASELINE_NAMES),
            "fit_source": "positions_train.jsonl + games_demo_train.jsonl (or games_demo.jsonl)",
            "snapshot_sha256": compute_snapshot_sha256(data_dir),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
        "baselines": results_by_baseline,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Multi-Seed Hierarchical Aggregation (e.g. S=3 seeds)
# ---------------------------------------------------------------------------


def aggregate_multi_seed_evaluations(
    eval_files: Sequence[Path],
    output_file: Path,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Combine per_root_predictions from multiple seed evaluations into one hierarchical report."""
    if not eval_files:
        raise ValueError("eval_files cannot be empty")

    all_records: list[dict[str, Any]] = []
    variants: Optional[list[str]] = None
    k_values: Optional[list[int]] = None
    num_draws: Optional[int] = None
    active_controls: Optional[list[str]] = None
    snapshot_shas: set[str] = set()
    seed_ids: list[int] = []

    for ef in eval_files:
        data = json.loads(Path(ef).read_text(encoding="utf-8"))
        meta = data["metadata"]
        snapshot_shas.add(str(meta["snapshot_sha256"]))
        seed_ids.append(int(meta["model_seed"]))
        if variants is None:
            variants = list(meta["variants"])
            k_values = [int(k) for k in meta["k_values"]]
            num_draws = int(meta["num_draws"])
            active_controls = list(meta["controls"])
        all_records.extend(data["per_root_predictions"])

    if len(snapshot_shas) != 1:
        raise ValueError(f"Cannot aggregate evaluations across differing snapshots: {snapshot_shas}")

    total_roots_by_variant: dict[str, int] = {}
    for v in variants or []:
        total_roots_by_variant[v] = len(
            {str(r["root_id"]) for r in all_records if str(r["variant"]) == v}
        )

    summary = _compute_cohort_summaries(
        all_records,
        variants=variants or [],
        k_values=k_values or [],
        num_draws=num_draws or 1,
        active_controls=active_controls or [],
        total_roots_by_variant=total_roots_by_variant,
        n_boot=n_boot,
        seed=seed,
    )
    out = {
        "metadata": {
            "source_files": [str(p) for p in eval_files],
            "model_seeds": sorted(set(seed_ids)),
            "n_seeds": len(set(seed_ids)),
            "snapshot_sha256": next(iter(snapshot_shas)),
            "variants": variants,
            "k_values": k_values,
            "num_draws": num_draws,
            "controls": active_controls,
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
        "summary": summary,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-study frozen evaluation suite for chess_icl_v2")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. select-checkpoint
    p_sel = sub.add_parser("select-checkpoint", help="Select best checkpoint on full validation fold")
    p_sel.add_argument("--data-dir", type=Path, required=True)
    p_sel.add_argument("--checkpoints-dir", type=Path, default=None)
    p_sel.add_argument("--checkpoints", type=Path, nargs="+", default=None)
    p_sel.add_argument("--output-manifest", type=Path, required=True)
    p_sel.add_argument(
        "--selection-regime",
        choices=["familiar", "warmup", "standard", "all_single", "all_variants"],
        default="familiar",
    )
    p_sel.add_argument("--variants", nargs="+", default=None)
    p_sel.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p_sel.add_argument("--num-draws", type=int, default=5)
    p_sel.add_argument("--controls", nargs="*", default=list(DEFAULT_CONTROLS))
    p_sel.add_argument("--notation", choices=["uci", "san"], default="san")
    p_sel.add_argument("--max-seq-len", type=int, default=32768)
    p_sel.add_argument("--max-roots", type=int, default=None)
    p_sel.add_argument("--seed", type=int, default=42)
    p_sel.add_argument("--gpu", type=int, default=0)
    p_sel.add_argument("--model-path", type=str, default=None)
    p_sel.add_argument("--arch", choices=["auto", "qwen", "scratch"], default="auto")

    # 2. evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a frozen manifest on test/val split")
    p_eval.add_argument("--manifest", type=Path, required=True)
    p_eval.add_argument("--data-dir", type=Path, required=True)
    p_eval.add_argument("--output-file", type=Path, required=True)
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.add_argument("--split", choices=["val", "test"], default="test")
    p_eval.add_argument("--variants", nargs="+", default=None)
    p_eval.add_argument("--notation", choices=["uci", "san"], default=None)
    p_eval.add_argument("--max-roots", type=int, default=None)
    p_eval.add_argument("--n-boot", type=int, default=1000)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--model-seed", type=int, default=0)
    p_eval.add_argument("--gpu", type=int, default=0)
    p_eval.add_argument("--model-path", type=str, default=None)
    p_eval.add_argument("--arch", choices=["auto", "qwen", "scratch"], default="auto")
    p_eval.add_argument("--token-match-tolerance", type=float, default=0.10)
    p_eval.add_argument("--max-search-attempts", type=int, default=200)
    p_eval.add_argument("--r256-subset-manifest", type=Path, default=None)
    p_eval.add_argument("--r256-positions-filename", type=str, default=None)
    p_eval.add_argument("--r256-data-dir", type=Path, default=None)

    # 3. select-r256-subset
    p_r256 = sub.add_parser(
        "select-r256-subset",
        help="Select outcome-independent test family subset for R=256 re-estimation",
    )
    p_r256.add_argument("--data-dir", type=Path, required=True)
    p_r256.add_argument("--output-manifest", type=Path, required=True)
    p_r256.add_argument("--families-per-variant", type=int, default=16)
    p_r256.add_argument("--variants", nargs="+", default=None)
    p_r256.add_argument("--split", choices=["val", "test"], default="test")
    p_r256.add_argument("--hash-salt", type=str, default="")

    # 4. run-baselines
    p_base = sub.add_parser("run-baselines", help="Run all 7 declared non-neural baselines")
    p_base.add_argument("--data-dir", type=Path, required=True)
    p_base.add_argument("--output-file", type=Path, required=True)
    p_base.add_argument("--manifest", type=Path, default=None)
    p_base.add_argument("--split", choices=["val", "test"], default="test")
    p_base.add_argument("--variants", nargs="+", default=None)
    p_base.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p_base.add_argument("--num-draws", type=int, default=5)
    p_base.add_argument("--n-boot", type=int, default=1000)
    p_base.add_argument("--seed", type=int, default=42)
    p_base.add_argument("--knn-top-k", type=int, default=10)
    p_base.add_argument("--knn-metric", choices=["cosine", "hamming"], default="cosine")
    p_base.add_argument("--max-train-per-variant", type=int, default=None)
    p_base.add_argument("--max-roots", type=int, default=None)

    # 5. aggregate-seeds
    p_agg = sub.add_parser("aggregate-seeds", help="Aggregate multiple seed evaluation JSONs")
    p_agg.add_argument("--eval-files", type=Path, nargs="+", required=True)
    p_agg.add_argument("--output-file", type=Path, required=True)
    p_agg.add_argument("--n-boot", type=int, default=1000)
    p_agg.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "select-checkpoint":
        select_checkpoint(
            data_dir=args.data_dir,
            output_manifest=args.output_manifest,
            checkpoints_dir=args.checkpoints_dir,
            checkpoint_paths=args.checkpoints,
            selection_regime=args.selection_regime,
            variants=args.variants,
            k_values=args.k_values,
            num_draws=args.num_draws,
            controls=args.controls,
            notation=args.notation,
            max_seq_len=args.max_seq_len,
            max_roots=args.max_roots,
            seed=args.seed,
            gpu=args.gpu,
            model_path=args.model_path,
            arch=args.arch,
        )
    elif args.command == "evaluate":
        evaluate_from_manifest(
            manifest_path=args.manifest,
            data_dir=args.data_dir,
            output_file=args.output_file,
            split=args.split,
            variants=args.variants,
            checkpoint_path=args.checkpoint,
            notation=args.notation,
            max_roots=args.max_roots,
            n_boot=args.n_boot,
            seed=args.seed,
            model_seed=args.model_seed,
            gpu=args.gpu,
            model_path=args.model_path,
            token_match_tolerance=args.token_match_tolerance,
            max_search_attempts=args.max_search_attempts,
            r256_subset_manifest=args.r256_subset_manifest,
            r256_positions_filename=args.r256_positions_filename,
            r256_data_dir=args.r256_data_dir,
            arch=args.arch,
        )
    elif args.command == "select-r256-subset":
        select_r256_subset(
            data_dir=args.data_dir,
            output_manifest=args.output_manifest,
            families_per_variant=args.families_per_variant,
            variants=args.variants,
            split=args.split,
            hash_salt=args.hash_salt,
        )
    elif args.command == "run-baselines":
        run_declared_baselines(
            data_dir=args.data_dir,
            output_file=args.output_file,
            split=args.split,
            manifest_path=args.manifest,
            variants=args.variants,
            k_values=args.k_values,
            num_draws=args.num_draws,
            n_boot=args.n_boot,
            seed=args.seed,
            knn_top_k=args.knn_top_k,
            knn_metric=args.knn_metric,
            max_train_per_variant=args.max_train_per_variant,
            max_roots=args.max_roots,
        )
    elif args.command == "aggregate-seeds":
        aggregate_multi_seed_evaluations(
            eval_files=args.eval_files,
            output_file=args.output_file,
            n_boot=args.n_boot,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
