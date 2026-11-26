#!/usr/bin/env python3
"""Unified long-training study trainer for Qwen2.5-7B LoRA and Scratch 494M Qwen2.

Supports:
  - Architecture: ``qwen`` (LoRA handoff + FP32 3-way head) or ``scratch`` (494M FP32 + BF16 autocast)
  - Phases: ``warmup``, ``main``, ``extend``
  - Exposures: ``standard``, ``familiar`` (4 TRAIN_VARIANTS), ``all_rules`` (13 SINGLE_VARIANTS)
  - Warmup-parent cloning via ``--init-weights-dir`` with fresh optimizer/scheduler & step_0 milestone
  - Atomic rolling full-state checkpointing (only ``ckpt_latest`` & ``ckpt_prev`` retained)
  - Graceful ``SIGTERM`` / ``SIGINT`` handling at the next completed optimizer boundary
  - Canonical unpadded ``batch_size=1`` (``attention_mask=None``) training and val-panel evaluation
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.long_study_sampler import (  # noqa: E402
    DeterministicEpisodeSampler,
    load_val_panel_episodes,
    verify_snapshot_manifest,
)
from chess_icl_v2.eval.metrics import (  # noqa: E402
    compute_argmax_accuracy,
    compute_brier,
    compute_nll,
)
from chess_icl_v2.models.qwen_model import (  # noqa: E402
    LoraMode,
    Notation,
    QwenICLClassifier,
    build_cosine_warmup_to_min_scheduler,
    build_long_study_optimizer,
    get_preferred_attn_implementation,
    load_qwen_tokenizer,
    resolve_qwen_model_path,
)
from chess_icl_v2.models.scratch_qwen2 import (  # noqa: E402
    ScratchQwen2Classifier,
)
from chess_icl_v2.variants import SINGLE_VARIANTS, TRAIN_VARIANTS  # noqa: E402

DEFAULT_MAIN_MILESTONES: tuple[int, ...] = (0, 200, 1000, 2500, 5000, 10000)
DEFAULT_WARMUP_MILESTONES: tuple[int, ...] = (0, 200, 1000)
DEFAULT_K_VALUES: tuple[int, ...] = (0, 1, 2, 4)

ARCH_DEFAULTS: dict[str, dict[str, float]] = {
    "qwen": {"lr": 1e-4, "weight_decay": 0.01},
    "scratch": {"lr": 3e-4, "weight_decay": 0.1},
}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def resolve_exposure_variants(
    exposure: str,
    explicit_variants: Optional[Sequence[str]] = None,
) -> list[str]:
    """Map exposure regime name (or explicit variants) to an ordered list of variants."""
    if explicit_variants:
        seen: set[str] = set()
        out: list[str] = []
        for v in explicit_variants:
            vs = str(v)
            if vs not in seen:
                seen.add(vs)
                out.append(vs)
        return out

    if exposure == "standard":
        return ["standard"]
    if exposure == "familiar":
        return list(TRAIN_VARIANTS)
    if exposure == "all_rules":
        return list(SINGLE_VARIANTS)
    raise ValueError(f"Unsupported exposure {exposure!r}; expected standard|familiar|all_rules")


def resolve_device(gpu: int = 0) -> torch.device:
    """Resolve CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        idx = int(gpu) if 0 <= int(gpu) < count else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _fsync_tree(root: Path) -> None:
    """Best-effort fsync of all regular files inside ``root``."""
    for item in root.rglob("*"):
        if item.is_file():
            try:
                _fsync_file(item)
            except OSError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text to ``path`` using temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Any, indent: int = 2) -> None:
    """Atomically serialize JSON to ``path``."""
    atomic_write_text(path, json.dumps(payload, indent=indent, sort_keys=False) + "\n")


def atomic_write_dir(target_dir: Path, writer_fn: Callable[[Path], None]) -> Path:
    """Atomically populate ``target_dir`` via a staging directory and swap."""
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    tmp_dir = parent / f".{target_dir.name}.tmp.{pid}"
    old_dir = parent / f".{target_dir.name}.old.{pid}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(old_dir, ignore_errors=True)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        writer_fn(tmp_dir)
        _fsync_tree(tmp_dir)
        if target_dir.exists():
            os.replace(target_dir, old_dir)
        os.replace(tmp_dir, target_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
    except Exception:
        if not target_dir.exists() and old_dir.exists():
            os.replace(old_dir, target_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return target_dir


def save_model_weights(model: nn.Module, arch: str, target_dir: Path) -> Path:
    """Atomically save model weights (LoRA+head for Qwen, full FP32 for Scratch)."""

    def _write(tmp_dir: Path) -> None:
        if arch == "qwen":
            model.save_adapter_and_head(tmp_dir)  # type: ignore[operator]
        elif arch == "scratch":
            model.save_weights(tmp_dir)  # type: ignore[operator]
        else:
            raise ValueError(f"Unsupported arch {arch!r}")

    return atomic_write_dir(target_dir, _write)


def load_model_weights(model: nn.Module, arch: str, weights_dir: Path) -> None:
    """Load saved weights into an existing model instance."""
    if arch == "qwen":
        model.load_adapter_and_head(weights_dir)  # type: ignore[operator]
    elif arch == "scratch":
        model.load_weights(weights_dir)  # type: ignore[operator]
    else:
        raise ValueError(f"Unsupported arch {arch!r}")


def compute_tokenizer_sha256(tokenizer: Any) -> str:
    """Compute a deterministic hash identifying tokenizer configuration and vocabulary."""
    h = hashlib.sha256()
    info = {
        "class": type(tokenizer).__name__,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    h.update(json.dumps(info, sort_keys=True).encode("utf-8"))
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
            for tok in sorted(vocab.keys()):
                h.update(f"{tok}:{vocab[tok]}\n".encode("utf-8"))
        except Exception:
            pass
    return h.hexdigest()


def compute_model_config_sha256(model: nn.Module, arch: str) -> str:
    """Compute a deterministic hash of model architecture config."""
    h = hashlib.sha256()
    h.update(arch.encode("utf-8"))
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        h.update(json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8"))
    if hasattr(model, "lora_mode"):
        h.update(str(getattr(model, "lora_mode")).encode("utf-8"))
    return h.hexdigest()


def compute_snapshot_report_sha256(report: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 over verified snapshot files and manifest."""
    h = hashlib.sha256()
    verified = report.get("verified_files", {})
    for rel in sorted(verified.keys()):
        entry = verified[rel]
        h.update(rel.encode("utf-8"))
        h.update(str(entry.get("sha256", "")).encode("utf-8"))
        h.update(str(entry.get("line_count", "")).encode("utf-8"))
    return h.hexdigest()


def compute_trainable_weights_sha256(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 over all trainable parameters."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        if not param.requires_grad:
            continue
        t = param.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        if t.dtype == torch.bfloat16:
            h.update(t.view(torch.uint16).numpy().tobytes())
        else:
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _resolve_lineage(
    init_weights_dir: Optional[Path],
    phase: str,
    exposure: str,
    exposed_variants: Sequence[str],
    seed: int,
    arch: str,
) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    if init_weights_dir is not None:
        parent_candidates = [
            init_weights_dir.parent / "training_summary.json",
            init_weights_dir.parent / "run_manifest.json",
            init_weights_dir / "lineage.json",
        ]
        for cand in parent_candidates:
            if cand.exists():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    parent_lin = data.get("exposure_lineage")
                    if isinstance(parent_lin, list):
                        lineage.extend(parent_lin)
                        break
                    if "phase" in data or "exposure" in data:
                        lineage.append(
                            {
                                "arch": data.get("arch", arch),
                                "phase": data.get("phase"),
                                "exposure": data.get("exposure"),
                                "exposed_variants": data.get("exposed_variants"),
                                "seed": data.get("seed"),
                                "source_dir": str(cand.parent),
                            }
                        )
                        break
                except Exception:
                    pass
        if not lineage:
            lineage.append({"init_weights_dir": str(init_weights_dir)})

    lineage.append(
        {
            "arch": arch,
            "phase": phase,
            "exposure": exposure,
            "exposed_variants": list(exposed_variants),
            "seed": int(seed),
            "init_weights_dir": str(init_weights_dir) if init_weights_dir else None,
        }
    )
    return lineage


def _is_valid_checkpoint_dir(ckpt_dir: Path) -> bool:
    meta_path = ckpt_dir / "checkpoint_meta.json"
    trainer_path = ckpt_dir / "trainer_state.pt"
    sampler_path = ckpt_dir / "sampler_state.json"
    weights_dir = ckpt_dir / "weights"
    if not (
        meta_path.exists()
        and trainer_path.exists()
        and sampler_path.exists()
        and weights_dir.is_dir()
    ):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("status") == "complete"
    except Exception:
        return False


def resolve_resume_checkpoint(
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> Path:
    """Locate the latest valid rolling checkpoint (ckpt_latest with fallback to ckpt_prev)."""
    if resume_from is not None:
        p = Path(resume_from)
        if _is_valid_checkpoint_dir(p):
            return p
        raise FileNotFoundError(f"Explicit resume checkpoint is missing or incomplete: {p}")

    ckpt_root = output_dir / "checkpoints"
    for name in ("ckpt_latest", "ckpt_prev"):
        cand = ckpt_root / name
        if _is_valid_checkpoint_dir(cand):
            return cand
    raise FileNotFoundError(
        f"No valid rolling checkpoint (ckpt_latest or ckpt_prev) found under {ckpt_root}"
    )


def save_rolling_checkpoint(
    *,
    output_dir: Path,
    arch: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: DeterministicEpisodeSampler,
    step: int,
    total_tokens_processed: int,
    total_episodes_processed: int,
    unique_roots_seen: set[str],
    unique_families_seen: set[str],
    gpu_process_seconds: float,
    wall_seconds: float,
    best_val_nll: float,
    best_val_step: Optional[int],
    exposure_lineage: list[dict[str, Any]],
    hashes: dict[str, str],
    run_config: dict[str, Any],
) -> Path:
    """Atomically save a full-state checkpoint and rotate keeping only ckpt_latest and ckpt_prev."""
    ckpt_root = output_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    ckpt_tmp = ckpt_root / f".ckpt_new.tmp.{pid}"
    ckpt_trash = ckpt_root / f".ckpt_trash.tmp.{pid}"
    ckpt_latest = ckpt_root / "ckpt_latest"
    ckpt_prev = ckpt_root / "ckpt_prev"

    # Clean up stale temp directories from any prior crashed attempt.
    for stale in ckpt_root.glob(".ckpt_*.tmp.*"):
        shutil.rmtree(stale, ignore_errors=True)

    ckpt_tmp.mkdir(parents=True, exist_ok=True)
    try:
        weights_dir = ckpt_tmp / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        if arch == "qwen":
            model.save_adapter_and_head(weights_dir)  # type: ignore[operator]
        elif arch == "scratch":
            model.save_weights(weights_dir)  # type: ignore[operator]
        else:
            raise ValueError(f"Unsupported arch {arch!r}")

        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        trainer_state = {
            "step": int(step),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng,
            "total_tokens_processed": int(total_tokens_processed),
            "total_episodes_processed": int(total_episodes_processed),
            "unique_roots_seen": sorted(unique_roots_seen),
            "unique_families_seen": sorted(unique_families_seen),
            "gpu_process_seconds": float(gpu_process_seconds),
            "wall_seconds": float(wall_seconds),
            "best_val_nll": float(best_val_nll),
            "best_val_step": best_val_step,
            "exposure_lineage": exposure_lineage,
            "hashes": hashes,
            "run_config": run_config,
        }
        trainer_pt = ckpt_tmp / "trainer_state.pt"
        torch.save(trainer_state, trainer_pt)

        sampler_json = ckpt_tmp / "sampler_state.json"
        atomic_write_json(sampler_json, sampler.state_dict())

        # Fsync all payload files before writing the completion marker.
        _fsync_tree(ckpt_tmp)
        atomic_write_json(
            ckpt_tmp / "checkpoint_meta.json",
            {
                "status": "complete",
                "step": int(step),
                "arch": arch,
                "timestamp": utc_now_iso(),
            },
        )
        _fsync_file(ckpt_tmp / "checkpoint_meta.json")

        # Atomic two-slot rotation: move ckpt_prev aside, demote ckpt_latest -> ckpt_prev,
        # promote ckpt_tmp -> ckpt_latest, then remove trash.
        if ckpt_prev.exists():
            os.replace(ckpt_prev, ckpt_trash)
        if ckpt_latest.exists():
            os.replace(ckpt_latest, ckpt_prev)
        os.replace(ckpt_tmp, ckpt_latest)
        shutil.rmtree(ckpt_trash, ignore_errors=True)
    except Exception:
        shutil.rmtree(ckpt_tmp, ignore_errors=True)
        if not ckpt_latest.exists() and ckpt_prev.exists():
            # Leave ckpt_prev available for fallback; do not discard valid state.
            pass
        raise

    return ckpt_latest


def load_rolling_checkpoint(
    *,
    ckpt_dir: Path,
    arch: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: DeterministicEpisodeSampler,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Restore model, optimizer, scheduler, RNGs, sampler, and telemetry from ``ckpt_dir``."""
    if not _is_valid_checkpoint_dir(ckpt_dir):
        raise ValueError(f"Invalid or incomplete checkpoint directory: {ckpt_dir}")

    load_model_weights(model, arch, ckpt_dir / "weights")
    trainer_state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu", weights_only=False)

    saved_hashes = trainer_state.get("hashes", {})
    for key in ("snapshot_sha256", "model_config_sha256", "tokenizer_sha256"):
        if saved_hashes.get(key) != expected_hashes.get(key):
            raise ValueError(
                f"Resume hash mismatch for {key}: checkpoint={saved_hashes.get(key)!r}, "
                f"current={expected_hashes.get(key)!r}"
            )

    optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    scheduler.load_state_dict(trainer_state["scheduler_state_dict"])

    sampler_state = json.loads((ckpt_dir / "sampler_state.json").read_text(encoding="utf-8"))
    sampler.load_state_dict(sampler_state)

    random.setstate(trainer_state["python_rng_state"])
    np.random.set_state(trainer_state["numpy_rng_state"])
    torch_rng = trainer_state["torch_rng_state"]
    if not isinstance(torch_rng, torch.ByteTensor):
        torch_rng = torch_rng.to(dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_rng)
    cuda_rng = trainer_state.get("cuda_rng_state")
    if cuda_rng is not None and torch.cuda.is_available():
        try:
            if len(cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng)
            elif len(cuda_rng) > 0:
                torch.cuda.set_rng_state(cuda_rng[0])
        except Exception:
            pass

    return trainer_state


def evaluate_val_panel(
    *,
    model: nn.Module,
    val_episodes: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_seq_len: int,
) -> dict[str, Any]:
    """Evaluate fixed validation panel using canonical unpadded batch_size=1 inference."""
    t0 = time.perf_counter()
    if not val_episodes:
        return {
            "n_episodes": 0,
            "nll": 0.0,
            "brier": 0.0,
            "accuracy": 0.0,
            "by_k": {},
            "by_variant": {},
            "eval_seconds": 0.0,
        }

    was_training = model.training
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    by_k_preds: dict[str, list[np.ndarray]] = {}
    by_k_targets: dict[str, list[np.ndarray]] = {}
    by_var_preds: dict[str, list[np.ndarray]] = {}
    by_var_targets: dict[str, list[np.ndarray]] = {}

    try:
        with torch.no_grad():
            for ep in val_episodes:
                encoded = tokenizer(
                    ep["prompt_text"],
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].to(device)
                seq_len = int(input_ids.shape[1])
                if seq_len > max_seq_len:
                    raise ValueError(
                        f"Validation episode length {seq_len} exceeds max_seq_len={max_seq_len}"
                    )
                out = model(input_ids=input_ids, attention_mask=None)
                prob = out.probs.squeeze(0).detach().cpu().numpy().astype(np.float64)
                y = np.asarray(ep["target_probs"], dtype=np.float64)

                preds.append(prob)
                targets.append(y)
                k_key = str(ep["k"])
                v_key = str(ep["variant"])
                by_k_preds.setdefault(k_key, []).append(prob)
                by_k_targets.setdefault(k_key, []).append(y)
                by_var_preds.setdefault(v_key, []).append(prob)
                by_var_targets.setdefault(v_key, []).append(y)
    finally:
        if was_training:
            model.train()

    p_mat = np.stack(preds, axis=0)
    y_mat = np.stack(targets, axis=0)
    mean_nll, _ = compute_nll(p_mat, y_mat)
    mean_brier = compute_brier(p_mat, y_mat)
    acc = compute_argmax_accuracy(p_mat, y_mat)["accuracy"]

    by_k: dict[str, dict[str, float]] = {}
    for k_key in sorted(by_k_preds.keys(), key=lambda s: int(s)):
        kp = np.stack(by_k_preds[k_key], axis=0)
        ky = np.stack(by_k_targets[k_key], axis=0)
        knll, _ = compute_nll(kp, ky)
        by_k[k_key] = {
            "n": int(len(kp)),
            "nll": float(knll),
            "brier": float(compute_brier(kp, ky)),
            "accuracy": float(compute_argmax_accuracy(kp, ky)["accuracy"]),
        }

    by_variant: dict[str, dict[str, float]] = {}
    for v_key in sorted(by_var_preds.keys()):
        vp = np.stack(by_var_preds[v_key], axis=0)
        vy = np.stack(by_var_targets[v_key], axis=0)
        vnll, _ = compute_nll(vp, vy)
        by_variant[v_key] = {
            "n": int(len(vp)),
            "nll": float(vnll),
            "brier": float(compute_brier(vp, vy)),
            "accuracy": float(compute_argmax_accuracy(vp, vy)["accuracy"]),
        }

    return {
        "n_episodes": int(len(preds)),
        "nll": float(mean_nll),
        "brier": float(mean_brier),
        "accuracy": float(acc),
        "by_k": by_k,
        "by_variant": by_variant,
        "eval_seconds": float(time.perf_counter() - t0),
    }


def _append_jsonl_line(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _truncate_trace_after_step(trace_path: Path, max_kept_step: int) -> None:
    """When resuming from an earlier checkpoint, trim trace records beyond ``max_kept_step``."""
    if not trace_path.exists():
        return
    kept_lines: list[str] = []
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
            if int(rec.get("step", -1)) <= max_kept_step:
                kept_lines.append(s)
        except Exception:
            continue
    text = ("\n".join(kept_lines) + "\n") if kept_lines else ""
    atomic_write_text(trace_path, text)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    return 0.0


def build_default_model_and_tokenizer(
    *,
    arch: str,
    device: torch.device,
    model_path: Optional[str] = None,
    lora_mode: LoraMode = "handoff",
    gradient_checkpointing: bool = True,
) -> tuple[nn.Module, Any]:
    """Instantiate the production Qwen or Scratch model and Qwen2.5 tokenizer."""
    resolved_qwen_path = resolve_qwen_model_path(model_path)
    tokenizer = load_qwen_tokenizer(resolved_qwen_path, local_files_only=True)

    if arch == "qwen":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = QwenICLClassifier(
            tokenizer=tokenizer,
            model_path=resolved_qwen_path,
            lora_mode=lora_mode,
            torch_dtype=torch_dtype,
            head_dtype=torch.float32,
            adapter_dtype=torch.float32,
            attn_implementation=get_preferred_attn_implementation(),
            local_files_only=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        model = model.to(device)
        return model, tokenizer

    if arch == "scratch":
        model = ScratchQwen2Classifier(
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            head_dtype=torch.float32,
            gradient_checkpointing=gradient_checkpointing,
            device=device,
        )
        return model, tokenizer

    raise ValueError(f"Unsupported arch {arch!r}")


def run_training(
    *,
    arch: str,
    phase: str,
    exposure: str,
    seed: int,
    snapshot_dir: Path,
    output_dir: Path,
    init_weights_dir: Optional[Path] = None,
    resume: bool = False,
    resume_from: Optional[Path] = None,
    max_steps: Optional[int] = None,
    stop_at_step: Optional[int] = None,
    eval_interval: Optional[int] = None,
    ckpt_interval: int = 250,
    milestones: Optional[Sequence[int]] = None,
    batch_size: int = 1,
    grad_accum: int = 8,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    val_k_values: Optional[Sequence[int]] = None,
    val_roots_per_variant: int = 32,
    val_seed: int = 0,
    max_seq_len: int = 32768,
    notation: Notation = "san",
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    min_lr_ratio: float = 0.1,
    warmup_steps: Optional[int] = None,
    max_grad_norm: float = 1.0,
    gpu: int = 0,
    model_path: Optional[str] = None,
    lora_mode: LoraMode = "handoff",
    gradient_checkpointing: bool = True,
    verify_manifest: bool = True,
    heartbeat_interval: int = 10,
    explicit_variants: Optional[Sequence[str]] = None,
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Execute or resume a long-study training phase."""
    if arch not in ARCH_DEFAULTS:
        raise ValueError(f"Unsupported arch {arch!r}; expected 'qwen' or 'scratch'")
    if phase not in ("warmup", "main", "extend"):
        raise ValueError(f"Unsupported phase {phase!r}; expected warmup|main|extend")
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")

    eff_max_steps = int(
        max_steps if max_steps is not None else (1000 if phase == "warmup" else 10000)
    )
    eff_eval_interval = int(
        eval_interval if eval_interval is not None else (100 if phase == "warmup" else 500)
    )
    if milestones is not None:
        eff_milestones = sorted({int(m) for m in milestones if int(m) >= 0})
    elif phase == "main":
        eff_milestones = list(DEFAULT_MAIN_MILESTONES)
    elif phase == "warmup":
        eff_milestones = list(DEFAULT_WARMUP_MILESTONES)
    else:
        eff_milestones = [0, eff_max_steps]

    eff_lr = float(lr if lr is not None else ARCH_DEFAULTS[arch]["lr"])
    eff_wd = float(weight_decay if weight_decay is not None else ARCH_DEFAULTS[arch]["weight_decay"])
    eff_k_values = [int(k) for k in k_values]
    eff_val_k_values = [int(k) for k in (val_k_values if val_k_values is not None else eff_k_values)]
    exposed_variants = resolve_exposure_variants(exposure, explicit_variants=explicit_variants)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(snapshot_dir)

    eff_device = device if device is not None else resolve_device(gpu)
    if eff_device.type == "cuda" and torch.cuda.is_available():
        cuda_idx = 0 if eff_device.index is None else int(eff_device.index)
        eff_device = torch.device("cuda", cuda_idx)
        torch.cuda.set_device(eff_device)
        torch.cuda.reset_peak_memory_stats(eff_device)

    set_global_seed(seed)
    t_load_start = time.perf_counter()
    if model is None or tokenizer is None:
        built_model, built_tok = build_default_model_and_tokenizer(
            arch=arch,
            device=eff_device,
            model_path=model_path,
            lora_mode=lora_mode,
            gradient_checkpointing=gradient_checkpointing,
        )
        model = model if model is not None else built_model
        tokenizer = tokenizer if tokenizer is not None else built_tok
    model_load_seconds = time.perf_counter() - t_load_start

    model = model.to(eff_device)
    model.train()

    if verify_manifest:
        snap_report = verify_snapshot_manifest(
            snap_dir,
            exposed_variants=exposed_variants,
            max_k=max(eff_k_values),
        )
        snapshot_sha256 = compute_snapshot_report_sha256(snap_report)
    else:
        manifest_text = (snap_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
        snapshot_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    sampler = DeterministicEpisodeSampler(
        snapshot_dir=snap_dir,
        tokenizer=tokenizer,
        exposed_variants=exposed_variants,
        k_values=eff_k_values,
        seed=seed,
        notation=notation,
        max_length=max_seq_len,
        strict_no_trim=True,
        verify_manifest=False,
    )
    val_episodes = load_val_panel_episodes(
        snapshot_dir=snap_dir,
        tokenizer=tokenizer,
        variants=exposed_variants,
        k_values=eff_val_k_values,
        roots_per_variant=val_roots_per_variant,
        seed=val_seed,
        notation=notation,
        max_length=max_seq_len,
        strict_no_trim=True,
        verify_manifest=False,
    )

    # Fresh phase initialization from --init-weights-dir (when NOT resuming).
    do_resume = bool(resume or resume_from is not None)
    if not do_resume and init_weights_dir is not None:
        load_model_weights(model, arch, Path(init_weights_dir))

    # Build fresh optimizer and cosine warmup-to-min scheduler.
    optimizer = build_long_study_optimizer(
        model,
        lr=eff_lr,
        weight_decay=eff_wd,
        betas=(0.9, 0.95),
    )
    scheduler = build_cosine_warmup_to_min_scheduler(
        optimizer,
        total_steps=eff_max_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    hashes = {
        "snapshot_sha256": snapshot_sha256,
        "model_config_sha256": compute_model_config_sha256(model, arch),
        "tokenizer_sha256": compute_tokenizer_sha256(tokenizer),
        "initial_weights_sha256": compute_trainable_weights_sha256(model),
    }
    run_config = {
        "arch": arch,
        "phase": phase,
        "exposure": exposure,
        "exposed_variants": exposed_variants,
        "seed": int(seed),
        "snapshot_dir": str(snap_dir),
        "init_weights_dir": str(init_weights_dir) if init_weights_dir else None,
        "max_steps": eff_max_steps,
        "eval_interval": eff_eval_interval,
        "ckpt_interval": int(ckpt_interval),
        "milestones": eff_milestones,
        "batch_size": int(batch_size),
        "grad_accum": int(grad_accum),
        "k_values": eff_k_values,
        "val_k_values": eff_val_k_values,
        "val_roots_per_variant": int(val_roots_per_variant),
        "max_seq_len": int(max_seq_len),
        "notation": notation,
        "lr": eff_lr,
        "weight_decay": eff_wd,
        "min_lr_ratio": float(min_lr_ratio),
        "warmup_steps": warmup_steps,
        "max_grad_norm": float(max_grad_norm),
    }

    completed_steps = 0
    total_tokens_processed = 0
    total_episodes_processed = 0
    unique_roots_seen: set[str] = set()
    unique_families_seen: set[str] = set()
    prior_gpu_seconds = 0.0
    prior_wall_seconds = 0.0
    best_val_nll = float("inf")
    best_val_step: Optional[int] = None
    val_history_path = out_dir / "val_panel_history.json"
    trace_path = out_dir / "train_trace.jsonl"
    heartbeat_path = out_dir / "heartbeat.json"
    val_history: list[dict[str, Any]] = []

    if do_resume:
        ckpt_dir = resolve_resume_checkpoint(out_dir, resume_from=resume_from)
        restored = load_rolling_checkpoint(
            ckpt_dir=ckpt_dir,
            arch=arch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            expected_hashes=hashes,
        )
        completed_steps = int(restored["step"])
        total_tokens_processed = int(restored["total_tokens_processed"])
        total_episodes_processed = int(restored["total_episodes_processed"])
        unique_roots_seen = set(restored["unique_roots_seen"])
        unique_families_seen = set(restored["unique_families_seen"])
        prior_gpu_seconds = float(restored.get("gpu_process_seconds", 0.0))
        prior_wall_seconds = float(restored.get("wall_seconds", 0.0))
        best_val_nll = float(restored.get("best_val_nll", float("inf")))
        best_val_step = restored.get("best_val_step")
        exposure_lineage = list(restored.get("exposure_lineage", []))
        hashes = dict(restored.get("hashes", hashes))

        _truncate_trace_after_step(trace_path, completed_steps)
        if val_history_path.exists():
            try:
                loaded_hist = json.loads(val_history_path.read_text(encoding="utf-8"))
                if isinstance(loaded_hist, list):
                    val_history = [h for h in loaded_hist if int(h.get("step", -1)) <= completed_steps]
                    atomic_write_json(val_history_path, val_history)
            except Exception:
                val_history = []
    else:
        exposure_lineage = _resolve_lineage(
            init_weights_dir=init_weights_dir,
            phase=phase,
            exposure=exposure,
            exposed_variants=exposed_variants,
            seed=seed,
            arch=arch,
        )
        # Fresh run: ensure step_0 milestone is saved if requested or cloned from parent.
        if 0 in eff_milestones or init_weights_dir is not None:
            save_model_weights(model, arch, out_dir / "milestones" / "step_00000")

    # Install graceful SIGTERM / SIGINT handlers when running on main thread.
    stop_flag = {"requested": False, "signal": None}
    is_main_thread = threading.current_thread() is threading.main_thread()
    prev_sigterm = signal.getsignal(signal.SIGTERM) if is_main_thread else None
    prev_sigint = signal.getsignal(signal.SIGINT) if is_main_thread else None

    def _handle_signal(signum: int, _frame: Any) -> None:
        stop_flag["requested"] = True
        stop_flag["signal"] = signum

    if is_main_thread:
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    session_start_wall = time.perf_counter()
    session_start_proc = time.process_time()
    last_train_loss: Optional[float] = None
    last_val_nll: Optional[float] = (
        float(val_history[-1]["nll"]) if val_history else None
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def _write_heartbeat(status: str, tokens_per_sec: float = 0.0) -> None:
        elapsed_wall = prior_wall_seconds + (time.perf_counter() - session_start_wall)
        elapsed_gpu = prior_gpu_seconds + (time.perf_counter() - session_start_wall)
        payload = {
            "timestamp": utc_now_iso(),
            "pid": os.getpid(),
            "status": status,
            "arch": arch,
            "phase": phase,
            "exposure": exposure,
            "seed": int(seed),
            "step": int(completed_steps),
            "max_steps": eff_max_steps,
            "total_tokens": int(total_tokens_processed),
            "total_episodes": int(total_episodes_processed),
            "tokens_per_sec": float(tokens_per_sec),
            "peak_vram_mb": _peak_vram_mb(eff_device),
            "wall_seconds": float(elapsed_wall),
            "gpu_process_seconds": float(elapsed_gpu),
            "last_train_loss": last_train_loss,
            "last_val_nll": last_val_nll,
        }
        atomic_write_json(heartbeat_path, payload)

    _write_heartbeat("running", tokens_per_sec=0.0)

    try:
        while completed_steps < eff_max_steps and not stop_flag["requested"]:
            if stop_at_step is not None and completed_steps >= int(stop_at_step):
                break
            step_t0 = time.perf_counter()
            sampler.set_step(completed_steps)
            optimizer.zero_grad(set_to_none=True)

            raw_loss_sum = 0.0
            tokens_this_step = 0
            episodes_this_step = 0
            input_hashes: list[str] = []
            variants_this_step: list[str] = []
            k_this_step: list[int] = []

            for _micro in range(grad_accum):
                if batch_size == 1:
                    prompt_text, target_probs, meta = sampler.sample_episode()
                    encoded = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                    )
                    input_ids = encoded["input_ids"].to(eff_device)
                    seq_len = int(input_ids.shape[1])
                    if seq_len > max_seq_len:
                        raise ValueError(
                            f"Sampled episode token length {seq_len} exceeds max_seq_len={max_seq_len}"
                        )
                    attention_mask = None
                    targets = torch.as_tensor(
                        target_probs, dtype=torch.float32, device=eff_device
                    ).unsqueeze(0)

                    tokens_this_step += seq_len
                    episodes_this_step += 1
                    input_hashes.append(str(meta["input_sha256"]))
                    variants_this_step.append(str(meta["variant"]))
                    k_this_step.append(int(meta["k"]))
                    unique_roots_seen.add(str(meta["root_id"]))
                    unique_families_seen.add(str(meta["family_id"]))
                else:
                    batch_prompts: list[str] = []
                    batch_targets: list[np.ndarray] = []
                    for _b in range(batch_size):
                        prompt_text, target_probs, meta = sampler.sample_episode()
                        batch_prompts.append(prompt_text)
                        batch_targets.append(target_probs)
                        episodes_this_step += 1
                        input_hashes.append(str(meta["input_sha256"]))
                        variants_this_step.append(str(meta["variant"]))
                        k_this_step.append(int(meta["k"]))
                        unique_roots_seen.add(str(meta["root_id"]))
                        unique_families_seen.add(str(meta["family_id"]))
                    encoded = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    )
                    seq_len = int(encoded["input_ids"].shape[1])
                    if seq_len > max_seq_len:
                        raise ValueError(
                            f"Batch episode token length {seq_len} exceeds max_seq_len={max_seq_len}"
                        )
                    input_ids = encoded["input_ids"].to(eff_device)
                    attention_mask = encoded["attention_mask"].to(eff_device)
                    targets = torch.as_tensor(
                        np.stack(batch_targets, axis=0),
                        dtype=torch.float32,
                        device=eff_device,
                    )
                    tokens_this_step += int(encoded["attention_mask"].sum().item())

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_probs=targets,
                )
                if out.loss is None or not torch.isfinite(out.loss):
                    raise RuntimeError(f"Non-finite training loss at step {completed_steps + 1}")
                raw_loss_sum += float(out.loss.detach().item())
                scaled_loss = out.loss / float(grad_accum)
                scaled_loss.backward()

            current_lr = float(optimizer.param_groups[0]["lr"])
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            completed_steps += 1
            sampler.set_step(completed_steps)
            total_tokens_processed += tokens_this_step
            total_episodes_processed += episodes_this_step
            mean_train_loss = raw_loss_sum / float(grad_accum)
            last_train_loss = mean_train_loss

            step_dt = max(1e-9, time.perf_counter() - step_t0)
            elapsed_wall = prior_wall_seconds + (time.perf_counter() - session_start_wall)
            elapsed_gpu = prior_gpu_seconds + (time.perf_counter() - session_start_wall)

            _append_jsonl_line(
                trace_path,
                {
                    "step": int(completed_steps),
                    "lr": current_lr,
                    "train_loss": float(mean_train_loss),
                    "tokens_this_step": int(tokens_this_step),
                    "total_tokens": int(total_tokens_processed),
                    "total_episodes": int(total_episodes_processed),
                    "unique_roots": len(unique_roots_seen),
                    "unique_families": len(unique_families_seen),
                    "variants": variants_this_step,
                    "k_values": k_this_step,
                    "input_hashes": input_hashes,
                    "step_seconds": float(step_dt),
                    "wall_seconds": float(elapsed_wall),
                    "gpu_process_seconds": float(elapsed_gpu),
                },
            )

            if completed_steps in eff_milestones:
                save_model_weights(
                    model,
                    arch,
                    out_dir / "milestones" / f"step_{completed_steps:05d}",
                )

            is_eval_step = (
                (eff_eval_interval > 0 and completed_steps % eff_eval_interval == 0)
                or completed_steps == eff_max_steps
            )
            if is_eval_step:
                val_metrics = evaluate_val_panel(
                    model=model,
                    val_episodes=val_episodes,
                    tokenizer=tokenizer,
                    device=eff_device,
                    max_seq_len=max_seq_len,
                )
                val_record = {
                    "step": int(completed_steps),
                    "total_tokens": int(total_tokens_processed),
                    "lr": current_lr,
                    "timestamp": utc_now_iso(),
                    **val_metrics,
                }
                val_history.append(val_record)
                last_val_nll = float(val_metrics["nll"])
                atomic_write_json(val_history_path, val_history)

                cand_dir = out_dir / "candidates" / f"step_{completed_steps:05d}"
                save_model_weights(model, arch, cand_dir)

                if val_metrics["nll"] < best_val_nll:
                    best_val_nll = float(val_metrics["nll"])
                    best_val_step = int(completed_steps)
                    save_model_weights(model, arch, out_dir / "best_weights")

            hit_stop_at = stop_at_step is not None and completed_steps >= int(stop_at_step)
            is_ckpt_step = (
                (ckpt_interval > 0 and completed_steps % ckpt_interval == 0)
                or completed_steps == eff_max_steps
                or hit_stop_at
                or stop_flag["requested"]
            )
            if is_ckpt_step:
                elapsed_wall = prior_wall_seconds + (time.perf_counter() - session_start_wall)
                elapsed_gpu = prior_gpu_seconds + (time.perf_counter() - session_start_wall)
                save_rolling_checkpoint(
                    output_dir=out_dir,
                    arch=arch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    step=completed_steps,
                    total_tokens_processed=total_tokens_processed,
                    total_episodes_processed=total_episodes_processed,
                    unique_roots_seen=unique_roots_seen,
                    unique_families_seen=unique_families_seen,
                    gpu_process_seconds=elapsed_gpu,
                    wall_seconds=elapsed_wall,
                    best_val_nll=best_val_nll,
                    best_val_step=best_val_step,
                    exposure_lineage=exposure_lineage,
                    hashes=hashes,
                    run_config=run_config,
                )

            if (
                completed_steps == 1
                or (heartbeat_interval > 0 and completed_steps % heartbeat_interval == 0)
                or is_eval_step
                or is_ckpt_step
            ):
                _write_heartbeat("running", tokens_per_sec=tokens_this_step / step_dt)

        # Ensure best_weights exists even if eval_interval > max_steps (e.g. tiny tests)
        if not (out_dir / "best_weights").exists():
            save_model_weights(model, arch, out_dir / "best_weights")

        final_status = (
            "interrupted_checkpointed"
            if (stop_flag["requested"] or (stop_at_step is not None and completed_steps < eff_max_steps))
            and completed_steps < eff_max_steps
            else "completed"
        )
        total_wall = prior_wall_seconds + (time.perf_counter() - session_start_wall)
        total_gpu = prior_gpu_seconds + (time.perf_counter() - session_start_wall)
        _write_heartbeat(
            final_status,
            tokens_per_sec=(
                total_tokens_processed / max(1e-9, total_wall)
                if total_tokens_processed > 0
                else 0.0
            ),
        )

        summary = {
            "status": final_status,
            "signal": stop_flag["signal"],
            "arch": arch,
            "phase": phase,
            "exposure": exposure,
            "exposed_variants": exposed_variants,
            "seed": int(seed),
            "step": int(completed_steps),
            "max_steps": eff_max_steps,
            "total_tokens_processed": int(total_tokens_processed),
            "total_episodes_processed": int(total_episodes_processed),
            "unique_roots_seen": len(unique_roots_seen),
            "unique_families_seen": len(unique_families_seen),
            "best_val_nll": None if math.isinf(best_val_nll) else float(best_val_nll),
            "best_val_step": best_val_step,
            "wall_seconds": float(total_wall),
            "gpu_process_seconds": float(total_gpu),
            "cpu_process_seconds": float(time.process_time() - session_start_proc),
            "model_load_seconds": float(model_load_seconds),
            "peak_vram_mb": _peak_vram_mb(eff_device),
            "hashes": hashes,
            "final_weights_sha256": compute_trainable_weights_sha256(model),
            "exposure_lineage": exposure_lineage,
            "run_config": run_config,
            "output_dir": str(out_dir),
            "finished_at": utc_now_iso(),
        }
        atomic_write_json(out_dir / "training_summary.json", summary)
        return summary
    finally:
        if is_main_thread:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for ``train_long_study.py``."""
    p = argparse.ArgumentParser(description="Unified long-study trainer for Chess-Variant ICL v2.")
    p.add_argument("--arch", choices=["qwen", "scratch"], required=True)
    p.add_argument("--phase", choices=["warmup", "main", "extend"], default="main")
    p.add_argument("--exposure", choices=["standard", "familiar", "all_rules"], default=None)
    p.add_argument("--variants", nargs="*", default=None, help="Optional explicit variant override")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--init-weights-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--ckpt-interval", type=int, default=250)
    p.add_argument("--milestones", type=int, nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p.add_argument("--val-k-values", type=int, nargs="+", default=None)
    p.add_argument("--val-roots-per-variant", type=int, default=32)
    p.add_argument("--val-seed", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--notation", choices=["san", "uci"], default="san")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--lora-mode", choices=["handoff", "full_lora"], default="handoff")
    p.add_argument("--heartbeat-interval", type=int, default=10)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--no-verify-manifest", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    exposure = args.exposure
    if exposure is None:
        exposure = "standard" if args.phase == "warmup" else "familiar"

    summary = run_training(
        arch=args.arch,
        phase=args.phase,
        exposure=exposure,
        seed=args.seed,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        init_weights_dir=args.init_weights_dir,
        resume=args.resume,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        ckpt_interval=args.ckpt_interval,
        milestones=args.milestones,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        k_values=args.k_values,
        val_k_values=args.val_k_values,
        val_roots_per_variant=args.val_roots_per_variant,
        val_seed=args.val_seed,
        max_seq_len=args.max_seq_len,
        notation=args.notation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gpu=args.gpu,
        model_path=args.model_path,
        lora_mode=args.lora_mode,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        verify_manifest=not args.no_verify_manifest,
        heartbeat_interval=args.heartbeat_interval,
        explicit_variants=args.variants,
    )
    print(json.dumps({"status": summary["status"], "step": summary["step"], "best_val_nll": summary["best_val_nll"]}))


if __name__ == "__main__":
    main()
