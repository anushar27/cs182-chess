"""Deterministic multi-GPU experiment orchestrator for chess_icl_v2.

Schedules Track 2 pilot, Track 2 muP scaling, Track 1 Qwen2.5-7B LoRA, and
baselines/figures waves across specified GPUs with zero intra-wave GPU collisions,
CUDA_VISIBLE_DEVICES pinning, offline HF flags for Track 1, wall-clock tracking,
and structured `experiment_manifest.json` output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_icl_v2.variants import (  # noqa: E402
    COMPOSITIONAL_VARIANTS,
    SINGLE_VARIANTS,
    TRAIN_VARIANTS,
)

VALID_PHASES: tuple[str, ...] = (
    "pilot",
    "track2_scaling",
    "track1_qwen",
    "baselines_and_figures",
)


@dataclass
class JobSpec:
    """Single executable experiment step inside the orchestration DAG."""

    job_id: str
    phase: str
    script: str
    cmd: list[str]
    depends_on: list[str] = field(default_factory=list)
    requires_gpu: bool = True
    env_overrides: dict[str, str] = field(default_factory=dict)
    assigned_gpu: Optional[int] = None
    wave_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _track2_eval_variants() -> list[str]:
    return list(SINGLE_VARIANTS) + list(COMPOSITIONAL_VARIANTS)


def build_pilot_jobs(
    *,
    data_dir: Path,
    runs_dir: Path,
    k_values: Sequence[int],
    n_draws: int,
    n_boot: int,
) -> list[JobSpec]:
    """Phase 1: Pilot gate on TINY & SMALL (standard warmup -> 2 move variants -> val eval)."""
    pilot_dir = runs_dir / "pilot"
    k_strs = [str(k) for k in k_values]
    pilot_vars = ["diagonal_rook", "straight_bishop"]
    jobs: list[JobSpec] = []

    for cfg in ("tiny", "small"):
        warm_dir = pilot_dir / f"{cfg}_warmup"
        cont_dir = pilot_dir / f"{cfg}_continue"
        eval_dir = pilot_dir / f"{cfg}_eval_val"

        warm_id = f"pilot_{cfg}_warmup"
        cont_id = f"pilot_{cfg}_continue"
        eval_id = f"pilot_{cfg}_eval"

        jobs.append(
            JobSpec(
                job_id=warm_id,
                phase="pilot",
                script="scripts_v2/train_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_track2.py",
                    "--mode",
                    "warmup",
                    "--config",
                    cfg,
                    "--use-mup",
                    "--attention-mode",
                    "causal",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(warm_dir),
                    "--seed",
                    "42",
                    "--lr",
                    "1e-3",
                    "--batch-size",
                    "16",
                    "--max-steps",
                    "400",
                    "--eval-interval",
                    "100",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[],
                requires_gpu=True,
                metadata={"config": cfg, "stage": "warmup"},
            )
        )
        jobs.append(
            JobSpec(
                job_id=cont_id,
                phase="pilot",
                script="scripts_v2/train_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_track2.py",
                    "--mode",
                    "continue",
                    "--config",
                    cfg,
                    "--use-mup",
                    "--attention-mode",
                    "causal",
                    "--variants",
                    *pilot_vars,
                    "--init-checkpoint",
                    str(warm_dir / "best.pt"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(cont_dir),
                    "--seed",
                    "42",
                    "--lr",
                    "5e-4",
                    "--batch-size",
                    "16",
                    "--max-steps",
                    "400",
                    "--eval-interval",
                    "100",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[warm_id],
                requires_gpu=True,
                metadata={"config": cfg, "stage": "continue"},
            )
        )
        jobs.append(
            JobSpec(
                job_id=eval_id,
                phase="pilot",
                script="scripts_v2/eval_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/eval_track2.py",
                    "--checkpoint",
                    str(cont_dir / "best.pt"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(eval_dir),
                    "--split",
                    "val",
                    "--variants",
                    *pilot_vars,
                    "--k-values",
                    *k_strs,
                    "--n-draws",
                    str(n_draws),
                    "--n-boot",
                    str(n_boot),
                    "--seed",
                    "42",
                    "--gpu",
                    "0",
                ],
                depends_on=[cont_id],
                requires_gpu=True,
                metadata={
                    "config": cfg,
                    "stage": "eval",
                    "gate_summary_path": str(eval_dir / "eval_summary.json"),
                },
            )
        )

    jobs.append(
        JobSpec(
            job_id="pilot_baselines_val",
            phase="pilot",
            script="scripts_v2/run_baselines.py",
            cmd=[
                sys.executable,
                "scripts_v2/run_baselines.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(pilot_dir / "baselines"),
                "--split",
                "val",
                "--variants",
                *pilot_vars,
                "--k-values",
                *k_strs,
                "--n-draws",
                str(n_draws),
                "--n-boot",
                str(n_boot),
                "--seed",
                "42",
            ],
            depends_on=[],
            requires_gpu=False,
            env_overrides={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            metadata={"baseline_summary_path": str(pilot_dir / "baselines" / "baselines_val.json")},
        )
    )
    return jobs


def build_track2_scaling_jobs(
    *,
    data_dir: Path,
    runs_dir: Path,
    split: str,
    k_values: Sequence[int],
    n_draws: int,
    n_boot: int,
) -> list[JobSpec]:
    """Phase 2: Track 2 muP scaling (TINY/SMALL/MEDIUM/LARGE), controls, & segment_bidir ablation."""
    t2_dir = runs_dir / "track2"
    k_strs = [str(k) for k in k_values]
    eval_vars = _track2_eval_variants()
    jobs: list[JobSpec] = []

    # 1. Common standard-chess warmups across all 4 scales
    warmup_ids: dict[str, str] = {}
    for cfg in ("tiny", "small", "medium", "large"):
        warm_id = f"t2_warmup_{cfg}"
        warmup_ids[cfg] = warm_id
        batch_sz = 16 if cfg in ("tiny", "small") else (8 if cfg == "medium" else 4)
        grad_acc = 1 if cfg in ("tiny", "small") else (2 if cfg == "medium" else 4)
        jobs.append(
            JobSpec(
                job_id=warm_id,
                phase="track2_scaling",
                script="scripts_v2/train_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_track2.py",
                    "--mode",
                    "warmup",
                    "--config",
                    cfg,
                    "--use-mup",
                    "--attention-mode",
                    "causal",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(t2_dir / f"warmup_{cfg}"),
                    "--seed",
                    "42",
                    "--lr",
                    "1e-3",
                    "--batch-size",
                    str(batch_sz),
                    "--grad-accum",
                    str(grad_acc),
                    "--max-steps",
                    "2000",
                    "--eval-interval",
                    "200",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[],
                requires_gpu=True,
                metadata={"config": cfg, "stage": "warmup"},
            )
        )

    # 2. Continuation training runs
    continue_specs: list[dict[str, Any]] = []
    for cfg in ("tiny", "small"):
        for seed in (42, 123, 456):
            continue_specs.append(
                {
                    "run_name": f"move_{cfg}_seed{seed}",
                    "cfg": cfg,
                    "seed": seed,
                    "variants": list(TRAIN_VARIANTS),
                    "attn": "causal",
                    "condition": "move_variants",
                }
            )
    for cfg in ("medium", "large"):
        continue_specs.append(
            {
                "run_name": f"move_{cfg}_seed42",
                "cfg": cfg,
                "seed": 42,
                "variants": list(TRAIN_VARIANTS),
                "attn": "causal",
                "condition": "move_variants",
            }
        )

    # Exposure upper bound, chess_only control, and segment_bidirectional ablation on SMALL
    continue_specs.extend(
        [
            {
                "run_name": "all_variants_small_seed42",
                "cfg": "small",
                "seed": 42,
                "variants": list(SINGLE_VARIANTS),
                "attn": "causal",
                "condition": "all_variants",
            },
            {
                "run_name": "chess_only_small_seed42",
                "cfg": "small",
                "seed": 42,
                "variants": ["standard"],
                "attn": "causal",
                "condition": "chess_only",
            },
            {
                "run_name": "seg_bidir_small_seed42",
                "cfg": "small",
                "seed": 42,
                "variants": list(TRAIN_VARIANTS),
                "attn": "segment_bidirectional",
                "condition": "segment_bidirectional",
            },
        ]
    )

    for spec in continue_specs:
        run_name = spec["run_name"]
        cfg = spec["cfg"]
        seed = spec["seed"]
        cont_dir = t2_dir / run_name
        eval_dir = t2_dir / f"eval_{run_name}_{split}"
        cont_id = f"t2_cont_{run_name}"
        eval_id = f"t2_eval_{run_name}"

        batch_sz = 16 if cfg in ("tiny", "small") else (8 if cfg == "medium" else 4)
        grad_acc = 1 if cfg in ("tiny", "small") else (2 if cfg == "medium" else 4)

        jobs.append(
            JobSpec(
                job_id=cont_id,
                phase="track2_scaling",
                script="scripts_v2/train_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_track2.py",
                    "--mode",
                    "continue",
                    "--config",
                    cfg,
                    "--use-mup",
                    "--attention-mode",
                    spec["attn"],
                    "--variants",
                    *spec["variants"],
                    "--init-checkpoint",
                    str(t2_dir / f"warmup_{cfg}" / "best.pt"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(cont_dir),
                    "--seed",
                    str(seed),
                    "--lr",
                    "5e-4",
                    "--batch-size",
                    str(batch_sz),
                    "--grad-accum",
                    str(grad_acc),
                    "--max-steps",
                    "2000",
                    "--eval-interval",
                    "200",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[warmup_ids[cfg]],
                requires_gpu=True,
                metadata={"config": cfg, "seed": seed, "condition": spec["condition"]},
            )
        )

        jobs.append(
            JobSpec(
                job_id=eval_id,
                phase="track2_scaling",
                script="scripts_v2/eval_track2.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/eval_track2.py",
                    "--checkpoint",
                    str(cont_dir / "best.pt"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(eval_dir),
                    "--split",
                    split,
                    "--variants",
                    *eval_vars,
                    "--k-values",
                    *k_strs,
                    "--n-draws",
                    str(n_draws),
                    "--controls",
                    "mismatched_context",
                    "corrupted_outcomes",
                    "corrupted_trajectories",
                    "--n-boot",
                    str(n_boot),
                    "--seed",
                    str(seed),
                    "--model-seed",
                    str(seed),
                    "--gpu",
                    "0",
                ],
                depends_on=[cont_id],
                requires_gpu=True,
                metadata={"config": cfg, "seed": seed, "condition": spec["condition"]},
            )
        )

    return jobs


def build_track1_qwen_jobs(
    *,
    data_dir: Path,
    runs_dir: Path,
    split: str,
    notation: str,
    k_values: Sequence[int],
    n_draws: int,
    n_boot: int,
) -> list[JobSpec]:
    """Phase 3: Track 1 Qwen2.5-7B-Instruct warmup, 6 continuation runs (GPUs 0..5), & eval."""
    t1_dir = runs_dir / "track1"
    k_strs = [str(k) for k in k_values]
    offline_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    jobs: list[JobSpec] = []

    # Warmups for handoff (Mode A) and full_lora (Mode B)
    for lora_mode in ("handoff", "full_lora"):
        warm_id = f"t1_qwen_warmup_{lora_mode}"
        warm_dir = t1_dir / f"warmup_{lora_mode}"
        jobs.append(
            JobSpec(
                job_id=warm_id,
                phase="track1_qwen",
                script="scripts_v2/train_qwen.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_qwen.py",
                    "--mode",
                    "warmup",
                    "--lora-mode",
                    lora_mode,
                    "--notation",
                    notation,
                    "--variants",
                    "standard",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(warm_dir),
                    "--seed",
                    "42",
                    "--lr",
                    "1e-4",
                    "--batch-size",
                    "1",
                    "--grad-accum",
                    "8",
                    "--max-steps",
                    "300",
                    "--eval-interval",
                    "50",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[],
                requires_gpu=True,
                env_overrides=dict(offline_env),
                metadata={"lora_mode": lora_mode, "stage": "warmup"},
            )
        )

    # Zero-shot base Qwen evaluation (independent of LoRA fine-tuning)
    jobs.append(
        JobSpec(
            job_id="t1_qwen_eval_zero_shot_base",
            phase="track1_qwen",
            script="scripts_v2/eval_qwen.py",
            cmd=[
                sys.executable,
                "scripts_v2/eval_qwen.py",
                "--zero-shot-base",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(t1_dir / f"eval_zero_shot_base_{split}"),
                "--split",
                split,
                "--variants",
                "all_variants",
                "compositional",
                "--k-values",
                *k_strs,
                "--num-draws",
                str(n_draws),
                "--controls",
                "mismatched_context",
                "corrupted_outcomes",
                "corrupted_trajectories",
                "--notation",
                notation,
                "--n-boot",
                str(n_boot),
                "--seed",
                "42",
                "--gpu",
                "0",
            ],
            depends_on=[],
            requires_gpu=True,
            env_overrides=dict(offline_env),
            metadata={"condition": "zero_shot_base"},
        )
    )

    # 6 continuation runs packed 1 run per GPU across GPUs 0-5
    cont_specs = [
        ("move_seed42", "handoff", "move_variants", 42),
        ("move_seed123", "handoff", "move_variants", 123),
        ("move_seed456", "handoff", "move_variants", 456),
        ("all_variants_seed42", "handoff", "all_variants", 42),
        ("chess_only_seed42", "handoff", "chess_only", 42),
        ("full_lora_seed42", "full_lora", "move_variants", 42),
    ]

    for run_name, lora_mode, variant_preset, seed in cont_specs:
        cont_id = f"t1_qwen_cont_{run_name}"
        eval_id = f"t1_qwen_eval_{run_name}"
        cont_dir = t1_dir / run_name
        eval_dir = t1_dir / f"eval_{run_name}_{split}"
        warm_dep = f"t1_qwen_warmup_{lora_mode}"

        jobs.append(
            JobSpec(
                job_id=cont_id,
                phase="track1_qwen",
                script="scripts_v2/train_qwen.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/train_qwen.py",
                    "--mode",
                    "continue",
                    "--lora-mode",
                    lora_mode,
                    "--notation",
                    notation,
                    "--variants",
                    variant_preset,
                    "--init-adapter-dir",
                    str(t1_dir / f"warmup_{lora_mode}" / "best_adapter"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(cont_dir),
                    "--seed",
                    str(seed),
                    "--lr",
                    "1e-4",
                    "--batch-size",
                    "1",
                    "--grad-accum",
                    "8",
                    "--max-steps",
                    "500",
                    "--eval-interval",
                    "50",
                    "--k-values",
                    *k_strs,
                    "--val-k-values",
                    "0",
                    "4",
                    "--gpu",
                    "0",
                ],
                depends_on=[warm_dep],
                requires_gpu=True,
                env_overrides=dict(offline_env),
                metadata={"lora_mode": lora_mode, "condition": variant_preset, "seed": seed},
            )
        )

        jobs.append(
            JobSpec(
                job_id=eval_id,
                phase="track1_qwen",
                script="scripts_v2/eval_qwen.py",
                cmd=[
                    sys.executable,
                    "scripts_v2/eval_qwen.py",
                    "--adapter-dir",
                    str(cont_dir / "best_adapter"),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(eval_dir),
                    "--split",
                    split,
                    "--variants",
                    "all_variants",
                    "compositional",
                    "--k-values",
                    *k_strs,
                    "--num-draws",
                    str(n_draws),
                    "--controls",
                    "mismatched_context",
                    "corrupted_outcomes",
                    "corrupted_trajectories",
                    "--notation",
                    notation,
                    "--n-boot",
                    str(n_boot),
                    "--seed",
                    str(seed),
                    "--model-seed",
                    str(seed),
                    "--gpu",
                    "0",
                ],
                depends_on=[cont_id],
                requires_gpu=True,
                env_overrides=dict(offline_env),
                metadata={"lora_mode": lora_mode, "condition": variant_preset, "seed": seed},
            )
        )

    return jobs


def build_baselines_and_figures_jobs(
    *,
    data_dir: Path,
    runs_dir: Path,
    split: str,
    k_values: Sequence[int],
    n_draws: int,
    n_boot: int,
    upstream_eval_ids: Sequence[str],
) -> list[JobSpec]:
    """Phase 4: Run val & test non-neural baselines + generate paper figures/tables."""
    k_strs = [str(k) for k in k_values]
    cpu_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    eval_vars = _track2_eval_variants()

    val_job = JobSpec(
        job_id="baselines_val",
        phase="baselines_and_figures",
        script="scripts_v2/run_baselines.py",
        cmd=[
            sys.executable,
            "scripts_v2/run_baselines.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir),
            "--split",
            "val",
            "--variants",
            *eval_vars,
            "--k-values",
            *k_strs,
            "--n-draws",
            str(n_draws),
            "--n-boot",
            str(n_boot),
            "--seed",
            "42",
        ],
        depends_on=[],
        requires_gpu=False,
        env_overrides=dict(cpu_env),
    )

    test_job = JobSpec(
        job_id="baselines_test",
        phase="baselines_and_figures",
        script="scripts_v2/run_baselines.py",
        cmd=[
            sys.executable,
            "scripts_v2/run_baselines.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir),
            "--split",
            "test",
            "--variants",
            *eval_vars,
            "--k-values",
            *k_strs,
            "--n-draws",
            str(n_draws),
            "--n-boot",
            str(n_boot),
            "--seed",
            "42",
        ],
        depends_on=[],
        requires_gpu=False,
        env_overrides=dict(cpu_env),
    )

    figs_job = JobSpec(
        job_id="make_paper_figures",
        phase="baselines_and_figures",
        script="scripts_v2/make_paper_figures.py",
        cmd=[
            sys.executable,
            "scripts_v2/make_paper_figures.py",
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(runs_dir / "paper_figures"),
            "--split",
            split,
        ],
        depends_on=["baselines_val", "baselines_test", *upstream_eval_ids],
        requires_gpu=False,
        env_overrides=dict(cpu_env),
    )
    return [val_job, test_job, figs_job]


def schedule_waves(jobs: Sequence[JobSpec], gpus: Sequence[int]) -> list[list[JobSpec]]:
    """Topologically partition jobs into execution waves with zero GPU collisions per wave."""
    if not gpus:
        raise ValueError("gpus list cannot be empty when scheduling GPU jobs")

    job_map = {j.job_id: j for j in jobs}
    completed: set[str] = set()
    remaining = [j.job_id for j in jobs]
    waves: list[list[JobSpec]] = []
    wave_idx = 0

    while remaining:
        ready = [
            job_map[jid]
            for jid in remaining
            if all((dep in completed) or (dep not in job_map) for dep in job_map[jid].depends_on)
        ]
        if not ready:
            raise RuntimeError(f"Cyclic or unsatisfied dependencies among jobs: {remaining}")

        gpu_ready = [j for j in ready if j.requires_gpu]
        cpu_ready = [j for j in ready if not j.requires_gpu]

        wave_jobs: list[JobSpec] = []
        for slot, job in enumerate(gpu_ready[: len(gpus)]):
            gpu_id = int(gpus[slot])
            job.assigned_gpu = gpu_id
            job.env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            job.wave_index = wave_idx
            wave_jobs.append(job)

        # Schedule CPU jobs concurrently alongside the GPU wave
        for job in cpu_ready:
            job.assigned_gpu = None
            job.wave_index = wave_idx
            wave_jobs.append(job)

        waves.append(wave_jobs)
        scheduled_ids = {j.job_id for j in wave_jobs}
        completed.update(scheduled_ids)
        remaining = [jid for jid in remaining if jid not in scheduled_ids]
        wave_idx += 1

    return waves


def evaluate_pilot_gate(runs_dir: Path) -> dict[str, Any]:
    """Verify SMALL pilot Context Gain exceeds both Board kNN and mismatched_context."""
    pilot_dir = runs_dir / "pilot"
    eval_path = pilot_dir / "small_eval_val" / "eval_summary.json"
    base_path = pilot_dir / "baselines" / "baselines_val.json"
    if not eval_path.exists() or not base_path.exists():
        return {"status": "skipped", "reason": "missing pilot eval or baseline JSON"}

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    base_data = json.loads(base_path.read_text(encoding="utf-8"))

    k_max = str(max(int(k) for k in eval_data["k_values"]))
    small_gain = float(eval_data["main_by_k"][k_max]["context_gain"]["macro_mean"])
    knn_gain = float(base_data["models"]["board_knn"]["by_k"][k_max]["macro"]["context_gain"])

    mismatch_by_k = eval_data.get("controls_by_k", {}).get("mismatched_context", {})
    mismatch_gain = (
        float(mismatch_by_k[k_max]["context_gain"]["macro_mean"])
        if k_max in mismatch_by_k
        else None
    )

    beats_knn = small_gain > knn_gain
    beats_mismatch = True if mismatch_gain is None else (small_gain > mismatch_gain)
    passed = beats_knn and beats_mismatch

    return {
        "status": "passed" if passed else "failed",
        "k_max": int(k_max),
        "small_context_gain": small_gain,
        "board_knn_context_gain": knn_gain,
        "mismatched_context_gain": mismatch_gain,
        "beats_knn": beats_knn,
        "beats_mismatched": beats_mismatch,
    }


def build_experiment_plan(
    *,
    phases: Sequence[str],
    gpus: Sequence[int],
    data_dir: Path,
    runs_dir: Path,
    split: str,
    notation: str,
    k_values: Sequence[int],
    n_draws: int,
    n_boot: int,
) -> tuple[list[JobSpec], list[list[JobSpec]]]:
    """Assemble requested phases into a validated, wave-scheduled DAG."""
    for p in phases:
        if p not in VALID_PHASES:
            raise ValueError(f"Unknown phase {p!r}; expected subset of {VALID_PHASES}")

    all_jobs: list[JobSpec] = []
    upstream_eval_ids: list[str] = []

    if "pilot" in phases:
        p_jobs = build_pilot_jobs(
            data_dir=data_dir,
            runs_dir=runs_dir,
            k_values=k_values,
            n_draws=n_draws,
            n_boot=n_boot,
        )
        all_jobs.extend(p_jobs)

    if "track2_scaling" in phases:
        t2_jobs = build_track2_scaling_jobs(
            data_dir=data_dir,
            runs_dir=runs_dir,
            split=split,
            k_values=k_values,
            n_draws=n_draws,
            n_boot=n_boot,
        )
        all_jobs.extend(t2_jobs)
        upstream_eval_ids.extend(j.job_id for j in t2_jobs if "_eval_" in j.job_id)

    if "track1_qwen" in phases:
        t1_jobs = build_track1_qwen_jobs(
            data_dir=data_dir,
            runs_dir=runs_dir,
            split=split,
            notation=notation,
            k_values=k_values,
            n_draws=n_draws,
            n_boot=n_boot,
        )
        all_jobs.extend(t1_jobs)
        upstream_eval_ids.extend(j.job_id for j in t1_jobs if "_eval_" in j.job_id)

    if "baselines_and_figures" in phases:
        bf_jobs = build_baselines_and_figures_jobs(
            data_dir=data_dir,
            runs_dir=runs_dir,
            split=split,
            k_values=k_values,
            n_draws=n_draws,
            n_boot=n_boot,
            upstream_eval_ids=upstream_eval_ids,
        )
        all_jobs.extend(bf_jobs)

    waves = schedule_waves(all_jobs, gpus=gpus)
    return all_jobs, waves


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)

    all_jobs, waves = build_experiment_plan(
        phases=args.phases,
        gpus=args.gpus,
        data_dir=data_dir,
        runs_dir=runs_dir,
        split=args.split,
        notation=args.notation,
        k_values=args.k_values,
        n_draws=args.n_draws,
        n_boot=args.n_boot,
    )

    manifest: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "phases": list(args.phases),
        "gpus": [int(g) for g in args.gpus],
        "policy_id": args.policy_id,
        "notation": args.notation,
        "data_dir": str(data_dir),
        "runs_dir": str(runs_dir),
        "split": args.split,
        "total_jobs": len(all_jobs),
        "total_waves": len(waves),
        "waves": [
            {
                "wave_index": w_idx,
                "jobs": [asdict(j) for j in wave],
            }
            for w_idx, wave in enumerate(waves)
        ],
        "execution_results": [],
        "pilot_gate": None,
        "total_wall_sec": 0.0,
    }

    if args.dry_run:
        print(
            f"[orchestrate] DRY RUN: {len(all_jobs)} jobs across {len(waves)} waves "
            f"(phases={args.phases}, gpus={args.gpus})"
        )
        for w_idx, wave in enumerate(waves):
            print(f"--- Wave {w_idx} ({len(wave)} jobs) ---")
            for j in wave:
                gpu_tag = f"GPU={j.assigned_gpu}" if j.assigned_gpu is not None else "CPU"
                print(f"  [{j.phase}] {j.job_id} ({gpu_tag}): {' '.join(j.cmd)}")
        return manifest

    runs_dir.mkdir(parents=True, exist_ok=True)
    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.perf_counter()
    results: list[dict[str, Any]] = []
    any_failed = False

    for w_idx, wave in enumerate(waves):
        procs: list[tuple[JobSpec, subprocess.Popen, Any, float, Path]] = []
        for job in wave:
            env = os.environ.copy()
            env.update(job.env_overrides)
            log_path = log_dir / f"{job.job_id}.log"
            fh = log_path.open("w", encoding="utf-8")
            t_job = time.perf_counter()
            proc = subprocess.Popen(
                job.cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            procs.append((job, proc, fh, t_job, log_path))

        wave_failed = False
        for job, proc, fh, t_job, log_path in procs:
            rc = proc.wait()
            elapsed = round(time.perf_counter() - t_job, 3)
            fh.close()
            results.append(
                {
                    "job_id": job.job_id,
                    "phase": job.phase,
                    "wave_index": w_idx,
                    "assigned_gpu": job.assigned_gpu,
                    "exit_code": int(rc),
                    "wall_sec": elapsed,
                    "log_path": str(log_path),
                }
            )
            if rc != 0:
                wave_failed = True
                any_failed = True

        if wave_failed and args.fail_fast:
            break

    if "pilot" in args.phases and not any_failed:
        manifest["pilot_gate"] = evaluate_pilot_gate(runs_dir)

    manifest["execution_results"] = results
    manifest["total_wall_sec"] = round(time.perf_counter() - t_global, 3)
    manifest["status"] = "failed" if any_failed else "completed"

    manifest_path = runs_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU experiment orchestrator for chess_icl_v2")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=list(VALID_PHASES),
        default=list(VALID_PHASES),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--policy-id", type=str, default="terminal_aware_v1")
    parser.add_argument("--notation", choices=["uci", "san"], default="san")
    parser.add_argument("--data-dir", type=Path, default=Path("data_v2"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_v2"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_orchestrator(args)


if __name__ == "__main__":
    main()
