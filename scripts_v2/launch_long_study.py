#!/usr/bin/env python3
"""Automated multi-GPU job queue & supervisor for the Chess-Variant ICL v2 long study.

Manages the 14-job DAG (6 warmup parents + 8 main slots):
  Warmup parents (1,000 updates each, exposure='standard'):
    - warmup_qwen_s{42,123,456}
    - warmup_scratch_s{42,123,456}

  Main jobs (10,000 updates each, cloned from matching warmup parent):
    - Slots 0-2: qwen familiar, seeds 42, 123, 456 -> warmup_qwen_s{42,123,456}
    - Slots 3-5: scratch familiar, seeds 42, 123, 456 -> warmup_scratch_s{42,123,456}
    - Slot 6: qwen all_rules, seed 42 -> warmup_qwen_s42
    - Slot 7: scratch all_rules, seed 42 -> warmup_scratch_s42

Features:
  - Dynamic free-GPU discovery via ``nvidia-smi`` without disturbing unrelated processes
  - Immutable per-job ``run_manifest.json``
  - ``CUDA_VISIBLE_DEVICES=<gpu>``, ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1``
  - Heartbeat & exit-code monitoring with automatic ``--resume`` retry on recoverable failures
  - Atomic ``launcher_state.json`` tracking + ``--dry-run`` + ``--max-concurrent``
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_v2.train_long_study import (  # noqa: E402
    DEFAULT_MAIN_MILESTONES,
    DEFAULT_WARMUP_MILESTONES,
    _is_valid_checkpoint_dir,
    atomic_write_json,
    utc_now_iso,
)

SEEDS: tuple[int, ...] = (42, 123, 456)


@dataclass(frozen=True)
class GPUStatus:
    """Snapshot of a single GPU's memory and utilization from ``nvidia-smi``."""

    index: int
    memory_used_mb: int
    memory_free_mb: int
    memory_total_mb: int
    utilization_pct: int


@dataclass
class LongStudyJob:
    """One node in the long-study training DAG."""

    job_id: str
    slot: Optional[int]
    arch: str
    phase: str
    exposure: str
    seed: int
    snapshot_dir: Path
    output_dir: Path
    max_steps: int
    eval_interval: int
    ckpt_interval: int
    milestones: list[int]
    grad_accum: int = 8
    batch_size: int = 1
    k_values: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    max_seq_len: int = 32768
    depends_on: list[str] = field(default_factory=list)
    init_weights_dir: Optional[Path] = None
    model_path: Optional[str] = None
    extra_args: list[str] = field(default_factory=list)

    def build_cmd(self, *, resume: bool = False) -> list[str]:
        """Build the ``train_long_study.py`` command for this job."""
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts_v2" / "train_long_study.py"),
            "--arch",
            self.arch,
            "--phase",
            self.phase,
            "--exposure",
            self.exposure,
            "--seed",
            str(self.seed),
            "--snapshot-dir",
            str(self.snapshot_dir),
            "--output-dir",
            str(self.output_dir),
            "--max-steps",
            str(self.max_steps),
            "--eval-interval",
            str(self.eval_interval),
            "--ckpt-interval",
            str(self.ckpt_interval),
            "--batch-size",
            str(self.batch_size),
            "--grad-accum",
            str(self.grad_accum),
            "--max-seq-len",
            str(self.max_seq_len),
            "--gpu",
            "0",
            "--k-values",
            *(str(k) for k in self.k_values),
        ]
        if self.milestones:
            cmd.extend(["--milestones", *(str(m) for m in self.milestones)])
        if self.init_weights_dir is not None and not resume:
            cmd.extend(["--init-weights-dir", str(self.init_weights_dir)])
        if self.model_path:
            cmd.extend(["--model-path", str(self.model_path)])
        if resume:
            cmd.append("--resume")
        if self.extra_args:
            cmd.extend(self.extra_args)
        return cmd

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "slot": self.slot,
            "arch": self.arch,
            "phase": self.phase,
            "exposure": self.exposure,
            "seed": self.seed,
            "snapshot_dir": str(self.snapshot_dir),
            "output_dir": str(self.output_dir),
            "init_weights_dir": str(self.init_weights_dir) if self.init_weights_dir else None,
            "depends_on": list(self.depends_on),
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "ckpt_interval": self.ckpt_interval,
            "milestones": list(self.milestones),
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "k_values": list(self.k_values),
            "max_seq_len": self.max_seq_len,
            "model_path": self.model_path,
            "cmd_initial": self.build_cmd(resume=False),
            "env_policy": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        }


def build_long_study_jobs(
    *,
    snapshot_dir: Path,
    runs_dir: Path,
    warmup_steps: int = 1000,
    main_steps: int = 10000,
    warmup_eval_interval: int = 100,
    main_eval_interval: int = 500,
    ckpt_interval: int = 250,
    grad_accum: int = 8,
    max_seq_len: int = 32768,
    model_path: Optional[str] = None,
    phases: str = "all",
    only_jobs: Optional[Sequence[str]] = None,
) -> list[LongStudyJob]:
    """Build the 6 warmup parents + 8 main logical slots DAG."""
    snap = Path(snapshot_dir)
    root = Path(runs_dir)

    warmup_milestones = sorted(
        {m for m in DEFAULT_WARMUP_MILESTONES if m <= warmup_steps} | {0, warmup_steps}
    )
    main_milestones = sorted(
        {m for m in DEFAULT_MAIN_MILESTONES if m <= main_steps} | {0, main_steps}
    )

    jobs: list[LongStudyJob] = []
    warmup_out_by_key: dict[tuple[str, int], Path] = {}
    warmup_id_by_key: dict[tuple[str, int], str] = {}

    # 6 warmup parents ordered by seed (42 first, then 123, 456) across both families
    # so prioritized slots 0/3 (seed 42) and 1/4 (seed 123) unlock earliest when GPUs are scarce.
    for seed in SEEDS:
        for arch in ("qwen", "scratch"):
            job_id = f"warmup_{arch}_s{seed}"
            out_dir = root / "warmup" / f"{arch}_seed{seed}"
            warmup_out_by_key[(arch, seed)] = out_dir
            warmup_id_by_key[(arch, seed)] = job_id
            if phases in ("all", "warmup"):
                jobs.append(
                    LongStudyJob(
                        job_id=job_id,
                        slot=None,
                        arch=arch,
                        phase="warmup",
                        exposure="standard",
                        seed=seed,
                        snapshot_dir=snap,
                        output_dir=out_dir,
                        max_steps=warmup_steps,
                        eval_interval=warmup_eval_interval,
                        ckpt_interval=ckpt_interval,
                        milestones=warmup_milestones,
                        grad_accum=grad_accum,
                        max_seq_len=max_seq_len,
                        depends_on=[],
                        init_weights_dir=None,
                        model_path=model_path if arch == "qwen" else None,
                    )
                )

    # 8 main logical slots prioritized per plan (Slots 0, 1, 3, 4 first, then 2, 5, 6, 7)
    main_specs: list[tuple[int, str, str, int]] = [
        (0, "qwen", "familiar", 42),
        (3, "scratch", "familiar", 42),
        (1, "qwen", "familiar", 123),
        (4, "scratch", "familiar", 123),
        (2, "qwen", "familiar", 456),
        (5, "scratch", "familiar", 456),
        (6, "qwen", "all_rules", 42),
        (7, "scratch", "all_rules", 42),
    ]
    if phases in ("all", "main"):
        for slot, arch, exposure, seed in main_specs:
            # Both familiar (matching seed) and all_rules (seed 42) depend on (arch, seed) warmup parent
            parent_id = warmup_id_by_key[(arch, seed)]
            parent_out = warmup_out_by_key[(arch, seed)]
            job_id = f"main_slot{slot}_{arch}_{exposure}_s{seed}"
            out_dir = root / "main" / f"slot{slot}_{arch}_{exposure}_seed{seed}"
            deps = [parent_id] if phases == "all" else []
            jobs.append(
                LongStudyJob(
                    job_id=job_id,
                    slot=slot,
                    arch=arch,
                    phase="main",
                    exposure=exposure,
                    seed=seed,
                    snapshot_dir=snap,
                    output_dir=out_dir,
                    max_steps=main_steps,
                    eval_interval=main_eval_interval,
                    ckpt_interval=ckpt_interval,
                    milestones=main_milestones,
                    grad_accum=grad_accum,
                    max_seq_len=max_seq_len,
                    depends_on=deps,
                    init_weights_dir=parent_out / "best_weights",
                    model_path=model_path if arch == "qwen" else None,
                )
            )

    if only_jobs:
        wanted = set(only_jobs)
        jobs = [j for j in jobs if j.job_id in wanted]
        kept_ids = {j.job_id for j in jobs}
        for j in jobs:
            j.depends_on = [d for d in j.depends_on if d in kept_ids]

    return jobs


def parse_nvidia_smi_output(raw_csv: str) -> list[GPUStatus]:
    """Parse ``nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits``."""
    statuses: list[GPUStatus] = []
    for line in raw_csv.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 5:
            continue
        try:
            statuses.append(
                GPUStatus(
                    index=int(parts[0]),
                    memory_used_mb=int(float(parts[1])),
                    memory_free_mb=int(float(parts[2])),
                    memory_total_mb=int(float(parts[3])),
                    utilization_pct=int(float(parts[4])),
                )
            )
        except ValueError:
            continue
    return statuses


def discover_gpus() -> list[GPUStatus]:
    """Query ``nvidia-smi`` for live GPU status without altering any running processes."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return parse_nvidia_smi_output(proc.stdout)


def select_free_gpus(
    gpus: Sequence[GPUStatus],
    *,
    busy_gpus: set[int] = frozenset(),  # type: ignore[assignment]
    max_used_mb: int = 5000,
    min_free_mb: int = 50000,
    max_util_pct: int = 10,
    allowed_gpus: Optional[Sequence[int]] = None,
) -> list[int]:
    """Filter GPUs that are genuinely idle and available for scheduling."""
    allowed_set = set(int(g) for g in allowed_gpus) if allowed_gpus is not None else None
    free: list[int] = []
    for g in gpus:
        if g.index in busy_gpus:
            continue
        if allowed_set is not None and g.index not in allowed_set:
            continue
        if g.memory_used_mb >= max_used_mb:
            continue
        if g.memory_free_mb <= min_free_mb:
            continue
        if g.utilization_pct >= max_util_pct:
            continue
        free.append(g.index)
    return sorted(free)


def simulate_dry_run_plan(
    jobs: Sequence[LongStudyJob],
    *,
    free_gpu_pool: Sequence[int],
    max_concurrent: Optional[int] = None,
) -> dict[str, Any]:
    """Compute topological wave packing for ``--dry-run`` inspection."""
    gpu_list = list(free_gpu_pool)
    cap = len(gpu_list)
    if max_concurrent is not None:
        cap = min(cap, max(1, int(max_concurrent)))

    remaining = {j.job_id: j for j in jobs}
    completed: set[str] = set()
    waves: list[dict[str, Any]] = []
    wave_idx = 0

    while remaining:
        ready = [
            j
            for j in remaining.values()
            if all(dep in completed for dep in j.depends_on)
        ]
        if not ready:
            raise ValueError(f"Dependency cycle or missing parent in jobs: {list(remaining.keys())}")

        batch = ready[:cap] if cap > 0 else ready
        assignments: list[dict[str, Any]] = []
        for idx, job in enumerate(batch):
            assigned_gpu = gpu_list[idx] if idx < len(gpu_list) else None
            assignments.append(
                {
                    "job_id": job.job_id,
                    "slot": job.slot,
                    "arch": job.arch,
                    "phase": job.phase,
                    "exposure": job.exposure,
                    "seed": job.seed,
                    "depends_on": list(job.depends_on),
                    "assigned_gpu": assigned_gpu,
                    "init_weights_dir": str(job.init_weights_dir) if job.init_weights_dir else None,
                    "output_dir": str(job.output_dir),
                }
            )
            completed.add(job.job_id)
            remaining.pop(job.job_id)
        waves.append({"wave": wave_idx, "jobs": assignments})
        wave_idx += 1

    return {
        "total_jobs": len(jobs),
        "free_gpu_pool": gpu_list,
        "max_concurrent": max_concurrent,
        "waves": waves,
    }


def _is_job_already_completed(job: LongStudyJob) -> bool:
    summary_path = job.output_dir / "training_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return (
            summary.get("status") == "completed"
            and int(summary.get("step", 0)) >= job.max_steps
        )
    except Exception:
        return False


def _has_resumable_checkpoint(job: LongStudyJob) -> bool:
    ckpt_root = job.output_dir / "checkpoints"
    return any(
        _is_valid_checkpoint_dir(ckpt_root / name)
        for name in ("ckpt_latest", "ckpt_prev")
    )


def _write_immutable_run_manifest(job: LongStudyJob) -> Path:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job.output_dir / "run_manifest.json"
    if not manifest_path.exists():
        payload = {
            **job.to_manifest_dict(),
            "created_at": utc_now_iso(),
        }
        atomic_write_json(manifest_path, payload)
    return manifest_path


def _read_heartbeat(job: LongStudyJob) -> Optional[dict[str, Any]]:
    hb_path = job.output_dir / "heartbeat.json"
    if not hb_path.exists():
        return None
    try:
        return json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _heartbeat_age_seconds(hb: Mapping[str, Any]) -> Optional[float]:
    ts = hb.get("timestamp")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def run_launcher(
    *,
    snapshot_dir: Path,
    runs_dir: Path,
    allowed_gpus: Optional[Sequence[int]] = None,
    max_concurrent: Optional[int] = None,
    max_retries: int = 3,
    poll_interval: float = 15.0,
    max_used_mb: int = 5000,
    min_free_mb: int = 50000,
    max_util_pct: int = 10,
    stale_heartbeat_seconds: float = 0.0,
    warmup_steps: int = 1000,
    main_steps: int = 10000,
    warmup_eval_interval: int = 100,
    main_eval_interval: int = 500,
    ckpt_interval: int = 250,
    grad_accum: int = 8,
    max_seq_len: int = 32768,
    model_path: Optional[str] = None,
    phases: str = "all",
    only_jobs: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Schedule, launch, monitor, and retry the long-study training DAG."""
    jobs = build_long_study_jobs(
        snapshot_dir=snapshot_dir,
        runs_dir=runs_dir,
        warmup_steps=warmup_steps,
        main_steps=main_steps,
        warmup_eval_interval=warmup_eval_interval,
        main_eval_interval=main_eval_interval,
        ckpt_interval=ckpt_interval,
        grad_accum=grad_accum,
        max_seq_len=max_seq_len,
        model_path=model_path,
        phases=phases,
        only_jobs=only_jobs,
    )
    job_by_id = {j.job_id: j for j in jobs}

    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    state_path = Path(runs_dir) / "launcher_state.json"

    discovered = discover_gpus()
    free_gpus_now = select_free_gpus(
        discovered,
        max_used_mb=max_used_mb,
        min_free_mb=min_free_mb,
        max_util_pct=max_util_pct,
        allowed_gpus=allowed_gpus,
    )

    if dry_run:
        if free_gpus_now:
            sim_pool = free_gpus_now
        elif allowed_gpus is not None:
            sim_pool = [int(g) for g in allowed_gpus]
        elif discovered:
            sim_pool = [g.index for g in discovered]
        else:
            sim_pool = list(range(8))

        plan = simulate_dry_run_plan(
            jobs,
            free_gpu_pool=sim_pool,
            max_concurrent=max_concurrent,
        )
        dry_state = {
            "mode": "dry_run",
            "timestamp": utc_now_iso(),
            "discovered_gpus": [asdict(g) for g in discovered],
            "free_gpus_now": free_gpus_now,
            "plan": plan,
            "jobs": {j.job_id: j.to_manifest_dict() for j in jobs},
        }
        atomic_write_json(state_path, dry_state)
        return dry_state

    # Initialize live job tracking state
    job_records: dict[str, dict[str, Any]] = {}
    for j in jobs:
        done = _is_job_already_completed(j)
        job_records[j.job_id] = {
            "job_id": j.job_id,
            "slot": j.slot,
            "arch": j.arch,
            "phase": j.phase,
            "exposure": j.exposure,
            "seed": j.seed,
            "depends_on": list(j.depends_on),
            "status": "completed" if done else "queued",
            "assigned_gpu": None,
            "pid": None,
            "retries": 0,
            "last_exit_code": 0 if done else None,
            "started_at": None,
            "finished_at": utc_now_iso() if done else None,
            "heartbeat": _read_heartbeat(j),
            "output_dir": str(j.output_dir),
        }

    active_procs: dict[str, tuple[subprocess.Popen[Any], Any, int]] = {}
    stop_requested = {"flag": False}

    def _on_signal(_sig: int, _frame: Any) -> None:
        stop_requested["flag"] = True

    prev_term = signal.signal(signal.SIGTERM, _on_signal)
    prev_int = signal.signal(signal.SIGINT, _on_signal)

    def _save_state() -> dict[str, Any]:
        payload = {
            "mode": "live",
            "updated_at": utc_now_iso(),
            "snapshot_dir": str(snapshot_dir),
            "runs_dir": str(runs_dir),
            "max_retries": int(max_retries),
            "max_concurrent": max_concurrent,
            "allowed_gpus": list(allowed_gpus) if allowed_gpus is not None else None,
            "jobs": job_records,
        }
        atomic_write_json(state_path, payload)
        return payload

    _save_state()

    try:
        while True:
            # 1. Poll active workers
            finished_ids: list[str] = []
            for jid, (proc, log_fh, gpu_idx) in active_procs.items():
                job = job_by_id[jid]
                rec = job_records[jid]
                hb = _read_heartbeat(job)
                if hb is not None:
                    rec["heartbeat"] = hb

                rc = proc.poll()
                if rc is None:
                    if stale_heartbeat_seconds > 0 and hb is not None:
                        age = _heartbeat_age_seconds(hb)
                        if age is not None and age > stale_heartbeat_seconds:
                            proc.terminate()
                    continue

                log_fh.close()
                rec["last_exit_code"] = int(rc)
                rec["pid"] = None
                rec["assigned_gpu"] = None
                finished_ids.append(jid)

                if rc == 0 and _is_job_already_completed(job):
                    rec["status"] = "completed"
                    rec["finished_at"] = utc_now_iso()
                elif not stop_requested["flag"] and rec["retries"] < max_retries:
                    rec["retries"] += 1
                    rec["status"] = "queued"
                else:
                    rec["status"] = "failed"
                    rec["finished_at"] = utc_now_iso()

            for jid in finished_ids:
                active_procs.pop(jid, None)

            if stop_requested["flag"]:
                for jid, (proc, _fh, _g) in active_procs.items():
                    if proc.poll() is None:
                        proc.terminate()
                deadline = time.time() + 30.0
                for jid, (proc, log_fh, _g) in list(active_procs.items()):
                    rem = max(0.1, deadline - time.time())
                    try:
                        proc.wait(timeout=rem)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    log_fh.close()
                    job_records[jid]["status"] = "interrupted"
                    job_records[jid]["pid"] = None
                    job_records[jid]["assigned_gpu"] = None
                active_procs.clear()
                return _save_state()

            # 2. Check termination condition
            pending_or_running = [
                jid
                for jid, r in job_records.items()
                if r["status"] in ("queued", "running")
            ]
            if not pending_or_running:
                return _save_state()

            # 3. Schedule ready jobs onto idle GPUs
            busy_gpus = {gpu_idx for _, _, gpu_idx in active_procs.values()}
            live_gpus = discover_gpus()
            free_gpus = select_free_gpus(
                live_gpus,
                busy_gpus=busy_gpus,
                max_used_mb=max_used_mb,
                min_free_mb=min_free_mb,
                max_util_pct=max_util_pct,
                allowed_gpus=allowed_gpus,
            )

            if max_concurrent is not None:
                slots_avail = max(0, int(max_concurrent) - len(active_procs))
                free_gpus = free_gpus[:slots_avail]

            if free_gpus:
                ready_jobs = [
                    job_by_id[jid]
                    for jid, rec in job_records.items()
                    if rec["status"] == "queued"
                    and all(job_records[dep]["status"] == "completed" for dep in job_by_id[jid].depends_on)
                ]
                # Fail any job whose dependency failed
                for jid, rec in job_records.items():
                    if rec["status"] == "queued":
                        if any(job_records[dep]["status"] == "failed" for dep in job_by_id[jid].depends_on):
                            rec["status"] = "failed_dependency"
                            rec["finished_at"] = utc_now_iso()

                for job, gpu_idx in zip(ready_jobs, free_gpus):
                    # Resolve fallback init_weights_dir if best_weights missing after warmup completion
                    if job.init_weights_dir is not None and not job.init_weights_dir.exists():
                        parent_dir = job.init_weights_dir.parent
                        cand = parent_dir / "milestones" / f"step_{warmup_steps:05d}"
                        if cand.exists():
                            job.init_weights_dir = cand

                    use_resume = _has_resumable_checkpoint(job)
                    _write_immutable_run_manifest(job)
                    cmd = job.build_cmd(resume=use_resume)

                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    env["HF_HUB_OFFLINE"] = "1"
                    env["TRANSFORMERS_OFFLINE"] = "1"
                    env.setdefault("OMP_NUM_THREADS", "1")
                    env.setdefault("MKL_NUM_THREADS", "1")

                    log_path = job.output_dir / "worker.log"
                    log_fh = log_path.open("a", encoding="utf-8")
                    log_fh.write(
                        f"\n[{utc_now_iso()}] LAUNCH gpu={gpu_idx} resume={use_resume} cmd={' '.join(cmd)}\n"
                    )
                    log_fh.flush()

                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                    )
                    active_procs[job.job_id] = (proc, log_fh, gpu_idx)
                    rec = job_records[job.job_id]
                    rec["status"] = "running"
                    rec["assigned_gpu"] = int(gpu_idx)
                    rec["pid"] = int(proc.pid)
                    rec["started_at"] = rec["started_at"] or utc_now_iso()

            _save_state()
            time.sleep(max(0.5, float(poll_interval)))
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-GPU launcher & supervisor for long study.")
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=Path("runs_v2/long_study"))
    p.add_argument("--allowed-gpus", type=int, nargs="*", default=None)
    p.add_argument("--max-concurrent", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--poll-interval", type=float, default=15.0)
    p.add_argument("--max-used-mb", type=int, default=5000)
    p.add_argument("--min-free-mb", type=int, default=50000)
    p.add_argument("--max-util-pct", type=int, default=10)
    p.add_argument("--stale-heartbeat-seconds", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--main-steps", type=int, default=10000)
    p.add_argument("--warmup-eval-interval", type=int, default=100)
    p.add_argument("--main-eval-interval", type=int, default=500)
    p.add_argument("--ckpt-interval", type=int, default=250)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--phases", choices=["all", "warmup", "main"], default="all")
    p.add_argument("--only-jobs", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    state = run_launcher(
        snapshot_dir=args.snapshot_dir,
        runs_dir=args.runs_dir,
        allowed_gpus=args.allowed_gpus,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        poll_interval=args.poll_interval,
        max_used_mb=args.max_used_mb,
        min_free_mb=args.min_free_mb,
        max_util_pct=args.max_util_pct,
        stale_heartbeat_seconds=args.stale_heartbeat_seconds,
        warmup_steps=args.warmup_steps,
        main_steps=args.main_steps,
        warmup_eval_interval=args.warmup_eval_interval,
        main_eval_interval=args.main_eval_interval,
        ckpt_interval=args.ckpt_interval,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        model_path=args.model_path,
        phases=args.phases,
        only_jobs=args.only_jobs,
        dry_run=args.dry_run,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
