"""CPU unit and integration tests for long-study training, preflight, and multi-GPU launcher."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import sys
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_v2.eval_long_study import discover_candidate_checkpoints  # noqa: E402
from scripts_v2.launch_long_study import (  # noqa: E402
    build_long_study_jobs,
    parse_nvidia_smi_output,
    run_launcher,
    select_free_gpus,
)
from scripts_v2.preflight_long_study import (  # noqa: E402
    _MockPreflightTokenizer,
    _build_proxy_qwen_model,
    _build_proxy_scratch_model,
    build_synthetic_preflight_snapshot,
    run_preflight,
)
from scripts_v2.train_long_study import (  # noqa: E402
    compute_trainable_weights_sha256,
    resolve_resume_checkpoint,
    run_training,
)


def _read_trace(trace_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("arch", ["scratch", "qwen"])
def test_warmup_clone_parent_fresh_main_optimizer_and_step0_milestone(
    tmp_path: Path, arch: str
) -> None:
    snap = build_synthetic_preflight_snapshot(tmp_path / "snapshot")
    tok = _MockPreflightTokenizer()
    cpu = torch.device("cpu")

    def _make_model(seed: int = 101) -> torch.nn.Module:
        if arch == "scratch":
            return _build_proxy_scratch_model(tok, cpu, seed=seed)
        return _build_proxy_qwen_model(tok, cpu, seed=seed)

    warmup_dir = tmp_path / f"{arch}_warmup"
    warm_summary = run_training(
        arch=arch,
        phase="warmup",
        exposure="standard",
        seed=42,
        snapshot_dir=snap,
        output_dir=warmup_dir,
        max_steps=4,
        eval_interval=2,
        ckpt_interval=2,
        milestones=[0, 2, 4],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=_make_model(101),
        tokenizer=tok,
        device=cpu,
    )
    assert warm_summary["status"] == "completed"
    assert warm_summary["step"] == 4
    assert (warmup_dir / "best_weights").is_dir()
    assert (warmup_dir / "val_panel_history.json").exists()
    assert (warmup_dir / "heartbeat.json").exists()

    # Capture warmup best_weights hash for exact step_0 verification
    probe_parent = _make_model(999)
    if arch == "scratch":
        probe_parent.load_weights(warmup_dir / "best_weights")  # type: ignore[operator]
    else:
        probe_parent.load_adapter_and_head(warmup_dir / "best_weights")  # type: ignore[operator]
    parent_best_sha = compute_trainable_weights_sha256(probe_parent)

    # Clone warmup best_weights into main with fresh optimizer & cosine scheduler
    main_dir = tmp_path / f"{arch}_main"
    main_summary = run_training(
        arch=arch,
        phase="main",
        exposure="familiar",
        seed=42,
        snapshot_dir=snap,
        output_dir=main_dir,
        init_weights_dir=warmup_dir / "best_weights",
        max_steps=4,
        eval_interval=2,
        ckpt_interval=2,
        milestones=[0, 2, 4],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        lr=1e-3,
        warmup_steps=2,
        min_lr_ratio=0.1,
        verify_manifest=True,
        model=_make_model(202),
        tokenizer=tok,
        device=cpu,
    )
    assert main_summary["status"] == "completed"
    assert main_summary["step"] == 4

    # Verify step_00000 milestone was written and equals warmup best_weights
    step0_dir = main_dir / "milestones" / "step_00000"
    assert step0_dir.is_dir()
    probe_step0 = _make_model(303)
    if arch == "scratch":
        probe_step0.load_weights(step0_dir)  # type: ignore[operator]
    else:
        probe_step0.load_adapter_and_head(step0_dir)  # type: ignore[operator]
    assert compute_trainable_weights_sha256(probe_step0) == parent_best_sha

    # Verify main started at step 0 with a FRESH LR schedule (0.5 * peak -> 1.0 * peak -> 0.55 * peak)
    main_trace = _read_trace(main_dir / "train_trace.jsonl")
    assert [r["step"] for r in main_trace] == [1, 2, 3, 4]
    assert main_trace[0]["lr"] == pytest.approx(0.5 * 1e-3, rel=1e-6)
    assert main_trace[1]["lr"] == pytest.approx(1.0 * 1e-3, rel=1e-6)
    assert main_trace[2]["lr"] == pytest.approx(1.0 * 1e-3, rel=1e-6)
    assert main_trace[3]["lr"] == pytest.approx(0.55 * 1e-3, rel=1e-6)

    # Verify exposure lineage records warmup -> main transition
    lineage = main_summary["exposure_lineage"]
    assert len(lineage) == 2
    assert lineage[0]["phase"] == "warmup" and lineage[0]["exposure"] == "standard"
    assert lineage[1]["phase"] == "main" and lineage[1]["exposure"] == "familiar"


@pytest.mark.parametrize("arch", ["scratch", "qwen"])
def test_atomic_save_resume_equivalence_across_optimizer_boundary(
    tmp_path: Path, arch: str
) -> None:
    snap = build_synthetic_preflight_snapshot(tmp_path / "snapshot")
    tok = _MockPreflightTokenizer()
    cpu = torch.device("cpu")

    def _make_model(seed: int = 77) -> torch.nn.Module:
        if arch == "scratch":
            return _build_proxy_scratch_model(tok, cpu, seed=seed)
        return _build_proxy_qwen_model(tok, cpu, seed=seed)

    # Continuous run 0 -> 5
    cont_dir = tmp_path / f"{arch}_continuous"
    cont_summary = run_training(
        arch=arch,
        phase="main",
        exposure="familiar",
        seed=123,
        snapshot_dir=snap,
        output_dir=cont_dir,
        max_steps=5,
        eval_interval=100,
        ckpt_interval=100,
        milestones=[],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=_make_model(77),
        tokenizer=tok,
        device=cpu,
    )

    # Segmented run: 0 -> 3 (stop_at_step=3 of a 5-step schedule, saving atomic checkpoint at step 3)
    res_dir = tmp_path / f"{arch}_resumed"
    run_training(
        arch=arch,
        phase="main",
        exposure="familiar",
        seed=123,
        snapshot_dir=snap,
        output_dir=res_dir,
        max_steps=5,
        stop_at_step=3,
        eval_interval=100,
        ckpt_interval=3,
        milestones=[],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=_make_model(77),
        tokenizer=tok,
        device=cpu,
    )

    # Resume from step 3 -> 5 into a fresh model object with perturbed trainable weights
    res_model_fresh = _make_model(77 if arch == "qwen" else 9999)
    with torch.no_grad():
        for p in res_model_fresh.parameters():
            if p.requires_grad:
                p.add_(0.25)

    res_summary = run_training(
        arch=arch,
        phase="main",
        exposure="familiar",
        seed=123,
        snapshot_dir=snap,
        output_dir=res_dir,
        resume=True,
        max_steps=5,
        eval_interval=100,
        ckpt_interval=5,
        milestones=[],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=res_model_fresh,
        tokenizer=tok,
        device=cpu,
    )

    cont_trace = _read_trace(cont_dir / "train_trace.jsonl")
    res_trace = _read_trace(res_dir / "train_trace.jsonl")
    assert len(cont_trace) == 5 and len(res_trace) == 5

    # Steps 4 and 5 must have identical sampled input_sha256, LR, loss, and cumulative stats
    for idx in (3, 4):
        assert cont_trace[idx]["input_hashes"] == res_trace[idx]["input_hashes"]
        assert cont_trace[idx]["lr"] == pytest.approx(res_trace[idx]["lr"], rel=0.0, abs=1e-12)
        assert cont_trace[idx]["train_loss"] == pytest.approx(
            res_trace[idx]["train_loss"], rel=1e-6, abs=1e-6
        )
        assert cont_trace[idx]["total_tokens"] == res_trace[idx]["total_tokens"]
        assert cont_trace[idx]["unique_roots"] == res_trace[idx]["unique_roots"]
        assert cont_trace[idx]["unique_families"] == res_trace[idx]["unique_families"]

    assert cont_summary["final_weights_sha256"] == res_summary["final_weights_sha256"]
    assert cont_summary["total_tokens_processed"] == res_summary["total_tokens_processed"]


def test_rolling_checkpoint_retention_fallback_milestones_candidates_and_sigterm(
    tmp_path: Path,
) -> None:
    snap = build_synthetic_preflight_snapshot(tmp_path / "snapshot")
    tok = _MockPreflightTokenizer()
    cpu = torch.device("cpu")

    out_dir = tmp_path / "qwen_rolling"
    summary = run_training(
        arch="qwen",
        phase="main",
        exposure="familiar",
        seed=42,
        snapshot_dir=snap,
        output_dir=out_dir,
        max_steps=6,
        eval_interval=2,
        ckpt_interval=2,
        milestones=[0, 2, 4, 6],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=_build_proxy_qwen_model(tok, cpu, seed=55),
        tokenizer=tok,
        device=cpu,
    )
    assert summary["step"] == 6

    # Only 2 rolling checkpoints (ckpt_latest @ step 6, ckpt_prev @ step 4)
    ckpt_entries = sorted(p.name for p in (out_dir / "checkpoints").iterdir())
    assert ckpt_entries == ["ckpt_latest", "ckpt_prev"]
    latest_meta = json.loads((out_dir / "checkpoints" / "ckpt_latest" / "checkpoint_meta.json").read_text())
    prev_meta = json.loads((out_dir / "checkpoints" / "ckpt_prev" / "checkpoint_meta.json").read_text())
    assert latest_meta["step"] == 6
    assert prev_meta["step"] == 4

    # Milestones and candidates saved cleanly and discoverable by eval_long_study
    assert sorted(p.name for p in (out_dir / "milestones").iterdir()) == [
        "step_00000",
        "step_00002",
        "step_00004",
        "step_00006",
    ]
    discovered = discover_candidate_checkpoints(out_dir / "candidates")
    assert [step for step, _ in discovered] == [2, 4, 6]

    # Corrupt ckpt_latest and verify automatic fallback to ckpt_prev (step 4)
    fallback_dir = tmp_path / "qwen_fallback"
    shutil.copytree(out_dir, fallback_dir)
    (fallback_dir / "checkpoints" / "ckpt_latest" / "checkpoint_meta.json").write_text(
        "{corrupted json", encoding="utf-8"
    )
    resolved = resolve_resume_checkpoint(fallback_dir)
    assert resolved.name == "ckpt_prev"

    resumed_summary = run_training(
        arch="qwen",
        phase="main",
        exposure="familiar",
        seed=42,
        snapshot_dir=snap,
        output_dir=fallback_dir,
        resume=True,
        max_steps=6,
        eval_interval=2,
        ckpt_interval=2,
        milestones=[0, 2, 4, 6],
        batch_size=1,
        grad_accum=2,
        k_values=(0, 1, 2, 4),
        val_roots_per_variant=2,
        max_seq_len=1024,
        verify_manifest=True,
        model=_build_proxy_qwen_model(tok, cpu, seed=55),
        tokenizer=tok,
        device=cpu,
    )
    assert resumed_summary["final_weights_sha256"] == summary["final_weights_sha256"]

    # Verify graceful SIGTERM stops at next optimizer boundary and saves atomic checkpoint
    sig_dir = tmp_path / "scratch_sigterm"
    # Test signal handling by raising SIGTERM inside a forward hook after run_training
    # installs its SIGTERM handler (raising earlier would terminate pytest itself).
    sig_model = _build_proxy_scratch_model(tok, cpu, seed=11)
    fired = {"done": False}

    def _send_sigterm_on_first_forward(_mod: Any, _inp: Any, _out: Any) -> None:
        if not fired["done"]:
            fired["done"] = True
            signal.raise_signal(signal.SIGTERM)

    hook = sig_model.classifier.register_forward_hook(_send_sigterm_on_first_forward)
    try:
        sig_summary = run_training(
            arch="scratch",
            phase="main",
            exposure="familiar",
            seed=42,
            snapshot_dir=snap,
            output_dir=sig_dir,
            max_steps=10,
            eval_interval=5,
            ckpt_interval=10,
            milestones=[],
            batch_size=1,
            grad_accum=1,
            k_values=(0, 1, 2, 4),
            val_roots_per_variant=2,
            max_seq_len=1024,
            verify_manifest=True,
            model=sig_model,
            tokenizer=tok,
            device=cpu,
        )
    finally:
        hook.remove()

    assert sig_summary["status"] == "interrupted_checkpointed"
    assert sig_summary["step"] == 1
    assert (sig_dir / "checkpoints" / "ckpt_latest" / "checkpoint_meta.json").exists()


def test_preflight_mock_small_all_five_checks(tmp_path: Path) -> None:
    report_path = tmp_path / "preflight_report.json"
    report = run_preflight(
        output_report=report_path,
        profile_steps=4,
        grad_accum=2,
        max_seq_len=1024,
        arch="all",
        mock_small=True,
    )
    assert report["status"] == "passed"
    assert report_path.exists()
    checks = report["checks"]
    assert checks["check1_params_grads_lineage"]["scratch"]["meta_494m_counts"]["total_params"] == 494_150_147
    assert checks["check1_params_grads_lineage"]["qwen"]["passed"] is True
    assert checks["check2_loss_padding_prompt_sampling"]["passed"] is True
    assert checks["check3_save_resume_equivalence"]["scratch"]["passed"] is True
    assert checks["check3_save_resume_equivalence"]["qwen"]["passed"] is True
    assert checks["check4_5_profile_and_projections"]["scratch"]["projections"]["main_10k_gpu_hours"] > 0
    assert checks["check4_5_profile_and_projections"]["qwen"]["projections"]["main_10k_gpu_hours"] > 0


def test_launch_long_study_dag_dependencies_gpu_packing_and_dry_run(tmp_path: Path) -> None:
    snap = tmp_path / "snapshot"
    runs_dir = tmp_path / "runs"

    jobs = build_long_study_jobs(snapshot_dir=snap, runs_dir=runs_dir)
    assert len(jobs) == 14
    by_id = {j.job_id: j for j in jobs}

    # 6 warmup parents
    warmup_jobs = [j for j in jobs if j.phase == "warmup"]
    assert len(warmup_jobs) == 6
    assert all(j.exposure == "standard" and j.depends_on == [] for j in warmup_jobs)

    # 8 main logical slots
    main_jobs = [j for j in jobs if j.phase == "main"]
    assert len(main_jobs) == 8

    # Slots 0-2: Qwen familiar seeds 42, 123, 456 -> matching Qwen warmup parents
    for slot, seed in [(0, 42), (1, 123), (2, 456)]:
        j = by_id[f"main_slot{slot}_qwen_familiar_s{seed}"]
        assert j.depends_on == [f"warmup_qwen_s{seed}"]
        assert j.init_weights_dir == runs_dir / "warmup" / f"qwen_seed{seed}" / "best_weights"

    # Slots 3-5: Scratch familiar seeds 42, 123, 456 -> matching Scratch warmup parents
    for slot, seed in [(3, 42), (4, 123), (5, 456)]:
        j = by_id[f"main_slot{slot}_scratch_familiar_s{seed}"]
        assert j.depends_on == [f"warmup_scratch_s{seed}"]
        assert j.init_weights_dir == runs_dir / "warmup" / f"scratch_seed{seed}" / "best_weights"

    # Slots 6-7: all_rules reference seed 42 -> seed-42 warmup parents
    q_all = by_id["main_slot6_qwen_all_rules_s42"]
    assert q_all.depends_on == ["warmup_qwen_s42"]
    assert q_all.init_weights_dir == runs_dir / "warmup" / "qwen_seed42" / "best_weights"

    s_all = by_id["main_slot7_scratch_all_rules_s42"]
    assert s_all.depends_on == ["warmup_scratch_s42"]
    assert s_all.init_weights_dir == runs_dir / "warmup" / "scratch_seed42" / "best_weights"

    # GPU discovery parsing & filtering (never pick occupied GPUs!)
    sample_csv = "\n".join(
        [
            "0, 32000, 49559, 81559, 45",  # occupied by unrelated process
            "1, 4, 81076, 81559, 0",       # free
            "2, 4, 81076, 81559, 0",       # free
            "3, 6000, 75000, 81559, 0",    # memory.used >= 5000 -> skip
            "4, 4, 81076, 81559, 25",      # utilization >= 10% -> skip
            "5, 4, 81076, 81559, 0",       # free
        ]
    )
    parsed = parse_nvidia_smi_output(sample_csv)
    assert select_free_gpus(parsed, busy_gpus={2}) == [1, 5]
    assert select_free_gpus(parsed, allowed_gpus=[0, 1, 2]) == [1, 2]

    # Dry-run execution writes launcher_state.json with topological waves & max_concurrent cap
    state = run_launcher(
        snapshot_dir=snap,
        runs_dir=runs_dir,
        allowed_gpus=[1, 2, 5, 6],
        max_concurrent=3,
        dry_run=True,
    )
    assert state["mode"] == "dry_run"
    assert (runs_dir / "launcher_state.json").exists()
    waves = state["plan"]["waves"]
    assert all(len(w["jobs"]) <= 3 for w in waves)
    # Wave 0/1 contain only warmup jobs because main jobs depend on warmup parents
    assert all(j["phase"] == "warmup" for j in waves[0]["jobs"])
    scheduled_ids = [j["job_id"] for w in waves for j in w["jobs"]]
    assert len(scheduled_ids) == 14 and len(set(scheduled_ids)) == 14
