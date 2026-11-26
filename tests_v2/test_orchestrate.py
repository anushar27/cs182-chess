"""Tests for multi-GPU experiment orchestrator (`scripts_v2/orchestrate_experiments.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_icl_v2.variants import COMPOSITIONAL_VARIANTS, SINGLE_VARIANTS
from scripts_v2 import (
    eval_qwen,
    eval_track2,
    make_paper_figures,
    orchestrate_experiments,
    run_baselines,
    train_qwen,
    train_track2,
)


def _validate_job_cmd_with_real_parser(job: dict) -> None:
    """Parse each generated CLI command using its target script's real argparse parser."""
    script = job["script"]
    cli_args = job["cmd"][2:]  # strip [python, script_path]

    if script == "scripts_v2/train_track2.py":
        parsed = train_track2.build_arg_parser().parse_args(cli_args)
        assert parsed.gpu == 0
    elif script == "scripts_v2/eval_track2.py":
        parsed = eval_track2.build_arg_parser().parse_args(cli_args)
        assert parsed.gpu == 0
    elif script == "scripts_v2/train_qwen.py":
        parsed = train_qwen.parse_args(cli_args)
        assert parsed.gpu == 0
    elif script == "scripts_v2/eval_qwen.py":
        parsed = eval_qwen.parse_args(cli_args)
        assert parsed.gpu == 0
    elif script == "scripts_v2/run_baselines.py":
        run_baselines.parse_args(cli_args)
    elif script == "scripts_v2/make_paper_figures.py":
        make_paper_figures.parse_args(cli_args)
    else:
        raise AssertionError(f"Unexpected script in DAG: {script}")


def _assert_wave_invariants(manifest: dict, allowed_gpus: list[int]) -> None:
    """Verify zero intra-wave GPU collisions, valid CUDA_VISIBLE_DEVICES, and DAG order."""
    wave_of: dict[str, int] = {}
    for wave in manifest["waves"]:
        w_idx = wave["wave_index"]
        gpu_jobs = [j for j in wave["jobs"] if j["requires_gpu"]]
        assigned = [j["assigned_gpu"] for j in gpu_jobs]

        # Zero GPU collisions within a wave
        assert len(assigned) == len(set(assigned)), f"GPU collision in wave {w_idx}: {assigned}"
        assert len(assigned) <= len(allowed_gpus)
        for g in assigned:
            assert g in allowed_gpus

        for j in wave["jobs"]:
            wave_of[j["job_id"]] = w_idx
            if j["requires_gpu"]:
                assert j["env_overrides"]["CUDA_VISIBLE_DEVICES"] == str(j["assigned_gpu"])
            if j["phase"] == "track1_qwen":
                assert j["env_overrides"]["HF_HUB_OFFLINE"] == "1"
                assert j["env_overrides"]["TRANSFORMERS_OFFLINE"] == "1"
            _validate_job_cmd_with_real_parser(j)

    for wave in manifest["waves"]:
        for j in wave["jobs"]:
            for dep in j["depends_on"]:
                if dep in wave_of:
                    assert wave_of[dep] < wave_of[j["job_id"]], (
                        f"Dependency {dep} (wave {wave_of[dep]}) not strictly before "
                        f"{j['job_id']} (wave {wave_of[j['job_id']]})"
                    )


@pytest.mark.parametrize("phase", list(orchestrate_experiments.VALID_PHASES))
def test_dry_run_each_phase_and_cli_compatibility(tmp_path: Path, phase: str):
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    manifest = orchestrate_experiments.main(
        [
            "--phases",
            phase,
            "--gpus",
            *[str(g) for g in gpus],
            "--dry-run",
            "--policy-id",
            "terminal_aware_v1",
            "--notation",
            "san",
            "--data-dir",
            str(tmp_path / "data"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert manifest["dry_run"] is True
    assert manifest["total_jobs"] > 0
    assert manifest["execution_results"] == []
    _assert_wave_invariants(manifest, allowed_gpus=gpus)


def test_full_dag_gpu_packing_and_coverage(tmp_path: Path):
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    manifest = orchestrate_experiments.main(
        [
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "--dry-run",
            "--notation",
            "uci",
            "--data-dir",
            str(tmp_path / "data_v2"),
            "--runs-dir",
            str(tmp_path / "runs_v2"),
        ]
    )
    _assert_wave_invariants(manifest, allowed_gpus=gpus)

    jobs_by_id = {
        j["job_id"]: j
        for wave in manifest["waves"]
        for j in wave["jobs"]
    }

    # Track 2 scaling coverage: 4 warmups, 3 seeds for TINY/SMALL, 1 for MED/LARGE, controls + seg_bidir
    for cfg in ("tiny", "small", "medium", "large"):
        assert f"t2_warmup_{cfg}" in jobs_by_id
    for seed in (42, 123, 456):
        assert f"t2_cont_tiny_seed{seed}" not in jobs_by_id  # explicit check below
        assert f"t2_cont_move_tiny_seed{seed}" in jobs_by_id
        assert f"t2_cont_move_small_seed{seed}" in jobs_by_id
    assert "t2_cont_move_medium_seed42" in jobs_by_id
    assert "t2_cont_move_large_seed42" in jobs_by_id
    assert "t2_cont_all_variants_small_seed42" in jobs_by_id
    assert "t2_cont_chess_only_small_seed42" in jobs_by_id
    assert "t2_cont_seg_bidir_small_seed42" in jobs_by_id

    # Verify Track 2 eval includes all 13 single variants + 3 compositional variants
    t2_eval_cmd = jobs_by_id["t2_eval_move_small_seed42"]["cmd"]
    for v in list(SINGLE_VARIANTS) + list(COMPOSITIONAL_VARIANTS):
        assert v in t2_eval_cmd

    # Track 1 Qwen coverage: 6 continuation jobs + zero-shot + evals
    qwen_cont_ids = [
        "t1_qwen_cont_move_seed42",
        "t1_qwen_cont_move_seed123",
        "t1_qwen_cont_move_seed456",
        "t1_qwen_cont_all_variants_seed42",
        "t1_qwen_cont_chess_only_seed42",
        "t1_qwen_cont_full_lora_seed42",
    ]
    for qid in qwen_cont_ids:
        assert qid in jobs_by_id
        assert "--notation" in jobs_by_id[qid]["cmd"]
        assert "uci" in jobs_by_id[qid]["cmd"]
    assert "t1_qwen_eval_zero_shot_base" in jobs_by_id

    # Verify Track 1 isolated dry-run packs the 6 continuation jobs onto GPUs 0-5 in one wave
    t1_only = orchestrate_experiments.main(
        [
            "--phases",
            "track1_qwen",
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data_v2"),
            "--runs-dir",
            str(tmp_path / "runs_v2"),
        ]
    )
    _assert_wave_invariants(t1_only, allowed_gpus=gpus)
    wave1_cont_gpus = [
        j["assigned_gpu"]
        for j in t1_only["waves"][1]["jobs"]
        if j["job_id"] in qwen_cont_ids
    ]
    assert sorted(wave1_cont_gpus) == [0, 1, 2, 3, 4, 5]


def test_wave_splitting_with_few_gpus(tmp_path: Path):
    """When fewer GPUs are passed than ready jobs, wave splitting prevents collisions."""
    manifest = orchestrate_experiments.main(
        [
            "--phases",
            "track2_scaling",
            "--gpus",
            "2",
            "5",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data_v2"),
            "--runs-dir",
            str(tmp_path / "runs_v2"),
        ]
    )
    _assert_wave_invariants(manifest, allowed_gpus=[2, 5])


def test_pilot_gate_check(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    eval_path = runs_dir / "pilot" / "small_eval_val" / "eval_summary.json"
    base_path = runs_dir / "pilot" / "baselines" / "baselines_val.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    eval_path.write_text(
        json.dumps(
            {
                "k_values": [0, 4],
                "main_by_k": {"4": {"context_gain": {"macro_mean": 0.12}}},
                "controls_by_k": {
                    "mismatched_context": {"4": {"context_gain": {"macro_mean": 0.01}}}
                },
            }
        ),
        encoding="utf-8",
    )
    base_path.write_text(
        json.dumps(
            {
                "models": {
                    "board_knn": {"by_k": {"4": {"macro": {"context_gain": 0.05}}}}
                }
            }
        ),
        encoding="utf-8",
    )
    res_pass = orchestrate_experiments.evaluate_pilot_gate(runs_dir)
    assert res_pass["status"] == "passed"
    assert res_pass["beats_knn"] and res_pass["beats_mismatched"]

    # Failing case: gain <= kNN
    eval_path.write_text(
        json.dumps(
            {
                "k_values": [0, 4],
                "main_by_k": {"4": {"context_gain": {"macro_mean": 0.03}}},
                "controls_by_k": {
                    "mismatched_context": {"4": {"context_gain": {"macro_mean": 0.01}}}
                },
            }
        ),
        encoding="utf-8",
    )
    res_fail = orchestrate_experiments.evaluate_pilot_gate(runs_dir)
    assert res_fail["status"] == "failed"
    assert not res_fail["beats_knn"]
