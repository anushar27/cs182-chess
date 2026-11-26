"""Smoke and integration test for scripts_v2/generate_data_v2.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts_v2.generate_data_v2 import main as generate_main


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_data_v2_tiny_end_to_end(tmp_path: Path):
    out_dir = tmp_path / "stage1_tiny"
    variants = ["standard", "diagonal_rook", "blocked_squares"]

    generate_main(
        [
            "--output-dir",
            str(out_dir),
            "--variants",
            *variants,
            "--policy-id",
            "terminal_aware_v1",
            "--horizon",
            "80",
            "--demo-games",
            "6",
            "--source-games",
            "9",
            "--train-roots",
            "4",
            "--val-roots",
            "2",
            "--test-roots",
            "2",
            "--R-train",
            "4",
            "--R-eval",
            "4",
            "--workers",
            "2",
            "--seed",
            "42",
            "--alt-demo-policy",
            "uniform_legal_v1",
        ]
    )

    manifest_path = out_dir / "dataset_manifest.json"
    audit_path = out_dir / "dataset_audit.json"
    assert manifest_path.exists()
    assert audit_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["passed"] is True
    assert audit["n_errors"] == 0
    assert manifest["audit_passed"] is True
    assert manifest["policy_id"] == "terminal_aware_v1"
    assert manifest["alt_demo_policy"] == "uniform_legal_v1"
    assert manifest["horizon"] == 80
    assert manifest["R_train"] == 4
    assert manifest["R_eval"] == 4
    assert manifest["elapsed_seconds"] > 0.0

    for variant in variants:
        v_dir = out_dir / variant
        demos = _load_jsonl(v_dir / "games_demo.jsonl")
        alt_demos = _load_jsonl(v_dir / "games_demo_alt_policy.jsonl")
        tr = _load_jsonl(v_dir / "positions_train.jsonl")
        va = _load_jsonl(v_dir / "positions_val.jsonl")
        te = _load_jsonl(v_dir / "positions_test.jsonl")

        assert len(demos) == 6
        assert len(alt_demos) == 6
        assert all(d["policy"] == "terminal_aware_v1" for d in demos)
        assert all(d["policy"] == "uniform_legal_v1" for d in alt_demos)
        assert all("states_at_plies" not in d for d in demos)

        assert 1 <= len(tr) <= 4
        assert 1 <= len(va) <= 2
        assert 1 <= len(te) <= 2

        for pos in tr + va + te:
            assert pos["ply"] >= 8
            assert pos["side_to_move"] in ("w", "b")
            assert isinstance(pos["material_diff_white"], int)
            assert pos["equal_material"] == (abs(pos["material_diff_white"]) <= 2)
            assert np.isclose(sum(pos["label_probs"]), 1.0)
            if variant == "blocked_squares":
                assert len(pos["blocked_squares"]) > 0

        v_man = manifest["variants"][variant]
        assert "dedup_stats" in v_man
        assert "splits" in v_man
