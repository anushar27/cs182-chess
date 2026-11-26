"""Evaluate Track 1 Qwen2.5-7B-Instruct LoRA (or zero-shot base) with controls & bootstrap CIs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chess_icl_v2.data.dataset_v2 import EvalEpisodeBuilder, discover_variants, load_jsonl  # noqa: E402
from chess_icl_v2.eval.controls import (  # noqa: E402
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_mismatched_context_episodes,
)
from chess_icl_v2.eval.metrics import (  # noqa: E402
    compute_brier_per_sample,
    compute_ece,
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

ALL_CONTROLS: tuple[str, ...] = (
    "mismatched_context",
    "corrupted_outcomes",
    "corrupted_trajectories",
)


def resolve_eval_variants(
    variants_arg: Optional[Sequence[str]],
    data_dir: Path,
    split: str,
) -> list[str]:
    """Expand preset variant groups or discover available variants under data_dir."""
    if not variants_arg:
        discovered = discover_variants(data_dir, split=split)
        return discovered if discovered else list(TEST_VARIANTS)

    resolved: list[str] = []
    for item in variants_arg:
        if item == "test_variants":
            resolved.extend(TEST_VARIANTS)
        elif item in ("train_variants", "move_variants"):
            resolved.extend(TRAIN_VARIANTS)
        elif item == "compositional":
            resolved.extend(COMPOSITIONAL_VARIANTS)
        elif item == "all_variants":
            resolved.extend(SINGLE_VARIANTS)
        elif item == "chess_only":
            resolved.append("standard")
        else:
            resolved.append(item)

    seen: set[str] = set()
    unique: list[str] = []
    for v in resolved:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _score_episodes(
    model: QwenICLClassifier,
    formatter: QwenPromptFormatter,
    episodes: Sequence[Mapping[str, Any]],
    condition: str,
    k: int,
    demo_draw: int,
    model_seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run model inference on a list of prepared episodes and build per-root records."""
    if not episodes:
        return []

    prompts = [
        formatter.format_prompt(
            ep["demos"],
            {
                "fen": ep["query_fen"],
                "blocked_squares": ep.get("blocked_squares"),
            },
        )
        for ep in episodes
    ]
    y_true = np.asarray([ep["y_true"] for ep in episodes], dtype=np.float64)
    p_pred = np.asarray(
        model.predict_proba(prompts, batch_size=max(1, batch_size)), dtype=np.float64
    )

    _, per_nll = compute_nll(p_pred, y_true)
    per_brier = compute_brier_per_sample(p_pred, y_true)
    pred_cls = np.argmax(p_pred, axis=-1)
    true_cls = np.argmax(y_true, axis=-1)

    records: list[dict[str, Any]] = []
    for idx, ep in enumerate(episodes):
        records.append(
            {
                "condition": condition,
                "k": int(k),
                "demo_draw": int(demo_draw),
                "model_seed": int(model_seed),
                "variant": str(ep["variant"]),
                "root_id": ep["root_id"],
                "source_game_id": ep.get("source_game_id", ep["root_id"]),
                "side_to_move": ep.get("side_to_move", "w"),
                "equal_material": bool(ep.get("equal_material", False)),
                "y_true": [float(v) for v in y_true[idx]],
                "p_pred": [float(v) for v in p_pred[idx]],
                "nll": float(per_nll[idx]),
                "brier": float(per_brier[idx]),
                "correct": float(1.0 if pred_cls[idx] == true_cls[idx] else 0.0),
            }
        )
    return records


def _summarize_records_for_k(
    records: Sequence[Mapping[str, Any]],
    n_boot: int,
    seed: int,
    k0_lookup: Optional[Mapping[tuple[str, int, Any], float]] = None,
) -> Optional[dict[str, Any]]:
    """Aggregate NLL, Brier, Accuracy, Calibration, and optional Context Gain for one K."""
    if not records:
        return None

    nll_agg = aggregate_evaluation_records(
        records, value_key="nll", n_boot=n_boot, seed=seed
    )
    brier_agg = aggregate_evaluation_records(
        records, value_key="brier", n_boot=n_boot, seed=seed + 11
    )
    acc_agg = aggregate_evaluation_records(
        records, value_key="correct", n_boot=n_boot, seed=seed + 23
    )

    p_mat = np.asarray([r["p_pred"] for r in records], dtype=np.float64)
    y_mat = np.asarray([r["y_true"] for r in records], dtype=np.float64)
    sides = [r.get("side_to_move", "w") for r in records]
    ece_stats = compute_ece(p_mat, y_mat)
    side_cal = compute_side_calibration(p_mat, y_mat, sides)

    out: dict[str, Any] = {
        "nll": nll_agg,
        "brier": brier_agg,
        "accuracy": acc_agg,
        "calibration": {
            "ece": ece_stats["ece"],
            "classwise_ece": ece_stats["classwise_ece"],
            "side_calibration": side_cal,
        },
        "n_records": len(records),
    }

    if k0_lookup is not None:
        gain_records: list[dict[str, Any]] = []
        for r in records:
            key = (str(r["variant"]), int(r["model_seed"]), r["root_id"])
            if key in k0_lookup:
                gain_records.append(
                    {
                        **r,
                        "delta_nll": float(k0_lookup[key] - r["nll"]),
                    }
                )
        if gain_records:
            out["context_gain"] = aggregate_evaluation_records(
                gain_records, value_key="delta_nll", n_boot=n_boot, seed=seed + 37
            )

    return out


def run_evaluation(
    *,
    data_dir: Path,
    output_dir: Path,
    adapter_dir: Optional[Path] = None,
    zero_shot_base: bool = False,
    split: Literal["val", "test"] = "test",
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (0, 1, 2, 4, 8, 16),
    num_draws: int = 5,
    controls: Sequence[str] = ALL_CONTROLS,
    outcome_corruption_mode: Literal["inverted", "scrambled", "all_draw"] = "inverted",
    trajectory_corruption_mode: Literal["shuffled", "mismatched"] = "shuffled",
    notation: Literal["uci", "san"] = "san",
    max_seq_len: int = 16384,
    batch_size: int = 4,
    max_roots: Optional[int] = None,
    n_boot: int = 1000,
    seed: int = 42,
    model_seed: int = 0,
    gpu: int = 0,
    model_path: Optional[str] = None,
    output_file: Optional[Path] = None,
    model: Optional[QwenICLClassifier] = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Run full Track 1 evaluation across variants, K values, demo draws, and controls."""
    if adapter_dir is None and not zero_shot_base and model is None:
        raise ValueError("Provide --adapter-dir or pass --zero-shot-base")

    set_global_seed(seed)
    device = resolve_device(gpu)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_variants = resolve_eval_variants(variants, data_dir=data_dir, split=split)
    if not eval_variants:
        raise ValueError(f"No evaluation variants found for split={split!r} in {data_dir}")

    active_controls = [
        c for c in controls if c and c.lower() != "none"
    ]
    for c in active_controls:
        if c not in ALL_CONTROLS:
            raise ValueError(f"Unsupported control {c!r}; expected one of {ALL_CONTROLS}")

    if tokenizer is None:
        resolved_path = resolve_qwen_model_path(model_path)
        tokenizer = load_qwen_tokenizer(resolved_path, local_files_only=True)

    if model is None:
        attn_impl = get_preferred_attn_implementation()
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if adapter_dir is not None and not zero_shot_base:
            model = QwenICLClassifier.from_adapter_and_head(
                adapter_dir,
                tokenizer=tokenizer,
                model_path=model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                local_files_only=True,
                is_trainable=False,
                device=device,
            )
        else:
            model = QwenICLClassifier(
                tokenizer=tokenizer,
                model_path=model_path,
                lora_mode="handoff",
                torch_dtype=torch_dtype,
                head_dtype=torch.float32,
                attn_implementation=attn_impl,
                local_files_only=True,
                gradient_checkpointing=False,
            ).to(device)
    else:
        model = model.to(device)

    model.eval()
    eff_max_len = min(int(max_seq_len), int(model.max_position_embeddings))
    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation=notation,
        max_length=eff_max_len,
    )
    builder = EvalEpisodeBuilder(data_dir)

    # Load all available demo pools for mismatched_context control
    all_demo_pools: dict[str, list[dict[str, Any]]] = {}
    if "mismatched_context" in active_controls:
        pool_variants = sorted(set(eval_variants) | set(discover_variants(data_dir, split=split)))
        for pv in pool_variants:
            demo_path = data_dir / pv / "games_demo.jsonl"
            if demo_path.exists():
                raw_pool = load_jsonl(demo_path)
                all_demo_pools[pv] = [
                    prepare_demo_for_notation(d, variant=pv, notation=notation)
                    for d in raw_pool
                ]

    all_records: list[dict[str, Any]] = []
    clean_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    controls_by_name_k: dict[str, dict[int, list[dict[str, Any]]]] = {
        c: defaultdict(list) for c in active_controls
    }

    sorted_ks = sorted(int(k) for k in k_values)
    for k in sorted_ks:
        if k == 0:
            # K=0 has empty demos regardless of draw_seed; evaluate once and replicate across draws
            k0_episodes: list[dict[str, Any]] = []
            for variant in eval_variants:
                eps, _ = builder.build(
                    variant=variant,
                    split=split,
                    k=0,
                    draw_seed=0,
                    max_roots=max_roots,
                    prefer_precomputed=True,
                )
                k0_episodes.extend(prepare_episodes_for_notation(eps, notation=notation))
            base_recs = _score_episodes(
                model=model,
                formatter=formatter,
                episodes=k0_episodes,
                condition="clean",
                k=0,
                demo_draw=0,
                model_seed=model_seed,
                batch_size=batch_size,
            )
            for draw_idx in range(max(1, num_draws)):
                for rec in base_recs:
                    rec_copy = dict(rec)
                    rec_copy["demo_draw"] = int(draw_idx)
                    all_records.append(rec_copy)
                    clean_by_k[0].append(rec_copy)
            continue

        for draw_idx in range(max(1, num_draws)):
            draw_episodes: list[dict[str, Any]] = []
            for variant in eval_variants:
                eps, _ = builder.build(
                    variant=variant,
                    split=split,
                    k=k,
                    draw_seed=draw_idx,
                    max_roots=max_roots,
                    prefer_precomputed=True,
                )
                draw_episodes.extend(prepare_episodes_for_notation(eps, notation=notation))

            clean_recs = _score_episodes(
                model=model,
                formatter=formatter,
                episodes=draw_episodes,
                condition="clean",
                k=k,
                demo_draw=draw_idx,
                model_seed=model_seed,
                batch_size=batch_size,
            )
            all_records.extend(clean_recs)
            clean_by_k[k].extend(clean_recs)

            ctrl_seed = seed + k * 1000 + draw_idx
            for ctrl in active_controls:
                if ctrl == "mismatched_context":
                    other_pools = {
                        pv: pool for pv, pool in all_demo_pools.items() if len(pool) > 0
                    }
                    ctrl_eps = make_mismatched_context_episodes(
                        draw_episodes, other_pools, seed=ctrl_seed
                    )
                elif ctrl == "corrupted_outcomes":
                    ctrl_eps = make_corrupted_outcome_episodes(
                        draw_episodes, mode=outcome_corruption_mode, seed=ctrl_seed
                    )
                elif ctrl == "corrupted_trajectories":
                    ctrl_eps = make_corrupted_trajectory_episodes(
                        draw_episodes, mode=trajectory_corruption_mode, seed=ctrl_seed
                    )
                else:
                    continue

                ctrl_recs = _score_episodes(
                    model=model,
                    formatter=formatter,
                    episodes=ctrl_eps,
                    condition=ctrl,
                    k=k,
                    demo_draw=draw_idx,
                    model_seed=model_seed,
                    batch_size=batch_size,
                )
                all_records.extend(ctrl_recs)
                controls_by_name_k[ctrl][k].extend(ctrl_recs)

    # Build K=0 paired root lookup for Delta NLL = NLL(0) - NLL(K)
    k0_lookup: Optional[dict[tuple[str, int, Any], float]] = None
    if 0 in clean_by_k and clean_by_k[0]:
        k0_lookup = {}
        for rec in clean_by_k[0]:
            if rec["demo_draw"] == 0:
                k0_lookup[(str(rec["variant"]), int(rec["model_seed"]), rec["root_id"])] = float(
                    rec["nll"]
                )

    clean_summary: dict[str, Any] = {}
    eq_summary: dict[str, Any] = {}
    for k in sorted_ks:
        clean_summary[str(k)] = _summarize_records_for_k(
            clean_by_k[k],
            n_boot=n_boot,
            seed=seed + k * 50,
            k0_lookup=k0_lookup if k > 0 else None,
        )
        eq_recs = [r for r in clean_by_k[k] if r.get("equal_material", False)]
        if eq_recs:
            eq_summary[str(k)] = _summarize_records_for_k(
                eq_recs,
                n_boot=n_boot,
                seed=seed + 500 + k * 50,
                k0_lookup=k0_lookup if k > 0 else None,
            )

    controls_summary: dict[str, Any] = {}
    for ctrl in active_controls:
        ctrl_dict: dict[str, Any] = {}
        for k in sorted_ks:
            if k == 0 or not controls_by_name_k[ctrl][k]:
                continue
            ctrl_k_summary = _summarize_records_for_k(
                controls_by_name_k[ctrl][k],
                n_boot=n_boot,
                seed=seed + 1000 + k * 50,
                k0_lookup=k0_lookup,
            )
            if ctrl_k_summary is not None and clean_summary.get(str(k)) is not None:
                ctrl_nll = ctrl_k_summary["nll"]["macro_mean"]
                clean_nll = clean_summary[str(k)]["nll"]["macro_mean"]
                ctrl_k_summary["delta_nll_vs_clean"] = float(ctrl_nll - clean_nll)
            ctrl_dict[str(k)] = ctrl_k_summary
        controls_summary[ctrl] = ctrl_dict

    payload = {
        "metadata": {
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "zero_shot_base": bool(zero_shot_base),
            "split": split,
            "notation": notation,
            "variants": eval_variants,
            "k_values": sorted_ks,
            "num_draws": int(num_draws),
            "controls": active_controls,
            "outcome_corruption_mode": outcome_corruption_mode,
            "trajectory_corruption_mode": trajectory_corruption_mode,
            "max_seq_len": eff_max_len,
            "model_seed": int(model_seed),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
        "summary": {
            "clean": clean_summary,
            "equal_material": eq_summary if eq_summary else None,
            "controls": controls_summary,
        },
        "per_root_predictions": all_records,
    }

    out_path = output_file if output_file is not None else (output_dir / f"eval_{split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[Eval] Saved evaluation report to {out_path} ({len(all_records)} records)")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track 1 Qwen2.5-7B-Instruct LoRA")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--zero-shot-base", action="store_true")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--num-draws", type=int, default=5)
    parser.add_argument("--controls", nargs="*", default=list(ALL_CONTROLS))
    parser.add_argument(
        "--outcome-corruption-mode",
        choices=["inverted", "scrambled", "all_draw"],
        default="inverted",
    )
    parser.add_argument(
        "--trajectory-corruption-mode",
        choices=["shuffled", "mismatched"],
        default="shuffled",
    )
    parser.add_argument("--notation", choices=["uci", "san"], default="san")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        zero_shot_base=args.zero_shot_base,
        split=args.split,
        variants=args.variants,
        k_values=args.k_values,
        num_draws=args.num_draws,
        controls=args.controls,
        outcome_corruption_mode=args.outcome_corruption_mode,
        trajectory_corruption_mode=args.trajectory_corruption_mode,
        notation=args.notation,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        max_roots=args.max_roots,
        n_boot=args.n_boot,
        seed=args.seed,
        model_seed=args.model_seed,
        gpu=args.gpu,
        model_path=args.model_path,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
