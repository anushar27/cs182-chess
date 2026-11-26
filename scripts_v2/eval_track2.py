"""Evaluate a trained Track 2 ChessTransformerV2 checkpoint with hierarchical uncertainty & controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_icl_v2.data.dataset_v2 import (
    EvalEpisodeBuilder,
    collate_episode_batch,
    discover_variants,
)
from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer
from chess_icl_v2.eval.controls import (
    make_corrupted_outcome_episodes,
    make_corrupted_trajectory_episodes,
    make_mismatched_context_episodes,
)
from chess_icl_v2.eval.metrics import (
    compute_argmax_accuracy,
    compute_brier_per_sample,
    compute_ece,
    compute_nll,
    compute_side_calibration,
)
from chess_icl_v2.eval.uncertainty import aggregate_evaluation_records
from chess_icl_v2.models.small_transformer import ChessTransformerV2


class Track2Predictor:
    """Unified Predictor wrapper around ChessTransformerV2 + ChessV2Tokenizer."""

    def __init__(
        self,
        model: ChessTransformerV2,
        tokenizer: Optional[ChessV2Tokenizer] = None,
        device: Optional[torch.device] = None,
        max_seq_len: Optional[int] = None,
        batch_size: int = 32,
        use_bf16: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or ChessV2Tokenizer()
        self.device = device or next(model.parameters()).device
        self.max_seq_len = int(max_seq_len or model.config.max_seq_len)
        self.batch_size = int(batch_size)
        self.use_bf16 = bool(
            use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, episodes: list[dict[str, Any]]) -> np.ndarray:
        """Predict [p_win, p_draw, p_loss] probabilities of shape (N, 3)."""
        if not episodes:
            return np.empty((0, 3), dtype=np.float64)

        encoded_items = []
        for ep in episodes:
            enc = self.tokenizer.encode_episode(
                demos=ep.get("demos", []),
                query_state={
                    "fen": ep["query_fen"],
                    "blocked_squares": ep.get("blocked_squares"),
                },
                max_len=self.max_seq_len,
            )
            encoded_items.append(
                (
                    enc["input_ids"],
                    enc["segment_ids"],
                    ep.get("y_true", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                )
            )

        chunks: list[np.ndarray] = []
        for start in range(0, len(encoded_items), self.batch_size):
            batch = collate_episode_batch(
                encoded_items[start : start + self.batch_size],
                pad_id=self.tokenizer.pad_id,
                pad_segment_id=-1,
            )
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            segment_ids = batch["segment_ids"].to(self.device)

            if self.use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    probs = self.model.predict_proba(
                        input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
                    )
            else:
                probs = self.model.predict_proba(
                    input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
            chunks.append(probs.float().cpu().numpy().astype(np.float64))

        return np.vstack(chunks)


def _score_episodes(
    predictor: Track2Predictor,
    episodes: list[dict[str, Any]],
    condition: str,
    k: int,
    draw_idx: int,
    draw_seed: int,
    model_seed: int,
    nll_k0_lookup: Optional[dict[tuple[str, Any], float]] = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Run predictor on episodes and build per-root metric records."""
    p_pred = predictor.predict_proba(episodes)
    y_true = np.asarray([ep["y_true"] for ep in episodes], dtype=np.float64)

    _, per_sample_nll = compute_nll(p_pred, y_true)
    per_sample_brier = compute_brier_per_sample(p_pred, y_true)
    pred_cls = np.argmax(p_pred, axis=-1)
    true_cls = np.argmax(y_true, axis=-1)

    records: list[dict[str, Any]] = []
    for idx, ep in enumerate(episodes):
        v = str(ep["variant"])
        rid = ep["root_id"]
        nll_val = float(per_sample_nll[idx])
        rec: dict[str, Any] = {
            "condition": condition,
            "variant": v,
            "k": int(k),
            "demo_draw": int(draw_idx),
            "draw_seed": int(draw_seed),
            "model_seed": int(model_seed),
            "root_id": rid,
            "source_game_id": ep.get("source_game_id", rid),
            "side_to_move": ep.get("side_to_move", "w"),
            "ply": int(ep.get("ply", 0)),
            "equal_material": bool(ep.get("equal_material", False)),
            "p_pred": [float(x) for x in p_pred[idx]],
            "y_true": [float(x) for x in y_true[idx]],
            "nll": nll_val,
            "brier": float(per_sample_brier[idx]),
            "correct": float(1.0 if pred_cls[idx] == true_cls[idx] else 0.0),
        }
        if nll_k0_lookup is not None and (v, rid) in nll_k0_lookup:
            rec["context_gain"] = float(nll_k0_lookup[(v, rid)] - nll_val)
        records.append(rec)

    return records, p_pred, y_true


def _summarize_condition_k(
    records: list[dict[str, Any]],
    p_preds: list[np.ndarray],
    y_trues: list[np.ndarray],
    sides: list[str],
    n_boot: int,
    seed: int,
    has_gain: bool,
) -> dict[str, Any]:
    """Compute hierarchical bootstrap CIs and calibration summaries for one (condition, K)."""
    p_all = np.vstack(p_preds) if p_preds else np.empty((0, 3), dtype=np.float64)
    y_all = np.vstack(y_trues) if y_trues else np.empty((0, 3), dtype=np.float64)

    summary: dict[str, Any] = {
        "n_records": len(records),
        "nll": aggregate_evaluation_records(records, value_key="nll", n_boot=n_boot, seed=seed),
        "brier": aggregate_evaluation_records(
            records, value_key="brier", n_boot=n_boot, seed=seed + 100
        ),
        "accuracy": aggregate_evaluation_records(
            records, value_key="correct", n_boot=n_boot, seed=seed + 200
        ),
        "ece": compute_ece(p_all, y_all),
        "argmax_summary": compute_argmax_accuracy(p_all, y_all),
        "side_calibration": compute_side_calibration(p_all, y_all, sides),
    }
    if has_gain and all("context_gain" in r for r in records):
        summary["context_gain"] = aggregate_evaluation_records(
            records, value_key="context_gain", n_boot=n_boot, seed=seed + 300
        )
    return summary


def resolve_device(gpu: Optional[int]) -> torch.device:
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if gpu is None and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Track 2 ChessTransformerV2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["mismatched_context", "corrupted_outcomes", "corrupted_trajectories"],
    )
    parser.add_argument(
        "--outcome-corruption-mode",
        choices=["inverted", "scrambled", "all_draw"],
        default="inverted",
    )
    parser.add_argument("--control-k", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.gpu)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ChessV2Tokenizer()
    model = ChessTransformerV2.load(args.checkpoint, map_location="cpu")
    predictor = Track2Predictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        use_bf16=args.bf16,
    )

    variants = (
        list(args.variants)
        if args.variants is not None
        else discover_variants(data_dir, split=args.split)
    )
    if not variants:
        raise ValueError(f"No variants found under {data_dir} for split={args.split!r}")

    k_values = sorted({int(k) for k in args.k_values})
    builder = EvalEpisodeBuilder(data_dir)

    # Preload demo pools across all available variants for mismatched_context controls
    all_available_variants = sorted(set(variants) | set(discover_variants(data_dir, split=args.split)))
    demo_pools = {v: builder.get_demo_pool(v) for v in all_available_variants}

    controls = [c for c in (args.controls or []) if c != "none"]
    all_records: list[dict[str, Any]] = []
    main_results_by_k: dict[str, Any] = {}
    equal_mat_results_by_k: dict[str, Any] = {}
    control_results: dict[str, dict[str, Any]] = {c: {} for c in controls}
    skipped_controls: dict[str, str] = {}

    # 1. Evaluate K=0 baseline first so identical-root Context Gain is available for all K>0
    nll_k0_lookup: dict[tuple[str, Any], float] = {}
    k0_episodes_by_variant: dict[str, list[dict[str, Any]]] = {}

    if 0 in k_values:
        k0_records: list[dict[str, Any]] = []
        k0_preds: list[np.ndarray] = []
        k0_trues: list[np.ndarray] = []
        k0_sides: list[str] = []

        for v in variants:
            eps, _ = builder.build(
                variant=v,
                split=args.split,
                k=0,
                draw_seed=args.seed,
                max_roots=args.max_roots,
            )
            k0_episodes_by_variant[v] = eps
            base_recs, p_pred, y_true = _score_episodes(
                predictor=predictor,
                episodes=eps,
                condition="main",
                k=0,
                draw_idx=0,
                draw_seed=args.seed,
                model_seed=args.model_seed,
            )
            for rec in base_recs:
                nll_k0_lookup[(v, rec["root_id"])] = rec["nll"]

            # Replicate across n_draws deterministically (K=0 has zero demos)
            for d_idx in range(args.n_draws):
                draw_seed = args.seed + d_idx
                for rec in base_recs:
                    r_copy = dict(rec)
                    r_copy["demo_draw"] = d_idx
                    r_copy["draw_seed"] = draw_seed
                    k0_records.append(r_copy)
                k0_preds.append(p_pred)
                k0_trues.append(y_true)
                k0_sides.extend([str(ep.get("side_to_move", "w")) for ep in eps])

        all_records.extend(k0_records)
        main_results_by_k["0"] = _summarize_condition_k(
            k0_records, k0_preds, k0_trues, k0_sides, n_boot=args.n_boot, seed=args.seed, has_gain=False
        )

        eq_k0_recs = [r for r in k0_records if r["equal_material"]]
        if eq_k0_recs:
            eq_mask = np.array([r["equal_material"] for r in k0_records], dtype=bool)
            p_k0_all = np.vstack(k0_preds)[eq_mask]
            y_k0_all = np.vstack(k0_trues)[eq_mask]
            s_k0_all = [s for s, m in zip(k0_sides, eq_mask) if m]
            equal_mat_results_by_k["0"] = _summarize_condition_k(
                eq_k0_recs, [p_k0_all], [y_k0_all], s_k0_all, n_boot=args.n_boot, seed=args.seed, has_gain=False
            )

    # 2. Evaluate all K > 0 (and controls)
    control_k_targets = (
        [args.control_k]
        if args.control_k is not None
        else [k for k in k_values if k > 0]
    )

    for k in k_values:
        if k == 0:
            continue

        k_records: list[dict[str, Any]] = []
        k_preds: list[np.ndarray] = []
        k_trues: list[np.ndarray] = []
        k_sides: list[str] = []

        ctrl_buffers: dict[str, dict[str, list[Any]]] = {
            c: {"records": [], "preds": [], "trues": [], "sides": []}
            for c in controls
        }

        for d_idx in range(args.n_draws):
            draw_seed = args.seed + d_idx
            draw_episodes_all_variants: list[dict[str, Any]] = []

            for v in variants:
                eps, _ = builder.build(
                    variant=v,
                    split=args.split,
                    k=k,
                    draw_seed=draw_seed,
                    max_roots=args.max_roots,
                )
                draw_episodes_all_variants.extend(eps)
                recs, p_pred, y_true = _score_episodes(
                    predictor=predictor,
                    episodes=eps,
                    condition="main",
                    k=k,
                    draw_idx=d_idx,
                    draw_seed=draw_seed,
                    model_seed=args.model_seed,
                    nll_k0_lookup=nll_k0_lookup if nll_k0_lookup else None,
                )
                k_records.extend(recs)
                k_preds.append(p_pred)
                k_trues.append(y_true)
                k_sides.extend([str(ep.get("side_to_move", "w")) for ep in eps])

            if k in control_k_targets:
                for ctrl in controls:
                    if ctrl == "mismatched_context":
                        has_other = all(
                            any(ov != v and len(demo_pools[ov]) > 0 for ov in demo_pools)
                            for v in variants
                        )
                        if not has_other:
                            skipped_controls["mismatched_context"] = (
                                "Requires >= 2 distinct variants with demo pools"
                            )
                            continue
                        ctrl_eps = make_mismatched_context_episodes(
                            draw_episodes_all_variants,
                            other_variant_demos_by_variant=demo_pools,
                            seed=draw_seed + 1000,
                        )
                    elif ctrl == "corrupted_outcomes":
                        ctrl_eps = make_corrupted_outcome_episodes(
                            draw_episodes_all_variants,
                            mode=args.outcome_corruption_mode,
                            seed=draw_seed + 2000,
                        )
                    elif ctrl == "corrupted_trajectories":
                        ctrl_eps = make_corrupted_trajectory_episodes(
                            draw_episodes_all_variants,
                            seed=draw_seed + 3000,
                            mode="shuffled",
                        )
                    else:
                        raise ValueError(f"Unknown control: {ctrl!r}")

                    c_recs, c_pred, c_true = _score_episodes(
                        predictor=predictor,
                        episodes=ctrl_eps,
                        condition=ctrl,
                        k=k,
                        draw_idx=d_idx,
                        draw_seed=draw_seed,
                        model_seed=args.model_seed,
                        nll_k0_lookup=nll_k0_lookup if nll_k0_lookup else None,
                    )
                    ctrl_buffers[ctrl]["records"].extend(c_recs)
                    ctrl_buffers[ctrl]["preds"].append(c_pred)
                    ctrl_buffers[ctrl]["trues"].append(c_true)
                    ctrl_buffers[ctrl]["sides"].extend(
                        [str(ep.get("side_to_move", "w")) for ep in ctrl_eps]
                    )

        all_records.extend(k_records)
        main_results_by_k[str(k)] = _summarize_condition_k(
            k_records,
            k_preds,
            k_trues,
            k_sides,
            n_boot=args.n_boot,
            seed=args.seed + k,
            has_gain=bool(nll_k0_lookup),
        )

        # Equal-material slice
        eq_mask = np.array([r["equal_material"] for r in k_records], dtype=bool)
        if np.any(eq_mask):
            eq_recs = [r for r, m in zip(k_records, eq_mask) if m]
            p_eq = np.vstack(k_preds)[eq_mask]
            y_eq = np.vstack(k_trues)[eq_mask]
            s_eq = [s for s, m in zip(k_sides, eq_mask) if m]
            equal_mat_results_by_k[str(k)] = _summarize_condition_k(
                eq_recs,
                [p_eq],
                [y_eq],
                s_eq,
                n_boot=args.n_boot,
                seed=args.seed + k + 500,
                has_gain=bool(nll_k0_lookup),
            )

        if k in control_k_targets:
            for ctrl in controls:
                buf = ctrl_buffers[ctrl]
                if not buf["records"]:
                    continue
                all_records.extend(buf["records"])
                control_results[ctrl][str(k)] = _summarize_condition_k(
                    buf["records"],
                    buf["preds"],
                    buf["trues"],
                    buf["sides"],
                    n_boot=args.n_boot,
                    seed=args.seed + k + 700,
                    has_gain=bool(nll_k0_lookup),
                )

    # Remove empty control dicts
    control_results = {k_name: v for k_name, v in control_results.items() if v}

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "variants": variants,
        "k_values": k_values,
        "n_draws": int(args.n_draws),
        "n_boot": int(args.n_boot),
        "model_config": {
            "d_model": model.config.d_model,
            "n_heads": model.config.n_heads,
            "n_layers": model.config.n_layers,
            "d_ff": model.config.d_ff,
            "use_mup": model.config.use_mup,
            "attention_mode": model.config.attention_mode,
            "param_count": model.count_params(),
        },
        "main_by_k": main_results_by_k,
        "equal_material_by_k": equal_mat_results_by_k,
        "controls_by_k": control_results,
        "skipped_controls": skipped_controls,
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return evaluate(args)


if __name__ == "__main__":
    main()
