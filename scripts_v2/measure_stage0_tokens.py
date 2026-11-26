#!/usr/bin/env python3
"""Measure real Qwen2.5-7B-Instruct (UCI vs SAN) and Track 2 ChessV2Tokenizer token lengths
on Stage 0 sample games (`runs_v2/stage0/sample_games.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from chess_icl_v2.data.tokenizer_v2 import ChessV2Tokenizer
from chess_icl_v2.models.qwen_model import QwenPromptFormatter, load_qwen_tokenizer

STANDARD_INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _summary_stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Qwen2.5-7B & ChessV2 token lengths on Stage 0 games.")
    parser.add_argument("--games-path", type=Path, default=Path("runs_v2/stage0/sample_games.jsonl"))
    parser.add_argument("--out-path", type=Path, default=Path("runs_v2/stage0/stage0_token_stats.json"))
    parser.add_argument("--episodes-per-group", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    games: list[dict[str, Any]] = []
    with args.games_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))

    print(f"Loaded {len(games)} sample games from {args.games_path}", flush=True)

    qwen_tok = load_qwen_tokenizer(local_files_only=True)
    fmt_uci = QwenPromptFormatter(tokenizer=qwen_tok, notation="uci", max_length=10_000_000)
    fmt_san = QwenPromptFormatter(tokenizer=qwen_tok, notation="san", max_length=10_000_000)
    t2_tok = ChessV2Tokenizer()

    # 1. Single-game formatted token lengths across all games
    single_uci: dict[str, list[int]] = {}
    single_san: dict[str, list[int]] = {}
    single_t2: dict[str, list[int]] = {}
    per_var_uci: dict[str, list[int]] = {}
    per_var_san: dict[str, list[int]] = {}
    per_var_t2: dict[str, list[int]] = {}
    t2_oov_counts: dict[str, int] = {}
    t2_oov_examples: list[str] = []

    def _safe_t2_demo_len(demo: dict[str, Any]) -> int:
        # Fixed 70 state tokens + 5 structural tokens ([DEMO_START],[STATE],[MOVES],[OUT_*],[DEMO_END]) + 1 per move
        try:
            body_t2, out_t2 = t2_tok.encode_demo(demo)
            return len(body_t2) + len(out_t2)
        except ValueError:
            var_name = demo["variant"]
            for mv in demo["moves"]:
                try:
                    t2_tok.encode_move(mv)
                except ValueError:
                    t2_oov_counts[var_name] = t2_oov_counts.get(var_name, 0) + 1
                    ex = f"{var_name}:{mv}"
                    if ex not in t2_oov_examples and len(t2_oov_examples) < 20:
                        t2_oov_examples.append(ex)
            return 75 + len(demo["moves"])

    for g in games:
        pid = g["policy"]
        var = g["variant"]

        txt_uci = fmt_uci.format_demo(g)
        txt_san = fmt_san.format_demo(g)
        n_uci = len(qwen_tok.encode(txt_uci, add_special_tokens=False))
        n_san = len(qwen_tok.encode(txt_san, add_special_tokens=False))
        n_t2 = _safe_t2_demo_len(g)

        single_uci.setdefault(pid, []).append(n_uci)
        single_san.setdefault(pid, []).append(n_san)
        single_t2.setdefault(pid, []).append(n_t2)

        per_var_uci.setdefault(var, []).append(n_uci)
        per_var_san.setdefault(var, []).append(n_san)
        per_var_t2.setdefault(var, []).append(n_t2)

    # 2. Full episode token lengths for K = 1, 4, 8, 16 per policy
    rng = random.Random(args.seed)
    by_pol_var: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for g in games:
        by_pol_var.setdefault((g["policy"], g["variant"]), []).append(g)

    k_values = (1, 4, 8, 16)
    episodes_stats: dict[str, dict[str, Any]] = {}

    policies = sorted({g["policy"] for g in games})
    variants = sorted({g["variant"] for g in games})

    dummy_query = {"fen": STANDARD_INITIAL_FEN, "blocked_squares": []}

    for pid in policies:
        episodes_stats[pid] = {}
        for k in k_values:
            uci_lens: list[int] = []
            san_lens: list[int] = []
            t2_lens: list[int] = []

            for _ in range(args.episodes_per_group):
                for var in variants:
                    pool = by_pol_var[(pid, var)]
                    chosen = [rng.choice(pool) for _ in range(k)]
                    q_obj = {
                        "fen": chosen[0]["init_fen"],
                        "blocked_squares": chosen[0].get("blocked_squares", []),
                    }
                    info_uci = fmt_uci.format_prompt_with_info(chosen, q_obj)
                    info_san = fmt_san.format_prompt_with_info(chosen, q_obj)
                    uci_lens.append(int(info_uci.num_tokens))
                    san_lens.append(int(info_san.num_tokens))

                    try:
                        ep_t2 = t2_tok.encode_episode(
                            chosen,
                            query_state=q_obj["fen"],
                            max_len=100000,
                            query_blocked_squares=q_obj["blocked_squares"],
                        )
                        t2_lens.append(len(ep_t2["input_ids"]))
                    except ValueError:
                        # [CLS] + K * (75 + len(moves)) + [QUERY, STATE, 70, SEP]
                        t2_lens.append(1 + sum(75 + len(c["moves"]) for c in chosen) + 73)

            episodes_stats[pid][f"K={k}"] = {
                "qwen_uci": _summary_stats(uci_lens),
                "qwen_san": _summary_stats(san_lens),
                "track2_chessv2": _summary_stats(t2_lens),
            }

    payload = {
        "num_games": len(games),
        "single_game_by_policy": {
            pid: {
                "qwen_uci": _summary_stats(single_uci[pid]),
                "qwen_san": _summary_stats(single_san[pid]),
                "track2_chessv2": _summary_stats(single_t2[pid]),
            }
            for pid in policies
        },
        "single_game_by_variant": {
            var: {
                "qwen_uci": _summary_stats(per_var_uci[var]),
                "qwen_san": _summary_stats(per_var_san[var]),
                "track2_chessv2": _summary_stats(per_var_t2[var]),
            }
            for var in variants
        },
        "episodes_by_policy_and_K": episodes_stats,
        "track2_oov_move_counts_by_variant": t2_oov_counts,
        "track2_oov_move_examples": t2_oov_examples,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Single-Game Token Lengths (Qwen2.5-7B UCI vs SAN vs Track 2 ChessV2Tokenizer) ===")
    for pid in policies:
        u = payload["single_game_by_policy"][pid]["qwen_uci"]
        s = payload["single_game_by_policy"][pid]["qwen_san"]
        t = payload["single_game_by_policy"][pid]["track2_chessv2"]
        print(
            f"{pid:<18} | Qwen-UCI mean={u['mean']:.1f} p95={u['p95']:.0f} max={u['max']:.0f} | "
            f"Qwen-SAN mean={s['mean']:.1f} p95={s['p95']:.0f} max={s['max']:.0f} | "
            f"Track2 mean={t['mean']:.1f} p95={t['p95']:.0f} max={t['max']:.0f}"
        )

    print("\n=== Full Episode Token Lengths by K ===")
    for pid in policies:
        for k in k_values:
            st = episodes_stats[pid][f"K={k}"]
            u, s, t = st["qwen_uci"], st["qwen_san"], st["track2_chessv2"]
            print(
                f"{pid:<18} K={k:<2} | Qwen-UCI mean={u['mean']:7.1f} p95={u['p95']:7.0f} max={u['max']:7.0f} | "
                f"Qwen-SAN mean={s['mean']:7.1f} p95={s['p95']:7.0f} max={s['max']:7.0f} | "
                f"Track2 mean={t['mean']:6.1f} p95={t['p95']:6.0f} max={t['max']:6.0f}"
            )
    print(f"\nTrack2 ChessV2Tokenizer OOV move counts by variant: {t2_oov_counts}")
    print(f"Track2 ChessV2Tokenizer OOV examples: {t2_oov_examples}")
    print(f"\nSaved token statistics to {args.out_path}")


if __name__ == "__main__":
    main()
