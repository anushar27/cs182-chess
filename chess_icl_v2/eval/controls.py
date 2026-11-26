"""Episode perturbation and control builders for chess_icl_v2.

All builders return deep-copied episode lists and never mutate inputs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

from .baselines import get_demo_moves, get_demo_outcome_white, get_query_fen

VALID_CORRUPTION_MODES = ("inverted", "scrambled", "all_draw")
OUTCOME_LABELS = ("win", "draw", "loss")


def _demo_length(demo: dict[str, Any]) -> int:
    """Measure demo length by token count if present, otherwise move count."""
    for key in ("token_length", "num_tokens"):
        if key in demo and demo[key] is not None:
            return int(demo[key])
    return len(get_demo_moves(demo))


def _canonical_outcome_str(demo: dict[str, Any]) -> str:
    """Return White-perspective discrete outcome label ('win', 'draw', or 'loss')."""
    vec = get_demo_outcome_white(demo)
    idx = int(np.argmax(vec))
    return OUTCOME_LABELS[idx]


def _set_demo_outcome(demo: dict[str, Any], new_outcome: str) -> None:
    """Update outcome fields on a copied demo dict to `new_outcome` ('win'/'draw'/'loss')."""
    if new_outcome not in OUTCOME_LABELS:
        raise ValueError(f"Invalid outcome {new_outcome!r}")
    idx = OUTCOME_LABELS.index(new_outcome)
    one_hot = [1.0 if j == idx else 0.0 for j in range(3)]

    updated_any = False
    for key in ("outcome", "result", "outcome_white", "winner"):
        if key in demo and demo[key] is not None:
            demo[key] = new_outcome
            updated_any = True
    if not updated_any:
        demo["outcome"] = new_outcome

    for key in ("outcome_probs_white", "y_white"):
        if key in demo and demo[key] is not None:
            demo[key] = one_hot


def _set_demo_moves(demo: dict[str, Any], new_moves: list[str]) -> None:
    """Update move sequence on a copied demo dict while preserving original key/type."""
    target_key = "moves"
    was_str = False
    for key in ("moves", "uci_moves", "san_moves"):
        if key in demo and demo[key] is not None:
            target_key = key
            was_str = isinstance(demo[key], str)
            break

    demo[target_key] = " ".join(new_moves) if was_str else list(new_moves)
    # Invalidate cached board states/tokens so downstream evaluators re-parse corrupted moves
    for cache_key in ("fens", "states", "final_fen", "tokens"):
        demo.pop(cache_key, None)


def make_mismatched_context_episodes(
    episodes: Sequence[dict[str, Any]],
    other_variant_demos_by_variant: Mapping[str, Sequence[dict[str, Any]]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Replace each episode's demos with mismatched-rule demos matched in K and length.

    For each episode with variant `v` and `K = len(ep['demos'])`, selects a mismatched
    variant `v' != v` (preferring pools with >= K demos, falling back to pooled
    mismatched variants if necessary) and chooses `K` demos closest in move/token
    length to the original demonstrations.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []

    for ep in episodes:
        ep_copy = copy.deepcopy(ep)
        orig_demos = ep_copy.get("demos", [])
        k = len(orig_demos)
        ep_variant = ep_copy.get("variant")

        if k == 0:
            ep_copy["control"] = "mismatched_context"
            ep_copy["control_source_variant"] = None
            out.append(ep_copy)
            continue

        candidate_variants = [
            v
            for v, pool in sorted(other_variant_demos_by_variant.items())
            if v != ep_variant and len(pool) > 0
        ]
        if not candidate_variants:
            raise ValueError(
                f"No mismatched-variant demos available for episode variant {ep_variant!r}"
            )

        rich_variants = [
            v for v in candidate_variants if len(other_variant_demos_by_variant[v]) >= k
        ]
        if rich_variants:
            chosen_variant = str(rich_variants[int(rng.integers(0, len(rich_variants)))])
            pool = list(other_variant_demos_by_variant[chosen_variant])
            source_tag: Any = chosen_variant
        else:
            pool = [
                d
                for v in candidate_variants
                for d in other_variant_demos_by_variant[v]
            ]
            source_tag = "pooled_mismatched"

        pool_lengths = np.array([_demo_length(d) for d in pool], dtype=np.int64)
        orig_lengths = [_demo_length(d) for d in orig_demos]

        # Match each original demo length greedily to closest available demo in pool
        available = np.ones(len(pool), dtype=bool)
        selected_indices: list[int] = [0] * k
        order = rng.permutation(k)

        for pos in order:
            target_len = orig_lengths[int(pos)]
            avail_idx = np.flatnonzero(available)
            if len(avail_idx) == 0:
                # Only reachable if total mismatched pool < K
                avail_idx = np.arange(len(pool))
            diffs = np.abs(pool_lengths[avail_idx] - target_len)
            min_diff = int(np.min(diffs))
            tied = avail_idx[diffs == min_diff]
            picked = int(tied[int(rng.integers(0, len(tied)))])
            selected_indices[int(pos)] = picked
            if np.any(available):
                available[picked] = False

        ep_copy["demos"] = [copy.deepcopy(pool[idx]) for idx in selected_indices]
        ep_copy["control"] = "mismatched_context"
        ep_copy["control_source_variant"] = source_tag
        out.append(ep_copy)

    return out


def make_corrupted_outcome_episodes(
    episodes: Sequence[dict[str, Any]],
    mode: Literal["inverted", "scrambled", "all_draw"],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Corrupt demonstration outcomes under mode in {'inverted', 'scrambled', 'all_draw'}.

    - 'inverted': flips White win <-> loss while keeping draws unchanged.
    - 'all_draw': sets every demonstration outcome to 'draw'.
    - 'scrambled': randomly permutes demonstration outcomes across the batch (or samples
      uniformly from {'win', 'draw', 'loss'} if all original outcomes are identical).
    """
    if mode not in VALID_CORRUPTION_MODES:
        raise ValueError(
            f"Unsupported corruption mode {mode!r}; expected one of {VALID_CORRUPTION_MODES}"
        )

    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(ep) for ep in episodes]

    if mode == "inverted":
        inv_map = {"win": "loss", "loss": "win", "draw": "draw"}
        for ep in out:
            for demo in ep.get("demos", []):
                soft_backup = {
                    key: list(demo[key])
                    for key in ("outcome_probs_white", "y_white")
                    if isinstance(demo.get(key), (list, tuple, np.ndarray))
                    and len(demo[key]) == 3
                }
                cur = _canonical_outcome_str(demo)
                _set_demo_outcome(demo, inv_map[cur])
                for key, vec in soft_backup.items():
                    demo[key] = [float(vec[2]), float(vec[1]), float(vec[0])]
            ep["control"] = "corrupted_outcome_inverted"
        return out

    if mode == "all_draw":
        for ep in out:
            for demo in ep.get("demos", []):
                _set_demo_outcome(demo, "draw")
            ep["control"] = "corrupted_outcome_all_draw"
        return out

    # mode == "scrambled"
    flat_refs: list[dict[str, Any]] = []
    orig_outcomes: list[str] = []
    for ep in out:
        for demo in ep.get("demos", []):
            flat_refs.append(demo)
            orig_outcomes.append(_canonical_outcome_str(demo))

    if flat_refs:
        if len(set(orig_outcomes)) > 1:
            perm = rng.permutation(len(orig_outcomes))
            scrambled = [orig_outcomes[int(idx)] for idx in perm]
        else:
            choices = rng.integers(0, 3, size=len(orig_outcomes))
            scrambled = [OUTCOME_LABELS[int(c)] for c in choices]

        for demo, new_out in zip(flat_refs, scrambled):
            _set_demo_outcome(demo, new_out)

    for ep in out:
        ep["control"] = "corrupted_outcome_scrambled"
    return out


def make_corrupted_trajectory_episodes(
    episodes: Sequence[dict[str, Any]],
    seed: int = 42,
    mode: Literal["shuffled", "mismatched"] = "shuffled",
) -> list[dict[str, Any]]:
    """Corrupt demonstration move trajectories while preserving original outcomes and lengths.

    - 'shuffled': randomly permutes moves within each demonstration (retaining exact length
      and outcome).
    - 'mismatched': replaces each demonstration's moves with a trajectory sliced/wrapped
      from another demonstration in `episodes` (preferring a different variant if present),
      retaining exact length and original outcome.
    """
    if mode not in {"shuffled", "mismatched"}:
        raise ValueError(f"Unsupported trajectory corruption mode {mode!r}")

    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(ep) for ep in episodes]

    if mode == "mismatched":
        # Build pool of (ep_idx, variant, moves) across all non-empty demos
        pool: list[tuple[int, Any, list[str]]] = []
        for ep_idx, ep in enumerate(out):
            v = ep.get("variant")
            for demo in ep.get("demos", []):
                mvs = get_demo_moves(demo)
                if mvs:
                    pool.append((ep_idx, v, list(mvs)))

    for ep_idx, ep in enumerate(out):
        ep_var = ep.get("variant")
        for demo in ep.get("demos", []):
            orig_moves = get_demo_moves(demo)
            length = len(orig_moves)
            if length == 0:
                continue

            if mode == "shuffled":
                new_moves = list(orig_moves)
                if length >= 2 and len(set(orig_moves)) > 1:
                    for _ in range(5):
                        perm = rng.permutation(length)
                        candidate = [orig_moves[int(i)] for i in perm]
                        if candidate != orig_moves:
                            new_moves = candidate
                            break
                    else:
                        # Fallback deterministic swap of two distinct moves
                        for j in range(1, length):
                            if new_moves[j] != new_moves[0]:
                                new_moves[0], new_moves[j] = new_moves[j], new_moves[0]
                                break
                _set_demo_moves(demo, new_moves)
            else:
                # mode == "mismatched"
                diff_var = [item for item in pool if item[1] != ep_var]
                diff_ep = [item for item in pool if item[0] != ep_idx]
                candidates = diff_var or diff_ep or pool
                if candidates:
                    chosen = candidates[int(rng.integers(0, len(candidates)))][2]
                    reps = (length + len(chosen) - 1) // len(chosen)
                    new_moves = (chosen * reps)[:length]
                else:
                    new_moves = list(orig_moves)
                _set_demo_moves(demo, new_moves)

        ep["control"] = f"corrupted_trajectory_{mode}"

    return out


def _count_text_tokens(tokenizer: Any, text: str, add_special_tokens: bool = False) -> int:
    """Count tokens using HF-like, custom mock, or character fallback tokenizers."""
    if tokenizer is None:
        return len(text)
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        except TypeError:
            return len(tokenizer.encode(text))
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        if isinstance(encoded, Mapping):
            return len(encoded["input_ids"])
        return len(encoded.input_ids)
    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer).__name__}")


class _TokenLengthMeasurer:
    """Measures single-demo and full-episode context token lengths consistently."""

    def __init__(self, tokenizer: Any, notation: Literal["uci", "san"] = "san") -> None:
        self.tokenizer = tokenizer
        self.notation = notation
        self._is_chess_v2 = hasattr(tokenizer, "encode_demo") and hasattr(
            tokenizer, "encode_episode"
        )
        self._formatter: Any = None
        if not self._is_chess_v2:
            from chess_icl_v2.models.qwen_model import QwenPromptFormatter

            self._formatter = QwenPromptFormatter(
                tokenizer=tokenizer,
                notation=notation,
                max_length=None,
            )

    def demo_token_length(self, demo: Mapping[str, Any]) -> int:
        if self._is_chess_v2:
            body, out = self.tokenizer.encode_demo(demo)
            return len(body) + len(out)
        text = self._formatter.format_demo(demo)
        return _count_text_tokens(self.tokenizer, text, add_special_tokens=False)

    def context_token_length(
        self,
        demos: Sequence[Mapping[str, Any]],
        query: Mapping[str, Any],
    ) -> int:
        if self._is_chess_v2:
            enc = self.tokenizer.encode_episode(
                demos=demos,
                query_state=query,
                max_len=10**9,
            )
            return len(enc["input_ids"])
        text = self._formatter.format_prompt(demos, query)
        return _count_text_tokens(self.tokenizer, text, add_special_tokens=True)


def _within_tolerance(cand_len: int, orig_len: int, tolerance: float) -> bool:
    lo = (1.0 - tolerance) * float(orig_len) - 1e-9
    hi = (1.0 + tolerance) * float(orig_len) + 1e-9
    return lo <= float(cand_len) <= hi


def _sample_distinct_assignment(
    eligible_per_pos: Sequence[Sequence[int]],
    orig_lengths: Sequence[int],
    pool_lengths: Sequence[int],
    rng: np.random.Generator,
    greedy: bool,
) -> Optional[list[int]]:
    """Assign K distinct indices from `eligible_per_pos` using MRV ordering."""
    k = len(eligible_per_pos)
    order = sorted(range(k), key=lambda j: (len(eligible_per_pos[j]), int(rng.integers(0, 10**6))))
    used: set[int] = set()
    chosen: list[int] = [-1] * k

    for pos in order:
        avail = [idx for idx in eligible_per_pos[pos] if idx not in used]
        if not avail:
            return None
        if greedy:
            target = orig_lengths[pos]
            min_diff = min(abs(pool_lengths[idx] - target) for idx in avail)
            tied = [idx for idx in avail if abs(pool_lengths[idx] - target) == min_diff]
            pick = int(tied[int(rng.integers(0, len(tied)))])
        else:
            pick = int(avail[int(rng.integers(0, len(avail)))])
        chosen[pos] = pick
        used.add(pick)
    return chosen


def make_token_matched_foreign_episodes(
    episodes: Sequence[dict[str, Any]],
    foreign_demos_by_variant: Mapping[str, Sequence[dict[str, Any]]],
    tokenizer: Any,
    notation: Literal["uci", "san"] = "san",
    tolerance: float = 0.10,
    max_search_attempts: int = 200,
    seed: int = 42,
    allow_pooled_foreign: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build +/-10% token-length matched foreign-rule control episodes with search audit.

    For each episode of variant `v` with `K` demonstrations, searches foreign-rule games
    (`v' != v`) such that:
      1. Each matched demo `j` has token length within `+/- tolerance` of original demo `j`:
         `(1 - tolerance) * orig_tok[j] <= cand_tok[j] <= (1 + tolerance) * orig_tok[j]`.
      2. Total formatted episode context token length is also within `+/- tolerance` of the
         original episode's formatted context token length.

    Bounded search failure is NEVER treated as proof that no match exists; every episode's
    `match_found` boolean and `match_failure_reason` are recorded in the returned search
    report, and only matched episodes are included in the returned feasible cohort.

    Returns:
        Tuple `(matched_episodes, search_report)`.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    if max_search_attempts < 1:
        raise ValueError(f"max_search_attempts must be >= 1, got {max_search_attempts}")

    rng = np.random.default_rng(seed)
    measurer = _TokenLengthMeasurer(tokenizer=tokenizer, notation=notation)

    # Precompute foreign demo token lengths once across the pool
    pool_by_variant: dict[str, list[dict[str, Any]]] = {}
    pool_lengths_by_variant: dict[str, list[int]] = {}
    for v, demos in sorted(foreign_demos_by_variant.items()):
        demo_list = list(demos)
        pool_by_variant[str(v)] = demo_list
        pool_lengths_by_variant[str(v)] = [measurer.demo_token_length(d) for d in demo_list]

    matched_episodes: list[dict[str, Any]] = []
    per_episode_status: list[dict[str, Any]] = []
    failure_reason_counts: dict[str, int] = {}

    for ep_idx, ep in enumerate(episodes):
        ep_variant = str(ep.get("variant"))
        root_id = ep.get("root_id", ep_idx)
        orig_demos = ep.get("demos", [])
        k = len(orig_demos)
        query_obj = {
            "fen": get_query_fen(ep),
            "blocked_squares": ep.get("blocked_squares"),
        }
        orig_demo_tokens = [measurer.demo_token_length(d) for d in orig_demos]
        orig_ctx_tokens = measurer.context_token_length(orig_demos, query_obj)

        if k == 0:
            ep_copy = copy.deepcopy(ep)
            ep_copy["control"] = "token_matched_foreign"
            ep_copy["control_source_variant"] = None
            ep_copy["match_found"] = True
            ep_copy["match_failure_reason"] = None
            ep_copy["orig_demo_tokens"] = []
            ep_copy["matched_demo_tokens"] = []
            ep_copy["orig_context_tokens"] = int(orig_ctx_tokens)
            ep_copy["matched_context_tokens"] = int(orig_ctx_tokens)
            ep_copy["context_token_rel_diff"] = 0.0
            matched_episodes.append(ep_copy)
            per_episode_status.append(
                {
                    "root_id": root_id,
                    "variant": ep_variant,
                    "k": 0,
                    "match_found": True,
                    "match_failure_reason": None,
                    "attempts_used": 0,
                    "control_source_variant": None,
                    "orig_context_tokens": int(orig_ctx_tokens),
                    "matched_context_tokens": int(orig_ctx_tokens),
                }
            )
            continue

        foreign_variants = [
            v for v, p in pool_by_variant.items() if v != ep_variant and len(p) > 0
        ]
        if not foreign_variants:
            reason = "no_foreign_variants_available"
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
            per_episode_status.append(
                {
                    "root_id": root_id,
                    "variant": ep_variant,
                    "k": int(k),
                    "match_found": False,
                    "match_failure_reason": reason,
                    "attempts_used": 0,
                    "control_source_variant": None,
                    "orig_context_tokens": int(orig_ctx_tokens),
                    "matched_context_tokens": None,
                }
            )
            continue

        # Identify single foreign variants with eligible candidates for all K positions
        qualifying_pools: list[tuple[str, list[dict[str, Any]], list[int], list[list[int]]]] = []
        shuffled_variants = [
            foreign_variants[int(i)] for i in rng.permutation(len(foreign_variants))
        ]
        for fv in shuffled_variants:
            f_pool = pool_by_variant[fv]
            f_lens = pool_lengths_by_variant[fv]
            if len(f_pool) < k:
                continue
            elig = [
                [idx for idx, clen in enumerate(f_lens) if _within_tolerance(clen, tlen, tolerance)]
                for tlen in orig_demo_tokens
            ]
            if all(len(e) > 0 for e in elig) and len({i for e in elig for i in e}) >= k:
                qualifying_pools.append((fv, f_pool, f_lens, elig))

        if allow_pooled_foreign:
            pooled_demos: list[dict[str, Any]] = []
            pooled_lens: list[int] = []
            for fv in shuffled_variants:
                pooled_demos.extend(pool_by_variant[fv])
                pooled_lens.extend(pool_lengths_by_variant[fv])
            if len(pooled_demos) >= k:
                pooled_elig = [
                    [
                        idx
                        for idx, clen in enumerate(pooled_lens)
                        if _within_tolerance(clen, tlen, tolerance)
                    ]
                    for tlen in orig_demo_tokens
                ]
                if (
                    all(len(e) > 0 for e in pooled_elig)
                    and len({i for e in pooled_elig for i in e}) >= k
                ):
                    qualifying_pools.append(
                        ("pooled_foreign", pooled_demos, pooled_lens, pooled_elig)
                    )

        if not qualifying_pools:
            reason = "insufficient_eligible_foreign_demos"
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
            per_episode_status.append(
                {
                    "root_id": root_id,
                    "variant": ep_variant,
                    "k": int(k),
                    "match_found": False,
                    "match_failure_reason": reason,
                    "attempts_used": 0,
                    "control_source_variant": None,
                    "orig_context_tokens": int(orig_ctx_tokens),
                    "matched_context_tokens": None,
                }
            )
            continue

        attempts_used = 0
        found = False
        chosen_source: Optional[str] = None
        chosen_demos: list[dict[str, Any]] = []
        chosen_demo_tokens: list[int] = []
        chosen_ctx_tokens: Optional[int] = None

        # Round 1: greedy closest-match attempt across qualifying pools within budget
        for src_tag, c_pool, c_lens, c_elig in qualifying_pools:
            if attempts_used >= max_search_attempts:
                break
            attempts_used += 1
            cand_idx = _sample_distinct_assignment(
                c_elig, orig_demo_tokens, c_lens, rng=rng, greedy=True
            )
            if cand_idx is None:
                continue
            cand_demos = [c_pool[i] for i in cand_idx]
            cand_ctx_tok = measurer.context_token_length(cand_demos, query_obj)
            if _within_tolerance(cand_ctx_tok, orig_ctx_tokens, tolerance):
                found = True
                chosen_source = src_tag
                chosen_demos = cand_demos
                chosen_demo_tokens = [c_lens[i] for i in cand_idx]
                chosen_ctx_tokens = int(cand_ctx_tok)
                break

        # Round 2: randomized distinct draws up to max_search_attempts
        while not found and attempts_used < max_search_attempts:
            pool_choice = qualifying_pools[int(rng.integers(0, len(qualifying_pools)))]
            src_tag, c_pool, c_lens, c_elig = pool_choice
            attempts_used += 1
            cand_idx = _sample_distinct_assignment(
                c_elig, orig_demo_tokens, c_lens, rng=rng, greedy=False
            )
            if cand_idx is None:
                continue
            cand_demos = [c_pool[i] for i in cand_idx]
            cand_ctx_tok = measurer.context_token_length(cand_demos, query_obj)
            if _within_tolerance(cand_ctx_tok, orig_ctx_tokens, tolerance):
                found = True
                chosen_source = src_tag
                chosen_demos = cand_demos
                chosen_demo_tokens = [c_lens[i] for i in cand_idx]
                chosen_ctx_tokens = int(cand_ctx_tok)
                break

        if found and chosen_ctx_tokens is not None:
            ep_copy = copy.deepcopy(ep)
            ep_copy["demos"] = [copy.deepcopy(d) for d in chosen_demos]
            ep_copy["control"] = "token_matched_foreign"
            ep_copy["control_source_variant"] = chosen_source
            ep_copy["match_found"] = True
            ep_copy["match_failure_reason"] = None
            ep_copy["orig_demo_tokens"] = [int(x) for x in orig_demo_tokens]
            ep_copy["matched_demo_tokens"] = [int(x) for x in chosen_demo_tokens]
            ep_copy["orig_context_tokens"] = int(orig_ctx_tokens)
            ep_copy["matched_context_tokens"] = int(chosen_ctx_tokens)
            ep_copy["context_token_rel_diff"] = float(
                abs(chosen_ctx_tokens - orig_ctx_tokens) / max(1, orig_ctx_tokens)
            )
            matched_episodes.append(ep_copy)
            per_episode_status.append(
                {
                    "root_id": root_id,
                    "variant": ep_variant,
                    "k": int(k),
                    "match_found": True,
                    "match_failure_reason": None,
                    "attempts_used": int(attempts_used),
                    "control_source_variant": chosen_source,
                    "orig_context_tokens": int(orig_ctx_tokens),
                    "matched_context_tokens": int(chosen_ctx_tokens),
                }
            )
        else:
            reason = "bounded_search_exhausted_without_match"
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
            per_episode_status.append(
                {
                    "root_id": root_id,
                    "variant": ep_variant,
                    "k": int(k),
                    "match_found": False,
                    "match_failure_reason": reason,
                    "attempts_used": int(attempts_used),
                    "control_source_variant": None,
                    "orig_context_tokens": int(orig_ctx_tokens),
                    "matched_context_tokens": None,
                }
            )

    n_total = len(episodes)
    n_matched = len(matched_episodes)
    n_failed = n_total - n_matched
    search_report = {
        "n_total": int(n_total),
        "n_matched": int(n_matched),
        "n_failed": int(n_failed),
        "failure_rate": float(n_failed / n_total) if n_total > 0 else 0.0,
        "tolerance": float(tolerance),
        "max_search_attempts": int(max_search_attempts),
        "failure_reason_counts": failure_reason_counts,
        "per_episode_status": per_episode_status,
        "note": (
            "Bounded matching search failure does not prove no match exists; "
            "only indicates whether a match was found within max_search_attempts."
        ),
    }
    return matched_episodes, search_report
