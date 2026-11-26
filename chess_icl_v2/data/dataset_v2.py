"""Dataset, deterministic eval episode builder, and batch collation for Track 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer_v2 import ChessV2Tokenizer, STANDARD_STARTING_FEN


def load_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stable_seed(*parts: Any) -> int:
    """Derive a deterministic 64-bit integer seed from arbitrary hashable parts."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def normalize_demo_record(demo: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw games_demo.jsonl record for both tokenizer_v2 and eval suites."""
    out = dict(demo)
    init_fen = (
        out.get("init_fen")
        or out.get("initial_fen")
        or out.get("start_fen")
        or out.get("initial_state")
        or STANDARD_STARTING_FEN
    )
    out["init_fen"] = init_fen
    out["initial_fen"] = init_fen
    if "outcome" not in out and "result" in out:
        out["outcome"] = out["result"]
    if "result" not in out and "outcome" in out:
        out["result"] = out["outcome"]
    if "moves" not in out:
        out["moves"] = []
    return out


def discover_variants(data_dir: Union[str, Path], split: str = "train") -> list[str]:
    """Discover variant subdirectories under data_dir containing games_demo and positions."""
    root = Path(data_dir)
    if not root.exists():
        return []
    found: list[str] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "games_demo.jsonl").exists() and (sub / f"positions_{split}.jsonl").exists():
            found.append(sub.name)
    return found


class EpisodeDatasetV2(Dataset):
    """PyTorch Dataset yielding (input_ids, segment_ids, label_probs) for Track 2 training.

    Modes:
        - ``standard_warmup``: samples standard-chess query roots and standard-chess
          demonstrations across ``k_values`` with ``k_probs``.
        - ``variant_mixture``: samples uniformly across ``variants``, then samples a query
          root, context size ``K``, and ``K`` distinct demonstrations from that variant.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        mode: Literal["standard_warmup", "variant_mixture"] = "variant_mixture",
        variants: Optional[Sequence[str]] = None,
        k_values: Sequence[int] = (0, 1, 2, 4, 8, 16),
        k_probs: Optional[Sequence[float]] = None,
        tokenizer: Optional[ChessV2Tokenizer] = None,
        max_seq_len: int = 8192,
        seed: int = 42,
        length: Optional[int] = None,
        standard_variant_name: str = "standard",
    ) -> None:
        super().__init__()
        if mode not in ("standard_warmup", "variant_mixture"):
            raise ValueError(f"Unsupported mode {mode!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode
        self.tokenizer = tokenizer or ChessV2Tokenizer()
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self._epoch = 0

        if mode == "standard_warmup":
            chosen_variants = [standard_variant_name]
        else:
            chosen_variants = (
                list(variants) if variants is not None else discover_variants(self.data_dir, split)
            )
            if not chosen_variants:
                raise ValueError(
                    f"No variants found under {self.data_dir} for split={split!r}"
                )

        self.variants: list[str] = chosen_variants
        self.k_values: list[int] = [int(k) for k in k_values]
        if not self.k_values:
            raise ValueError("k_values cannot be empty")

        if k_probs is None:
            probs = np.full(len(self.k_values), 1.0 / len(self.k_values), dtype=np.float64)
        else:
            probs = np.asarray(k_probs, dtype=np.float64)
            if probs.shape != (len(self.k_values),):
                raise ValueError(
                    f"k_probs shape {probs.shape} must match k_values ({len(self.k_values)},)"
                )
            if np.any(probs < 0) or probs.sum() <= 0:
                raise ValueError("k_probs must be non-negative and sum to > 0")
            probs = probs / probs.sum()
        self.k_probs: np.ndarray = probs

        self.demos_by_variant: dict[str, list[dict[str, Any]]] = {}
        self.positions_by_variant: dict[str, list[dict[str, Any]]] = {}
        total_positions = 0

        for v in self.variants:
            v_dir = self.data_dir / v
            raw_demos = load_jsonl(v_dir / "games_demo.jsonl")
            raw_positions = load_jsonl(v_dir / f"positions_{split}.jsonl")
            if not raw_positions:
                raise ValueError(f"Empty positions_{split}.jsonl for variant {v!r}")
            self.demos_by_variant[v] = [normalize_demo_record(d) for d in raw_demos]
            self.positions_by_variant[v] = raw_positions
            total_positions += len(raw_positions)

        self._length = int(length) if length is not None else total_positions

    def set_epoch(self, epoch: int) -> None:
        """Advance epoch counter so sampling varies across epochs deterministically."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(_stable_seed(self.seed, self._epoch, idx))

        v_idx = int(rng.integers(0, len(self.variants)))
        variant = self.variants[v_idx]

        pos_pool = self.positions_by_variant[variant]
        pos = pos_pool[int(rng.integers(0, len(pos_pool)))]

        k = int(rng.choice(self.k_values, p=self.k_probs))
        demo_pool = self.demos_by_variant[variant]
        k_eff = min(k, len(demo_pool))

        if k_eff > 0:
            src_gid = pos.get("source_game_id")
            candidate_indices = [
                i for i, d in enumerate(demo_pool) if d.get("game_id") != src_gid
            ]
            if len(candidate_indices) < k_eff:
                candidate_indices = list(range(len(demo_pool)))
            chosen = rng.choice(candidate_indices, size=k_eff, replace=False)
            demos = [demo_pool[int(i)] for i in chosen]
        else:
            demos = []

        query_state = {
            "fen": pos["fen"],
            "blocked_squares": pos.get("blocked_squares"),
        }
        ep = self.tokenizer.encode_episode(
            demos=demos,
            query_state=query_state,
            max_len=self.max_seq_len,
        )

        input_ids = torch.tensor(ep["input_ids"], dtype=torch.long)
        segment_ids = torch.tensor(ep["segment_ids"], dtype=torch.long)
        label_probs = torch.tensor(pos["label_probs"], dtype=torch.float32)
        return input_ids, segment_ids, label_probs


class EvalEpisodeBuilder:
    """Deterministic builder for evaluation episodes over identical query roots.

    Produces episode dicts compatible with both ``ChessV2Tokenizer`` and
    ``chess_icl_v2.eval.{baselines,controls,metrics,uncertainty}``.
    """

    def __init__(self, data_dir: Union[str, Path]) -> None:
        self.data_dir = Path(data_dir)
        self._demos_cache: dict[str, list[dict[str, Any]]] = {}
        self._positions_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def get_demo_pool(self, variant: str) -> list[dict[str, Any]]:
        if variant not in self._demos_cache:
            raw = load_jsonl(self.data_dir / variant / "games_demo.jsonl")
            self._demos_cache[variant] = [normalize_demo_record(d) for d in raw]
        return self._demos_cache[variant]

    def get_positions(self, variant: str, split: str) -> list[dict[str, Any]]:
        key = (variant, split)
        if key not in self._positions_cache:
            self._positions_cache[key] = load_jsonl(
                self.data_dir / variant / f"positions_{split}.jsonl"
            )
        return self._positions_cache[key]

    def _try_load_precomputed(
        self,
        variant: str,
        split: str,
        k: int,
        draw_seed: int,
    ) -> Optional[list[dict[str, Any]]]:
        """Load precomputed eval_episodes_<K>.jsonl if present and matching split/draw."""
        path = self.data_dir / variant / f"eval_episodes_{k}.jsonl"
        if not path.exists():
            return None
        rows = load_jsonl(path)
        filtered = [
            r
            for r in rows
            if r.get("split", split) == split
            and int(r.get("draw_seed", draw_seed)) == int(draw_seed)
        ]
        if not filtered:
            return None
        out: list[dict[str, Any]] = []
        for r in filtered:
            ep = dict(r)
            ep["demos"] = [normalize_demo_record(d) for d in ep.get("demos", [])]
            if "query_fen" not in ep and "fen" in ep:
                ep["query_fen"] = ep["fen"]
            if "y_true" not in ep and "label_probs" in ep:
                ep["y_true"] = ep["label_probs"]
            out.append(ep)
        return out

    def build(
        self,
        variant: str,
        split: str,
        k: int,
        draw_seed: int = 0,
        max_roots: Optional[int] = None,
        prefer_precomputed: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build `(episodes, query_meta)` deterministically for `(variant, split, k, draw_seed)`."""
        if prefer_precomputed:
            pre = self._try_load_precomputed(variant, split, k, draw_seed)
            if pre is not None:
                if max_roots is not None:
                    pre = pre[:max_roots]
                meta = [
                    {
                        "root_id": ep["root_id"],
                        "variant": ep.get("variant", variant),
                        "split": split,
                        "query_fen": ep["query_fen"],
                        "side_to_move": ep.get("side_to_move"),
                        "y_true": ep["y_true"],
                        "source_game_id": ep.get("source_game_id", ep["root_id"]),
                        "ply": ep.get("ply", 0),
                        "material_diff_white": ep.get("material_diff_white", 0),
                        "equal_material": bool(ep.get("equal_material", False)),
                        "blocked_squares": ep.get("blocked_squares"),
                    }
                    for ep in pre
                ]
                return pre, meta

        positions = self.get_positions(variant, split)
        if max_roots is not None:
            positions = positions[:max_roots]
        demo_pool = self.get_demo_pool(variant)

        rng = np.random.default_rng(_stable_seed(variant, split, k, draw_seed))
        episodes: list[dict[str, Any]] = []
        query_meta: list[dict[str, Any]] = []

        for pos in positions:
            k_eff = min(int(k), len(demo_pool))
            if k_eff > 0:
                src_gid = pos.get("source_game_id")
                cand = [i for i, d in enumerate(demo_pool) if d.get("game_id") != src_gid]
                if len(cand) < k_eff:
                    cand = list(range(len(demo_pool)))
                chosen_idx = rng.choice(cand, size=k_eff, replace=False)
                demos = [dict(demo_pool[int(i)]) for i in chosen_idx]
            else:
                demos = []

            ep = {
                "root_id": pos["root_id"],
                "variant": pos.get("variant", variant),
                "split": split,
                "k": int(k),
                "draw_seed": int(draw_seed),
                "demos": demos,
                "query_fen": pos["fen"],
                "blocked_squares": pos.get("blocked_squares"),
                "side_to_move": pos.get("side_to_move"),
                "y_true": pos["label_probs"],
                "source_game_id": pos.get("source_game_id", pos["root_id"]),
                "ply": pos.get("ply", 0),
                "material_diff_white": pos.get("material_diff_white", 0),
                "equal_material": bool(pos.get("equal_material", False)),
            }
            meta = {
                "root_id": ep["root_id"],
                "variant": ep["variant"],
                "split": split,
                "query_fen": ep["query_fen"],
                "side_to_move": ep["side_to_move"],
                "y_true": ep["y_true"],
                "source_game_id": ep["source_game_id"],
                "ply": ep["ply"],
                "material_diff_white": ep["material_diff_white"],
                "equal_material": ep["equal_material"],
                "blocked_squares": ep["blocked_squares"],
            }
            episodes.append(ep)
            query_meta.append(meta)

        return episodes, query_meta


def collate_episode_batch(
    batch: Sequence[Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], Mapping[str, Any]]],
    pad_id: int = 0,
    pad_segment_id: int = -1,
) -> dict[str, torch.Tensor]:
    """Pad a batch of (input_ids, segment_ids, label_probs) items into tensors."""
    if not batch:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
            "segment_ids": torch.empty((0, 0), dtype=torch.long),
            "label_probs": torch.empty((0, 3), dtype=torch.float32),
        }

    ids_list: list[torch.Tensor] = []
    segs_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for item in batch:
        if isinstance(item, Mapping):
            ids_t = torch.as_tensor(item["input_ids"], dtype=torch.long)
            segs_t = torch.as_tensor(item["segment_ids"], dtype=torch.long)
            lbl_t = torch.as_tensor(item["label_probs"], dtype=torch.float32)
        else:
            ids_t = torch.as_tensor(item[0], dtype=torch.long)
            segs_t = torch.as_tensor(item[1], dtype=torch.long)
            lbl_t = torch.as_tensor(item[2], dtype=torch.float32)
        ids_list.append(ids_t)
        segs_list.append(segs_t)
        labels_list.append(lbl_t)

    max_len = max(t.shape[0] for t in ids_list)
    bsz = len(ids_list)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    segment_ids = torch.full((bsz, max_len), pad_segment_id, dtype=torch.long)
    label_probs = torch.stack(labels_list, dim=0)

    for i, (ids_t, segs_t) in enumerate(zip(ids_list, segs_list)):
        t_len = ids_t.shape[0]
        input_ids[i, :t_len] = ids_t
        attention_mask[i, :t_len] = 1
        segment_ids[i, :t_len] = segs_t

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "label_probs": label_probs,
    }


__all__ = [
    "EpisodeDatasetV2",
    "EvalEpisodeBuilder",
    "collate_episode_batch",
    "discover_variants",
    "load_jsonl",
    "normalize_demo_record",
]
