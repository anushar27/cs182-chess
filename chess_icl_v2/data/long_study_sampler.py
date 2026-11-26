"""Checksum-verified snapshot loader, deterministic episode sampler, and val builders."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence, Union

import chess
import numpy as np

from ..models.qwen_model import Notation, QwenPromptFormatter, STANDARD_INITIAL_FEN
from ..variants import ALL_VARIANTS, get_board
from .dataset_v2 import load_jsonl, normalize_demo_record

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrqk]?$")


def _stable_seed(*parts: Any) -> int:
    """Derive a deterministic 64-bit integer seed from arbitrary hashable parts."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _file_sha256_and_lines(path: Path) -> tuple[str, int, int]:
    """Compute SHA-256 hex digest, physical line count, and byte size for ``path``."""
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    # Physical line count matching standard newline-terminated JSONL files.
    line_count = len(raw.splitlines()) if raw else 0
    return sha256, line_count, len(raw)


def _normalize_manifest_files(
    files_obj: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Normalize ``manifest['files']`` (dict or list) into ``(rel_path, meta)`` pairs."""
    if isinstance(files_obj, Mapping):
        entries: list[tuple[str, dict[str, Any]]] = []
        for rel_path, meta in files_obj.items():
            if not isinstance(meta, Mapping):
                raise ValueError(f"Manifest file entry for {rel_path!r} must be a mapping")
            entries.append((str(rel_path), dict(meta)))
        return entries

    if isinstance(files_obj, list):
        entries = []
        for idx, item in enumerate(files_obj):
            if not isinstance(item, Mapping):
                raise ValueError(f"Manifest files[{idx}] must be a mapping")
            rel_path = item.get("path") or item.get("rel_path") or item.get("file")
            if not rel_path:
                raise ValueError(f"Manifest files[{idx}] missing 'path' field")
            entries.append((str(rel_path), dict(item)))
        return entries

    raise ValueError("snapshot_manifest.json 'files' must be a dict or list of file entries")


def _resolve_variant_file(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    variant: str,
    filename: str,
    role_key: str,
) -> Optional[Path]:
    """Resolve a variant split file inside ``snapshot_dir`` using manifest or standard layout."""
    variants_meta = manifest.get("variants")
    if isinstance(variants_meta, Mapping):
        v_entry = variants_meta.get(variant)
        if isinstance(v_entry, Mapping):
            explicit = v_entry.get(role_key) or v_entry.get(filename)
            if explicit:
                return snapshot_dir / str(explicit)

    for cand in (
        snapshot_dir / variant / filename,
        snapshot_dir / "variants" / variant / filename,
    ):
        if cand.exists():
            return cand
    return None


def _discover_snapshot_variants(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    """Determine exposed variants from manifest or snapshot directory layout."""
    exposed = manifest.get("exposed_variants")
    if isinstance(exposed, list) and exposed:
        return [str(v) for v in exposed]

    variants_field = manifest.get("variants")
    if isinstance(variants_field, Mapping) and variants_field:
        return sorted(str(v) for v in variants_field.keys())
    if isinstance(variants_field, list) and variants_field:
        return [str(v) for v in variants_field]

    found: list[str] = []
    for base in (snapshot_dir, snapshot_dir / "variants"):
        if not base.exists():
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "games_demo_train.jsonl").exists() and (sub / "positions_train.jsonl").exists():
                if sub.name not in found:
                    found.append(sub.name)
    return found


def _validate_positions_records(
    positions: Sequence[Mapping[str, Any]],
    variant: str,
    split_name: str,
) -> None:
    """Validate that every position has complete finite 3-way labels and 0 unresolved rollouts."""
    if not positions:
        raise ValueError(f"Exposed variant {variant!r} has empty {split_name} positions")

    for idx, pos in enumerate(positions):
        root_id = pos.get("root_id", f"idx={idx}")
        n_unres = pos.get("n_unresolved")
        if n_unres is not None and int(n_unres) > 0:
            raise ValueError(
                f"[{variant}/{split_name}] Position {root_id!r} has incomplete rollouts: "
                f"n_unresolved={n_unres}"
            )
        unres_cnt = pos.get("unresolved_count")
        if unres_cnt is not None and int(unres_cnt) > 0:
            raise ValueError(
                f"[{variant}/{split_name}] Position {root_id!r} has incomplete rollouts: "
                f"unresolved_count={unres_cnt}"
            )
        probs = pos.get("label_probs")
        if probs is None:
            raise ValueError(
                f"[{variant}/{split_name}] Position {root_id!r} has label_probs=None"
            )
        if not isinstance(probs, (list, tuple)) or len(probs) != 3:
            raise ValueError(
                f"[{variant}/{split_name}] Position {root_id!r} has invalid label_probs={probs!r}"
            )
        vals = [float(x) for x in probs]
        if any(not math.isfinite(v) or v < 0.0 for v in vals) or sum(vals) <= 0.0:
            raise ValueError(
                f"[{variant}/{split_name}] Position {root_id!r} has non-probability label_probs={probs!r}"
            )


def verify_snapshot_manifest(
    snapshot_dir: Union[str, Path],
    exposed_variants: Optional[Sequence[str]] = None,
    max_k: Optional[int] = None,
) -> dict[str, Any]:
    """Verify SHA-256 checksums, line counts, label completeness, and demo counts for a snapshot.

    Fails loudly if ``snapshot_manifest.json`` or any tracked file is missing, modified,
    has unresolved labels (``label_probs is None`` or ``n_unresolved > 0``), or if any
    exposed rule is empty or lacks enough training demos for ``max_k``.
    """
    root = Path(snapshot_dir)
    if not root.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {root}")
    manifest_path = root / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing snapshot_manifest.json in {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("snapshot_manifest.json must contain a JSON object")

    if int(manifest.get("n_unresolved") or 0) > 0:
        raise ValueError(
            f"Snapshot manifest reports n_unresolved={manifest['n_unresolved']} > 0"
        )

    files_entries = _normalize_manifest_files(manifest.get("files"))
    if not files_entries:
        raise ValueError("snapshot_manifest.json 'files' cannot be empty")

    root_resolved = root.resolve()
    verified_files: dict[str, dict[str, Any]] = {}
    for rel_path, meta in files_entries:
        fpath = (root / rel_path).resolve()
        if not fpath.is_relative_to(root_resolved):
            raise ValueError(f"Manifest path escapes snapshot_dir: {rel_path!r}")
        if not fpath.exists():
            raise FileNotFoundError(f"Manifest file missing: {root / rel_path}")

        expected_sha = meta.get("sha256")
        if not expected_sha or not isinstance(expected_sha, str):
            raise ValueError(f"Manifest entry for {rel_path!r} missing 'sha256'")

        actual_sha, actual_lines, actual_bytes = _file_sha256_and_lines(fpath)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {rel_path!r}: expected {expected_sha}, got {actual_sha}"
            )

        expected_lines = None
        for key in ("line_count", "n_lines", "num_lines", "lines"):
            if key in meta and meta[key] is not None:
                expected_lines = int(meta[key])
                break
        if expected_lines is None:
            raise ValueError(f"Manifest entry for {rel_path!r} missing 'line_count'")
        if actual_lines != expected_lines:
            raise ValueError(
                f"Line count mismatch for {rel_path!r}: expected {expected_lines}, got {actual_lines}"
            )

        expected_bytes = meta.get("byte_size") if meta.get("byte_size") is not None else meta.get("size_bytes")
        if expected_bytes is not None and actual_bytes != int(expected_bytes):
            raise ValueError(
                f"Byte size mismatch for {rel_path!r}: expected {expected_bytes}, got {actual_bytes}"
            )

        norm_rel = fpath.relative_to(root_resolved).as_posix()
        verified_files[norm_rel] = {
            "sha256": actual_sha,
            "line_count": actual_lines,
            "byte_size": actual_bytes,
        }

    eff_variants = (
        [str(v) for v in exposed_variants]
        if exposed_variants is not None
        else _discover_snapshot_variants(root, manifest)
    )
    if not eff_variants:
        raise ValueError(f"No exposed variants found in snapshot {root}")

    eff_max_k = int(manifest.get("max_k", 4)) if max_k is None else int(max_k)
    per_variant_counts: dict[str, dict[str, int]] = {}

    for variant in eff_variants:
        demo_train_path = _resolve_variant_file(
            root, manifest, variant, "games_demo_train.jsonl", "games_demo_train"
        )
        pos_train_path = _resolve_variant_file(
            root, manifest, variant, "positions_train.jsonl", "positions_train"
        )
        if demo_train_path is None or not demo_train_path.exists():
            raise FileNotFoundError(
                f"Missing games_demo_train.jsonl for exposed variant {variant!r} in {root}"
            )
        if pos_train_path is None or not pos_train_path.exists():
            raise FileNotFoundError(
                f"Missing positions_train.jsonl for exposed variant {variant!r} in {root}"
            )

        for req_path in (demo_train_path, pos_train_path):
            req_rel = req_path.resolve().relative_to(root_resolved).as_posix()
            if req_rel not in verified_files:
                raise ValueError(
                    f"Required snapshot file {req_rel!r} is not tracked in snapshot_manifest.json"
                )

        train_demos = load_jsonl(demo_train_path)
        train_positions = load_jsonl(pos_train_path)
        if not train_demos:
            raise ValueError(f"Exposed variant {variant!r} has empty games_demo_train.jsonl")
        if len(train_demos) < eff_max_k:
            raise ValueError(
                f"Exposed variant {variant!r} has only {len(train_demos)} training demos, "
                f"fewer than max_k={eff_max_k}"
            )
        _validate_positions_records(train_positions, variant=variant, split_name="positions_train")

        counts: dict[str, int] = {
            "games_demo_train": len(train_demos),
            "positions_train": len(train_positions),
        }

        demo_val_path = _resolve_variant_file(
            root, manifest, variant, "games_demo_val.jsonl", "games_demo_val"
        )
        pos_val_path = _resolve_variant_file(
            root, manifest, variant, "positions_val.jsonl", "positions_val"
        )
        if demo_val_path is not None and demo_val_path.exists():
            val_rel = demo_val_path.resolve().relative_to(root_resolved).as_posix()
            if val_rel not in verified_files:
                raise ValueError(
                    f"Snapshot file {val_rel!r} is not tracked in snapshot_manifest.json"
                )
            val_demos = load_jsonl(demo_val_path)
            if len(val_demos) < eff_max_k:
                raise ValueError(
                    f"Exposed variant {variant!r} has only {len(val_demos)} val demos, "
                    f"fewer than max_k={eff_max_k}"
                )
            counts["games_demo_val"] = len(val_demos)

        if pos_val_path is not None and pos_val_path.exists():
            val_rel = pos_val_path.resolve().relative_to(root_resolved).as_posix()
            if val_rel not in verified_files:
                raise ValueError(
                    f"Snapshot file {val_rel!r} is not tracked in snapshot_manifest.json"
                )
            val_positions = load_jsonl(pos_val_path)
            _validate_positions_records(val_positions, variant=variant, split_name="positions_val")
            counts["positions_val"] = len(val_positions)

        per_variant_counts[variant] = counts

    return {
        "snapshot_dir": str(root),
        "manifest": dict(manifest),
        "exposed_variants": eff_variants,
        "verified_files": verified_files,
        "per_variant_counts": per_variant_counts,
    }


def _uci_to_san_if_needed(demo: Mapping[str, Any], variant: str, notation: Notation) -> dict[str, Any]:
    """Normalize a demo record and populate ``moves_san`` if missing and ``moves`` is UCI."""
    out = normalize_demo_record(demo)
    if notation != "san":
        return out

    existing_san = out.get("moves_san") or out.get("san_moves")
    raw_moves = [str(m) for m in out.get("moves", [])]
    if existing_san is not None and len(existing_san) == len(raw_moves):
        out["moves_san"] = [str(m) for m in existing_san]
        return out

    if not raw_moves or variant not in ALL_VARIANTS:
        return out
    if not all(_UCI_RE.match(m.strip()) for m in raw_moves):
        return out

    init_fen = out.get("initial_fen") or STANDARD_INITIAL_FEN
    fen_arg = None if init_fen == STANDARD_INITIAL_FEN else init_fen
    blocked = out.get("blocked_squares")
    kwargs: dict[str, Any] = {}
    if blocked:
        kwargs["blocked"] = [int(sq) for sq in blocked]
    try:
        board = get_board(variant, fen=fen_arg, **kwargs)
    except TypeError:
        board = get_board(variant, fen=fen_arg)

    san_moves: list[str] = []
    for uci in raw_moves:
        try:
            mv = chess.Move.from_uci(uci.strip())
            san_moves.append(board.san(mv))
            board.push(mv)
        except Exception:
            return out
    out["moves_san"] = san_moves
    return out


def _demo_identifier(demo: Mapping[str, Any], variant: str, idx: int) -> str:
    return str(demo.get("game_id") or demo.get("demo_id") or f"{variant}_demo_{idx}")


def _position_family_id(pos: Mapping[str, Any]) -> str:
    return str(pos.get("family_id") or pos.get("source_game_id") or pos.get("root_id"))


def _eligible_demo_indices(
    demo_pool: Sequence[Mapping[str, Any]],
    query_pos: Mapping[str, Any],
) -> list[int]:
    """Return demo indices that do not share ``family_id`` or ``source_game_id`` with ``query_pos``."""
    q_family = query_pos.get("family_id")
    q_source_gid = query_pos.get("source_game_id")
    if q_family is None and q_source_gid is None:
        return list(range(len(demo_pool)))

    eligible: list[int] = []
    for idx, demo in enumerate(demo_pool):
        d_family = demo.get("family_id")
        d_gid = demo.get("game_id")
        d_source_gid = demo.get("source_game_id")
        if q_family is not None and d_family is not None and d_family == q_family:
            continue
        if q_source_gid is not None and (
            d_gid == q_source_gid or d_source_gid == q_source_gid or d_family == q_source_gid
        ):
            continue
        if q_family is not None and (d_gid == q_family or d_source_gid == q_family):
            continue
        eligible.append(idx)
    return eligible


class DeterministicEpisodeSampler:
    """Checksum-verified, independently seeded, serializable episode sampler for long studies.

    Samples episodes with:
      - equal weighting across ``exposed_variants`` (independent of target values),
      - uniform K mixture across ``k_values`` (default ``(0, 1, 2, 4)``),
      - dynamic resampling of full training-only demos (``games_demo_train.jsonl``)
        with family/source-game exclusion,
      - strict no-trimming ``QwenPromptFormatter(notation="san", max_length=32768, strict_no_trim=True)``,
      - dedicated ``np.random.Generator`` + ``state_dict()`` / ``load_state_dict()`` for exact resume.
    """

    def __init__(
        self,
        snapshot_dir: Union[str, Path],
        tokenizer: Any,
        exposed_variants: Optional[Sequence[str]] = None,
        k_values: Sequence[int] = (0, 1, 2, 4),
        seed: int = 42,
        notation: Notation = "san",
        max_length: int = 32768,
        strict_no_trim: bool = True,
        verify_manifest: bool = True,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.k_values: list[int] = [int(k) for k in k_values]
        if not self.k_values or any(k < 0 for k in self.k_values):
            raise ValueError(f"k_values must be non-empty non-negative ints, got {k_values!r}")

        max_k = max(self.k_values)
        if verify_manifest:
            report = verify_snapshot_manifest(
                self.snapshot_dir,
                exposed_variants=exposed_variants,
                max_k=max_k,
            )
            self.manifest: dict[str, Any] = report["manifest"]
            self.exposed_variants: list[str] = list(report["exposed_variants"])
        else:
            manifest_path = self.snapshot_dir / "snapshot_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing snapshot_manifest.json in {self.snapshot_dir}")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.exposed_variants = (
                [str(v) for v in exposed_variants]
                if exposed_variants is not None
                else _discover_snapshot_variants(self.snapshot_dir, self.manifest)
            )
            if not self.exposed_variants:
                raise ValueError(f"No exposed variants resolved for {self.snapshot_dir}")

        self.seed: int = int(seed)
        self.rng: np.random.Generator = np.random.Generator(np.random.PCG64(self.seed))
        self.draw_counter: int = 0
        self.step_counter: int = 0

        self.formatter = QwenPromptFormatter(
            tokenizer=tokenizer,
            notation=notation,
            max_length=max_length,
            strict_no_trim=strict_no_trim,
        )

        self.demos_by_variant: dict[str, list[dict[str, Any]]] = {}
        self.positions_by_variant: dict[str, list[dict[str, Any]]] = {}
        for variant in self.exposed_variants:
            demo_path = _resolve_variant_file(
                self.snapshot_dir,
                self.manifest,
                variant,
                "games_demo_train.jsonl",
                "games_demo_train",
            )
            pos_path = _resolve_variant_file(
                self.snapshot_dir,
                self.manifest,
                variant,
                "positions_train.jsonl",
                "positions_train",
            )
            if demo_path is None or not demo_path.exists():
                raise FileNotFoundError(f"Missing games_demo_train.jsonl for {variant!r}")
            if pos_path is None or not pos_path.exists():
                raise FileNotFoundError(f"Missing positions_train.jsonl for {variant!r}")

            raw_demos = load_jsonl(demo_path)
            raw_positions = load_jsonl(pos_path)
            if len(raw_demos) < max_k:
                raise ValueError(
                    f"Variant {variant!r} has {len(raw_demos)} training demos < max_k={max_k}"
                )
            _validate_positions_records(raw_positions, variant=variant, split_name="positions_train")

            norm_demos: list[dict[str, Any]] = []
            for idx, d in enumerate(raw_demos):
                nd = _uci_to_san_if_needed(d, variant=variant, notation=notation)
                nd["_demo_id"] = _demo_identifier(nd, variant=variant, idx=idx)
                norm_demos.append(nd)
            self.demos_by_variant[variant] = norm_demos
            self.positions_by_variant[variant] = raw_positions

    def set_step(self, step: int) -> None:
        """Set explicit optimizer step counter."""
        self.step_counter = int(step)

    def advance_step(self, n: int = 1) -> int:
        """Increment optimizer step counter by ``n`` and return the updated value."""
        self.step_counter += int(n)
        return self.step_counter

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON/torch-serializable snapshot of RNG and counters."""
        return {
            "seed": self.seed,
            "exposed_variants": list(self.exposed_variants),
            "k_values": list(self.k_values),
            "draw_counter": self.draw_counter,
            "step_counter": self.step_counter,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact RNG state and counters for bit-identical mid-run resume."""
        saved_variants = state.get("exposed_variants")
        if saved_variants is not None and list(saved_variants) != self.exposed_variants:
            raise ValueError(
                f"Sampler exposed_variants mismatch: saved={saved_variants!r}, "
                f"current={self.exposed_variants!r}"
            )
        saved_k = state.get("k_values")
        if saved_k is not None and [int(k) for k in saved_k] != self.k_values:
            raise ValueError(
                f"Sampler k_values mismatch: saved={saved_k!r}, current={self.k_values!r}"
            )
        self.draw_counter = int(state["draw_counter"])
        self.step_counter = int(state["step_counter"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def sample_episode(self) -> tuple[str, np.ndarray, dict[str, Any]]:
        """Sample one training episode and return ``(prompt_text, target_probs, metadata)``."""
        v_idx = int(self.rng.integers(0, len(self.exposed_variants)))
        variant = self.exposed_variants[v_idx]

        pos_pool = self.positions_by_variant[variant]
        p_idx = int(self.rng.integers(0, len(pos_pool)))
        pos = pos_pool[p_idx]

        k_idx = int(self.rng.integers(0, len(self.k_values)))
        k = int(self.k_values[k_idx])

        demo_pool = self.demos_by_variant[variant]
        if k > 0:
            eligible = _eligible_demo_indices(demo_pool, pos)
            if len(eligible) < k:
                raise ValueError(
                    f"Variant {variant!r} has only {len(eligible)} eligible demos after "
                    f"family/source exclusion for root {pos.get('root_id')!r}, need k={k}"
                )
            chosen_indices = self.rng.choice(eligible, size=k, replace=False)
            demos = [demo_pool[int(i)] for i in chosen_indices]
        else:
            demos = []

        query = {
            "fen": pos.get("fen") or pos.get("query_fen"),
            "blocked_squares": pos.get("blocked_squares"),
        }
        formatted = self.formatter.format_prompt_with_info(demos, query)
        prompt_text = formatted.text
        target_probs = np.asarray(pos["label_probs"], dtype=np.float32)

        draw_idx = self.draw_counter
        self.draw_counter += 1

        input_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        demo_ids = [str(d["_demo_id"]) for d in demos]
        metadata: dict[str, Any] = {
            "variant": variant,
            "family_id": _position_family_id(pos),
            "root_id": str(pos.get("root_id", f"{variant}_pos_{p_idx}")),
            "k": k,
            "demo_ids": demo_ids,
            "input_sha256": input_sha256,
            "num_tokens": formatted.num_tokens,
            "draw_counter": draw_idx,
            "step_counter": self.step_counter,
        }
        return prompt_text, target_probs, metadata


def _build_val_episodes_for_positions(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    variants: Sequence[str],
    positions_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    k_values: Sequence[int],
    formatter: QwenPromptFormatter,
    notation: Notation,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically build formatted validation episodes from ``games_demo_val.jsonl``."""
    episodes: list[dict[str, Any]] = []
    max_k = max(int(k) for k in k_values)

    for variant in variants:
        demo_val_path = _resolve_variant_file(
            snapshot_dir, manifest, variant, "games_demo_val.jsonl", "games_demo_val"
        )
        if demo_val_path is None or not demo_val_path.exists():
            raise FileNotFoundError(
                f"Missing games_demo_val.jsonl for variant {variant!r} in {snapshot_dir}"
            )
        raw_val_demos = load_jsonl(demo_val_path)
        if len(raw_val_demos) < max_k:
            raise ValueError(
                f"Variant {variant!r} has {len(raw_val_demos)} val demos < max_k={max_k}"
            )
        demo_pool: list[dict[str, Any]] = []
        for idx, d in enumerate(raw_val_demos):
            nd = _uci_to_san_if_needed(d, variant=variant, notation=notation)
            nd["_demo_id"] = _demo_identifier(nd, variant=variant, idx=idx)
            demo_pool.append(nd)

        for k in k_values:
            k_int = int(k)
            for p_idx, pos in enumerate(positions_by_variant[variant]):
                root_id = str(pos.get("root_id", f"{variant}_val_{p_idx}"))
                if k_int > 0:
                    ep_rng = np.random.Generator(
                        np.random.PCG64(_stable_seed(seed, variant, k_int, root_id))
                    )
                    eligible = _eligible_demo_indices(demo_pool, pos)
                    if len(eligible) < k_int:
                        raise ValueError(
                            f"Variant {variant!r} has only {len(eligible)} eligible val demos "
                            f"for root {root_id!r}, need k={k_int}"
                        )
                    chosen = ep_rng.choice(eligible, size=k_int, replace=False)
                    demos = [demo_pool[int(i)] for i in chosen]
                else:
                    demos = []

                query_fen = str(pos.get("fen") or pos.get("query_fen"))
                blocked = pos.get("blocked_squares")
                formatted = formatter.format_prompt_with_info(
                    demos,
                    {"fen": query_fen, "blocked_squares": blocked},
                )
                prompt_text = formatted.text
                target_probs = np.asarray(pos["label_probs"], dtype=np.float32)
                episodes.append(
                    {
                        "prompt_text": prompt_text,
                        "target_probs": target_probs,
                        "y_true": [float(x) for x in pos["label_probs"]],
                        "variant": variant,
                        "family_id": _position_family_id(pos),
                        "root_id": root_id,
                        "source_game_id": pos.get("source_game_id"),
                        "k": k_int,
                        "demo_ids": [str(d["_demo_id"]) for d in demos],
                        "demos": demos,
                        "query_fen": query_fen,
                        "blocked_squares": blocked,
                        "side_to_move": pos.get("side_to_move"),
                        "equal_material": bool(pos.get("equal_material", False)),
                        "input_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                        "num_tokens": formatted.num_tokens,
                    }
                )
    return episodes


def build_full_val_episodes(
    snapshot_dir: Union[str, Path],
    tokenizer: Any,
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (0, 1, 2, 4),
    seed: int = 0,
    notation: Notation = "san",
    max_length: int = 32768,
    strict_no_trim: bool = True,
    max_roots_per_variant: Optional[int] = None,
    verify_manifest: bool = True,
) -> list[dict[str, Any]]:
    """Build deterministic full validation episodes from ``games_demo_val.jsonl`` & ``positions_val.jsonl``."""
    root = Path(snapshot_dir)
    k_list = [int(k) for k in k_values]
    if not k_list or any(k < 0 for k in k_list):
        raise ValueError(f"k_values must be non-empty non-negative ints, got {k_values!r}")

    if verify_manifest:
        report = verify_snapshot_manifest(root, exposed_variants=variants, max_k=max(k_list))
        manifest = report["manifest"]
        eff_variants = list(report["exposed_variants"])
    else:
        manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
        eff_variants = (
            [str(v) for v in variants]
            if variants is not None
            else _discover_snapshot_variants(root, manifest)
        )

    positions_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in eff_variants:
        pos_val_path = _resolve_variant_file(
            root, manifest, variant, "positions_val.jsonl", "positions_val"
        )
        if pos_val_path is None or not pos_val_path.exists():
            raise FileNotFoundError(
                f"Missing positions_val.jsonl for variant {variant!r} in {root}"
            )
        val_positions = load_jsonl(pos_val_path)
        _validate_positions_records(val_positions, variant=variant, split_name="positions_val")
        if max_roots_per_variant is not None:
            val_positions = val_positions[: int(max_roots_per_variant)]
        positions_by_variant[variant] = val_positions

    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation=notation,
        max_length=max_length,
        strict_no_trim=strict_no_trim,
    )
    return _build_val_episodes_for_positions(
        snapshot_dir=root,
        manifest=manifest,
        variants=eff_variants,
        positions_by_variant=positions_by_variant,
        k_values=k_list,
        formatter=formatter,
        notation=notation,
        seed=seed,
    )


def load_val_panel_episodes(
    snapshot_dir: Union[str, Path],
    tokenizer: Any,
    variants: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (0, 1, 2, 4),
    roots_per_variant: Optional[int] = 32,
    seed: int = 0,
    notation: Notation = "san",
    max_length: int = 32768,
    strict_no_trim: bool = True,
    verify_manifest: bool = True,
) -> list[dict[str, Any]]:
    """Deterministically load or construct a fixed validation panel from ``games_demo_val.jsonl`` & ``positions_val.jsonl``."""
    root = Path(snapshot_dir)
    k_list = [int(k) for k in k_values]
    if not k_list or any(k < 0 for k in k_list):
        raise ValueError(f"k_values must be non-empty non-negative ints, got {k_values!r}")

    if verify_manifest:
        report = verify_snapshot_manifest(root, exposed_variants=variants, max_k=max(k_list))
        manifest = report["manifest"]
        eff_variants = list(report["exposed_variants"])
    else:
        manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
        eff_variants = (
            [str(v) for v in variants]
            if variants is not None
            else _discover_snapshot_variants(root, manifest)
        )

    panel_ids_by_variant: Optional[dict[str, set[str]]] = None
    panel_json = None
    for panel_name in ("val_panel_manifest.json", "val_panel.json"):
        cand = root / panel_name
        if cand.exists():
            panel_json = cand
            break
    if panel_json is not None:
        panel_payload = json.loads(panel_json.read_text(encoding="utf-8"))
        if isinstance(panel_payload, Mapping):
            root_map = panel_payload.get("root_ids_by_variant") or panel_payload.get("variants")
            if isinstance(root_map, Mapping):
                parsed_map: dict[str, set[str]] = {}
                for v, entry in root_map.items():
                    if isinstance(entry, Mapping):
                        rids = entry.get("selected_root_ids") or entry.get("root_ids") or []
                    else:
                        rids = entry
                    parsed_map[str(v)] = {str(r) for r in rids}
                panel_ids_by_variant = parsed_map

    positions_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in eff_variants:
        pos_val_path = _resolve_variant_file(
            root, manifest, variant, "positions_val.jsonl", "positions_val"
        )
        if pos_val_path is None or not pos_val_path.exists():
            raise FileNotFoundError(
                f"Missing positions_val.jsonl for variant {variant!r} in {root}"
            )
        val_positions = load_jsonl(pos_val_path)
        _validate_positions_records(val_positions, variant=variant, split_name="positions_val")

        if panel_ids_by_variant is not None and variant in panel_ids_by_variant:
            allowed_ids = panel_ids_by_variant[variant]
            selected = [p for p in val_positions if str(p.get("root_id")) in allowed_ids]
        elif roots_per_variant is not None and len(val_positions) > int(roots_per_variant):
            panel_rng = np.random.Generator(
                np.random.PCG64(_stable_seed(seed, "val_panel", variant))
            )
            chosen_idx = sorted(
                int(i)
                for i in panel_rng.choice(
                    len(val_positions), size=int(roots_per_variant), replace=False
                )
            )
            selected = [val_positions[i] for i in chosen_idx]
        else:
            selected = val_positions

        positions_by_variant[variant] = selected

    formatter = QwenPromptFormatter(
        tokenizer=tokenizer,
        notation=notation,
        max_length=max_length,
        strict_no_trim=strict_no_trim,
    )
    return _build_val_episodes_for_positions(
        snapshot_dir=root,
        manifest=manifest,
        variants=eff_variants,
        positions_by_variant=positions_by_variant,
        k_values=k_list,
        formatter=formatter,
        notation=notation,
        seed=seed,
    )


__all__ = [
    "DeterministicEpisodeSampler",
    "build_full_val_episodes",
    "load_val_panel_episodes",
    "verify_snapshot_manifest",
]
