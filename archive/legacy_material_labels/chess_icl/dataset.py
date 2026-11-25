"""PyTorch Dataset classes for chess pretraining, variant continue-training, and evaluation."""

import json
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .data import ChessTokenizer, load_games, load_positions

LABEL_MAP = {"win": 0, "draw": 1, "loss": 2}
INV_LABEL_MAP = {0: "win", 1: "draw", 2: "loss"}


class ChessPretrainDataset(Dataset):
    """(position, W/D/L label) pairs for standard chess pretraining."""

    def __init__(
        self,
        positions_path: Union[str, Path],
        tokenizer: ChessTokenizer,
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.positions = load_positions(positions_path)
        self.label_map = LABEL_MAP

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.positions[idx]
        input_ids = self.tokenizer.encode_input([], pos["fen"])
        label = self.label_map.get(pos.get("label", "draw"), 1)

        # Truncate / pad
        input_ids = input_ids[: self.max_len]
        padding = [self.tokenizer.pad_id] * (self.max_len - len(input_ids))
        padded_ids = input_ids + padding

        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ChessVariantDataset(Dataset):
    """Mixed standard + variant games with K-shot context.

    Each sample: K demo games from one variant + query position + label.
    Standard chess samples have K=0.
    Variant samples draw K uniformly from [1, k_max].
    """

    def __init__(
        self,
        standard_path: Union[str, Path],
        variant_paths: Sequence[Union[str, Path]],
        tokenizer: ChessTokenizer,
        k_max: int = 8,
        max_len: int = 4096,
        mix_ratio: float = 0.5,
        dataset_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.k_max = k_max
        self.max_len = max_len
        self.mix_ratio = mix_ratio
        self.label_map = LABEL_MAP

        # Load standard positions
        std_p = Path(standard_path)
        if std_p.is_dir():
            positions_file = std_p / "positions_train.jsonl"
            if not positions_file.exists():
                positions_file = std_p / "positions_val.jsonl"
        else:
            positions_file = std_p
        self.standard_positions = load_positions(positions_file)

        # Load variant data: {variant_name: (games, positions)}
        self.variant_data = {}
        for vpath in variant_paths:
            vp = Path(vpath)
            name = vp.stem
            games = load_games(vp / "games_demo.jsonl")
            positions = load_positions(vp / "positions_eval.jsonl")
            if games and positions:
                self.variant_data[name] = (games, positions)

        self.variant_names = list(self.variant_data.keys())
        self._length = dataset_size if dataset_size else max(len(self.standard_positions), 10000)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        use_standard = (
            len(self.variant_names) == 0
            or random.random() < self.mix_ratio
            or not self.variant_data
        )

        if use_standard:
            pos = random.choice(self.standard_positions)
            input_ids = self.tokenizer.encode_input([], pos["fen"])
            label = self.label_map.get(pos.get("label", "draw"), 1)
        else:
            v_name = random.choice(self.variant_names)
            games, positions = self.variant_data[v_name]
            K = random.randint(1, min(self.k_max, len(games)))
            demos = random.sample(games, K)
            pos = random.choice(positions)
            input_ids = self.tokenizer.encode_input(demos, pos["fen"])
            label = self.label_map.get(pos.get("label", "draw"), 1)

        input_ids = input_ids[: self.max_len]
        padding = [self.tokenizer.pad_id] * (self.max_len - len(input_ids))
        padded_ids = input_ids + padding

        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ChessEvalDataset:
    """Fixed evaluation set for one variant. Deterministic demo sampling."""

    def __init__(self, variant_dir: Union[str, Path], tokenizer: ChessTokenizer):
        self.tokenizer = tokenizer
        self.variant_dir = Path(variant_dir)
        self.demo_pool = load_games(self.variant_dir / "games_demo.jsonl")
        self.demo_pool_weak = load_games(self.variant_dir / "games_demo_weak.jsonl")
        self.positions = load_positions(self.variant_dir / "positions_eval.jsonl")
        self.label_map = LABEL_MAP

    def get_eval_batch(
        self,
        K: int,
        seed: int,
        max_len: int = 4096,
        weak_demos: bool = False,
    ) -> tuple[list[list[int]], list[int]]:
        """Return (input_ids, labels) for all eval positions with K demos."""
        pool = self.demo_pool_weak if weak_demos else self.demo_pool
        rng = random.Random(seed)

        if K == 0 or len(pool) == 0:
            demos = []
        else:
            # Nested sampling: sort or sample deterministically
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            selected_indices = indices[: min(K, len(pool))]
            demos = [pool[i] for i in selected_indices]

        all_inputs = []
        all_labels = []
        for pos in self.positions:
            ids = self.tokenizer.encode_input(demos, pos["fen"])
            ids = ids[:max_len]
            all_inputs.append(ids)
            all_labels.append(self.label_map.get(pos.get("label", "draw"), 1))

        return all_inputs, all_labels
