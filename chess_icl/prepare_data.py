"""Backward-compatible wrapper for data preparation."""

from pathlib import Path
from typing import Optional
from scripts.generate_data import generate_variant_dataset, generate_standard_dataset


def prepare_all_data(output_dir: str = "data", count_per_variant: int = 2000, seed: int = 42):
    """Generate all data for standard chess and 13 variants."""
    from chess_icl.variants import VARIANTS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating standard chess dataset...")
    generate_standard_dataset(output_dir=out / "standard", count=10000, seed=seed)

    print("Generating variant datasets...")
    for var_name in VARIANTS:
        print(f"Generating variant {var_name}...")
        generate_variant_dataset(
            variant_name=var_name,
            output_dir=out / var_name,
            heuristic_count=count_per_variant,
            random_count=min(500, count_per_variant // 4),
            eval_pos_count=500,
            seed=seed,
        )
