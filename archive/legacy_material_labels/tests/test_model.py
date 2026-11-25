"""Unit tests for model architecture, forward/backward passes, and muP."""

import copy
import pytest
import torch
from chess_icl.model import ChessTransformer, TINY, SMALL, MEDIUM, LARGE, ModelConfig


def test_model_shapes():
    config = copy.deepcopy(TINY)
    config.vocab_size = 128
    model = ChessTransformer(config, pad_token_id=0)

    # Batch of size 4, sequence length 32
    input_ids = torch.randint(1, 120, (4, 32))
    logits = model(input_ids)
    assert logits.shape == (4, 3)


def test_backward_pass():
    config = copy.deepcopy(TINY)
    config.vocab_size = 128
    model = ChessTransformer(config, pad_token_id=0)

    input_ids = torch.randint(1, 120, (2, 16))
    labels = torch.tensor([0, 2], dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    logits = model(input_ids)
    loss = criterion(logits, labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_overfit_small_batch():
    """Model should easily overfit 5 samples in < 100 steps."""
    torch.manual_seed(42)
    config = copy.deepcopy(TINY)
    config.vocab_size = 128
    model = ChessTransformer(config, pad_token_id=0)

    input_ids = torch.randint(1, 120, (5, 20))
    labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(60):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    preds = model(input_ids).argmax(-1)
    assert (preds == labels).sum().item() >= 4


def test_parameter_counts():
    t_model = ChessTransformer(TINY)
    s_model = ChessTransformer(SMALL)
    l_model = ChessTransformer(LARGE)
    assert t_model.count_params() < 2_000_000
    assert 4_000_000 < s_model.count_params() < 10_000_000
    assert 190_000_000 < l_model.count_params() < 220_000_000


def test_mup_coordinate_check():
    """Verify that under muP, initial output logits remain O(1) across model widths."""
    torch.manual_seed(42)
    input_ids = torch.randint(1, 120, (8, 32))

    tiny_cfg = copy.deepcopy(TINY)
    tiny_cfg.vocab_size = 128
    tiny_cfg.use_mup = True
    tiny_model = ChessTransformer(tiny_cfg)

    small_cfg = copy.deepcopy(SMALL)
    small_cfg.vocab_size = 128
    small_cfg.use_mup = True
    small_model = ChessTransformer(small_cfg)

    med_cfg = copy.deepcopy(MEDIUM)
    med_cfg.vocab_size = 128
    med_cfg.use_mup = True
    med_model = ChessTransformer(med_cfg)

    large_cfg = copy.deepcopy(LARGE)
    large_cfg.vocab_size = 128
    large_cfg.use_mup = True
    large_model = ChessTransformer(large_cfg)

    with torch.no_grad():
        tiny_logits = tiny_model(input_ids)
        small_logits = small_model(input_ids)
        med_logits = med_model(input_ids)
        large_logits = large_model(input_ids)

    # Logits should have O(1) std across all widths (no explosion)
    assert 0.01 < tiny_logits.std().item() < 2.0
    assert 0.01 < small_logits.std().item() < 2.0
    assert 0.01 < med_logits.std().item() < 2.0
    assert 0.01 < large_logits.std().item() < 2.0

    # Ensure no NaNs or Infs
    assert torch.isfinite(tiny_logits).all()
    assert torch.isfinite(small_logits).all()
    assert torch.isfinite(med_logits).all()
    assert torch.isfinite(large_logits).all()


def test_mup_optimizer_groups():
    """Verify that muP optimizer scales hidden layer learning rates inversely with width."""
    base_lr = 1e-3

    # Tiny: d_model = 128, d_base = 128 -> r = 1.0
    tiny_cfg = copy.deepcopy(TINY)
    tiny_cfg.use_mup = True
    tiny_model = ChessTransformer(tiny_cfg)
    tiny_opt = tiny_model.get_optimizer(base_lr=base_lr)

    # Small: d_model = 256, d_base = 128 -> r = 2.0
    small_cfg = copy.deepcopy(SMALL)
    small_cfg.use_mup = True
    small_model = ChessTransformer(small_cfg)
    small_opt = small_model.get_optimizer(base_lr=base_lr)

    # Large: d_model = 1024, d_base = 128 -> r = 8.0
    large_cfg = copy.deepcopy(LARGE)
    large_cfg.use_mup = True
    large_model = ChessTransformer(large_cfg)
    large_opt = large_model.get_optimizer(base_lr=base_lr)

    # In Tiny, hidden lr is base_lr / 1.0 = 1e-3
    tiny_hidden_lr = tiny_opt.param_groups[1]["lr"]
    assert pytest.approx(tiny_hidden_lr, rel=1e-5) == base_lr

    # In Small, hidden lr is base_lr / 2.0 = 5e-4
    small_hidden_lr = small_opt.param_groups[1]["lr"]
    assert pytest.approx(small_hidden_lr, rel=1e-5) == base_lr / 2.0

    # In Large, hidden lr is base_lr / 8.0 = 1.25e-4
    large_hidden_lr = large_opt.param_groups[1]["lr"]
    assert pytest.approx(large_hidden_lr, rel=1e-5) == base_lr / 8.0

    # Embedding and readout lr should remain base_lr
    assert pytest.approx(tiny_opt.param_groups[0]["lr"], rel=1e-5) == base_lr
    assert pytest.approx(small_opt.param_groups[0]["lr"], rel=1e-5) == base_lr
    assert pytest.approx(large_opt.param_groups[0]["lr"], rel=1e-5) == base_lr


def test_mup_overfit():
    """muP model should stably train and overfit."""
    torch.manual_seed(42)
    config = copy.deepcopy(SMALL)
    config.vocab_size = 128
    config.use_mup = True
    model = ChessTransformer(config, pad_token_id=0)

    input_ids = torch.randint(1, 120, (6, 20))
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    optimizer = model.get_optimizer(base_lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(50):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    preds = model(input_ids).argmax(-1)
    assert (preds == labels).sum().item() >= 5
