"""Unit tests for model architecture and forward/backward passes."""

import pytest
import torch
from chess_icl.model import ChessTransformer, TINY, SMALL, ModelConfig


def test_model_shapes():
    config = TINY
    config.vocab_size = 128
    model = ChessTransformer(config, pad_token_id=0)

    # Batch of size 4, sequence length 32
    input_ids = torch.randint(1, 120, (4, 32))
    logits = model(input_ids)
    assert logits.shape == (4, 3)


def test_backward_pass():
    config = TINY
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
    config = TINY
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
    assert t_model.count_params() < 2_000_000
    assert 4_000_000 < s_model.count_params() < 10_000_000
