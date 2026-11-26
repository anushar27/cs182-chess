"""Unit tests for Track 2 tokenizer (ChessV2Tokenizer) and model (ChessTransformerV2)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from chess_icl_v2.data import ChessV2Tokenizer, collate_episodes
from chess_icl_v2.data.tokenizer_v2 import STANDARD_STARTING_FEN, STATE_TOKEN_LEN
from chess_icl_v2.models import (
    ChessTransformerV2,
    LARGE,
    MEDIUM,
    ModelConfigV2,
    SMALL,
    TINY,
)
from chess_icl_v2.models.small_transformer import apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# Tokenizer v2 tests
# ---------------------------------------------------------------------------


def test_tokenizer_vocab_and_special_tokens():
    tok = ChessV2Tokenizer()
    assert tok.pad_id == 0
    for name in (
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[DEMO_START]",
        "[DEMO_END]",
        "[STATE]",
        "[MOVES]",
        "[OUT_W]",
        "[OUT_D]",
        "[OUT_L]",
        "[QUERY]",
    ):
        assert name in tok.token_to_id

    move_tokens = [t for t in tok.vocab if t.startswith("MOVE_")]
    assert len(move_tokens) == 4096 + 1280 == 5376
    assert tok.vocab_size == 5473


def test_state_roundtrip_and_blocked_barriers():
    tok = ChessV2Tokenizer()
    fens = [
        STANDARD_STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w Kq - 4 9",
        "8/8/8/3k4/8/8/4K3/8 b - - 0 40",
    ]
    for fen in fens:
        ids = tok.encode_state(fen)
        assert len(ids) == STATE_TOKEN_LEN == 70
        decoded = tok.decode_state(ids)
        expected_4_fields = " ".join(fen.split()[:4])
        assert decoded == expected_4_fields

    # Barrier 'X' via FEN string and via blocked_squares list
    blocked_fen = "rnbqkbnr/pppppppp/2X5/8/4X3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ids_from_fen = tok.encode_state(blocked_fen)
    assert len(ids_from_fen) == 70
    assert tok.decode_state(ids_from_fen) == " ".join(blocked_fen.split()[:4])

    # c6 = rank 5, file 2 -> 5*8 + 2 = 42; e4 = rank 3, file 4 -> 3*8 + 4 = 28
    ids_from_arg = tok.encode_state(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        blocked_squares=["c6", 28],
    )
    assert ids_from_arg == ids_from_fen
    assert tok.extract_blocked_squares(ids_from_arg) == [28, 42]


def test_one_token_per_move_roundtrip():
    tok = ChessV2Tokenizer()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "a1h8", "h8a1", "e7e8q", "a7b8n", "c2c1r", "h2g1b"]
    for m in moves:
        tid = tok.encode_move(m)
        assert isinstance(tid, int)
        assert tok.decode_move(tid) == m

    encoded_seq = tok.encode_moves(moves)
    assert len(encoded_seq) == len(moves)
    assert tok.decode_moves(encoded_seq) == moves


def test_extended_pawn_and_antichess_promotions_and_stage0_zero_oov():
    import json

    tok = ChessV2Tokenizer()
    special_promos = ["b3b1n", "a6a8q", "a7a8k", "h2h1k", "f6f8k", "c3c1k"]
    for mv in special_promos:
        tid = tok.encode_move(mv)
        assert isinstance(tid, int)
        assert tok.decode_move(tid) == mv

    stage0_path = Path("runs_v2/stage0/sample_games.jsonl")
    if stage0_path.exists():
        n_games = 0
        n_moves = 0
        oov_moves: list[str] = []
        with stage0_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_games += 1
                rec = json.loads(line)
                for uci in rec.get("moves_uci") or rec.get("moves") or []:
                    n_moves += 1
                    try:
                        tid = tok.encode_move(uci)
                        assert tok.decode_move(tid) == uci.strip().lower()
                    except ValueError:
                        oov_moves.append(uci)
        assert n_games == 2520
        assert n_moves > 0
        assert oov_moves == [], f"Found {len(oov_moves)} OOV moves: {oov_moves[:10]}"


def test_episode_segments_and_k16_capacity():
    tok = ChessV2Tokenizer()
    sample_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"] * 13 + ["b5a4", "g8f6"]
    assert len(sample_moves) == 80

    demos = [
        {"initial_fen": STANDARD_STARTING_FEN, "moves": sample_moves, "result": r}
        for r in (["win", "draw", "loss"] * 6)[:16]
    ]
    query_fen = "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4"
    ep = tok.encode_episode(demos, query_fen, max_len=8192)

    assert not ep["was_truncated"]
    assert ep["num_demos_kept"] == 16
    assert len(ep["input_ids"]) < 3000  # Easily fits in max_seq_len=8192

    ids = ep["input_ids"]
    segs = ep["segment_ids"]
    assert ids[0] == tok.cls_id
    assert segs[0] == 0
    assert ids[-1] == tok.sep_id

    # 1 [CLS] + 16 * 2 (body + outcome) + 1 (query) = 34 segments (0..33)
    assert segs[-1] == 33
    # Verify each outcome token is in an isolated segment separate from its demo body
    for pos, tid in enumerate(ids):
        if tid in (tok.out_w_id, tok.out_d_id, tok.out_l_id):
            assert segs[pos] > segs[pos - 1]


def test_no_query_truncation_guarantee():
    tok = ChessV2Tokenizer()
    long_moves = ["e2e4", "e7e5"] * 30  # 60 moves per demo -> 135 tokens per demo
    demos = [{"moves": long_moves, "result": "win"} for _ in range(5)]
    query_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"

    # Tight budget: fits 1 [CLS] + 1 demo (135) + 73 query = 209 <= 250
    ep = tok.encode_episode(demos, query_fen, max_len=250)
    assert ep["was_truncated"] is True
    assert ep["num_demos_kept"] == 1
    assert len(ep["input_ids"]) <= 250

    # Verify the query state is 100% intact at the end
    q_start = ep["query_start_idx"]
    q_slice = ep["input_ids"][q_start:]
    assert q_slice[0] == tok.query_id
    assert q_slice[1] == tok.state_id
    assert q_slice[-1] == tok.sep_id
    assert len(q_slice) == 2 + STATE_TOKEN_LEN + 1
    assert tok.decode_state(q_slice[2:-1]) == " ".join(query_fen.split()[:4])

    # With allow_trim_demo_prefix=True, an extra demo prefix can be trimmed while query stays intact
    ep_trimmed = tok.encode_episode(
        demos, query_fen, max_len=250, allow_trim_demo_prefix=True
    )
    assert ep_trimmed["was_truncated"] is True
    assert len(ep_trimmed["input_ids"]) <= 250
    q_slice_t = ep_trimmed["input_ids"][ep_trimmed["query_start_idx"] :]
    assert tok.decode_state(q_slice_t[2:-1]) == " ".join(query_fen.split()[:4])

    # Refuses max_len smaller than query overhead instead of truncating query
    with pytest.raises(ValueError, match="un-truncatable query"):
        tok.encode_episode(demos, query_fen, max_len=50)


def test_collate_episodes():
    tok = ChessV2Tokenizer()
    ep1 = tok.encode_episode([{"moves": ["e2e4"], "result": "win"}], STANDARD_STARTING_FEN)
    ep2 = tok.encode_episode(
        [{"moves": ["e2e4", "e7e5", "g1f3"], "result": "loss"}], STANDARD_STARTING_FEN
    )
    batch = collate_episodes([ep1, ep2], pad_token_id=tok.pad_id)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    pad_len = len(ep2["input_ids"]) - len(ep1["input_ids"])
    assert batch["attention_mask"][0][-pad_len:] == [0] * pad_len
    assert batch["input_ids"][0][-pad_len:] == [tok.pad_id] * pad_len


# ---------------------------------------------------------------------------
# Model v2 tests (RoPE, causal vs segment_bidirectional, soft CE, muP)
# ---------------------------------------------------------------------------


def test_rope_relative_position_property():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 16, 32
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    from chess_icl_v2.models.small_transformer import RotaryEmbedding

    rope = RotaryEmbedding(head_dim=D, max_seq_len=64)
    cos, sin = rope(T, q.device)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Inner product of identical vectors at positions (2, 5) vs (7, 10) (both delta = -3)
    q_same = q[:, :, :1, :].expand(B, H, T, D).clone()
    k_same = k[:, :, :1, :].expand(B, H, T, D).clone()
    qr, kr = apply_rotary_pos_emb(q_same, k_same, cos, sin)
    dot_2_5 = (qr[0, 0, 2] * kr[0, 0, 5]).sum()
    dot_7_10 = (qr[0, 0, 7] * kr[0, 0, 10]).sum()
    assert torch.allclose(dot_2_5, dot_7_10, atol=1e-5)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape


def test_model_shapes_predict_proba_and_param_counts():
    tok = ChessV2Tokenizer()
    assert TINY.vocab_size == tok.vocab_size

    t_model = ChessTransformerV2(TINY, pad_token_id=tok.pad_id)
    s_model = ChessTransformerV2(SMALL, pad_token_id=tok.pad_id)
    m_model = ChessTransformerV2(MEDIUM, pad_token_id=tok.pad_id)
    l_model = ChessTransformerV2(LARGE, pad_token_id=tok.pad_id)

    t_params = t_model.count_params()
    s_params = s_model.count_params()
    m_params = m_model.count_params()
    l_params = l_model.count_params()

    assert 1_000_000 < t_params < 2_000_000
    assert 6_000_000 < s_params < 10_000_000
    assert 20_000_000 < m_params < 30_000_000
    assert 195_000_000 < l_params < 220_000_000

    x = torch.randint(1, tok.vocab_size, (3, 96))
    t_model.eval()
    logits = t_model(x)
    probs = t_model.predict_proba(x)
    assert logits.shape == (3, 3)
    assert probs.shape == (3, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_causal_and_segment_bidirectional_no_leak():
    tok = ChessV2Tokenizer()
    demos = [
        {"moves": ["e2e4", "e7e5", "g1f3"], "result": "win"},
        {"moves": ["d2d4", "d7d5"], "result": "loss"},
    ]
    query_fen = STANDARD_STARTING_FEN
    ep = tok.encode_episode(demos, query_fen, max_len=512)

    input_ids = torch.tensor([ep["input_ids"]], dtype=torch.long)
    segment_ids = torch.tensor([ep["segment_ids"]], dtype=torch.long)

    # 1. Verify segment_bidirectional requires segment_ids
    cfg_seg = copy.deepcopy(TINY)
    cfg_seg.attention_mode = "segment_bidirectional"
    cfg_seg.dropout = 0.0
    model = ChessTransformerV2(cfg_seg, pad_token_id=tok.pad_id)
    model.eval()

    with pytest.raises(ValueError, match="segment_ids must be provided"):
        model(input_ids)

    with torch.no_grad():
        base_feats, _ = model.forward_features(input_ids, segment_ids=segment_ids)
        base_logits = model(input_ids, segment_ids=segment_ids)

    # Locate outcome of demo 1 (segment 2) and query start (segment 5)
    seg2_positions = (segment_ids[0] == 2).nonzero(as_tuple=True)[0]
    out1_pos = int(seg2_positions[0].item())
    assert input_ids[0, out1_pos].item() == tok.out_w_id

    # Leak Test A: Mutating Demo 1's outcome token [OUT_W] -> [OUT_L]
    # Must NOT leak into Segment 0 ([CLS]) or Segment 1 (Demo 1 body),
    # while Demo 2 (Segment 3) and Query (Segment 5) DO see the change.
    mutated_out_ids = input_ids.clone()
    mutated_out_ids[0, out1_pos] = tok.out_l_id
    with torch.no_grad():
        mut_out_feats, _ = model.forward_features(mutated_out_ids, segment_ids=segment_ids)

    earlier_mask = segment_ids[0] <= 1
    assert torch.allclose(
        base_feats[:, earlier_mask, :], mut_out_feats[:, earlier_mask, :], atol=0.0, rtol=0.0
    )
    assert (base_feats[:, segment_ids[0] >= 3, :] - mut_out_feats[:, segment_ids[0] >= 3, :]).abs().max() > 1e-4

    # Leak Test B: Mutating a token inside Query (Segment 5)
    # Must NOT change activations of ANY earlier segment (0..4)
    q_pos = ep["query_start_idx"] + 5
    mutated_q_ids = input_ids.clone()
    mutated_q_ids[0, q_pos] = tok.token_to_id["Q"]
    with torch.no_grad():
        mut_q_feats, _ = model.forward_features(mutated_q_ids, segment_ids=segment_ids)

    pre_query_mask = segment_ids[0] < 5
    assert torch.allclose(
        base_feats[:, pre_query_mask, :], mut_q_feats[:, pre_query_mask, :], atol=0.0, rtol=0.0
    )

    # Bidirectionality Test: Mutating a later token inside Segment 1 (e.g., last move of Demo 1)
    # SHOULD change earlier tokens in Segment 1 (unlike causal mode)!
    seg1_positions = (segment_ids[0] == 1).nonzero(as_tuple=True)[0]
    last_seg1_pos = int(seg1_positions[-1].item())
    first_seg1_pos = int(seg1_positions[0].item())
    mut_within_ids = input_ids.clone()
    mut_within_ids[0, last_seg1_pos] = tok.encode_move("a2a3")
    with torch.no_grad():
        mut_within_feats, _ = model.forward_features(mut_within_ids, segment_ids=segment_ids)
    assert (base_feats[0, first_seg1_pos] - mut_within_feats[0, first_seg1_pos]).abs().max() > 1e-4

    # Padding invariance test in segment_bidirectional AND causal modes
    padded_ids = torch.cat([input_ids, torch.full((1, 17), tok.pad_id, dtype=torch.long)], dim=1)
    padded_segs = torch.cat([segment_ids, torch.full((1, 17), int(segment_ids[0, -1]), dtype=torch.long)], dim=1)
    padded_mask = torch.cat(
        [torch.ones_like(input_ids), torch.zeros((1, 17), dtype=torch.long)], dim=1
    )
    with torch.no_grad():
        padded_logits = model(padded_ids, attention_mask=padded_mask, segment_ids=padded_segs)
    assert torch.allclose(base_logits, padded_logits, atol=1e-5)

    # Same padding invariance in causal mode
    cfg_causal = copy.deepcopy(TINY)
    cfg_causal.attention_mode = "causal"
    cfg_causal.dropout = 0.0
    causal_model = ChessTransformerV2(cfg_causal, pad_token_id=tok.pad_id)
    causal_model.eval()
    with torch.no_grad():
        c_base = causal_model(input_ids)
        c_pad = causal_model(padded_ids, attention_mask=padded_mask)
    assert torch.allclose(c_base, c_pad, atol=1e-5)


def test_soft_target_ce_and_backward_pass():
    torch.manual_seed(42)
    cfg = copy.deepcopy(TINY)
    cfg.dropout = 0.0
    model = ChessTransformerV2(cfg, pad_token_id=0)

    input_ids = torch.randint(1, cfg.vocab_size, (4, 64))
    target_probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.34, 0.33, 0.33],
        ],
        dtype=torch.float32,
    )

    logits = model(input_ids)
    loss = model.compute_loss(logits, target_probs)
    expected = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    # Verify soft-target overfitting
    opt = model.get_optimizer(base_lr=3e-3)
    for _ in range(40):
        opt.zero_grad()
        out = model(input_ids)
        l = model.compute_loss(out, target_probs)
        l.backward()
        opt.step()

    with torch.no_grad():
        probs = model.predict_proba(input_ids)
    kl = F.kl_div(probs.log(), target_probs, reduction="batchmean")
    assert kl.item() < 0.05


def test_mup_coordinate_check_and_1step_stability():
    """Verify initial logits are O(1) and 1-step updates stay O(1) across TINY/SMALL/MEDIUM/LARGE."""
    torch.manual_seed(123)
    input_ids = torch.randint(1, TINY.vocab_size, (6, 48))
    target_probs = torch.tensor([[0.6, 0.3, 0.1]] * 6, dtype=torch.float32)

    configs = [("TINY", TINY), ("SMALL", SMALL), ("MEDIUM", MEDIUM), ("LARGE", LARGE)]
    init_stds = {}
    delta_h_means = {}
    delta_logit_stds = {}

    base_lr = 1e-3
    for name, base_cfg in configs:
        cfg = copy.deepcopy(base_cfg)
        cfg.dropout = 0.0
        cfg.use_mup = True
        model = ChessTransformerV2(cfg, pad_token_id=0)
        model.eval()

        opt = model.get_optimizer(base_lr=base_lr, weight_decay=0.0)
        r = float(cfg.d_model) / float(cfg.d_base)

        # Verify exact muP LR scaling on optimizer param groups
        lrs = [g["lr"] for g in opt.param_groups]
        assert pytest.approx(lrs[0], rel=1e-6) == base_lr  # token_emb
        assert pytest.approx(lrs[1], rel=1e-6) == base_lr / r  # hidden decay

        with torch.no_grad():
            h0, _ = model.forward_features(input_ids)
            logits0 = model(input_ids)

        assert torch.isfinite(logits0).all()
        init_std = logits0.std().item()
        init_stds[name] = init_std
        assert 0.02 < init_std < 1.5, f"{name} init logit std out of O(1) range: {init_std}"

        # 1 AdamW step
        opt.zero_grad()
        logits_step = model(input_ids)
        loss = model.compute_loss(logits_step, target_probs)
        loss.backward()
        opt.step()

        with torch.no_grad():
            h1, _ = model.forward_features(input_ids)
            logits1 = model(input_ids)

        dh = (h1 - h0).abs().mean().item()
        dlogit = (logits1 - logits0).abs().mean().item()
        delta_h_means[name] = dh
        delta_logit_stds[name] = dlogit

        assert torch.isfinite(h1).all() and torch.isfinite(logits1).all()
        assert 0.05 < dh < 1.5, f"{name} 1-step hidden delta out of O(1) range: {dh}"
        assert 0.05 < dlogit < 1.5, f"{name} 1-step logit delta out of O(1) range: {dlogit}"

    # Verify both hidden delta and logit delta remain width-invariant across TINY (r=1) -> LARGE (r=8)
    ratio_dh = max(delta_h_means.values()) / min(delta_h_means.values())
    ratio_dlogit = max(delta_logit_stds.values()) / min(delta_logit_stds.values())
    assert ratio_dh < 2.0, f"Hidden update delta diverged across widths: {delta_h_means}"
    assert ratio_dlogit < 1.25, f"Logit update delta diverged across widths: {delta_logit_stds}"


def test_save_and_load_roundtrip(tmp_path: Path):
    cfg = copy.deepcopy(TINY)
    cfg.attention_mode = "segment_bidirectional"
    cfg.dropout = 0.0
    model = ChessTransformerV2(cfg, pad_token_id=0)
    model.eval()

    ckpt_path = tmp_path / "model_v2.pt"
    model.save(ckpt_path)
    loaded = ChessTransformerV2.load(ckpt_path)
    loaded.eval()

    x = torch.randint(1, cfg.vocab_size, (2, 24))
    segs = torch.zeros_like(x)
    segs[:, 12:] = 1
    with torch.no_grad():
        assert torch.allclose(
            model(x, segment_ids=segs), loaded(x, segment_ids=segs), atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward_backward():
    cfg = copy.deepcopy(SMALL)
    cfg.attention_mode = "segment_bidirectional"
    model = ChessTransformerV2(cfg, pad_token_id=0).cuda()

    x = torch.randint(1, cfg.vocab_size, (2, 256), device="cuda")
    segs = torch.arange(256, device="cuda").unsqueeze(0).expand(2, 256) // 64
    targets = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]], device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x, segment_ids=segs)
        loss = model.compute_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss).all()
