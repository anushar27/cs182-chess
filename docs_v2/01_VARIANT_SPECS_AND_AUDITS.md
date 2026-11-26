# Variant Specifications, v1 Post-Mortem, and Verification Audits (`chess_icl_v2`)

**Date:** 2026-11-25  
**Package:** `chess_icl_v2/variants/`  
**Verification Status:** ALL 19 unit/fuzz/property tests PASS; 300-position differential fuzzing PASS; 17-variant pure-Python reference & property audit PASS.

---

## 1. Architecture Overview

`chess_icl_v2/variants/` replaces v1's wrapper-based `VariantBoard` with a direct subclass of `python-chess`'s `chess.Board`:

- [`specs.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/chess_icl_v2/variants/specs.py): Frozen `VariantSpec` definitions, train/test/compositional splits, and explicit metadata flags (`is_setup_only_control`, `is_visibly_identifiable`, `no_check_rules`, `uses_blocked_squares`, `is_compositional`).
- [`board.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/chess_icl_v2/variants/board.py): `VariantBoard(chess.Board)` overriding `attackers_mask`, `attacks_mask`, `checkers_mask`, `is_check`, `is_into_check`, `was_into_check`, `generate_castling_moves`, `generate_pseudo_legal_moves`, `generate_legal_moves`, `legal_moves`, `is_pseudo_legal`, `is_castling`, `is_checkmate`, `is_stalemate`, `has_insufficient_material`, `outcome`, `is_game_over`, plus `canonical_state_key()`, `to_state_dict()`, `from_state_dict()`, `board_matrix_64()`, `board_fen_with_barriers()`, and `evaluate_material()`.
- [`variants_impl.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/chess_icl_v2/variants/variants_impl.py): Concrete subclasses for `standard`, 13 single variants, and 3 compositional variants, plus `get_board()`.
- [`audit.py`](file:///data/backups/rganapa/Sebastian/cs182-chess/chess_icl_v2/variants/audit.py): Differential fuzzing (`chess.Board`, `chess.variant.AntichessBoard`), an independent from-scratch pure-Python `(rank, file)` reference move/attack verifier, property audits (replay round-trip, state-dict round-trip, perspective-swap symmetry), and `ms/ply` throughput benchmarking.

### Design Decision: Direct `chess.Board` Subclass & Declarative Rule Knobs
Instead of calling `chess.Board.is_into_check()` (which delegates to `_slider_blockers` and `_generate_evasions` assuming standard chess slider geometry), `VariantBoard` uses declarative bitboard rule knobs (`_rook_mode`, `_bishop_mode`, `_queen_mode`, `_knight_mode`, `_king_mode`, `_pawn_forward_capture`, `_pawn_extended_push`, `_pawn_long_opening_push`, `_dark_square_immunity`, `_convert_captures`, `_no_check_rules`, `_require_both_kings`, `_uses_blocked_squares`) and simulates move occupancy directly on 64-bit bitboards inside `is_into_check(move)` using the variant's own `attackers_mask()`. This eliminates all hidden standard-chess slider/pin assumptions while preserving `0.08–0.34 ms/ply` Python rollout speed.

---

## 2. Formal Variant Specifications (17 Total)

### 2.1 Standard Baseline (1)

| Name | Split | Category | Formal Rule |
|---|---|---|---|
| `standard` | `standard` | `STANDARD` | Standard FIDE chess (`chess.Board`). |

### 2.2 Train Variants (4)

| Name | Category | Formal Rule & Edge-Case Contract |
|---|---|---|
| `diagonal_rook` | `MOVE` | Rooks slide, attack, check, and pin along diagonals only; orthogonal rook moves/attacks are disabled. Standard castling remains valid if king and corner rook are unmoved and path is clear/unattacked. Insufficient material treats diagonal rooks identically to bishops (e.g., `K+R_diag vs K` is insufficient; `K+R_light+R_dark vs K` is sufficient). |
| `straight_bishop` | `MOVE` | Bishops slide, attack, check, and pin along ranks and files only; diagonal bishop moves/attacks are disabled. Insufficient material treats straight bishops as sufficient mating material (like rooks). |
| `weakened_queen` | `MOVE` | Queens slide, attack, check, and pin along ranks and files only; diagonal queen movement and attacks are disabled. |
| `extended_pawn` | `MOVE` | Pawns may advance 1 or 2 unobstructed squares forward from any rank (yielding 4 promotions `Q/R/B/N` if landing on the backrank). `ep_square` is set only on a 2-square push from the pawn's starting rank (rank 2 for White, rank 7 for Black) so FEN/en-passant rank invariants remain standard. Diagonal captures and checks are standard. |

### 2.3 Test Variants (9)

| Name | Category | Flags | Formal Rule & Edge-Case Contract |
|---|---|---|---|
| `short_knight` | `MOVE` | — | Knights jump via **Ferz + Dabbaba** offsets `(±1,±1), (±2,0), (0,±2)`, ignoring intermediate occupancy, and attack/check along those same 8 offsets. Because all 8 offsets preserve `(rank+file)%2`, short knights are strictly color-bound (treated like bishops in `has_insufficient_material`). |
| `mobile_king` | `MOVE` | — | Kings slide, attack, check, and pin any unobstructed distance along all 8 orthogonal and diagonal rays (queen movement). Castling remains valid when rights exist and path is clear/unattacked; `_to_chess960`, `_from_chess960`, and `is_castling` are overridden so non-castling king slides (`e1g1`, `e1c1`, `e1h1`, `e1a1`) never falsely trigger castling once castling rights are lost or blocked. |
| `piece_capture` | `CAPTURE` | — | On any capture by a non-king piece (including en passant), the captured non-king piece is converted to the capturer's color on `move.to_square`, while the origin square becomes empty. Captures by a King preserve the King on `move.to_square` (preventing self-elimination). Updates `castling_rights` (`clean_castling_rights()`), `ep_square`, `halfmove_clock` (zeroed on capture), `fullmove_number`, and `_stack` (for `pop()` and repetition draws). Missing-king custom FENs terminate immediately via `is_variant_end()`. |
| `forward_pawn_capture` | `CAPTURE` | — | Pawns may capture an enemy piece 1 square directly ahead (with backrank promotion) in addition to standard diagonal captures and en passant. Consequently, pawns attack and deliver check 1 square directly ahead (`BB_BACKWARD_PAWN_STEP` in `attackers_mask`) as well as diagonally. |
| `dark_square_immunity` | `CAPTURE` | — | Non-king pieces standing on dark squares `(rank + file) % 2 == 0` (e.g., `a1`, `c3`, `e5`) cannot be captured (including en passant when the captured pawn stands on a dark square). **Kings remain vulnerable to check and checkmate on all squares** so games cannot degenerate into immortal kings on dark squares; immune checkers on dark squares cannot be captured to escape check. |
| `blocked_squares` | `BOARD` | `is_visibly_identifiable`, `uses_blocked_squares` | Eight barrier squares are sampled deterministically from `seed` over squares `16..47` (`a3..h6`, ranks 3–6), independent of FEN occupancy so midgame FEN loading never drifts. Barriers block landing, sliding, 2-square pawn pushes, castling paths, and ray-attacks/checks/pins; knights may leap over barriers. Barriers are rendered as `'X'` in `board_matrix_64()` and `board_fen_with_barriers()`, and preserved in `to_state_dict()`, `from_state_dict()`, and `canonical_state_key()`. Only `K vs K` is treated as insufficient material. |
| `long_board` | `BOARD` | — | Frozen **8x8 Extended Opening Pawn Push approximation**: pawns on their starting rank (rank 2 for White, rank 7 for Black) may push `1..6` squares forward provided intermediate and destination squares are clear and the destination is not the promotion rank (effectively up to 5 squares from rank 2→7 or 7→2). `ep_square` is set only on a 2-square push from the starting rank. |
| `knight_army` | `PIECES` | `is_setup_only_control`, `is_visibly_identifiable` | **Setup-only null control.** When initialized without a custom FEN (`fen=None` / default start), places White King on `e1` + 15 White Knights on `a1..h2` and Black King on `e8` + 15 Black Knights on `a7..h8`, with castling rights `'-'`. Passed custom FENs are never overwritten. All movement, capture, and check rules are standard. |
| `antichess` | `OBJECTIVE` | `no_check_rules` | Subclasses `chess.variant.AntichessBoard`: captures (including en passant) are mandatory whenever available; no check, no checkmate, no castling; pawns may promote to King; losing all pieces or being stalemated wins for the side to move. |

### 2.4 Compositional Variants (3)

| Name | Components | Formal Rule & Edge-Case Contract |
|---|---|---|
| `diagonal_rook_and_piece_capture` | `diagonal_rook` + `piece_capture` | Rooks slide and check diagonally only; captured non-king pieces convert to the capturer's color on the capture square (preserving King capturers and missing-king termination). |
| `mobile_king_and_dark_immunity` | `mobile_king` + `dark_square_immunity` | Kings move and check like queens along all 8 rays; non-king pieces on dark squares `(rank+file)%2 == 0` cannot be captured, while kings remain vulnerable to check and checkmate everywhere. |
| `short_knight_and_antichess` | `short_knight` + `antichess` | Knights jump via **Ferz + Dabbaba** `(±1,±1), (±2,0), (0,±2)` (matching `short_knight`!), combined with full antichess rules (mandatory captures, king promotions, no check, no castling, stalemate=win). Overrides `has_insufficient_material` for color-bound short knights when only knights remain. |

---

## 3. Post-Mortem of v1 Bugs Fixed in v2

1. **Missing variant-aware check/attack/checkmate/stalemate detection (Fix 1):**  
   v1 delegated `is_into_check`, `is_game_over`, and `result` to an unmodified `chess.Board`, causing `DiagonalRook`, `StraightBishop`, `WeakenedQueen`, `MobileKing`, `ShortKnight`, `ForwardPawnCapture`, `DarkSquareImmunity`, and `BlockedSquares` to detect checks, pins, castling safety, checkmates, and stalemates using standard chess movement. v2 overrides `attackers_mask`, `attacks_mask`, `checkers_mask`, `is_check`, `is_into_check`, `generate_legal_moves`, `is_checkmate`, `is_stalemate`, `has_insufficient_material`, and `outcome` across all variants.
2. **`KnightArmyBoard` overwrote passed FENs (Fix 2):**  
   v1 ran `_setup()` unconditionally, wiping out any midgame FEN passed to `get_board("knight_army", fen=...)`. v2 uses `starting_fen = "nnnnknnn/nnnnnnnn/8/8/8/8/NNNNNNNN/NNNNKNNN w - - 0 1"` and overrides `reset()`, applying the 15-knight army only when initialized without a custom FEN.
3. **`BlockedSquaresBoard` seed drift, invisible barriers, and unblocked rays (Fix 3):**  
   v1 sampled barriers over `empty` squares of whatever FEN was passed (causing midgame FENs to change barrier locations), did not block ray-attacks/checks/pins through barriers, and hid barriers from FEN/state keys. v2 samples deterministically from `range(16, 48)` (`a3..h6`), ORs `blocked_mask` into ray occupancy in `attackers_mask` and `attacks_mask`, renders `'X'` in `board_matrix_64()` and `board_fen_with_barriers()`, and persists `blocked` in `to_state_dict()` / `from_state_dict()` / `canonical_state_key()`.
4. **`ShortKnight` vs `ShortKnightAntichess` offset mismatch (Fix 4):**  
   v1 used Ferz + Dabbaba `(±1,±1),(±2,0),(0,±2)` in `ShortKnightBoard` but Ferz-only `(±1,±1)` in `ShortKnightAntichessBoard`. v2 freezes Ferz + Dabbaba everywhere and fixes color-bound insufficient-material detection for `short_knight_and_antichess`.
5. **`AntichessBoard` incomplete rules (Fix 5):**  
   v1 lacked king promotions, permitted castling from custom FENs, and used ad-hoc material/stalemate logic. v2 subclasses `chess.variant.AntichessBoard` directly for 100% parity with `python-chess`.
6. **`PieceCaptureBoard` state corruption & disappearing-king bug (Fix 6):**  
   v1 failed to zero `halfmove_clock` on capture, failed to clean `castling_rights` when capturing corner rooks, missed en-passant conversion, and replaced a capturing King with the captured piece (suicidally deleting the player's own King). v2 hooks `chess.Board._push_capture` (preserving `castling_rights`, `ep_square`, `halfmove_clock`, `fullmove_number`, and `_stack` for `pop()`/repetitions) and guards against converting when the capturer is a King.
7. **`LongBoardBoard` frozen specification & `MobileKing` castling collision (Fix 7 + additional fix):**  
   v2 freezes `long_board` as the 8x8 Extended Opening Pawn Push approximation (`1..6` squares from starting rank excluding promotion rank; `ep_square` on 2-square push only) and overrides `MobileKingBoard._to_chess960` / `_from_chess960` / `is_castling` so `e1h1` / `e1g1` slides never erroneously invoke castling once castling rights are lost. Also fixed `StraightBishopBoard` / `DiagonalRookBoard` `has_insufficient_material` (which previously misclassified `K+StraightBishop vs K` as insufficient and `K+DiagonalRook vs K` as sufficient).

---

## 4. Observed Audit & Differential Fuzzing Results

All results below were observed directly from running:
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m chess_icl_v2.variants.audit`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests_v2/test_variants_v2.py -v` (**19 passed in 8.64s**)

### 4.1 Differential Fuzzing vs `python-chess` (300 positions each)

| Variant | Reference Class | Total Pos | Near-Terminal Pos | Legal Move Match | Check Match | Terminal Match | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `standard` | `chess.Board` | 300 | 150 | 300 / 300 | 300 / 300 | 300 / 300 | **PASS** |
| `antichess` | `chess.variant.AntichessBoard` | 300 | 150 | 300 / 300 | 300 / 300 | 300 / 300 | **PASS** |

### 4.2 Independent Pure-Python Reference & Property Audit (All 17 Variants)

Each variant was audited across **20 random trajectories × up to 15 plies + 40 near-terminal positions = 340 positions**, plus **300 `to_state_dict`/`from_state_dict` round-trips**, **300 perspective-swap symmetry checks** (vertical mirror + color flip; 180° rotation + color flip for `dark_square_immunity` variants), **20 full replay/pop round-trips**, and a **400-ply rollout throughput benchmark**.

| Variant | Pos | Near-Term | MoveOK | ChkOK | TermOK | DictRT | SymOK | RepRT | ms/ply | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `standard` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.120 | **PASS** |
| `diagonal_rook` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.232 | **PASS** |
| `straight_bishop` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.241 | **PASS** |
| `weakened_queen` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.232 | **PASS** |
| `extended_pawn` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.284 | **PASS** |
| `short_knight` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.251 | **PASS** |
| `mobile_king` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.265 | **PASS** |
| `piece_capture` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.262 | **PASS** |
| `forward_pawn_capture` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.307 | **PASS** |
| `dark_square_immunity` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.245 | **PASS** |
| `blocked_squares` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.217 | **PASS** |
| `long_board` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.262 | **PASS** |
| `knight_army` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.122 | **PASS** |
| `antichess` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.083 | **PASS** |
| `diagonal_rook_and_piece_capture` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.199 | **PASS** |
| `mobile_king_and_dark_immunity` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.336 | **PASS** |
| `short_knight_and_antichess` | 340 | 40 | 340 | 340 | 340 | 300 | 300 | 20 | 0.092 | **PASS** |
