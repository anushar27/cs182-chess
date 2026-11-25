# Frozen simulator specifications

Rules use `research_rules_v1`, except capture conversion, corrected and frozen as
`research_rules_v2_royal_conversion`. The correction checks the captured rook's
color when distinguishing castling from a king capture; every capture-v2 dataset
and the all-rule reference were regenerated. See [artifact status](ARTIFACT_STATUS.md).

These are **research definitions**, based on the source branch at
`e6aa1fece489ac23552e51dbed67985fc60c9f5a`. They resolve ambiguous legacy behavior
explicitly. A result obtained from the old wrappers is not a v2 result.

All boards have 8 files and 8 ranks. Standard chess is a separate warmup/reference
control, in addition to the 13 project variants. Unless a row changes it, movement,
promotion (Q/R/B/N), en passant, royal kings, and castling follow standard chess.
Check and king safety use the active variant's attacks. Royal kings cannot be
captured. Pinned pieces still attack for purposes of king safety, as in chess.

| Identifier | Exact change | Important boundary decisions |
|---|---|---|
| diagonal_rook | Rooks slide diagonally, stopping at the first piece/barrier | Castling disabled because the rook no longer has rook movement; the independent reference uses the same convention |
| straight_bishop | Bishops slide orthogonally | Rooks and castling unchanged |
| weakened_queen | Queens slide orthogonally only | Promotion to Q produces this modified queen |
| extended_pawn | Pawns may push one or two squares from any rank, with clear intermediate/destination squares | Two-step arrival on the final rank promotes; every two-step creates a normal diagonal en-passant opportunity on the intervening square, even off the starting rank |
| short_knight | Knights jump by (±1,±1), (±2,0), (0,±2), i.e. ferz+dabbaba FD | No orthodox (2,1) jump; the same offsets generate attacks |
| mobile_king | Royal king slides as a queen; only destination king safety applies to ordinary slides | Castling disabled; a slide past attacked intermediate squares is allowed. Disable python-chess's automatic king/corner castling UCI conversions |
| piece_capture | On capture, the moving piece disappears and the victim remains at the destination with the mover's color | En passant removes the victim and places its recolored pawn on the landing square. Conversion overrides a capture-promotion's selected type; quiet promotions are normal. Captures reset the halfmove clock. Castling is not a capture. Kings remain royal |
| forward_pawn_capture | Pawns additionally capture one square straight forward | Standard diagonal captures, starting double pushes, promotions, en passant remain. Forward capture attacks can give check |
| dark_square_immunity | A piece on a dark square cannot be captured | Applies to kings, so a king on a dark square cannot be checked. For en passant test the victim's square. Quiet moves to dark squares are allowed. Reflecting ranks changes square colors and is not a rule-preserving symmetry |
| blocked_squares | Eight empty initial squares are permanent impassable barriers | Derive once from the game's seed, then serialize the exact squares; never regenerate them from a later FEN. Sliding attacks, pawn double paths, and castling paths are blocked; knights can jump over barriers. No move lands on a barrier |
| extended_initial_pawn | Legacy `long_board`, renamed: on the ordinary 8×8 board, a pawn on its starting rank may push up to six squares through clear squares, stopping before the final rank on multi-pushes | This is **not 8×20 chess**. A single step promotes normally. En passant is available after a two-step only. No hidden first-move flag: eligibility follows current rank, as in the old implementation |
| knight_army | White: king e1 and knights on the other 15 squares of ranks 1–2; Black: king e8 and knights on the other 15 squares of ranks 7–8 | Castling rights initially absent. All movement and outcomes are orthodox. Supplying a FEN must never replace it with the initial army. Setup-only null control |
| antichess | The pinned `chess.variant.AntichessBoard` rules | Forced capture, nonroyal king, king promotion, no castling; side with no pieces or no legal moves wins. Use this precise variant, not the distinct SuicideBoard material-tiebreak convention |

## Ending games and claims

Standard chess and Knight Army retain python-chess insufficient-material rules.
Other custom royal variants do **not** reuse orthodox insufficient-material
heuristics: they may be unsound for altered movement or captures. Their draws are
stalemate, automatic fivefold repetition, or 150 halfmoves without a pawn move or
capture (75-move rule), with mate taking precedence. This is a deliberate, versioned
adjudication convention; it can delay termination in dead positions. Antichess
uses its pinned board's terminal rules and material rules. Optional threefold and
50-move claims are never made (`claim_draw=False`), because a move-only policy does
not specify a claim action. No resignation or agreed draws.

The rollout budget is an administrative stopping point, **not a game rule**.
`unresolved` is never a draw or a material label. A terminal result exactly on the
last allowed ply is terminal, including a genuine rule-defined draw.

## State and audits

Persist adapter/rule version, initial FEN, complete legal move history and barrier
metadata. Replaying restores repetitions, side, promotions, castling, en passant,
clocks, and move numbering. `State.key` hashes all of this; FEN alone is insufficient.
Copies and push/pop preserve barriers and history. Public input removes rule IDs
and provenance but retains physical state and history. Consequently K=0 can still
contain evidence about rules in the query history; it is not a history-free board
condition. Barrier geometry and Knight Army setup are visible by design.

`audit.py` checks replay, save/load, push/pop, copy-related replay and the applicable
color/rank symmetry on reachable states. Fairy-Stockfish via pinned pyffish compares
legal moves for standard, Knight Army, antichess and seven configured movement
variants. For independently encountered no-move terminals, it also compares result
signs. Optional-draw conventions and insufficient-material implementations differ;
this is **not** a full differential terminal-status equivalence claim. Separate
golden tests cover the declared automatic draws, mate, antichess ending, special
captures and barriers. Four custom rules (conversion, immunity, barriers, and the
renamed approximation) currently have property/golden coverage, not independent
engine equivalence. Do not describe random fuzzing as proof of correctness.

The pyffish custom pawn reference needs Betza `ifmnD` for its initial double-step;
setting only double-step region metadata did not add that move to a custom pawn.
Wall-square FENs exist in Fairy-Stockfish; an exact barrier adapter is not yet
validated, so blocked squares remain in-house for the reported audit.
