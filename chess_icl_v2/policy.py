import random
from .rules import outcome

POLICIES = ("uniform_legal_v1", "terminal_aware_v1")


def choose(board, rng: random.Random, policy):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy}")
    legal = sorted(board.legal_moves, key=lambda m: m.uci())
    if not legal:
        raise ValueError("Policy called on a state with no legal moves")
    if policy == "terminal_aware_v1":
        wins = []
        turn = board.turn
        for move in legal:
            board.push(move)
            result = outcome(board)
            board.pop()
            if result is not None and result.winner == turn:
                wins.append(move)
        if wins:
            legal = wins
    return legal[rng.randrange(len(legal))]
