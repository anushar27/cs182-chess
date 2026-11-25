import numpy as np
import json
from functools import lru_cache
from .common import canonical
from .data import states_in_game
from .state import State, board_distance


def material(state):
    side,squares=_root_geometry(canonical(state.to_dict()))
    values={'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}
    white=sum(value*(squares.count(piece)-squares.count(piece.lower())) for piece,value in values.items())
    return white if side else -white


def material_features(query, demos):
    side,squares=_root_geometry(canonical(query['state']))
    ours='PNBRQK' if side else 'pnbrqk'
    theirs='pnbrqk' if side else 'PNBRQK'
    return np.array([1.]+[squares.count(piece)/8 for piece in ours+theirs])


def histogram_features(query, demos):
    counts = np.array([sum(g["white_result"] == c for g in demos) for c in ("W", "D", "L")], float)
    counts /= max(1, len(demos))
    lengths = [len(g["moves"]) for g in demos] or [0]
    side = 1. if _root_geometry(canonical(query['state']))[0] else -1.
    return np.array([1., side, len(demos) / 16, np.mean(lengths) / 320,
                     np.std(lengths) / 320, *counts, *(counts * side)])


class SoftRegression:
    def __init__(self, features=material_features):
        self.features = features

    def fit(self, pairs, steps=500, lr=.1, l2=.001):
        if not pairs or any(q["split"] != "train" for q, _ in pairs):
            raise ValueError("Baseline fit requires nonempty training data only")
        x = np.stack([self.features(q, d) for q, d in pairs])
        y = np.stack([q["label"]["target_wdl"] for q, _ in pairs])
        self.weights = np.zeros((x.shape[1], 3))
        for _ in range(steps):
            logits = x @ self.weights
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            self.weights -= lr * (x.T @ (p - y) / len(y) + l2 * self.weights)
        return self

    def predict(self, query, demos):
        logits = self.features(query, demos) @ self.weights
        p = np.exp(logits - logits.max())
        return (p / p.sum()).tolist()


def frequency(train_queries):
    if not train_queries or any(q["split"] != "train" for q in train_queries):
        raise ValueError("Training-frequency baseline needs training queries")
    return np.mean([q["label"]["target_wdl"] for q in train_queries], axis=0).tolist()


def demo_prior(query, demos, majority=False):
    counts = np.array([sum(g["white_result"] == c for g in demos) for c in ("W", "D", "L")], float)
    if majority and demos:
        counts = (counts == counts.max()).astype(float)
    counts += 1.  # Declared Laplace smoothing avoids infinite NLL.
    if not State.from_dict(query["state"]).board().turn:
        counts = counts[::-1]
    return (counts / counts.sum()).tolist()


@lru_cache(maxsize=512)
def _knn_geometries(serialized_game):
    return tuple(state.geometry for state in states_in_game(json.loads(serialized_game)))


@lru_cache(maxsize=4096)
def _root_geometry(serialized_state):
    return State.from_dict(json.loads(serialized_state)).geometry


def knn(query, demos, neighbors=5):
    side, pieces = _root_geometry(canonical(query["state"]))
    candidates = []
    for g in demos:
        geometry = _knn_geometries(canonical({"initial":g["initial"],"moves":g["moves"]}))
        result = g["white_result"]
        if not side and result != "D":
            result = {"W": "L", "L": "W"}[result]
        for turn, squares in geometry:
            if turn == side:
                distance = sum(x != y for x,y in zip(pieces,squares))
                candidates.append((distance,g["family_id"],result))
    counts = np.ones(3)
    for _, _, result in sorted(candidates)[:neighbors]:
        counts[("W", "D", "L").index(result)] += 1
    return (counts / counts.sum()).tolist()
