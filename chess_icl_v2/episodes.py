from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import random
from .common import canonical, digest, seed_for
from .state import State

CONDITIONS = ("zero", "relevant", "mismatched", "corrupt_outcomes", "corrupt_trajectories")


@dataclass
class Episode:
    input_ids: list[int]
    text: str
    target: list[float] | None
    metadata: dict


class ByteTokenizer:
    """Smoke-test vocabulary only. Not a substitute for measuring Qwen tokens."""
    pad_token_id = 0
    vocab_size = 257

    def encode(self, text, **kwargs):
        return [b + 1 for b in text.encode("utf-8")]


@lru_cache(maxsize=4096)
def _encode_cached(tokenizer, text):
    return tuple(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=2048)
def _query_suffix(serialized_state):
    state = State.from_dict(json.loads(serialized_state))
    return "<QUERY>\n" + canonical(state.public()) + "\n</QUERY>\n<PREDICT_WDL>"


@lru_cache(maxsize=4096)
def _demo_block(serialized_initial, moves, result):
    initial=State.from_dict(json.loads(serialized_initial)).public()
    return "<DEMO>\n"+canonical({'initial':initial,'moves_uci':list(moves),'white_outcome':result})+"\n</DEMO>\n"


def demo_text(g, condition, rng):
    moves=list(g['moves']);result=g['white_result']
    if condition=='corrupt_outcomes':
        result=rng.choice([v for v in ('W','D','L') if v!=result])
    if condition=='corrupt_trajectories':rng.shuffle(moves)
    return _demo_block(canonical(g['initial']),tuple(moves),result)


def build_episode(query, demonstrations, tokenizer, k, max_tokens, condition="relevant", draw=0):
    if condition not in CONDITIONS or k < 0:
        raise ValueError("Invalid context condition or K")
    if condition == "zero" and k != 0:
        raise ValueError("zero condition requires K=0")
    if condition != "zero" and k == 0:
        condition = "zero"
    suffix = _query_suffix(canonical(query["state"]))
    encode = lambda s: _encode_cached(tokenizer, s)
    if len(encode(suffix)) > max_tokens:
        raise ValueError("Query alone exceeds context budget; never truncate the query")
    same = condition != "mismatched"
    pool = [g for g in demonstrations if g["split"] == query["split"] and
            g["status"] == "terminal" and g["family_id"] != query["family_id"] and
            ((g["variant"] == query["variant"]) == same) and
            g["policy_id"] == query["label"]["policy_id"]]
    if len({g["family_id"] for g in pool}) != len(pool):
        raise ValueError("Duplicate demonstration families")
    rng = random.Random(seed_for("demo_draw_v1", query["id"], draw))
    if k > len(pool):
        raise ValueError(f"Need {k} independent complete demos; only {len(pool)} available")
    chosen = rng.sample(pool, k)
    text = "".join(demo_text(g, condition, rng) for g in chosen) + suffix
    ids = list(encode(text))
    if len(ids) > max_tokens:
        raise ValueError(f"Full K={k} context requires {len(ids)} tokens > {max_tokens}; report infeasible K")
    return Episode(ids, text, query["label"]["target_wdl"],
                   {"query_id": query["id"], "family_id": query["family_id"],
                    "variant": query["variant"], "split": query["split"], "k": k,
                    "condition": condition, "draw": draw, "tokens": len(ids),
                    "demo_families": [g["family_id"] for g in chosen], "input_sha256": digest(text)})


def paired_episodes(query, demos, tokenizer, k, max_tokens, draw=0, tolerance=.1):
    """Deterministic best-length mismatched draw; fail if controls cannot be matched."""
    relevant = build_episode(query, demos, tokenizer, k, max_tokens, "relevant", draw)
    candidates = []
    zero = build_episode(query, demos, tokenizer, 0, max_tokens, "zero", draw)
    rlen = len(relevant.input_ids) - len(zero.input_ids)
    if k == 1:
        pool=[g for g in demos if g['split']==query['split'] and g['status']=='terminal' and
              g['variant']!=query['variant'] and g['family_id']!=query['family_id'] and
              g['policy_id']==query['label']['policy_id']]
        # Isolated demo length ranks candidates only. Admission always checks the complete encoded input.
        def key(g):
            block=demo_text(g,'mismatched',random.Random(0))
            try:
                length=len(_encode_cached(tokenizer,block))
            except ValueError:  # Structured scratch vocabulary requires a complete query suffix.
                length=len(_encode_cached(tokenizer,block+zero.text))-len(zero.input_ids)
            return abs(length-rlen),seed_for(query['id'],draw,g['family_id'])
        for g in sorted(pool,key=key):
            try:
                ep=build_episode(query,[g],tokenizer,1,max_tokens,'mismatched',draw)
                mlen=len(ep.input_ids)-len(zero.input_ids)
                if abs(mlen-rlen)<=tolerance*max(1,rlen):
                    candidates.append(ep)
                    break
            except ValueError:
                continue
    else:
        for attempt in range(64):
            try:
                ep = build_episode(query, demos, tokenizer, k, max_tokens, "mismatched", draw * 64 + attempt)
                candidates.append(ep)
            except ValueError:
                continue
    if not candidates:
        raise ValueError("No full-game mismatched context meets K/policy/split and exact token-length tolerance")
    best = min(candidates, key=lambda e: abs(len(e.input_ids) - len(relevant.input_ids)))
    # Match DEMO length, not query length (a large query must not hide a bad match).
    zero = build_episode(query, demos, tokenizer, 0, max_tokens, "zero", draw)
    rlen = len(relevant.input_ids) - len(zero.input_ids)
    mlen = len(best.input_ids) - len(zero.input_ids)
    if abs(mlen - rlen) > tolerance * max(1, rlen):
        raise ValueError("Mismatched control fails demonstration-token length tolerance")
    best.metadata["candidate_draw"] = best.metadata["draw"]
    best.metadata["draw"] = draw
    return zero, relevant, best
