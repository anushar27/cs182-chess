from __future__ import annotations

from collections import Counter
import random
import time
from .common import digest, provenance, seed_for
from .rollout import continuation, estimate
from .rules import make_board
from .state import State, board_distance, capture, position_signature, is_near

SPLITS = ("train", "validation", "test")


def split_for(family):
    bucket = seed_for("family_fold_v1", family) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def family_seeds(master, variant, pool, split, count):
    found, i = 0, 0
    while found < count:
        family = digest([master, variant, pool, i])
        if split_for(family) == split:
            yield family, seed_for(master, variant, pool, i)
            found += 1
        i += 1


def game(variant, policy, seed, family, split, pool, max_plies):
    state = capture(variant, make_board(variant, seed=seed))
    run = continuation(state, policy, seed, max_plies)
    return {"family_id": family, "split": split, "pool": pool,
            "variant": variant, "initial": state.to_dict(), **run}


def states_in_game(g):
    state = State.from_dict(g["initial"])
    b = state.board()
    yield state
    for uci in g["moves"]:
        b.push_uci(uci)
        yield capture(state.variant, b)


def generate(config):
    required = {"seed", "variants", "policy_id", "max_plies", "replicates",
                "demo_families_per_split", "query_families_per_split", "near_distance", "workers"}
    if set(config) != required:
        raise ValueError(f"Config keys must be {sorted(required)}")
    if config["near_distance"] < 0:
        raise ValueError("Near duplicate threshold cannot be negative")
    start = time.perf_counter()
    data = {"schema": "chess_icl_v2.1", "provenance": provenance(config),
            "demonstrations": [], "source_games": [], "queries": [], "exclusions": []}
    # Generate all demo pools before choosing queries; no test-outcome-dependent filtering.
    for variant in config["variants"]:
        for split in SPLITS:
            for family, seed in family_seeds(config["seed"], variant, "demonstration", split,
                                             config["demo_families_per_split"]):
                data["demonstrations"].append(game(variant, config["policy_id"], seed, family,
                    split, "demonstration", config["max_plies"]))
    demo_states = [s for g in data["demonstrations"] for s in states_in_game(g)]
    accepted = []
    for variant in config["variants"]:
        for split in SPLITS:
            for family, seed in family_seeds(config["seed"], variant, "query_source", split,
                                             config["query_families_per_split"]):
                g = game(variant, config["policy_id"], seed, family, split,
                         "query_source", config["max_plies"])
                data["source_games"].append(g)
                states = list(states_in_game(g))
                # One adjacent white/black pair per family; selection uses source moves only.
                choices = list(range(8, len(states) - 2))  # all earlier prefixes precede source termination
                if not choices:
                    data["exclusions"].append({"family_id": family, "reason": "no_root_pair"})
                    continue
                ply = random.Random(seed_for(seed, "root_selection")).choice(choices)
                for state in (states[ply], states[ply + 1]):
                    comparisons = demo_states + [s for s, fold in accepted if fold != split]
                    near = any(is_near(state, other, config["near_distance"])
                               for other in comparisons)
                    duplicate = any(position_signature(state) == position_signature(s)
                                    for s, _ in accepted)
                    if near or duplicate:
                        data["exclusions"].append({"family_id": family, "state_key": state.key,
                                                   "reason": "near_duplicate"})
                        continue
                    accepted.append((state, split))
                    label = estimate(state, config["policy_id"], config["replicates"],
                                     config["max_plies"], config["seed"], config["workers"])
                    data["queries"].append({"id": digest([family, state.key]), "family_id": family,
                        "split": split, "variant": variant, "state": state.to_dict(), "label": label})
    data["generation_seconds"] = time.perf_counter() - start
    data["integrity"] = audit_dataset(data)
    return data


def audit_dataset(data):
    pools, folds, signatures = {}, {}, {}
    errors = []
    for g in data["demonstrations"] + data["source_games"]:
        f = g["family_id"]
        if f in pools:
            errors.append("duplicate game family")
        if f in pools and pools[f] != g["pool"]:
            errors.append("family crosses pools")
        if f in folds and folds[f] != g["split"]:
            errors.append("family crosses folds")
        if split_for(f) != g["split"]:
            errors.append("incorrect family fold")
        pools[f], folds[f] = g["pool"], g["split"]
        states = list(states_in_game(g))
        if states[-1].board().fen(en_passant="fen") != g["final_fen"]:
            errors.append("game replay mismatch")
    demo_states = [s for g in data["demonstrations"] for s in states_in_game(g)]
    accepted = []
    threshold = data["provenance"]["config"]["near_distance"]
    for q in data["queries"]:
        s = State.from_dict(q["state"])
        if q["variant"] != s.variant:
            errors.append("query rule/state mismatch")
        if pools.get(q["family_id"]) != "query_source" or folds.get(q["family_id"]) != q["split"]:
            errors.append("query source family mismatch")
        key = position_signature(s)
        if key in signatures:
            errors.append("duplicate query position")
        signatures[key] = q["split"]
        compare = demo_states + [other for other, fold in accepted if fold != q["split"]]
        if any(is_near(s, other, threshold)
               for other in compare):
            errors.append("near duplicate crosses pools/folds")
        accepted.append((s, q["split"]))
        label = q["label"]
        if label["unresolved"] and label["target_wdl"] is not None:
            errors.append("censored label became target")
        counts = [sum(r['root_result'] == c for r in label['runs']) for c in ('W','D','L')]
        if counts != label['counts_wdl'] or label['replicates'] != len(label['runs']):
            errors.append("label trace/count mismatch")
        if len({r['seed'] for r in label['runs']}) != label['replicates']:
            errors.append("reused replicate seeds")
        if label["replicates"] != sum(counts) + label["unresolved"]:
            errors.append("label count mismatch")
        expected_target = [n/label['replicates'] for n in counts] if label['unresolved'] == 0 else None
        if label['target_wdl'] != expected_target:
            errors.append("incorrect empirical probability target")
        for r in label['runs']:
            expected = r['white_result']
            if not s.geometry[0] and expected in ('W','L'):
                expected = {'W':'L','L':'W'}[expected]
            if expected != r['root_result'] or (r['status'] == 'unresolved') != (expected is None):
                errors.append("outcome perspective/status mismatch")
    if errors:
        raise ValueError(sorted(set(errors)))
    return {"passed": True, "queries": len(data["queries"]), "families": len(pools),
            "near_distance": threshold, "scope": "all demo states vs roots; cross-fold query roots"}


def summarize(data):
    rows = []
    for variant in data["provenance"]["config"]["variants"]:
        queries = [q for q in data["queries"] if q["variant"] == variant]
        runs = [r for q in queries for r in q["label"]["runs"]]
        plies = sum(len(r["moves"]) for r in runs)
        secs = sum(r["seconds"] for r in runs)
        counts = Counter(r["root_result"] or "unresolved" for r in runs)
        rows.append({"variant": variant, "queries": len(queries), "continuations": len(runs),
            "root_outcome_counts": dict(counts), "unresolved_fraction": counts["unresolved"] / max(1, len(runs)),
            "usable_complete_labels": sum(q["label"]["target_wdl"] is not None for q in queries),
            "mean_plies": plies / max(1, len(runs)), "seconds_per_ply": secs / max(1, plies),
            "projected_core_hours_per_million_continuations": 1e6 * secs / max(1, len(runs)) / 3600,
            "sides": dict(Counter("white" if State.from_dict(q["state"]).board().turn else "black"
                                  for q in queries)),
            "terminal_demo_games": sum(g["variant"] == variant and g["status"] == "terminal"
                                       for g in data["demonstrations"])})
    return {"exploratory_only": True, "rows": rows, "generation_seconds": data["generation_seconds"],
            "exclusions": len(data["exclusions"]), "integrity": data["integrity"]}
