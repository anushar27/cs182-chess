from collections import defaultdict
import numpy as np


def validate_probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError(f"Invalid WDL probability vector: {values}")
    return p


def score(target, prediction):
    y, p = validate_probabilities(target), validate_probabilities(prediction)
    return {"nll": float(-np.dot(y, np.log(np.maximum(p, 1e-12)))),
            "brier": float(np.square(p - y).sum()),
            "expected_outcome_brier": float(1 + np.dot(p,p) - 2 * np.dot(p,y)),
            "l1": float(np.abs(p - y).sum()),
            "argmax_accuracy": float(np.argmax(y) == np.argmax(p)),
            "target_entropy": float(-np.dot(y, np.log(np.maximum(y, 1e-12))))}


def aggregate(rows):
    if not rows:
        return {"n": 0}
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(score(row["target"], row["prediction"]))
    per_variant = {v: {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
                   for v, scores in by_variant.items()}
    return {"n": len(rows), "per_variant": per_variant,
            "macro": {k: float(np.mean([s[k] for s in per_variant.values()]))
                      for k in next(iter(per_variant.values()))},
            "target_argmax_counts": np.bincount([np.argmax(r["target"]) for r in rows], minlength=3).tolist(),
            "argmax_ties": int(sum(sum(np.isclose(r["target"], max(r["target"]))) > 1 for r in rows))}


def paired_gains(rows, condition="relevant", k=1):
    zeros = {(r["query_id"], r["model_seed"]): r for r in rows if r["k"] == 0}
    gains = []
    for r in rows:
        if r["condition"] != condition or r["k"] != k:
            continue
        z = zeros.get((r["query_id"], r["model_seed"]))
        if z is None or z["target"] != r["target"]:
            raise ValueError("Context gain requires identical roots, targets and model seeds")
        gains.append({**r, "gain": score(r["target"], z["prediction"])["nll"] -
                                       score(r["target"], r["prediction"])["nll"]})
    return gains


def hierarchical_ci(gains, seed=0, replicates=1000):
    if not gains:
        return {"status": "no paired data"}
    rng = np.random.default_rng(seed)
    seeds = sorted({r["model_seed"] for r in gains})
    variants = sorted({r["variant"] for r in gains})
    if any(not any(r["model_seed"] == s and r["variant"] == v for r in gains)
           for s in seeds for v in variants):
        raise ValueError("Incomplete seed-by-variant matrix")
    prepared={}
    for model_seed in seeds:
        for v in variants:
            rows=[r for r in gains if r['model_seed']==model_seed and r['variant']==v]
            draws=sorted({r['draw'] for r in rows})
            entries={}
            for draw in draws:
                dr=[r for r in rows if r['draw']==draw]
                families=sorted({r['family_id'] for r in dr})
                entries[draw]=(np.array([sum(r['gain'] for r in dr if r['family_id']==f) for f in families]),
                               np.array([sum(r['family_id']==f for r in dr) for f in families]))
            prepared[(model_seed,v)]=(draws,entries)
    values=[]
    for _ in range(replicates):
        seed_values=[]
        for model_seed in rng.choice(seeds,len(seeds),replace=True):
            variant_values=[]
            for v in variants:
                draws,entries=prepared[(model_seed,v)];draw_values=[]
                for draw in rng.choice(draws,len(draws),replace=True):
                    totals,counts=entries[draw]
                    sampled=rng.choice(len(totals),len(totals),replace=True)
                    draw_values.append(float(totals[sampled].sum()/counts[sampled].sum()))
                variant_values.append(np.mean(draw_values))
            seed_values.append(np.mean(variant_values))
        values.append(np.mean(seed_values))
    per_variant = [np.mean([r["gain"] for r in gains if r["variant"] == v]) for v in variants]
    return {"macro_gain": float(np.mean(per_variant)), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "model_seeds": len(seeds), "demo_draws": len({r["draw"] for r in gains}),
            "bootstrap_replicates": replicates,
            "warning": "Single-seed intervals omit between-seed uncertainty" if len(seeds) < 2 else None,
            "mc_label_noise": "not included; report multinomial uncertainty separately"}


def calibration(rows, bins=10):
    """Classwise reliability against soft empirical labels; no independence claim."""
    output={}
    for c,name in enumerate(('win','draw','loss')):
        values=[]
        for i in range(bins):
            selected=[r for r in rows if min(int(r['prediction'][c]*bins),bins-1)==i]
            if selected:
                predicted=float(np.mean([r['prediction'][c] for r in selected]))
                observed=float(np.mean([r['target'][c] for r in selected]))
                values.append({'bin':i,'n':len(selected),'mean_prediction':predicted,'mean_target':observed})
        output[name]={'bins':values,'weighted_absolute_gap':sum(v['n']*abs(v['mean_prediction']-v['mean_target']) for v in values)/max(1,len(rows))}
    return output


def paired_context_difference(rows,k=1):
    mismatched={(r['query_id'],r['model_seed'],r['draw']):r for r in rows if r['condition']=='mismatched' and r['k']==k}
    gains=[]
    for row in rows:
        if row['condition']!='relevant' or row['k']!=k:continue
        other=mismatched.get((row['query_id'],row['model_seed'],row['draw']))
        if other is None or other['target']!=row['target']:
            raise ValueError('Relevance comparison requires paired roots/seeds/demo draws')
        gains.append({**row,'gain':score(row['target'],other['prediction'])['nll']-score(row['target'],row['prediction'])['nll']})
    return gains
