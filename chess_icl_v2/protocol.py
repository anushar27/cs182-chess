from .common import digest
from .rules import RULES, TRAIN_VARIANTS


def gates(data, audits, pilot=None, profile=None, min_replicates=32):
    variants = data['provenance']['config']['variants']
    queries = data['queries']
    checks = {
        'all_requested_rules_present': set(variants) == set(RULES),
        'simulator_audits': audits.get('passed', False) and all(audits['variants'].get(v, {}).get('passed', False) for v in variants),
        'dataset_integrity': data['integrity']['passed'],
        'no_censored_targets': bool(queries) and all(q['label']['unresolved'] == 0 for q in queries),
        'minimum_replicates': bool(queries) and all(q['label']['replicates'] >= min_replicates for q in queries),
        'both_sides_all_folds': False,
        'pilot_context_gain': bool(pilot and len(pilot.get('familiar_variants', [])) >= 2 and pilot.get('k0_context_gain', 0) > 0 and pilot.get('context_gain_over_knn', 0) > 0 and
                                   pilot.get('relevant_gain_over_mismatched', 0) > 0),
        'measured_h100_profile': bool(profile and profile.get('device_name', '').find('H100') >= 0 and
                                     profile.get('passed', False)),
    }
    diverse = 0
    for variant in TRAIN_VARIANTS:
        counts = [sum(q['label']['counts_wdl'][i] for q in queries if q['variant'] == variant and q['split'] != 'test') for i in range(3)]
        diverse += sum(n > 0 for n in counts) >= 2
    checks['outcome_variation_two_familiar_rules'] = diverse >= 2
    from .state import State
    checks['both_sides_all_folds'] = all(
        {State.from_dict(q['state']).geometry[0] for q in queries if q['variant'] == v and q['split'] == split} == {True, False}
        for v in variants for split in ('train', 'validation', 'test'))
    return {'passed': all(checks.values()), 'checks': checks, 'dataset_sha256': digest(data),
            'simulator_audit_sha256': digest(audits), 'min_replicates': min_replicates,
            'note': 'Necessary gates only. See preregistration for manual protocol-freeze requirements.'}


def can_seal(gate, exploratory=False):
    if gate.get('passed'):
        return True
    required={'all_requested_rules_present','simulator_audits','dataset_integrity',
              'no_censored_targets','minimum_replicates','both_sides_all_folds',
              'measured_h100_profile','outcome_variation_two_familiar_rules'}
    return bool(exploratory and required.issubset(gate.get('checks',{})) and
                all(gate['checks'][key] for key in required))
