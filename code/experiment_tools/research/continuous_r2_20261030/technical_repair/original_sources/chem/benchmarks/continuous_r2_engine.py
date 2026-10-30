"""R2 policy: prospective refresh validation and bounded evidence inheritance.

No objective provider or LLM is called here. The B4-subset toggle is a separate
ablation; all GP, acquisition, budget, and tie primitives remain frozen.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from . import revised_continuous_engine as r1
from . import continuous_benchmark_engine as old
from . import continuous_benchmark_data as data

ROOT = Path('data/runs/continuous_r2_20261030')
METHODS = ('New_Agent_R2', 'New_Agent_R2_Subset')
BOOTSTRAP = {'SnAr': 2, 'HPLC': 3}
PARAMETERS = {
    'version': 'Continuous_R2', 'seed': 0, 'initial_queries': 4, 'additional_queries': 16,
    'GP_acquisition_LGBO_budget_ties': 'unchanged frozen continuous adapter and prefix seeds',
    'bootstrap_replay': BOOTSTRAP,
    'refresh': {'maximum': 3, 'maximum_before_query9': 2, 'last_query': 14, 'minimum_gap': 2,
                'deployment': 'refresh round executes BO; proposal is shadow-tested on subsequent feedback before next-round promotion',
                'validation': 'one informative frozen shadow prediction must improve paired squared error over GP; empty knowledge withdraws preferences',
                'rejected': 'keep BO suspension; await another authorized event refresh'},
    'negative_inheritance': {'overlap': 'intersection / smaller full-pool geometric membership set',
                'minimum_overlap': .5, 'mean': 'min(0.5, current means of overlapping banks with negative actual deployed gain)',
                'variance': 'at least 0.25, or larger elapsed source variance',
                'shadow_is_not_actual_KF': True},
    'actual_KF': deepcopy(r1.PARAMETERS['scalar_KF']),
    'later_joint_test': {'query': 6, 'allow_keep_BO': True, 'skip_during_prospective_BO_validation': True,
                        'candidate_geometry_and_acquisition_gate': 'unchanged R1'},
    'subset_ablation_only': {'minimum_BO_shadow_observations': 2, 'positive_sum_and_latest_gain': True,
                'rank_sensitivity': 'same eligible candidate at half and full current coefficient; not a confidence interval',
                'negative_actual_gain_resets_subset_evidence': True},
    'interpretation': 'one frozen version and one independent subset toggle; no post-outcome tuning',
}


def seed_cache(pool):
    target = ROOT/'cache'/pool['task']; target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for parent in [Path('data/runs/continuous_20261029/cache'), r1.ROOT/'cache']:
        for source in sorted((parent/pool['task']).glob('*.pt')):
            destination = target/source.name
            if not destination.exists():
                shutil.copyfile(source, destination); copied += 1
            else:
                assert data.filehash(source) == data.filehash(destination), source
    return copied


class Cache(old.Cache):
    def __init__(self, pool):
        super().__init__(pool)
        self.directory = ROOT/'cache'/pool['task']
        self.directory.mkdir(parents=True, exist_ok=True)


def choose(d, scores, mask=None):
    js = np.flatnonzero(d['eligible'] if mask is None else mask); assert len(js)
    js = js[scores[js] == scores[js].max()]
    js = js[d['information_proxy'][js] == d['information_proxy'][js].max()]
    js = js[d['stats']['sd_y'][js] == d['stats']['sd_y'][js].max()]
    return int(js[np.argmin(d['tie_priority'][js])])


def spec_mask(pool, spec):
    if not spec:
        return np.zeros(len(pool['ids']), dtype=bool)
    z = np.asarray(pool['z']); result = np.ones(len(z), dtype=bool)
    for j, axis in enumerate(pool['axes']):
        lo, hi = spec['allowed'][axis]; result &= (z[:, j] >= lo) & (z[:, j] <= hi)
    return result


def inherit_bank(pool, state, spec, t):
    key = old.trust.geometry_key(spec)
    if not key:
        return None, {'geometry_key': None, 'sources': [], 'initial_mean': 0.}
    if key in state['banks']:
        bank = deepcopy(state['banks'][key])
        return bank, {'geometry_key': key, 'existing_geometry': True, 'sources': [], 'initial_mean': bank['mean']}
    bank = r1.new_bank(t); mask = spec_mask(pool, spec); sources = []
    for source_key, entry in sorted(state.get('geometry_ledger', {}).items()):
        if not entry['negative_actual_count'] or source_key not in state['banks']:
            continue
        other = spec_mask(pool, entry['region']); denominator = min(int(mask.sum()), int(other.sum()))
        overlap = float(np.sum(mask & other)/denominator) if denominator else 0.
        if overlap < .5:
            continue
        prior = r1.forecast(state['banks'][source_key], t)
        bank['mean'] = min(bank['mean'], prior['mean'])
        bank['variance'] = max(bank['variance'], prior['variance'])
        sources.append({'geometry_key': source_key, 'overlap_of_smaller_set': overlap,
                        'negative_actual_count': entry['negative_actual_count'],
                        'source_mean': prior['mean'], 'source_variance': prior['variance']})
    return bank, {'geometry_key': key, 'existing_geometry': False, 'sources': sources,
                  'initial_mean': bank['mean'], 'initial_variance': bank['variance'],
                  'rule': PARAMETERS['negative_inheritance']}


def register_feedback(state, event):
    key = event['bank_key']
    if key is None:
        return
    entry = state.setdefault('geometry_ledger', {}).setdefault(key, {
        'region': deepcopy(event['region']), 'negative_actual_count': 0, 'actual_updates': 0})
    entry['actual_updates'] += int(event['measurement']['feature'] != 0.)
    if event['measurement']['actual_paired_SE_gain'] < -1e-12:
        entry['negative_actual_count'] += 1
        entry['last_negative_query'] = event['iteration']
    entry['last_query'] = event['iteration']


def suspend(d, reason):
    d.setdefault('pre_suspension_proposal', d['normal_proposal'])
    d.update(selected=d['reference'], normal_proposal=d['reference'],
             score=d['base'], guided_meta=d['base_meta'], amplitude=0., gate_open=False,
             fallback=True, shift_std=np.zeros_like(d['raw_shift_std']), suspension_reason=reason)
    return d


def subset_permission(state, d):
    book = state.get('subset_evidence', {}).get(d['bank_key'], {})
    rows = book.get('observations', [])
    gains = [r['paired_SE_gain'] for r in rows]
    allowed = bool(len(rows) >= 2 and sum(gains) > 1e-12 and gains[-1] > 1e-12)
    return {'allowed': allowed, 'BO_shadow_count': len(rows), 'sum_gain': float(sum(gains)),
            'latest_gain': gains[-1] if gains else None,
            'evidence_scope': 'retained canonical geometry, prospective BO-only shadow predictions'}


def setup(cache, state, t):
    d = r1.setup(cache, state, t)
    lineage = None
    if d['bank_key'] and d['bank_key'] not in state['banks']:
        bank, lineage = inherit_bank(cache.pool, state, d['region'], t)
        state['banks'][d['bank_key']] = bank
        state.setdefault('bank_initializations', []).append({'query': t, **lineage})
        d = r1.setup(cache, state, t)
    d['inheritance'] = lineage
    d['subset_recovery'] = {'enabled': state['method'] == METHODS[1], 'applied': False}
    conflict = bool(d['diagnosis'].revoked_ids) or d['diagnosis'].conflict_kind != 'none'
    if state.get('active_suspended'):
        return suspend(d, 'pending_or_rejected_refresh_proposal')
    if state['method'] == METHODS[1] and conflict and d['bank_key'] and d['retained']:
        permission = subset_permission(state, d)
        d['subset_recovery'].update(permission)
        if permission['allowed'] and d['admission']['admitted']:
            alpha = d['prior']['amplitude']
            scores, meta = cache.score(state['history'], alpha*d['raw_shift_std'])
            half, _ = cache.score(state['history'], .5*alpha*d['raw_shift_std'])
            full_j, half_j = choose(d, scores), choose(d, half)
            stable = full_j == half_j
            d['subset_recovery'].update(rank_stable=stable,
                full_proposal_id=cache.pool['ids'][full_j], half_proposal_id=cache.pool['ids'][half_j])
            if stable:
                d.update(selected=full_j, normal_proposal=full_j, score=scores, guided_meta=meta,
                         amplitude=alpha, gate_open=True, fallback=False, shift_std=alpha*d['raw_shift_std'])
                d['subset_recovery']['applied'] = True
    return d


def promote(state, t):
    pending = state.get('pending')
    if not pending or pending['status'] != 'validated':
        return None
    assert pending['created_query'] < t and pending['checks'][-1]['query'] < t
    state['knowledge'] = deepcopy(pending['knowledge'])
    if pending['bank_key']:
        state['banks'].setdefault(pending['bank_key'], deepcopy(pending['initial_bank']))
    receipt = {'query': t, 'proposal_created_query': pending['created_query'],
               'geometry_key': pending['bank_key'], 'knowledge': deepcopy(pending['knowledge']),
               'initial_bank': deepcopy(pending['initial_bank']), 'inheritance': deepcopy(pending['inheritance']),
               'validated_by': deepcopy(pending['checks'][-1]), 'evidence_kind': 'shadow_only_not_actual_KF_gain'}
    entry = state['region_trust'].get(pending['bank_key'])
    check = pending['checks'][-1]
    if entry and entry['blocked'] and check['after_observation_count'] > entry['blocked_observation_count'] and check['positive_evidence']:
        entry['pending_evidence'] = {'after_observation_count': check['after_observation_count'], **deepcopy(check)}
    state.setdefault('promotions', []).append(deepcopy(receipt))
    state.setdefault('proposal_archive', []).append(deepcopy(pending))
    state['pending'] = None; state['active_suspended'] = False
    return receipt


def refresh_permission(state, t):
    quota = r1.refresh_permission(state, t)
    if state.get('pending') and state['pending']['status'] == 'awaiting_feedback':
        quota.update(allowed=False, reason='pending_prospective_validation')
    return quota


def stage_refresh(pool, state, t, answer, key, reasons, quota):
    knowledge = deepcopy(answer['plan']['knowledge'])
    spec = old.region(pool, knowledge, [h['id'] for h in knowledge['clauses']])
    bank, inheritance = inherit_bank(pool, state, spec, t)
    if state.get('pending'):
        state.setdefault('proposal_archive', []).append(deepcopy(state['pending']))
    state['calls'].append({'iteration': t, 'reasons': list(reasons), 'quota_before': deepcopy(quota),
        'knowledge_before': deepcopy(state['knowledge']), 'answer': deepcopy(answer), 'context_sha256': key,
        'deployment_policy': 'pending until prospective BO feedback', 'inheritance': deepcopy(inheritance)})
    state['pending'] = {'created_query': t, 'knowledge': knowledge, 'region': spec,
        'bank_key': old.trust.geometry_key(spec), 'initial_bank': bank, 'inheritance': inheritance,
        'status': 'awaiting_feedback', 'checks': [], 'context_sha256': key}
    state['active_suspended'] = True


def pending_forecast(cache, state, d, t):
    pending = state.get('pending')
    if not pending or pending['status'] != 'awaiting_feedback':
        return None
    raw, meta = cache.field(state['history'], pending['region'])
    coefficient = float(pending['initial_bank']['mean']) if pending['initial_bank'] else 0.
    # Zero inherited deployment remains zero. A separate .5 shadow probe can
    # validate geometry; it never increases the actual KF bank on its own.
    probe = coefficient if coefficient > 0. else .5
    return {'knowledge': deepcopy(pending['knowledge']), 'region': deepcopy(pending['region']),
            'bank_key': pending['bank_key'], 'inherited_deployment_coefficient': coefficient,
            'shadow_probe_coefficient': probe, 'raw_shift_std': raw,
            'field_meta': meta, 'evidence_kind': 'prospective_shadow_only'}


def finish_pending(state, frozen, d, t, y, candidate_id):
    if frozen is None:
        return None
    assert d['selected'] == d['reference'] and not d['gate_open']
    j = d['selected']; mu = float(d['stats']['mu_y'][j])
    raw = float(frozen['raw_shift_std'][j]*d['stats']['std'])
    shift = frozen['shadow_probe_coefficient']*raw
    gain = float((y-mu)**2-(y-mu-shift)**2)
    empty = frozen['region'] is None
    informative = abs(shift) > 1e-12
    positive = bool(informative and gain > 1e-12)
    receipt = {'query': t, 'candidate_id': candidate_id, 'GP_mean_y': mu,
        'raw_shift_y': raw, 'frozen_probe_coefficient': frozen['shadow_probe_coefficient'],
        'frozen_probe_prediction_y': mu+shift, 'observed_y': float(y), 'paired_SE_gain': gain,
        'positive_evidence': positive, 'informative': informative,
        'after_observation_count': len(state['history'])+1, 'evidence_kind': 'shadow_not_deployed',
        'inherited_deployment_coefficient_unchanged': frozen['inherited_deployment_coefficient']}
    state['pending']['checks'].append(receipt)
    state['pending']['status'] = 'validated' if positive or empty else 'rejected' if informative else 'awaiting_feedback'
    if empty:
        receipt['withdrawal_without_positive_claim'] = True
    return receipt


def subset_feedback(state, d, t, y):
    key = d['bank_key']; conflict = bool(d['diagnosis'].revoked_ids) or d['diagnosis'].conflict_kind != 'none'
    if not key or not conflict or d['gate_open'] or d['selected'] != d['reference']:
        return None
    j = d['selected']; mu = float(d['stats']['mu_y'][j])
    raw = float(d['raw_shift_std'][j]*d['stats']['std'])
    coefficient = d['prior']['amplitude'] if d['prior']['amplitude'] > 0. else .5
    shift = coefficient*raw
    if abs(shift) <= 1e-12:
        return None
    receipt = {'query': t, 'geometry_key': key, 'GP_mean_y': mu, 'raw_shift_y': raw,
        'frozen_probe_coefficient': coefficient, 'frozen_probe_prediction_y': mu+shift,
        'observed_y': float(y), 'paired_SE_gain': float((y-mu)**2-(y-mu-shift)**2),
        'evidence_kind': 'prospective_BO_only_shadow_not_actual_KF'}
    state.setdefault('subset_evidence', {}).setdefault(key, {'observations': []})['observations'].append(receipt)
    return receipt
