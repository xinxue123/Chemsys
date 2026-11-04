"""Independent, read-only audit of frozen knowledge-loop artifacts.

No runner/engine/policy imports, LLM calls, model fitting, or oracle-value reads.
Input/oracle files are hashed for custody only. Report generation never modifies
the experiment directories. PASS means only the explicitly listed checks pass;
it is neither proof of GP fitting correctness nor a scientific performance gate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import exp, log, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.special import expit, log_ndtr, logsumexp


ROOT = Path(__file__).resolve().parents[2]
AUDIT_IMPLEMENTATION_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()
SEED1_REGISTRY_PATH = 'experiments/reports/knowledge_loop_20261101/v3_seed1_input_registration.json'
SEED1_REGISTRY_SHA256 = '97235716a5a0bb9157b0c1f4da723af4145534a18438f8f3872513cdfc736937'
SEED1_RULE_PATH = 'experiments/reports/knowledge_loop_20261101/v3_seed1_rule_registration.json'
SEED1_RULE_SHA256 = '6897a3d08ad9842a2ad36487fd2c667c4037ec5a407e5873a9a17380b668b59b'
CHECKLIST = [
    'Frozen protocol, source/config/input custody and candidate-only pool schema',
    'Four fixed initial IDs, 16 ordered unique queries, CP prefix/hash and state history',
    'Stored GP arrays finite; exact physical-feature tie priority; BO argmax and budget set',
    'Frozen shadow hash, semantic version, GP mean/variance and conditional moments',
    'Independent prospective log-score gain, information threshold, weight and ledger replay',
    'LLM prompt whitelist, prefix-only observations/evidence, eligible menu and response hashes',
    'Selection mode and recorded scalar scores; full knowledge argmax needs joint covariance',
    'SearchAUC/final best computed separately from technical validity and matched BO',
]
LIMITATIONS = [
    'Stored marginal GP arrays are checked, not independently refitted or kernel-recomputed.',
    'Without full contrast covariance/combined-score artifacts, guided and verification argmax are not independently proven.',
    'CP hashes establish internal prospective ordering; filesystem timestamps alone cannot prove original wall-clock causality.',
    'No-oracle audit covers public input/prompt schemas and source call order, not an OS-level I/O trace.',
    'Seed-0 development results are not independent scientific confirmation or causal module attribution.',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')).hexdigest()


def filehash(path):
    with Path(path).open('rb') as stream:
        value = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-11, rtol=1e-10):
    require(np.isclose(actual, expected, atol=atol, rtol=rtol),
            f'{label}: {actual!r} != {expected!r}')


def exact(actual, expected, label):
    require(digest(actual) == digest(expected), label)


def trial_seeds(protocol):
    trial = protocol.get('trial_seed', protocol['seed'])
    tie = protocol.get('tie_seed', trial)
    require(type(trial) is int and trial in (0, 1) and type(protocol['seed']) is int
            and protocol['seed'] == trial, 'unregistered trial seed')
    require(type(tie) is int and tie == trial, 'unregistered tie stream')
    require(type(protocol.get('numerical_seed', 0)) is int and protocol.get('numerical_seed', 0) == 0
            and type(protocol.get('menu_seed', 0)) is int and protocol.get('menu_seed', 0) == 0,
            'GP numerical/menu stream changed')
    require((trial == 1) == (protocol['variant'] == 'v3_seed1_confirm'), 'seed1 must use the registered confirmation variant')
    return trial, tie


def seed1_registration_audit(root, protocol, pools):
    """Validate the pre-target rule and exact input-only initial design."""
    trial, _ = trial_seeds(protocol)
    if trial == 0:
        return None
    require(protocol['initial_registry_path'] == SEED1_REGISTRY_PATH
            and protocol['initial_registry_sha256'] == SEED1_REGISTRY_SHA256
            and protocol['initial_rule_path'] == SEED1_RULE_PATH
            and protocol['initial_rule_sha256'] == SEED1_RULE_SHA256,
            'confirmation registration differs from the frozen pre-target registration')
    require(filehash(root / SEED1_REGISTRY_PATH) == protocol['initial_registry_file_sha256'], 'registration file SHA')
    registry, rule = read(root / SEED1_REGISTRY_PATH), read(root / SEED1_RULE_PATH)
    for obj, expected in ((registry, SEED1_REGISTRY_SHA256), (rule, SEED1_RULE_SHA256)):
        require(obj['registration_sha256'] == expected
                and digest({k: v for k, v in obj.items() if k != 'registration_sha256'}) == expected,
                'registration content SHA')
        require(obj['initial_y_read'] is False and obj['oracle_access'] is False
                and obj['LLM_calls'] == 0 and obj['GP_fits'] == 0, 'registration reports prior target/model access')
        require(obj['trial_seed'] == 1 and obj['tie_seed'] == 1 and obj['menu_seed'] == 0, 'registered seed streams')
    require(registry['rule_registration_sha256'] == SEED1_RULE_SHA256, 'input-to-rule registration link')
    require(rule['tasks'] == protocol['tasks'] == [r['task'] for r in registry['tasks']], 'registered tasks changed')
    require(rule['arms'] == protocol['supported_modes'] == ['bo', 'full']
            and protocol['supported_conditions'] == [rule['condition']] == ['agent_initial_terra'], 'unregistered confirmation arm/condition')
    require(rule['initial'] == protocol['initial'] == 4 and rule['queries'] == protocol['queries'] == 16, 'registered confirmation budget')
    require(protocol['prompt_sha256'] == rule['prompt_sha256'], 'confirmation system prompt changed')
    require(protocol['numerical_seed_policy'] == rule['native_gp_numerical_seed_policy'], 'native numerical GP seed policy changed')
    for rel, expected in rule['core_source_sha256'].items():
        if not rel.endswith('/knowledge_gp.py'):
            require(protocol['source_sha256'][rel] == expected, 'registered V3 mechanism/prompt changed: ' + rel)
    manifests = {r['task']: r for r in protocol['inputs']}
    require(set(manifests) == set(pools) == set(protocol['tasks']), 'registered input tasks missing or duplicated')
    for row in registry['tasks']:
        pool, manifest = pools[row['task']], manifests[row['task']]
        require(manifest['files_sha256'] == {row['input_pool']: row['input_pool_sha256']}, 'confirmation input custody differs from registration')
        require(filehash(root / row['input_pool']) == row['input_pool_sha256'], 'registered pool content SHA')
        require(row['candidate_count'] == len(pool['ids']) and row['axes'] == pool['axes'] and row['kinds'] == pool['kinds'], 'registered input schema')
        order = sorted(range(len(pool['ids'])), key=lambda j: pool['features'][j])
        positions = np.random.default_rng(np.random.SeedSequence([1, 0])).choice(len(order), 4, replace=False)
        indices = [order[int(j)] for j in positions]
        ids = [pool['ids'][j] for j in indices]
        features, points = [list(pool['features'][j]) for j in indices], [pool['points'][j] for j in indices]
        require(ids == row['initial_ids'] == manifest['initial_ids'], 'initial design differs from pre-target draw')
        exact(features, row['initial_features'], 'registered initial features')
        exact(points, row['initial_points'], 'registered initial physical points')
        require(digest(features) == row['initial_features_sha256'] == manifest['initial_features_sha256'], 'registered initial feature SHA')
        require(digest([list(pool['features'][j]) for j in order]) == row['sorted_pool_features_sha256'], 'registered candidate feature set SHA')
        require(manifest['unit'] == pool['unit'] and manifest['group'] == pool['group']
                and manifest['knowledge_conditions'] == ['agent_initial_terra'], 'confirmation input contract')
    return dict(result='PASS_INPUT_REGISTRATION', trial_seed=1, tie_seed=1, numerical_seed=0, menu_seed=0,
                initial_registry_sha256=SEED1_REGISTRY_SHA256, initial_rule_sha256=SEED1_RULE_SHA256,
                initial_designs=len(manifests), targets_inspected=False)


def initial_commit_audit(path, state, protocol, pool, manifest):
    trial, _ = trial_seeds(protocol)
    if trial == 0 and 'initial_commit_sha256' not in state:
        return  # Historical seed0 did not promise an initial_commit artifact.
    source = next(rel for rel in manifest['files_sha256'] if rel.endswith('/pool.json'))
    ids = manifest['initial_ids']
    expected = dict(task=state['task'], mode=state['mode'], condition=state['condition'], seed=trial,
        protocol_sha256=protocol['protocol_sha256'], initial_registry_sha256=protocol.get('initial_registry_sha256'),
        initial_ids=ids, initial_points=[pool['points'][pool['ix'][i]] for i in ids],
        initial_features=[list(pool['features'][pool['ix'][i]]) for i in ids], input_pool_sha256=manifest['files_sha256'][source])
    expected['commit_sha256'] = digest(expected)
    actual = read(path.parent / 'initial_commit.json')
    exact(actual, expected, 'initial input-only commitment changed or includes non-input data')
    require(state['initial_commit_sha256'] == expected['commit_sha256'], 'state initial commitment SHA')
    require([h['candidate_id'] for h in state['history'][:4]] == ids, 'initial outcomes differ from committed points')


def native_seed_audit(pool, commit, saved_fit_seed=None):
    """The trial/tie stream never enters the preregistered GP/MC seed recipe."""
    if pool['group'] == 'P0':
        offset = int(pool['task'][-1]) * 10000
    elif pool['group'] == 'P1':
        offset = (int(digest({'task': pool['raw']['task_id']})[:7], 16) % 1000000) * 100
    else:
        require(pool['task'] in ('SnAr', 'HPLC'), 'unknown numerical-seed task')
        offset = (1 if pool['task'] == 'SnAr' else 2) * 10000
    expected_fit = 28102026 + offset + commit['query']
    expected_mc = 38102026 + offset + commit['query']
    meta = commit['GP_acquisition_meta']
    require(meta['fit_seed'] == expected_fit and meta['mc_seed'] == expected_mc, 'native numerical GP/MC seed changed')
    if saved_fit_seed is not None:
        require(saved_fit_seed == expected_fit, 'cached model numerical seed changed')


def public_pool(root, item):
    """Decode physical features without importing production adapters."""
    path = next(root / name for name in item['files_sha256'] if name.endswith('/pool.json'))
    raw = read(path)
    group = item['group']
    if group == 'P0':
        axes = ['t_res', 'temperature', 'catalyst_loading', 'catalyst']
        kinds = ['continuous'] * 3 + ['categorical']
        features = [tuple(map(float, z[:3])) + (p['catalyst'],)
                    for z, p in zip(raw['z'], raw['points'])]
        scale, unit = 100., 'simulated yield percent'
    elif group == 'P1':
        axes = raw['axes']
        kinds = ['categorical'] * len(axes)
        features = [tuple(p[a] for a in axes) for p in raw['points']]
        scale, unit = 100., raw['objective_unit']
    else:
        axes = raw['axes']
        kinds = ['continuous'] * len(axes)
        features = [tuple(map(float, z)) for z in raw['z']]
        scale, unit = float(raw['objective_scale']), raw['objective_unit']
    require(len(raw['ids']) == item['candidate_count'], 'candidate count differs from manifest')
    require(len(set(raw['ids'])) == len(raw['ids']), 'duplicate candidate IDs')
    require(len(set(features)) == len(features), 'duplicate physical features')
    require(all(set(point) <= set(axes) for point in raw['points']),
            'candidate point contains a non-input field')
    return dict(ids=raw['ids'], ix={v: i for i, v in enumerate(raw['ids'])},
                axes=axes, kinds=kinds, features=features, points=raw['points'], scale=scale, unit=unit, group=group, raw=raw)


def in_scope(pool, h, index):
    for axis, value in zip(pool['axes'], pool['features'][index]):
        spec = h['scope']['axes'][axis]
        if spec['kind'] == 'continuous':
            lo, hi = spec['bounds']
            if not lo - 1e-12 <= value <= hi + 1e-12:
                return False
        elif not any(digest(value) == digest(a) for a in spec['allowed']):
            return False
    return True


def penalty(h):
    return sum(min(0., e['delta_log_score']) for e in h.get('inherited_negative_evidence', []) if e['applicable'])


def weight(h):
    effective = h['ledger']['score'] + penalty(h)
    if h['status'] != 'active' or not h['ledger']['events'] or effective <= 0:
        return 0.
    return min(h['ledger'].get('cap', .5), max(0., float(2 * expit(effective) - 1)))


def key(h):
    return h.get('id', h.get('hypothesis_id')), h['version']


def validate_ledger(h, history, max_round):
    """Recompute accumulated evidence from frozen event predictions, not receipts."""
    semantic = {k: h[k] for k in ('plus_features', 'minus_features', 'focus', 'scope')}
    exact(h['semantic_key'], digest(semantic), 'changed semantic contract')
    score, seen, last = 0., [], 0
    for e in h['ledger']['events']:
        t = e['query_round']
        require(last < t <= max_round and t >= h['created_round'], 'event is retrospective/out of order')
        require(e['point_id'] not in seen, 'duplicate evidence point')
        require(e['point_id'] == history[t + 3]['candidate_id'], 'event point is not its actual experiment')
        close(e['observed_y'], history[t + 3]['y'], 'event observed_y')
        require(key(e) == key(h), 'event from wrong hypothesis version')
        exact(e['event_id'], digest({k: v for k, v in e.items() if k != 'event_id'}), 'event hash')
        pred = e['prediction']
        exact(pred['prediction_sha256'], digest({k: v for k, v in pred.items() if k != 'prediction_sha256'}), 'prediction hash')
        require(pred['semantic_key'] == h['semantic_key'] and key(pred) == key(h), 'prediction version/semantic mismatch')
        require(pred['query_round'] == t and pred['point_id'] == e['point_id'], 'prediction event mismatch')
        expected = score_gain(pred, e['observed_y']) if e['informative'] else None
        if expected is None:
            require(e['delta_log_score'] is None, 'uninformative event has score')
        else:
            close(e['delta_log_score'], expected, 'event gain')
            score += expected
        seen.append(e['point_id'])
        last = t
    require(sorted(seen) == h['ledger']['seen'], 'ledger seen mismatch')
    close(h['ledger']['score'], score, 'accumulated ledger score')
    close(h['ledger']['weight'], weight(h), 'ledger weight')
    require(len({e['event_id'] for e in h.get('inherited_negative_evidence', [])}) == len(h.get('inherited_negative_evidence', [])), 'duplicate inherited penalty')
    for e in h.get('inherited_negative_evidence', []):
        require(e['delta_log_score'] < 0 and e['query_round'] < h['created_round'], 'invalid inherited negative evidence')


def score_gain(p, y):
    vz = p['contrast_var'] + p['preference_noise_sd'] ** 2
    rem = max(vz - p['cov_y_contrast'] ** 2 / p['gp_var_y'], 1e-15 * vz)
    beta = (p['contrast_mean'] + p['cov_y_contrast'] / p['gp_var_y'] * (y - p['gp_mean_y'])) / sqrt(rem)
    return float(log_ndtr(beta) - log_ndtr(p['contrast_mean'] / sqrt(vz)))


def shadow_feedback(pool, h, sh, receipt, observation, t, arrays):
    p = sh['prediction']
    require(key(sh) == key(h) == key(p) == key(receipt), 'shadow/evidence version mismatch')
    require(p['point_id'] == observation['candidate_id'] and p['query_round'] == t, 'shadow is not prequery selection')
    require(p['point_id'] not in h['ledger']['seen'] + h.get('excluded_observed_ids', []), 'reused evidence')
    exact(p['prediction_sha256'], digest({k: v for k, v in p.items() if k != 'prediction_sha256'}), 'frozen shadow hash')
    require(p['semantic_key'] == h['semantic_key'], 'shadow semantic mismatch')
    index = pool['ix'][p['point_id']]
    close(p['gp_mean_y'], arrays['mu_y'][index], 'shadow GP mean')
    close(p['gp_var_y'], arrays['sd_predictive_y'][index] ** 2, 'shadow GP predictive variance')
    close(p['contrast_mean'], arrays['mu_y'][pool['ix'][h['plus_id']]] - arrays['mu_y'][pool['ix'][h['minus_id']]], 'contrast mean')
    require(p['contrast_var'] >= 0 and p['gp_var_y'] > 0, 'invalid variances')
    vz = p['contrast_var'] + p['preference_noise_sd'] ** 2
    rem = vz - p['cov_y_contrast'] ** 2 / p['gp_var_y']
    require(rem >= -1e-8 * vz, 'joint covariance is not positive semidefinite')
    info = .5 * log(vz / max(rem, 1e-15 * vz))
    close(sh['information_nats'], info, 'information')
    require(sh['informative'] == bool(in_scope(pool, h, index) and info >= .01), 'informative gate')
    alpha = p['contrast_mean'] / sqrt(vz)
    mills = exp(-.5 * alpha ** 2 - .5 * log(2 * pi) - float(log_ndtr(alpha)))
    if 'knowledge_mean_y' in p:
        close(p['knowledge_mean_y'], p['gp_mean_y'] + p['cov_y_contrast'] / sqrt(vz) * mills, 'conditional mean')
        close(p['knowledge_var_y'], max(0., p['gp_var_y'] - p['cov_y_contrast'] ** 2 / vz * mills * (mills + alpha)), 'conditional variance')
    before = weight(h)
    expected_gain = score_gain(p, observation['y']) if sh['informative'] else None
    if expected_gain is None:
        require(receipt['delta_log_score'] is None, 'uninformative feedback has gain')
    else:
        close(receipt['delta_log_score'], expected_gain, 'feedback independent gain')
    # Verify arithmetic above, then retain its exact serialized value for hashes;
    # harmless libm rounding differences must not manufacture custody failures.
    gain = receipt['delta_log_score']
    event = dict(hypothesis_id=h['id'], version=h['version'], point_id=p['point_id'], query_round=t,
                 prediction=deepcopy(p), observed_y=float(observation['y']), informative=sh['informative'], delta_log_score=gain)
    event['event_id'] = digest(event)
    h['ledger']['events'].append(event)
    h['ledger']['seen'] = sorted(h['ledger']['seen'] + [p['point_id']])
    if gain is not None:
        h['ledger']['score'] += gain
    h['ledger']['weight'] = weight(h)
    expected = {k: event[k] for k in ('event_id', 'hypothesis_id', 'version', 'point_id', 'query_round', 'informative', 'delta_log_score')}
    expected.update(weight_before=before, weight_after=weight(h), score_after=h['ledger']['score'], inherited_negative_penalty=penalty(h))
    require(set(receipt) == set(expected), 'feedback receipt schema')
    for name, value in expected.items():
        if isinstance(value, float):
            close(receipt[name], value, f'feedback {name}')
        else:
            exact(receipt[name], value, f'feedback {name}')
    h['ledger']['weight'] = receipt['weight_after']


def initial_prompt_hypotheses(pool, prior):
    """Reconstruct prior public contracts without calling the prior/policy code."""
    result = []
    if prior is None:
        return result
    for clause in prior['clauses']:
        a, b = clause['plus_features'], clause['minus_features']
        focus, axes = deepcopy(clause['focus']), {}
        for j, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
            if kind == 'continuous':
                bounds = (focus[axis]['bounds'] if axis in focus else
                          [max(0., a[j] - .25, b[j] - .25), min(1., a[j] + .25, b[j] + .25)])
                axes[axis] = dict(kind=kind, bounds=list(map(float, bounds)))
            else:
                allowed = sorted(focus[axis]['allowed'], key=digest) if axis in focus else [a[j]]
                if axis in focus:
                    focus[axis]['allowed'] = allowed
                axes[axis] = dict(kind=kind, allowed=allowed)
        contract = dict(plus_features=a, minus_features=b, focus=focus,
                        scope=dict(axes=axes, background_radius=.25))
        require(digest(contract) == clause['semantic_key'], 'prior public-contract reconstruction')
        result.append(dict(contract, id=clause['hypothesis_id'], version=clause['version'], parent=None,
                           plus_id=clause['plus_id'], minus_id=clause['minus_id'], status='active',
                           ledger=dict(score=0., weight=0., events=[], seen=[], cap=.5), inherited_negative_evidence=[]))
    return result


def refreshed_prompt_hypotheses(pool, previous, eligible, t):
    """Independent current-round availability transition; no new evidence."""
    result, events = deepcopy(previous), []
    js = np.flatnonzero(eligible)
    for h in result:
        if h['status'] == 'retracted':
            continue
        count = sum(in_scope(pool, h, int(j)) for j in js)
        status = 'active' if count else 'dormant'
        if status != h['status']:
            events.append(dict(round=t, id=h['id'], version=h['version'], from_status=h['status'],
                               to_status=status, reason='eligible_scope_overlap' if count else 'no_eligible_scope_overlap'))
        h['status'] = status
        h['coverage'] = dict(round=t, eligible_count=count)
        h['ledger']['weight'] = weight(h)
    return result, events


def domain_context_audit(domain, pool):
    """Exact safe schema, physical units/ranges and actual nominal labels."""
    task = pool['task']
    top = {'task', 'group', 'reaction_or_process', 'description', 'objective', 'substrate_scope',
           'inputs', 'candidate_policy', 'knowledge_policy', 'candidate_count'}
    require(set(domain) == top, 'unapproved domain field (including source references)')
    require(domain['task'] == task and domain['group'] == pool['group'], 'domain task/group')
    require(domain['candidate_count'] == len(pool['ids']), 'domain candidate count')
    def prose(value):
        require(isinstance(value, str) and 0 < len(value) <= 2000, 'domain narrative is not bounded plain text')
    for name in ('description', 'candidate_policy', 'knowledge_policy'):
        prose(domain[name])
    continuous = {
        'P0': [('t_res', 's', [60., 600.]), ('temperature', 'degC', [30., 110.]),
               ('catalyst_loading', 'mol%', [.5, 2.5])],
        'SnAr': [('tau', 'min', [.5, 2.]), ('equiv_pldn', 'equiv', [1., 5.]),
                 ('conc_dfnb', 'mol/L', [.1, .5]), ('temperature', 'degC', [30., 120.])],
        'HPLC': [('sample_loop', 'mL', [0., .08]), ('additional_volume', 'mL', [0., .06]),
                 ('tubing_volume', 'mL', [.1, .9]), ('sample_flow', 'mL/min', [.5, 2.5]),
                 ('push_speed', 'Hz', [80., 150.]), ('wait_time', 's', [.5, 10.])],
    }
    objectives = {
        'P0': ('Suzuki-Miyaura cross-coupling', 'yield', 'percent'),
        'Perera_Suzuki': ('Suzuki-Miyaura cross-coupling', 'product_UV_area', 'percent'),
        'Doyle_Buchwald_Hartwig': ('Buchwald-Hartwig C-N coupling (amination)', 'reaction_yield', 'percent'),
        'SnAr': ('Nucleophilic aromatic substitution in a plug-flow reactor', 'space_time_yield', 'kg m^-3 h^-1'),
        'HPLC': ('Automated HPLC sample-injection process', 'peak_area', 'source arbitrary peak-area units'),
    }
    registry_key = 'P0' if pool['group'] == 'P0' else task
    process, objective, unit = objectives[registry_key]
    require(domain['reaction_or_process'] == process, 'public process identity')
    require(set(domain['objective']) == {'name', 'unit', 'direction', 'meaning'}, 'unapproved objective field')
    require(domain['objective']['name'] == objective and domain['objective']['unit'] == unit
            and domain['objective']['direction'] == 'maximize', 'public objective/units/direction')
    prose(domain['objective']['meaning'])
    require(set(domain['substrate_scope']) == {'substrates_are_optimization_variables', 'identity_status', 'statement'}, 'unapproved substrate field')
    require(domain['substrate_scope']['substrates_are_optimization_variables'] == (pool['group'] == 'P1'), 'public substrate optimization scope')
    prose(domain['substrate_scope']['identity_status'])
    prose(domain['substrate_scope']['statement'])
    require([r['name'] for r in domain['inputs']] == pool['axes'], 'domain input axes')
    ranges = {axis: (unit, bounds) for axis, unit, bounds in continuous.get(registry_key, [])}
    for j, row in enumerate(domain['inputs']):
        axis, kind = pool['axes'][j], pool['kinds'][j]
        require(row['kind'] == kind, 'domain variable kind')
        prose(row['meaning'])
        common = {'name', 'kind', 'meaning', 'unit'}
        if kind == 'continuous':
            allowed = common | {'declared_input_bounds'}
            if 'bounds' in pool['raw']:
                allowed.add('executable_pool_bounds')
                exact(row['executable_pool_bounds'], pool['raw']['bounds'][axis], 'domain executable range')
            require(set(row) == allowed, 'unapproved continuous metadata field')
            expected_unit, bounds = ranges[axis]
            require(row['unit'] == expected_unit, 'domain physical unit')
            exact(row['declared_input_bounds'], bounds, 'domain physical range')
            require(all(bounds[0] - 1e-10 <= float(p[axis]) <= bounds[1] + 1e-10 for p in pool['points']), 'physical input outside declared range')
        else:
            allowed = common | {'representation', 'ordered', 'identity_status', 'levels'}
            if pool['group'] == 'P0':
                allowed.add('interpretation_limit')
            require(set(row) == allowed, 'unapproved category metadata field')
            require(row['unit'] is None and row['ordered'] is False, 'category is assigned numeric unit/order')
            representation = 'opaque label' if pool['group'] == 'P0' else 'SMILES'
            require(row['representation'] == representation, 'unsupported molecular identity representation')
            identity = ('opaque experimental label; molecular identity unavailable' if pool['group'] == 'P0'
                        else 'supplied molecular structure string')
            require(row['identity_status'] == identity, 'category identity status')
            if pool['group'] == 'P0':
                prose(row['interpretation_limit'])
            real = {digest(x[j]) for x in pool['features']}
            require(len(row['levels']) == len(real) and {digest(x) for x in row['levels']} == real, 'invented/missing input category')
            if 'levels' in pool['raw']:
                exact(row['levels'], pool['raw']['levels'][axis], 'category labels do not match public pool')
    def reject_references(value):
        if isinstance(value, dict):
            for name, child in value.items():
                require(name != 'source_references', 'source references exposed to LLM')
                reject_references(child)
        elif isinstance(value, list):
            for child in value:
                reject_references(child)
    reject_references(domain)


def prompt_audit(folder, p, pool, call, prefix, c, previous, eligible):
    context = read(folder / 'requests' / call['context_sha256'] / 'context.json')
    require(digest(context) == call['context_sha256'], 'LLM context hash')
    require(context['model'] == p['model'] and context['reasoning_effort'] == p['reasoning_effort'], 'LLM model/effort')
    require(digest(context['system']) == p['prompt_sha256'], 'system prompt drift')
    public = context['public']
    expected_fields = {'task', 'objective_unit', 'query', 'remaining_queries', 'menu_policy', 'axes', 'own_observations',
                       'candidate_menu', 'knowledge_ledger', 'actionable_coverage', 'feedback_note'}
    if p.get('domain_context') == 'public_process_units_v1':
        expected_fields.add('domain_context')
        domain_context_audit(public['domain_context'], pool)
    else:
        require(p.get('domain_context') is None, 'unknown domain-context protocol')
    require(set(public) == expected_fields, 'unapproved LLM top-level field')
    require(public['task'] == pool['task'] and public['query'] == c['query'] and public['remaining_queries'] == 17 - c['query'], 'LLM task/round')
    require(public['objective_unit'] == pool['unit'], 'public objective unit differs from input contract')
    exact(public['own_observations'], [{'point': pool['points'][pool['ix'][h['candidate_id']]], 'objective': h['raw_value']} for h in prefix], 'LLM observations not exact own prefix')
    exact(public['axes'], [{'name': a, 'type': k} for a, k in zip(pool['axes'], pool['kinds'])], 'LLM axes')
    require([r['id'] for r in public['candidate_menu']] == c['menu_ids'], 'LLM menu != committed menu')
    for row in public['candidate_menu']:
        require(set(row) == {'id', 'point', 'normalized_continuous'}, 'unapproved menu field/target')
        j = pool['ix'][row['id']]
        exact(row['point'], pool['points'][j], 'menu physical point')
        exact(row['normalized_continuous'], {a: pool['features'][j][k] for k, (a, kind) in enumerate(zip(pool['axes'], pool['kinds'])) if kind == 'continuous'}, 'menu normalized coordinates')
    allowed = {'id', 'version', 'parent', 'plus_id', 'minus_id', 'focus', 'scope', 'status', 'score', 'weight', 'events', 'event_units', 'inherited_counterevidence_count'}
    for h in public['knowledge_ledger']:
        require(set(h) <= allowed, 'GP/hidden prediction field in public ledger')
        for event in h['events']:
            require(set(event) <= {'point_id', 'query_round', 'observed_y', 'informative', 'delta_log_score'}, 'hidden prediction in public event')
            require(1 <= event['query_round'] < c['query'], 'future feedback in LLM prompt')
            observed = prefix[event['query_round'] + 3]
            require(event['point_id'] == observed['candidate_id'], 'LLM feedback point mismatch')
            close(event['observed_y'], observed['y'], 'LLM feedback value')
        if previous is not None:
            source = next((x for x in previous if key(x) == key(h)), None)
            require(source is not None, 'public ledger version absent from own state')
            for name in ('focus', 'scope', 'status', 'plus_id', 'minus_id'):
                exact(h[name], source[name], f'public ledger {name}')
            close(h['score'], source['ledger']['score'], 'public ledger score')
            close(h['weight'], source['ledger']['weight'], 'public ledger weight')
            exact(h['events'], [{k: e[k] for k in ('point_id', 'query_round', 'observed_y', 'informative', 'delta_log_score')} for e in source['ledger']['events']], 'public ledger events')
        else:
            require(not h['events'] and h['score'] == 0 and h['weight'] == 0, 'initial prior has retrospective evidence')
    if previous is not None:
        require({key(h) for h in public['knowledge_ledger']} == {key(h) for h in previous}, 'LLM ledger omitted/added version')
    allowed_coverage = {'id', 'version', 'weight', 'eligible_inside_scope', 'can_change_rank_at_cap', 'eligibility_state'}
    require(all(set(r) == allowed_coverage for r in public['actionable_coverage']), 'unapproved public coverage field')
    if p.get('prompt_state') == 'current_round_coverage':
        require(previous is not None, 'missing prepared prompt state')
        live = {key(h): h for h in previous if h['status'] != 'retracted'}
        require({key(r) for r in public['actionable_coverage']} == live.keys(), 'public coverage omitted/added live version')
        for row in public['actionable_coverage']:
            h = live[key(row)]
            count = sum(in_scope(pool, h, int(j)) for j in np.flatnonzero(eligible))
            require(row['eligible_inside_scope'] == count, 'public current coverage count')
            close(row['weight'], weight(h), 'public current coverage weight')
            require((h['status'] == 'active') == bool(count), 'public current coverage status')
            require(type(row['can_change_rank_at_cap']) is bool, 'invalid public rank-effect flag')
            expected = ('dormant_no_coverage' if not count else 'covered' if row['can_change_rank_at_cap']
                        else 'covered_without_current_rank_effect')
            require(row['eligibility_state'] == expected, 'public current coverage classification')
    receipt = read(folder / 'requests' / call['context_sha256'] / 'validated.json')
    exact(receipt, call['receipt'], 'saved LLM receipt differs')
    response = read(folder / 'requests' / call['context_sha256'] / f"response_{receipt['attempt']:02d}.json")
    require(digest(response) == receipt['response_sha256'], 'LLM response hash')
    exact(json.loads(response['output_text']), receipt['proposal'], 'response/proposal mismatch')
    exact(c['proposal'], receipt['proposal'], 'committed proposal mismatch')


def joint_selection_audit(folder, pool, manifest, prefix, c, arrays, arm, verification_used, fit, parameters):
    """Load state_dict only; verify direct GP algebra and independently rerank.

    No production helper, optimizer, oracle reader or cache writer is imported.
    Native posterior contrasts reproduce the run's arithmetic order; a separate
    dense conditioning formula checks their values and engineering units.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    torch.set_num_threads(1)
    saved = torch.load(folder / 'cache' / pool['task'] / (c['GP_fit_key'] + '.pt'),
                       map_location='cpu', weights_only=True)
    require(saved['key'] == c['GP_fit_key'], 'cached model key')
    native_seed_audit(pool, c, saved['fit_seed'])
    raw = pool['raw']
    if manifest['group'] == 'P0':
        labels = ['P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7']
        features = np.column_stack((np.asarray(raw['z'])[:, :3].clip(1e-6, 1 - 1e-6),
                                    [saved['category_map'][labels.index(p['catalyst'])] for p in raw['points']]))
        categories = [3]
    elif manifest['group'] == 'P1':
        features = np.asarray([[saved['mapping'][a][p[a]] for a in pool['axes']] for p in raw['points']], dtype=float)
        categories = list(range(len(pool['axes'])))
    else:
        features = np.asarray(raw['z'], dtype=float)
        categories = None
    X = torch.tensor(features, dtype=torch.double)
    ordered = sorted(prefix, key=(lambda h: pool['features'][pool['ix'][h['candidate_id']]]) if manifest['group'] == 'P2' else (lambda h: h['candidate_id']))
    train = X[[pool['ix'][h['candidate_id']] for h in ordered]]
    y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
    outer_mean, outer_std = y.mean(), y.std(unbiased=False).clamp_min(1e-12)
    ys = (y - outer_mean) / outer_std
    if 'train_candidate_ids' in saved:
        require(saved['train_candidate_ids'] == [h['candidate_id'] for h in ordered], 'cached model fitted on another prefix')
        require(torch.equal(torch.as_tensor(saved['train_X'], dtype=torch.double), train), 'cached GP train X')
    elif 'history' in saved:
        exact(saved['history'], [{'candidate_id': h['candidate_id'], 'y': float(h['y'])} for h in ordered], 'cached GP training history')
    m = MixedSingleTaskGP(train, ys, cat_dims=categories) if categories is not None else SingleTaskGP(train, ys)
    m.load_state_dict(saved['state'])
    m.eval()
    exact({name: parameter.detach().cpu().tolist() for name, parameter in m.named_parameters()}, fit,
          'stored fit-parameter receipt differs from cached model')
    n = len(pool['ids'])
    with torch.no_grad():
        K = m.covar_module(train).to_dense() + torch.eye(len(train), dtype=torch.double) * m.likelihood.noise.reshape(())
        cross = m.covar_module(X, train).to_dense()
        solve = torch.linalg.solve(K, cross.T)
        scale2 = m.outcome_transform.stdvs.reshape(()).square() * outer_std.square()
        mu = m.mean_module(X) + cross @ torch.linalg.solve(K, m.train_targets - m.mean_module(train))
        mu = (mu * m.outcome_transform.stdvs.reshape(()) + m.outcome_transform.means.reshape(())) * outer_std + outer_mean
        # BoTorch's nominal kernel forms a full pairwise matrix even for diag;
        # bound that temporary allocation instead of creating N-by-N-by-D data.
        diagonal = torch.cat([m.covar_module(part, diag=True) for part in X.split(256)])
        var = (diagonal - (cross * solve.T).sum(-1)) * scale2
        pred_var = var + m.likelihood.noise.reshape(()) * scale2
        np.testing.assert_allclose(arrays['mu_y'], mu.numpy(), atol=1e-9, rtol=1e-7, err_msg='direct GP mean')
        np.testing.assert_allclose(arrays['sd_y'] ** 2, var.numpy(), atol=1e-9, rtol=1e-6, err_msg='direct GP variance')
        np.testing.assert_allclose(arrays['sd_predictive_y'] ** 2, pred_var.numpy(), atol=1e-9, rtol=1e-6, err_msg='direct GP observation variance')
        moments = {}
        for h in c['hypotheses_at_commit']:
            if h['status'] == 'retracted':
                continue
            anchors = [pool['ix'][h['plus_id']], pool['ix'][h['minus_id']]]
            covariance = m.covar_module(X, X[anchors]).to_dense() - cross @ solve[:, anchors]
            direct_cov = (covariance[:, 0] - covariance[:, 1]) * scale2
            direct_vd = float(direct_cov[anchors[0]] - direct_cov[anchors[1]])
            covs, vd = direct_cov.numpy().copy(), None
            # Native joint posterior is needed only on executable scope points,
            # the BO reference and the selected point. Direct algebra above
            # already covers the whole pool and avoids N-by-N prediction work.
            needed = sorted({j for j in np.flatnonzero(arrays['eligible']) if in_scope(pool, h, int(j))}
                            | {pool['ix'][c['BO_reference_id']], pool['ix'][c['selected_id']]})
            # Match only the posterior arithmetic (not the production decision implementation).
            for start in range(0, len(needed), 128):
                indices = needed[start:start + 128]
                part = X[indices]
                joint = m.posterior(torch.cat((part, X[anchors]))).mvn.covariance_matrix.numpy() * float(outer_std ** 2)
                count = len(part)
                covs[indices] = joint[:count, count] - joint[:count, count + 1]
                current = max(0., float(joint[count, count] + joint[count + 1, count + 1] - 2 * joint[count, count + 1]))
                if vd is not None:
                    close(current, vd, 'chunk contrast variance', atol=1e-8, rtol=1e-6)
                vd = current
            np.testing.assert_allclose(covs, direct_cov.numpy(), atol=1e-9, rtol=1e-6, err_msg='direct contrast covariance')
            close(vd, max(0., direct_vd), 'direct contrast variance', atol=1e-9, rtol=1e-6)
            moments[key(h)] = (arrays['mu_y'][anchors[0]] - arrays['mu_y'][anchors[1]], vd, covs)
    base, eligible, ties = arrays['base_logei'], arrays['eligible'], arrays['tie_priority']
    def choose(scores):
        js = np.flatnonzero(eligible & (scores == scores[eligible].max()))
        return int(js[np.argmin(ties[js])])
    def information(vy, vd, cov):
        vz = vd + .06 ** 2
        return .5 * log(vz / max(vz - cov ** 2 / max(vy, 1e-18), 1e-15 * vz))
    bo = choose(base)
    scores_by_key = {}
    nodes, ws = np.polynomial.hermite.hermgauss(64)
    log_weights, best = np.log(ws / sqrt(pi)), max(h['y'] for h in prefix)
    for row in c['coverage']:
        h = next(h for h in c['hypotheses_at_commit'] if key(h) == key(row))
        js = np.array([j for j in np.flatnonzero(eligible) if in_scope(pool, h, int(j))], dtype=int)
        if not len(js):
            require(not row['can_change_rank_at_cap'], 'empty scope has rank effect')
            continue
        md, vd, covs = moments[key(h)]
        scores = base.copy()
        vz = vd + .06 ** 2
        for j in js:
            vx = max(float(arrays['sd_y'][j]) ** 2, 1e-18)
            draws = arrays['mu_y'][j] + sqrt(2 * vx) * nodes
            improvement = np.maximum(draws - best, 0.)
            positive = improvement > 0
            if not positive.any():
                continue
            common = log_weights[positive] + np.log(improvement[positive])
            rem = max(vz - covs[j] ** 2 / vx, 1e-15 * vz)
            z = (md + covs[j] / vx * (draws[positive] - arrays['mu_y'][j])) / sqrt(rem)
            offset = logsumexp(common + log_ndtr(z)) - logsumexp(common) - log_ndtr(md / sqrt(vz))
            if abs(offset) <= 1e-12:
                offset = 0.
            scores[j] += float(np.clip(offset, -log(4), log(4)))
        offset = scores - base
        correction = np.logaddexp(log(.5), log(.5) + offset)
        correction[np.abs(offset) <= 1e-12] = 0.
        require(row['can_change_rank_at_cap'] == (choose(base + correction) != bo), 'independent cap rank-effect assessment')
        scores_by_key[key(h)] = scores
    combined = base.copy()
    deployed = c['deployed_versions']
    if deployed:
        total = sum(h['weight'] for h in deployed)
        weights = [h['weight'] * min(1., .5 / total) for h in deployed]
        offsets = np.stack([scores_by_key[key(h)] - base for h in deployed])
        correction = logsumexp(np.vstack([np.full(n, log(1 - sum(weights)))] + [log(w) + offset for w, offset in zip(weights, offsets)]), axis=0)
        correction[np.all(np.abs(offsets) <= 1e-12, axis=0)] = 0.
        combined += correction
    selected = choose(combined)
    mode = 'knowledge_guided' if selected != bo else 'base_bo'
    verification = None
    if (arm == 'full' and selected == bo
            and c['query'] <= parameters['last_verification_round']
            and verification_used < parameters['maximum_optional_verifications']):
        candidates = []
        for row in c['coverage']:
            h = next(h for h in c['hypotheses_at_commit'] if key(h) == key(row))
            if (row['weight'] > 0 or not row['can_change_rank_at_cap']
                    or row['eligible_inside_scope'] < parameters['minimum_remaining_scope_candidates']
                    or h['ledger']['score'] + penalty(h) < 0):
                continue
            _, vd, covs = moments[key(h)]
            bo_info = information(arrays['sd_predictive_y'][bo] ** 2, vd, covs[bo])
            for j in np.flatnonzero(eligible):
                if j == bo or not in_scope(pool, h, int(j)):
                    continue
                gap = float(base[bo] - base[j])
                info = information(arrays['sd_predictive_y'][j] ** 2, vd, covs[j])
                if (gap <= min(c['epsilon'], parameters['verification_maximum_gap'])
                        and info - bo_info >= parameters['minimum_incremental_information']):
                    candidates.append((info - bo_info, -gap, -ties[j], int(j), key(h), info))
        if candidates:
            verification = max(candidates, key=lambda x: x[:3])
            selected, mode = verification[3], 'optional_verification'
    require(pool['ids'][selected] == c['selected_id'] and mode == c['mode'], 'independently reconstructed selection differs')
    close(c['combined_logei'], combined[selected], 'independent mixture score', atol=1e-9, rtol=1e-8)
    if verification:
        require(key(c['verification']) == verification[4], 'independent verification hypothesis')
        close(c['verification']['incremental_information'], verification[0], 'independent verification information', atol=1e-8, rtol=1e-6)
    for sh in c['shadows']:
        md, vd, covs = moments[key(sh)]
        close(sh['prediction']['contrast_var'], vd, 'shadow independent contrast variance', atol=1e-9, rtol=1e-6)
        close(sh['prediction']['cov_y_contrast'], covs[selected], 'shadow independent covariance', atol=1e-9, rtol=1e-6)


def knowledge_roles(path, state, parameters):
    """Describe audited actions/weights separately from their actual selection role."""
    proposed, accepted, rejected, evidence = Counter(), Counter(), Counter(), Counter()
    timeline, deployment_rounds, max_before, max_after = [], 0, 0., 0.
    for decision in state['decisions']:
        t = decision['query']
        c = read(path.parent / 'rounds' / f'{t:02d}_commit.json')
        require(digest(c) == decision['commit_sha256'], 'role summary commitment hash')
        for action in (c['proposal'] or {}).get('actions', []):
            proposed[action['op']] += 1
        reviews = []
        for review in c['reviews']:
            if 'op' not in review:
                continue
            if review['accepted']:
                accepted[review['outcome']] += 1
            else:
                rejected[review['op']] += 1
            reviews.append({k: review[k] for k in ('op', 'id', 'version', 'accepted', 'outcome') if k in review})
        before = max((r['weight'] for r in c['coverage']), default=0.)
        after = max((r['weight_after'] for r in decision['evidence']), default=0.)
        max_before, max_after = max(max_before, before), max(max_after, after)
        deployed = c['deployed_versions']
        total = sum(h['weight'] for h in deployed)
        scale = min(1., parameters['maximum_deployment_weight'] / total) if total else 0.
        deployment_rounds += bool(deployed)
        receipts = []
        for receipt in decision['evidence']:
            gain = receipt['delta_log_score']
            evidence['uninformative' if gain is None else 'positive' if gain > 0 else 'negative' if gain < 0 else 'zero'] += 1
            receipts.append({k: receipt[k] for k in ('hypothesis_id', 'version', 'delta_log_score', 'weight_before', 'weight_after')})
        timeline.append(dict(query=t, actual_role=c['mode'], selected_id=c['selected_id'],
            BO_reference_id=c['BO_reference_id'], observed_objective=decision['raw_value'],
            proposal_reviews=reviews, positive_weight_versions=sum(r['weight'] > 0 for r in c['coverage']),
            deployed_versions=[dict(h, effective_mixture_weight=h['weight'] * scale) for h in deployed],
            total_effective_mixture_weight=total * scale,
            maximum_weight_before_selection=before, maximum_weight_after_feedback=after,
            evidence=receipts))
    return dict(proposed_actions=dict(proposed), accepted_outcomes=dict(accepted), rejected_actions=dict(rejected),
        evidence_counts=dict(evidence), maximum_weight_before_selection=max_before,
        maximum_weight_after_feedback=max_after, mixture_calculation_rounds=deployment_rounds,
        actual_knowledge_guided_rounds=[r['query'] for r in timeline if r['actual_role'] == 'knowledge_guided'],
        optional_verification_rounds=[r['query'] for r in timeline if r['actual_role'] == 'optional_verification'],
        interpretation='Nonzero evidence weight or mixture calculation is not an actual knowledge-guided choice. Optional verification and weighted knowledge selection remain separate roles.',
        rounds=timeline)


def audit_run(folder, path, p, pool, manifest, include_incomplete=False, joint=False):
    state = read(path)
    parameters = p['parameters']
    trial, tie_seed = trial_seeds(p)
    history, decisions = state['history'], state['decisions']
    prior = None
    if state['condition'] != 'agent_initial_terra':
        prior = read(path.parent / 'prior_receipt.json')
        require(digest(prior) == state['prior_receipt_sha256'], 'synthetic prior receipt hash')
        require(prior['status'] == 'PASS' and prior['created_hypotheses'] == 4 and not prior['uninstantiable_clauses'], 'incomplete pressure prior')
    require(state['protocol_sha256'] == p['protocol_sha256'], 'state protocol')
    require(type(state['seed']) is int and state['seed'] == trial and len(history) == 4 + len(decisions), 'seed/budget/history length')
    if trial == 1:
        require(state['mode'] in p['supported_modes'] and state['condition'] in p['supported_conditions'], 'unregistered confirmation run')
    require(len(decisions) == 16 or include_incomplete and 0 < len(decisions) < 16 and state['status'] != 'COMPLETE', 'invalid completed/query budget')
    require([h['candidate_id'] for h in history[:4]] == manifest['initial_ids'], 'initial IDs changed')
    initial_commit_audit(path, state, p, pool, manifest)
    require(len({h['candidate_id'] for h in history}) == len(history), 'duplicate experiment')
    require(all(h['candidate_id'] in pool['ix'] for h in history), 'unknown observed point')
    for h in history:
        require(np.isfinite(h['y']) and np.isfinite(h['raw_value']), 'nonfinite observation')
        close(h['raw_value'], h['y'] * pool['scale'], 'engineering scale')
    previous, calls, seen_predictions = None, [], set()
    modes, limitations, forecasts = Counter(), set(), 0
    for t, d in enumerate(decisions, 1):
        label = f'{state["task"]}/{state["mode"]}/{state["condition"]}/q{t}'
        cp = path.parent / 'rounds' / f'{t:02d}_commit.json'
        c, prefix, observation = read(cp), history[:t + 3], history[t + 3]
        require(c['query'] == d['query'] == t, label + ': CP sequence')
        require(c['history_sha256'] == digest(prefix), label + ': prefix hash')
        require(c['protocol_sha256'] == p['protocol_sha256'] and digest(c) == d['commit_sha256'], label + ': CP hash')
        require(c['selected_id'] == observation['candidate_id'] == d['selected_id'], label + ': selected ID')
        require(d['mode'] == c['mode'] and d['BO_reference_id'] == c['BO_reference_id'], label + ': decision mode/reference')
        close(d['y'], observation['y'], label + ': decision y')
        close(d['raw_value'], observation['raw_value'], label + ': decision objective')
        close(d['incumbent_before_y'], max(h['y'] for h in prefix), label + ': incumbent before')
        close(d['incumbent_after_y'], max(h['y'] for h in history[:t + 4]), label + ': incumbent after')
        with np.load(cp.with_name(f'{t:02d}_GP.npz'), allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        n = len(pool['ids'])
        require(all(a.shape == (n,) and np.isfinite(a).all() for a in arrays.values()), label + ': invalid GP array')
        require(np.all(arrays['sd_y'] >= 0) and np.all(arrays['sd_predictive_y'] >= arrays['sd_y'] - 1e-12), label + ': GP noise variances')
        available = ~np.isin(pool['ids'], [h['candidate_id'] for h in prefix])
        order = sorted(range(n), key=lambda j: pool['features'][j])
        ties = np.empty(n, dtype=np.int64)
        ties[order] = np.random.default_rng(np.random.SeedSequence([tie_seed, t])).permutation(n)
        require(np.array_equal(ties, arrays['tie_priority']), label + ': physical-feature tie priority')
        require(np.array_equal(available, arrays['available']), label + ': queried points available')
        base = arrays['base_logei']
        maximizers = np.flatnonzero(available & (base == base[available].max()))
        bo = int(maximizers[np.argmin(ties[maximizers])])
        selected = pool['ix'][c['selected_id']]
        epsilon = log(4) * (16 - t) / 15
        eligible = available & (base[bo] - base <= epsilon)
        require(np.array_equal(eligible, arrays['eligible']), label + ': eligible set')
        prompt_previous = previous
        if p.get('prompt_state') == 'current_round_coverage':
            before_prompt = previous if previous is not None else initial_prompt_hypotheses(pool, prior)
            prompt_previous, expected_events = refreshed_prompt_hypotheses(pool, before_prompt, eligible, t)
            exact(c['pre_call_coverage_events'], expected_events, label + ': pre-call coverage transitions')
        else:
            require(p.get('prompt_state') is None, 'unknown prompt-state protocol')
        require(eligible[selected] and c['eligible_count'] == int(eligible.sum()), label + ': selected outside budget')
        require(c['BO_reference_id'] == pool['ids'][bo], label + ': incorrect BO argmax')
        close(c['epsilon'], epsilon, label + ': epsilon')
        close(c['base_logei'], base[selected], label + ': selected base score')
        close(c['reference_logei'], base[bo], label + ': reference score')
        close(c['opportunity_gap'], base[bo] - base[selected], label + ': gap')
        require(c['GP_fit_key'] == c['GP_acquisition_meta']['fit_key'], label + ': GP fit key')
        fit = read(cp.with_name(f'{t:02d}_fit.json'))
        require(fit and all(np.isfinite(np.asarray(v)).all() for v in fit.values()), label + ': nonfinite fit parameters')
        require(len(c['menu_ids']) == len(set(c['menu_ids'])) and all(eligible[pool['ix'][i]] for i in c['menu_ids']), label + ': menu includes queried/ineligible point')
        hyps = deepcopy(c['hypotheses_at_commit'])
        hmap = {key(h): h for h in hyps}
        require(len(hmap) == len(hyps), label + ': duplicate hypothesis version')
        for h in hyps:
            validate_ledger(h, history, t - 1)
            for side in ('plus', 'minus'):
                anchor = pool['ix'][h[side + '_id']]
                exact(h[side + '_features'], list(pool['features'][anchor]), label + ': anchor features')
                require(in_scope(pool, h, anchor), label + ': anchor outside own scope')
            for inherited in h.get('inherited_negative_evidence', []):
                source_h = hmap[(inherited['source_id'], inherited['source_version'])]
                event = next(e for e in source_h['ledger']['events'] if e['event_id'] == inherited['event_id'])
                for name in ('point_id', 'query_round', 'prediction', 'observed_y', 'delta_log_score'):
                    exact(inherited[name], event[name], label + ': inherited evidence changed')
                old_contract = {k: source_h[k] for k in ('plus_id', 'minus_id', 'plus_features', 'minus_features', 'focus', 'scope')}
                exact(inherited['source_contract'], old_contract, label + ': inherited source contract')
                applicable = set(old_contract['focus']) == set(h['focus'])
                for axis, spec in h['focus'].items():
                    if axis not in old_contract['focus'] or old_contract['focus'][axis]['direction'] != spec['direction']:
                        applicable = False
                    elif spec['direction'] == 'prefer_plus':
                        axis_index = pool['axes'].index(axis)
                        applicable &= all(digest(old_contract[side + '_features'][axis_index]) == digest(h[side + '_features'][axis_index]) for side in ('plus', 'minus'))
                applicable &= in_scope(pool, h, pool['ix'][inherited['point_id']])
                require(bool(applicable) == inherited['applicable'], label + ': inherited applicability')
            if previous is not None:
                old = next((x for x in previous if key(x) == key(h)), None)
                if old is not None:
                    exact(h['ledger']['events'], old['ledger']['events'], label + ': prequery evidence rewrite')
                    exact(h['semantic_key'], old['semantic_key'], label + ': version contract rewrite')
                else:
                    require(h['created_round'] == t and not h['ledger']['events'] and h['ledger']['score'] == 0, label + ': new version has retrospective credit')
        if previous is not None:
            require({key(h) for h in previous} <= hmap.keys(), label + ': historical version disappeared')
        require({key(r) for r in c['coverage']} == {key(h) for h in hyps if h['status'] != 'retracted'}, label + ': coverage skipped a live hypothesis')
        for row in c['coverage']:
            h = hmap[key(row)]
            close(row['weight'], weight(h), label + ': coverage weight')
            count = sum(in_scope(pool, h, int(j)) for j in np.flatnonzero(eligible))
            require(row['eligible_inside_scope'] == count, label + ': coverage count')
            require(bool(count) == (h['status'] == 'active'), label + ': dormancy inconsistent with coverage')
        expected_deployed = [{k: r[k] for k in ('id', 'version', 'weight')} for r in c['coverage'] if r['weight'] > 0 and r['eligible_inside_scope']] if state['mode'] in ('full', 'deployment') else []
        exact(c['deployed_versions'], expected_deployed, label + ': deployed versions')
        if c['deployed_versions']:
            limitations.add('Knowledge mixture argmax not independently reconstructed from full joint posterior.')
        if c['mode'] == 'base_bo':
            require(selected == bo, label + ': base mode selected non-BO point')
            if not c['deployed_versions']:
                close(c['combined_logei'], base[selected], label + ': no-knowledge combined score', atol=0., rtol=0.)
        elif c['mode'] == 'knowledge_guided':
            require(selected != bo and bool(c['deployed_versions']), label + ': guided mode without deployed supported knowledge')
            require(c['combined_logei'] >= base[selected] - log(4) - 1e-12 and c['combined_logei'] <= base[selected] + log(4) + 1e-12, label + ': combined shift exceeds bound')
        elif c['mode'] == 'optional_verification':
            require(state['mode'] == 'full' and selected != bo
                    and t <= parameters['last_verification_round']
                    and modes['optional_verification'] < parameters['maximum_optional_verifications'], label + ': verification quota/round')
            require(c['opportunity_gap'] <= min(epsilon, parameters['verification_maximum_gap']) + 1e-12, label + ': verification opportunity budget')
            v = c['verification']
            require(v and v['incremental_information'] >= parameters['minimum_incremental_information'], label + ': verification information')
            h = hmap[key(v)]
            require(weight(h) == 0 and h['ledger']['score'] + penalty(h) >= 0, label + ': verification of refuted/weighted claim')
            row = next(r for r in c['coverage'] if key(r) == key(v))
            require(row['eligible_inside_scope'] >= parameters['minimum_remaining_scope_candidates'] and row['can_change_rank_at_cap'], label + ': non-actionable verification')
            limitations.add('Optional verification global information argmax not independently reconstructed.')
        else:
            raise AssertionError(label + ': unknown decision mode')
        if c['mode'] != 'optional_verification':
            require(c['verification'] is None, label + ': unexpected verification metadata')
        if joint:
            joint_selection_audit(folder, pool, manifest, prefix, c, arrays, state['mode'], modes['optional_verification'], fit, parameters)
        modes[c['mode']] += 1
        require(c['calls_at_commit'][:len(calls)] == calls and len(c['calls_at_commit']) <= len(calls) + 1, label + ': call prefix rewritten')
        if len(c['calls_at_commit']) > len(calls):
            call = c['calls_at_commit'][-1]
            require(call['query'] == t and state['mode'] != 'bo', label + ': call query/mode')
            require(not calls or t - calls[-1]['query'] >= 3, label + ': LLM call spacing')
            require(t >= 11 or len(c['calls_at_commit']) <= 3, label + ': late-call reserve consumed')
            prompt_audit(folder, p, pool, call, prefix, c, prompt_previous, eligible)
        else:
            require(c['proposal'] is None, label + ': proposal without LLM call')
        calls = c['calls_at_commit']
        require(len(calls) <= 4, label + ': LLM call budget')
        require(len(c['shadows']) == len(d['evidence']), label + ': shadow/receipt count')
        require({key(s) for s in c['shadows']} == {key(h) for h in hyps if h['status'] != 'retracted'}, label + ': missing live prediction')
        for sh, receipt in zip(c['shadows'], d['evidence']):
            require(sh['prediction']['prediction_sha256'] not in seen_predictions, label + ': prediction reused')
            seen_predictions.add(sh['prediction']['prediction_sha256'])
            shadow_feedback(pool, hmap[key(sh)], sh, receipt, observation, t, arrays)
            forecasts += 1
        previous = hyps
    exact(state['hypotheses'], previous, 'final hypotheses/state ledger differs from independent replay')
    exact(state['calls'], calls, 'final call history')
    require(state['verification_used'] == modes['optional_verification'], 'final verification count')
    require(state['verification_used'] <= parameters['maximum_optional_verifications'], 'protocol verification budget exceeded')
    require(state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0, 'operators compiled inside loop')
    values = np.array([h['raw_value'] for h in history])
    curve = np.maximum.accumulate(values)
    return dict(task=state['task'], group=manifest['group'], mode=state['mode'], condition=state['condition'], status=state['status'],
                trial_seed=trial, tie_seed=tie_seed, numerical_seed=0, initial_history_sha256=digest(history[:4]),
                technical_checks='PASS', queries=len(decisions), selection_modes=dict(modes), forecasts=forecasts,
                argmax_scope='FULL_RECONSTRUCTED_GP_AND_SELECTION' if joint else 'FULL_BO' if not limitations else 'BASE_AND_RECORDED_KNOWLEDGE_FIELDS',
                limitations=[] if joint else sorted(limitations), SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
                first_best_total=int(np.flatnonzero(values == curve[-1])[0]) + 1,
                knowledge_roles=knowledge_roles(path, state, parameters),
                run=path.relative_to(ROOT).as_posix())


def source_order_audit(root):
    """Static checks complement, but do not replace, runtime no-oracle tests."""
    tree = ast.parse((root / 'code/chemist_system/chem/benchmarks/knowledge_study.py').read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
    loops = [n for n in ast.walk(run) if isinstance(n, ast.While)]
    require(len(loops) == 1, 'unrecognized runner experiment-loop structure')
    writes, reveals = [], []
    for node in ast.walk(loops[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == 'write' and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'cp':
            writes.append(node.lineno)
        if node.func.attr == 'read_observation':
            reveals.append(node.lineno)
    require(len(writes) == len(reveals) == 1 and writes[0] < reveals[0], 'commit write is not textually before oracle reveal')
    loop_nodes = {id(n) for n in ast.walk(loops[0])}
    initial_writes, initial_reveals = [], []
    for node in ast.walk(run):
        if not isinstance(node, ast.Call) or id(node) in loop_nodes:
            continue
        if (isinstance(node.func, ast.Name) and node.func.id == 'initial_commit'
                and any(k.arg == 'create' and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords)):
            initial_writes.append(node.lineno)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'read_observation':
            initial_reveals.append(node.lineno)
    gp = ast.parse((root / 'code/chemist_system/chem/benchmarks/knowledge_gp.py').read_text(encoding='utf-8'))
    for node in gp.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ('load_task', 'GPAdapter'):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    require(call.func.attr != 'read_observation', 'input/GP adapter calls oracle reader')
            require(not any(isinstance(s, ast.Constant) and isinstance(s.value, str) and 'oracle.npz' in s.value for s in ast.walk(node)), 'input/GP adapter names oracle file')
    initial_order = (len(initial_writes) == len(initial_reveals) == 1 and initial_writes[0] < initial_reveals[0])
    return dict(commit_write_line=writes[0], selected_oracle_read_line=reveals[0],
                initial_commit_before_initial_oracle=initial_order,
                initial_commit_line=initial_writes[0] if len(initial_writes) == 1 else None,
                initial_oracle_read_line=initial_reveals[0] if len(initial_reveals) == 1 else None,
                result='PASS_STATIC_SOURCE_ONLY')


def main():
    started = datetime.now(timezone.utc).isoformat()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default='v2')
    ap.add_argument('--allow-source-drift', action='store_true', help='Audit old artifact internals while reporting source drift explicitly')
    ap.add_argument('--include-incomplete', action='store_true', help='Also audit already committed/observed rounds of active runs')
    ap.add_argument('--joint', action='store_true', help='Read cached state_dict; independently check direct GP conditioning, contrast/GH/mixture/verification argmax; never fit/write cache')
    ap.add_argument('--task', action='append', help='Audit only these canonical task names')
    ap.add_argument('--run', action='append', help='Audit only these exact run directory names')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    require(args.variant and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.variant), 'invalid variant')
    folder = ROOT / 'experiments/runs/knowledge_loop_20261101' / args.variant
    p = read(folder / 'protocol.json')
    trial, tie_seed = trial_seeds(p)
    require(p['protocol_sha256'] == digest({k: v for k, v in p.items() if k != 'protocol_sha256'}), 'protocol self hash')
    drift = [rel for rel, expected in p['source_sha256'].items() if not (ROOT / rel).exists() or filehash(ROOT / rel) != expected]
    current_sources = {(ROOT / 'code/chemist_system/knowledge_loop.py').relative_to(ROOT).as_posix()}
    current_sources.update(path.relative_to(ROOT).as_posix() for path in (ROOT / 'code/chemist_system/chem').rglob('*.py'))
    drift.extend(sorted(current_sources - p['source_sha256'].keys()))
    require(args.allow_source_drift or not drift, 'frozen source changed; use --allow-source-drift only for historical artifact-internal audit')
    goal_path = p.get('goal_config_path', 'code/resources/knowledge_loop_goal_20261101.json')
    require(filehash(ROOT / goal_path) == p['goal_config_sha256'], 'goal config changed')
    goal = read(ROOT / goal_path)
    for item in p['inputs']:
        for rel, expected in item['files_sha256'].items():
            require(filehash(ROOT / rel) == expected, f'frozen input changed: {rel}')
    source = source_order_audit(ROOT)
    if trial == 1:
        require(source['initial_commit_before_initial_oracle'], 'confirmation runner lacks initial commit before initial oracle access')
    items = {i['task']: i for i in p['inputs']}
    pools = {task: dict(public_pool(ROOT, item), task=task) for task, item in items.items()}
    registration = seed1_registration_audit(ROOT, p, pools)
    rows, pending, failures = [], [], []
    for path in sorted((folder / 'runs').glob('*/state.json')):
        state = read(path)
        if args.run and path.parent.name not in args.run:
            continue
        if args.task and state['task'] not in args.task:
            continue
        if state['status'] != 'COMPLETE':
            pending.append(dict(run=path.parent.name, status=state['status'], queries=len(state['decisions'])))
            if not args.include_incomplete or not state['decisions']:
                continue
        try:
            rows.append(audit_run(folder, path, p, pools[state['task']], items[state['task']], args.include_incomplete, args.joint))
            if args.joint:
                print(json.dumps(dict(audited=path.parent.name, queries=rows[-1]['queries'], result='PASS_LISTED_CHECKS'), ensure_ascii=False), flush=True)
        except (AssertionError, ValueError, KeyError, IndexError, StopIteration) as error:
            failures.append(dict(run=path.parent.name, error=str(error)))
    baseline = {r['task']: r for r in rows if r['mode'] == 'bo' and r['status'] == 'COMPLETE'}
    for row in rows:
        if row['mode'] == 'bo' or row['status'] != 'COMPLETE' or row['task'] not in baseline:
            continue
        b = baseline[row['task']]
        require(row['initial_history_sha256'] == b['initial_history_sha256'], 'matched arms have different initial observations')
        row['AUC_delta_vs_matched_BO'] = row['SearchAUC'] - b['SearchAUC']
        row['best_delta_vs_matched_BO'] = row['final_best'] - b['final_best']
        row['performance_noninferior_development'] = row['AUC_delta_vs_matched_BO'] >= -1e-9 and row['best_delta_vs_matched_BO'] >= -1e-9
    real_comparisons = []
    for mode in sorted({r['mode'] for r in rows if r['mode'] != 'bo'}):
        comparable = [r for r in rows if r['mode'] == mode and r['condition'] == 'agent_initial_terra'
                      and 'AUC_delta_vs_matched_BO' in r]
        real_comparisons.append(dict(mode=mode, completed_compared_tasks=len(comparable),
            SearchAUC_strict_wins=sum(r['AUC_delta_vs_matched_BO'] > 1e-9 for r in comparable),
            SearchAUC_ties=sum(abs(r['AUC_delta_vs_matched_BO']) <= 1e-9 for r in comparable),
            SearchAUC_losses=sum(r['AUC_delta_vs_matched_BO'] < -1e-9 for r in comparable),
            final_best_losses=sum(r['best_delta_vs_matched_BO'] < -1e-9 for r in comparable)))
    report = dict(variant=args.variant, started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=p['protocol_sha256'],
                  trial_seed=trial, tie_seed=tie_seed, numerical_seed=0, initial_registration=registration,
                  audit_implementation_sha256=AUDIT_IMPLEMENTATION_SHA256, technical_status='FAIL' if failures else 'PASS_LISTED_CHECKS' if rows else 'NO_COMPLETE_RUNS',
                  independent_arithmetic=True, no_oracle_values_read=True, source_drift=drift, static_source=source,
                  checklist=CHECKLIST, joint_reconstruction=args.joint,
                  limitations=(["GP fit optimum and native MC512 qLogEI numerical implementation are not independently certified by this audit."] + LIMITATIONS[2:]) if args.joint else LIMITATIONS,
                  rows=rows, pending=pending, failures=failures, goal_config_path=goal_path,
                  completed_full_scenarios=sum(r['mode'] == 'full' and r['status'] == 'COMPLETE' for r in rows),
                  primary_task_count=goal.get('task_count', len(p['tasks'])),
                  registered_agent_scenarios=goal.get('agent_scenarios'),
                  real_agent_development_comparisons=real_comparisons,
                  performance_goal_achieved=None, goal_evaluated_by_this_audit=False,
                  performance_note='Technical PASS is not a performance claim. Real-Agent win/tie/loss counts and signed per-task differences are descriptive; stress conditions are separate. Goal promotion and independent confirmation are not decided by this audit.')
    output = (args.output or ROOT / 'experiments/reports/knowledge_loop_20261101' / f'{args.variant}_independent_audit.json').resolve()
    require(not output.is_relative_to(folder.resolve()), 'audit output must not modify experiment run directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(technical_status=report['technical_status'], audited_runs=len(rows), pending=len(pending),
                          failures=failures, source_drift=len(drift), report=str(output), performance_goal_achieved=None), ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
