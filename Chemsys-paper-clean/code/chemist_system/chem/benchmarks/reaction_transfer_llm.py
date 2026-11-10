"""Named nominal conditions, checked observed evidence, and bounded API repair."""
from copy import deepcopy
from pathlib import Path
import json
import math
import time

import httpx

from . import deepseek_suzuki_client as api
from . import deepseek_small_suzuki_study as transport
from . import reaction_transfer_data as data
from .suzuki_domain import digest

PARAMETERS = deepcopy(transport.PARAMETERS)


def domain(pool):
    return {a: [{'id': f'{a}_{i+1:02d}', 'source_label': label} for i, label in enumerate(pool['levels'][a])]
            for a in pool['axes']}


def token_maps(pool):
    forward = {a: {r['source_label']: r['id'] for r in rows} for a, rows in domain(pool).items()}
    reverse = {a: {value: key for key, value in rows.items()} for a, rows in forward.items()}
    return forward, reverse


def public_point(pool, point):
    forward, _ = token_maps(pool)
    return {a: forward[a][point[a]] for a in pool['axes']}


def decode_point(pool, point):
    _, reverse = token_maps(pool)
    if not isinstance(point, dict) or set(point) != set(pool['axes']):
        raise ValueError('point requires exactly the named categorical axes')
    result = {}
    for a in pool['axes']:
        if not isinstance(point[a], str) or point[a] not in reverse[a]:
            raise ValueError('unknown category ID')
        result[a] = reverse[a][point[a]]
    return result


def observations(pool, state, include_gp=True):
    lookup = {k: i for i, k in enumerate(pool['ids'])}
    result = []
    for i, h in enumerate(state['history'], 1):
        row = {'index': i, 'point': public_point(pool, pool['points'][lookup[h['candidate_id']]]),
               'yield_percent': round(h['y']*100, 3)}
        if include_gp and i > 4 and 'measurement' in state['decisions'][i-5]:
            m = state['decisions'][i-5]['measurement']
            row.update(prequery_GP_mean_percent=round(m['prequery_mu_y']*100, 3),
                prequery_observation_SD_pp=round(m['prequery_predictive_sd_y']*100, 3),
                prequery_GP_residual_pp=round(m['residual']*100, 3),
                unscaled_region_shift_pp=round(m['raw_region_shift_y']*100, 3),
                joint_innovation_z=round(m['innovation_z'], 4))
        result.append(row)
    return result


def public_knowledge(pool, knowledge):
    forward, _ = token_maps(pool)
    return {'confidence': knowledge['confidence'], 'clauses': [
        {'id': h['id'], 'axis': h['axis'], 'allowed': [forward[h['axis']][v] for v in h['allowed']]}
        for h in knowledge['clauses']]}


def public_request(pool, state, decision, reasons, phase='refresh'):
    from .reaction_transfer_engine import argmax
    primary = pool['axes'][0]
    top = []
    # Up to three/current primary-category level, capped at24, using GP only.
    by_level = []
    for level in pool['levels'][primary]:
        candidates = [j for j, p in enumerate(pool['points']) if p[primary] == level and decision['available'][j]]
        by_level.append(sorted(candidates, key=lambda j: (-float(decision['base'][j]), pool['ids'][j]))[:3])
    chosen = [rows[k] for k in range(3) for rows in by_level if k < len(rows)][:24]
    for j in chosen:
        top.append({'point': public_point(pool, pool['points'][j]),
            'GP_mean_percent': round(float(decision['stats']['mu_y'][j])*100, 3),
            'GP_latent_SD_pp': round(float(decision['stats']['sd_y'][j])*100, 3),
            'base_logEI': round(float(decision['base'][j]), 5)})
    return {'task': state['task'], 'phase': phase, 'reaction_class': pool['reaction_class'],
        'fixed_substrates': pool['fixed_substrates'], 'objective_unit': pool['objective_unit'],
        'domain': domain(pool), 'fixed_slots': {f'H{i+1}': a for i, a in enumerate(pool['axes'])},
        'iteration': len(state['decisions'])+1, 'remaining_queries': 16-len(state['decisions']),
        'reasons': reasons, 'observations': observations(pool, state),
        'current_knowledge': public_knowledge(pool, state['knowledge']),
        'GP_summary': top, 'B4_diagnosis': decision['diagnosis'].to_dict(),
        'calibration_bank': decision['bank'], 'calibration_forecast': decision['prior'],
        'deployed_amplitude': decision['amplitude'], 'conflict_fallback': decision['fallback'],
        'calibration_units': {'b': 'yield fraction', 'a': 'dimensionless',
            'P': 'covariance in [yield fraction, dimensionless] coordinates',
            'GP_and_region_display': 'percentage points; divide by100 for the filter'},
        'all_inputs_categorical': True}


def validate_plan(raw, pool, knowledge, observed, grounded):
    if not isinstance(raw, dict) or set(raw) != {'actions', 'summary'}:
        raise ValueError('expected actions,summary')
    if not isinstance(raw['summary'], str) or not 1 <= len(raw['summary']) <= 1600:
        raise ValueError('invalid summary')
    if not isinstance(raw['actions'], list) or len(raw['actions']) > len(pool['axes']):
        raise ValueError('at most one action per axis')
    slots = {f'H{i+1}': axis for i, axis in enumerate(pool['axes'])}
    old = {h['id']: h for h in knowledge['clauses']}
    by_index = {row['index']: row for row in observed}
    _, reverse = token_maps(pool)
    seen, result, facts = set(), {}, []
    for row in raw['actions']:
        if not isinstance(row, dict) or set(row) != {'id', 'operation', 'allowed', 'reason', 'support'}:
            raise ValueError('invalid action fields')
        key, op = row['id'], row['operation']
        if key not in slots or key in seen:
            raise ValueError('unknown or duplicate fixed slot')
        axis = slots[key]
        seen.add(key)
        if not isinstance(row['reason'], str) or not 1 <= len(row['reason']) <= 1200:
            raise ValueError('invalid concise reason')
        if not isinstance(row['support'], list) or not row['support']:
            raise ValueError('nonempty observed support required')
        cited = []
        for fact in row['support']:
            if grounded:
                if not isinstance(fact, dict) or set(fact) != {'index', 'axis', 'category_id', 'yield_percent'}:
                    raise ValueError('expected typed categorical evidence')
                i = fact['index']
                if type(i) != int or i not in by_index:
                    raise ValueError('future/unobserved evidence')
                if fact['axis'] != axis or fact['category_id'] != by_index[i]['point'][axis]:
                    raise ValueError('observed named category mismatch')
                value = fact['yield_percent']
                if type(value) not in (int, float) or not math.isfinite(value) or abs(value-by_index[i]['yield_percent']) > 1e-8:
                    raise ValueError('observed yield mismatch')
                facts.append({'slot': key, **deepcopy(fact)})
            else:
                i = fact
                if type(i) != int or i not in by_index:
                    raise ValueError('future/unobserved evidence')
            if i in cited:
                raise ValueError('duplicate support')
            cited.append(i)
        if op in ('retain', 'revoke'):
            if key not in old or row['allowed'] != []:
                raise ValueError('retain/revoke require active slot and allowed=[]')
            if op == 'retain':
                result[key] = deepcopy(old[key])
        elif op in ('add', 'revise'):
            if (op == 'add') != (key not in old):
                raise ValueError('add vacant, revise active')
            allowed = row['allowed']
            if not isinstance(allowed, list) or not 1 <= len(allowed) < len(pool['levels'][axis]):
                raise ValueError('allowed must be a nontrivial nominal subset')
            if any(not isinstance(v, str) or v not in reverse[axis] for v in allowed) or len(set(allowed)) != len(allowed):
                raise ValueError('unknown or duplicate allowed category')
            clause = {'id': key, 'axis': axis, 'allowed': sorted(reverse[axis][v] for v in allowed)}
            if op == 'revise' and clause == old[key]:
                raise ValueError('identical geometry must retain')
            result[key] = clause
        else:
            raise ValueError('unknown operation')
    if not set(old) <= seen:
        raise ValueError('account for every active clause')
    return {'actions': deepcopy(raw['actions']), 'summary': raw['summary'], 'checked_facts': facts,
        'knowledge': {'confidence': knowledge['confidence'], 'clauses': [result[k] for k in sorted(result)]}}


COMMON = '''You assist one chemistry experiment optimization task using ONLY its observed experiments.
All varying axes are categorical. Category IDs and internal numeric codes are identifiers, NOT ranks,
distances or concentrations. Each ID is mapped once to its exact source name or SMILES in domain.
Do not invent structures behind incomplete names, temperatures, reaction times, or variables absent from domain.
Do not identify the benchmark, recall its outcomes, use another trajectory, or treat GP predictions as observations.
Fixed substrates must remain fixed. Literal source label None denotes an intentional no-ligand/no-base control.
Objective units are supplied; UV area percentage is not asserted to be isolated chemical yield.
Use concise rationales, not private reasoning. Factual evidence is not proof of chemical causality.
'''

REGION_COMMON = COMMON + '''
Update an uncertain preferred REGION, the intersection of at most one subset per named axis.
Every method starts with the same supplied knowledge within its condition. Confidence is fixed and cannot be edited.
The region shifts only the common GP mean through the official regional covariance formula; it does not change covariance.
A separate KF state [b,a] per actual region estimates residual y-mu_GP using [1,unscaled region field].
b is a nuisance offset, not deployed; a is clipped to[0,1]. An acquisition-budget conflict may force deployment to zero.
The unscaled shadow region is still calibrated during fallback. Old prequery residuals come from different GP fits.
B4 diagnoses compatibility with the acquisition budget, not chemical truth. Revoke unsupported or GP-redundant preferences.
Return exactly {"actions":[...],"summary":"..."}. Each action has EXACTLY id,operation,allowed,reason,support.
Use fixed_slots without changing an axis. Account for every active slot with retain/revise/revoke.
Vacant slots may be add or omitted. At most one action per slot.
retain/revoke use allowed=[]; retain preserves existing geometry automatically.
add/revise uses a nonempty proper subset of that axis's known category IDs; no unknown IDs or duplicates.
reason<=1200 chars, summary<=1600 chars; prefer short rationales and1..3 supporting observations.
You may revoke all knowledge. No experiment occurs during repair. Output only the JSON object.
'''

REGION_GROUNDED = REGION_COMMON + '''
support must be a nonempty list of EXACT typed observed facts:
{"index":<observed 1-based index>,"axis":<the fixed slot's named axis>,
 "category_id":<copy that observed row's category ID for THIS axis>,"yield_percent":<copy displayed yield>}
No duplicate indices within an action. Copy the displayed values exactly, without recomputation.
All facts are checked against this request's own observed table before applying your update.
'''

REGION_LEGACY = REGION_COMMON + '''
support must be a nonempty list of 1-based integer observed indices. No future or duplicate indices.
'''

POINT_SYSTEM = COMMON + '''
Propose one next nominal experiment using the named domain and your own observed history.
Balance unresolved conditions and refinement of high-yield conditions. Supplied preferences are uncertain.
No GP scores or uncertainty estimates are provided to you. A separate Justify reviewer may reject your proposal.
Return exactly {"point":{<every axis>:<known category ID>},"reason":"...","support":[<observed indices>]}.
reason must be concise. support is a nonempty list of observed 1-based indices, no future evidence.
Already queried/absent combinations are projected onto the nearest unused measured combination using Hamming distance,
with fixed lexical candidate-ID ties, before BO review. No experiment occurs during schema repair.
Output only the JSON object.'''


def validate_point(raw, pool, history):
    if not isinstance(raw, dict) or set(raw) != {'point', 'reason', 'support'}:
        raise ValueError('expected point,reason,support')
    point = decode_point(pool, raw['point'])
    if not isinstance(raw['reason'], str) or not raw['reason'].strip() or len(raw['reason']) > 1600:
        raise ValueError('invalid reason')
    support = raw['support']
    if not isinstance(support, list) or not support or any(type(i) != int or not 1 <= i <= len(history) for i in support):
        raise ValueError('unobserved point support')
    return {**raw, 'decoded_point': point}


def get(client, context, folder, system, validator):
    folder = Path(folder)
    if (folder/'validated.json').exists():
        saved = data.read(folder/'validated.json')
        assert saved['context_sha256'] == context['context_sha256']
        assert saved['plan'] == validator(saved['raw'])
        return saved
    base = [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(context['public'], ensure_ascii=False, separators=(',', ':'))}]
    messages, received = base, 0
    for attempt in range(1, 9):
        directory = folder/f'attempt{attempt:02d}'
        payload = dict(PARAMETERS, messages=messages)
        if (directory/'request.json').exists():
            assert data.read(directory/'request.json') == payload
        else:
            data.write(directory/'request.json', payload)
        if (directory/'transport_error.json').exists():
            if not data.read(directory/'transport_error.json')['retryable']:
                raise RuntimeError('nonretryable API resource failure')
            continue
        if (directory/'response.json').exists():
            answer = data.read(directory/'response.json')
        else:
            try:
                answer = transport.complete(client, messages)
                data.write(directory/'response.json', answer)
            except Exception as error:
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                retryable = isinstance(error, (httpx.TimeoutException, httpx.TransportError)) or status in (429,500,502,503,504)
                data.write(directory/'transport_error.json', {'error': api.safe_error(error, client.config), 'retryable': retryable})
                if not retryable:
                    raise RuntimeError('nonretryable API resource failure') from None
                if attempt < 8:
                    time.sleep(min(5*2**(attempt-1), 30))
                continue
        received += 1
        try:
            if answer['model'] != api.MODEL or len(answer['choices']) != 1 or answer['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('model mismatch or incomplete response')
            raw = json.loads(answer['choices'][0]['message']['content'])
            plan = validator(raw)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            text = type(error).__name__+': '+str(error)[:220]
            data.write(directory/'validation_error.json', {'error': text})
            if received >= 3:
                raise RuntimeError('received response limit exhausted') from None
            content = answer['choices'][0]['message'].get('content') if answer.get('choices') else '{}'
            messages = base + [{'role': 'assistant', 'content': content or '{}'},
                {'role': 'user', 'content': 'Validation failed: '+text+'. Correct only the JSON from this same observed data; no experiment occurred.'}]
            continue
        valid = {'raw': raw, 'plan': plan, 'accepted_attempt': attempt, 'received_responses': received,
            'context_sha256': context['context_sha256'], 'response_sha256': data.filehash(directory/'response.json'),
            'response_model': answer['model'], 'response_id': answer['id']}
        data.write(folder/'validated.json', valid)
        return valid
    raise RuntimeError('HTTP attempt limit exhausted')
