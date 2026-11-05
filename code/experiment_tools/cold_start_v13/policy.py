"""Event editing and experiment-level safeguards around the frozen controller."""
from copy import deepcopy

from knowledge_rank_v9 import policy as native
from reasoning_memory_v12.engine import category_opportunity
from . import control, adapter

PARAMETERS = dict(control.PARAMETERS, maximum_live_notes=4, maximum_memory_calls=4,
                  minimum_edit_gap=3, late_edit_round=15,
                  trajectory_review_query_budget=2, continuous_change_threshold=.25)


def incumbent(pool, history):
    return min(history, key=lambda r: (-r['raw_value'], pool.features[pool.ix[r['candidate_id']]]))


def thresholds(pool):
    return {a: PARAMETERS['continuous_change_threshold'] for a, k in zip(pool.axes, pool.kinds) if k == 'continuous'}


def public_note(note):
    public = control.public_note(note)
    public.update(claim=note['claim'], focus=note['focus'], birth_reference_id=note['reference_id'],
                  birth_reference_objective=note['reference_objective'],
                  recent_feedback=deepcopy(note['events'][-3:]))
    return public


def public_context(pool, notes, history, ids, query, review_queries):
    public = adapter.public_context(pool, notes, history, ids, query, change_thresholds=thresholds(pool))
    by_key = {(n['id'], n['version']): n for n in notes}
    for row in public['knowledge']:
        row.update(public_note(by_key[row['id'], row['version']]))
        if len(review_queries) >= PARAMETERS['trajectory_review_query_budget']:
            row['eligible_review_candidates'] = []
    public['review_query_budget_remaining'] = max(0, PARAMETERS['trajectory_review_query_budget'] - len(review_queries))
    public['continuous_material_change_threshold_normalized'] = PARAMETERS['continuous_change_threshold']
    return public


def validate_decision(root, pool, notes, state, context, ids, query, raw):
    if len(state['review_queries']) >= PARAMETERS['trajectory_review_query_budget']:
        if any(r.get('use') == 'review_test' for r in raw.get('knowledge_uses', []) if isinstance(r, dict)):
            raise ValueError('The trajectory-wide review experiment budget is exhausted')
    return adapter.commit_decision(root, pool, notes, state['history'], context, ids, query, raw,
                                   change_thresholds=thresholds(pool))


def edit_event(pool, state, ordered, query):
    if not state['history']:
        return dict(allowed=False, reasons=['no_observed_incumbent_yet'])
    calls = state['edit_calls']
    if not calls:
        return dict(allowed=True, reasons=['initial_knowledge'])
    if len(calls) >= PARAMETERS['maximum_memory_calls']:
        return dict(allowed=False, reasons=['edit_budget_exhausted'])
    if query - calls[-1] < PARAMETERS['minimum_edit_gap']:
        return dict(allowed=False, reasons=['minimum_edit_gap'])
    if query < PARAMETERS['late_edit_round'] and len(calls) >= PARAMETERS['maximum_memory_calls'] - 1:
        return dict(allowed=False, reasons=['reserved_late_edit'])
    reasons = []
    recent = state['decisions'][-3:]
    if len(recent) == 3 and recent[0]['incumbent_before_y'] is not None and recent[-1]['incumbent_after_y'] <= recent[0]['incumbent_before_y'] + 1e-12:
        reasons.append('stagnation')
    events = [e for d in recent[-2:] for e in d['feedback']]
    if any(e['calibration']['Brier_gain'] < -.05 for e in events):
        reasons.append('negative_prediction')
    if any(e['recommendation_before'] != e['recommendation_after'] and
           e['recommendation_after'] == 'suppressed' for e in events):
        reasons.append('recommendation_suppressed')
    if any(d['incumbent_before_y'] is not None and d['incumbent_after_y'] > d['incumbent_before_y'] + 1e-12 for d in recent[-2:]):
        reasons.append('new_incumbent_evidence')
    if category_opportunity(pool, state, ordered)[1]:
        reasons.append('category_concentration_without_contrasts')
    live = [n for n in state['notes'] if n['recommendation']['permission'] != 'withdrawn']
    for i, note in enumerate(live):
        for other in live[i+1:]:
            if note['focus'] == other['focus'] and note['effect'] != other['effect']:
                if any(control.in_scope(note['scope'], adapter.conditions(pool, pool.ids[int(j)])) and
                       control.in_scope(other['scope'], adapter.conditions(pool, pool.ids[int(j)])) for j in ordered):
                    reasons.append('opposed_overlapping_rules')
    return dict(allowed=bool(reasons), reasons=sorted(set(reasons)))


def apply_edits(raw, pool, state, query):
    if (not isinstance(raw, dict) or set(raw) != {'summary', 'actions'} or
            type(raw['summary']) is not str or not 1 <= len(raw['summary']) <= 1000 or
            not isinstance(raw['actions'], list) or len(raw['actions']) > 3):
        raise ValueError('Invalid memory editor envelope')
    result, reviews, touched = deepcopy(state['notes']), [], set()
    available = set(native.next_ids(result))
    ref = incumbent(pool, state['history'])
    for index, action in enumerate(raw['actions']):
        review = dict(action=index, accepted=False)
        try:
            if not isinstance(action, dict) or set(action) != native.ACTION_KEYS | {'evidence_queries'}:
                raise ValueError('Invalid action fields')
            op, hid = action['op'], action['id']
            review.update(op=op, id=hid)
            if op not in ('add', 'retain', 'revise', 'retract') or type(hid) is not str or hid in touched:
                raise ValueError('Invalid or repeated action')
            if type(action['reason']) is not str or not 1 <= len(action['reason']) <= 800:
                raise ValueError('Concise evidence reason required')
            evidence = action['evidence_queries']
            if not isinstance(evidence, list) or any(type(q) is not int or not 1 <= q < query for q in evidence):
                raise ValueError('Evidence must refer to completed own queries')
            old = next((n for n in reversed(result) if n['id'] == hid), None)
            if op == 'add' and (old is not None or hid not in available):
                raise ValueError('Use a supplied unused ID')
            if op != 'add' and old is None:
                raise ValueError('Unknown note')
            if op in ('retain', 'retract'):
                if any(action[k] is not None for k in ('claim', 'effect', 'reference_id')) or action['focus'] != [] or action['scope'] != {}:
                    raise ValueError('Retain/retract cannot rewrite semantics')
                if old['recommendation']['permission'] == 'withdrawn':
                    raise ValueError('Retired note requires an explicit revision')
                if op == 'retract':
                    adverse = {f['query'] for f in old['recommendation']['risk_facts']}
                    adverse.update(e['query'] for e in old['events'] if e['calibration']['Brier_gain'] < -1e-12)
                    if not set(evidence) & adverse:
                        raise ValueError('Withdrawal requires recorded adverse evidence')
                    result[result.index(old)] = control.withdraw(old, query, 'LLM: ' + action['reason'])
                touched.add(hid)
                review.update(accepted=True, outcome=op, version=old['version'])
                reviews.append(review)
                continue
            if (type(action['claim']) is not str or not 1 <= len(action['claim']) <= 900 or
                    action['effect'] not in ('promote', 'avoid') or not isinstance(action['focus'], list) or
                    not 1 <= len(action['focus']) <= 2 or len(set(action['focus'])) != len(action['focus']) or
                    set(action['focus']) - set(pool.axes) or action['reference_id'] != ref['candidate_id']):
                raise ValueError('Invalid claim, focus or current-incumbent birth reference')
            scope = native.normalize_scope(pool, action['scope'])
            if sum(control.in_scope(scope, dict(zip(pool.axes, row))) for row in pool.features) < 2:
                raise ValueError('Scope must cover at least two public pool inputs')
            semantic = dict(claim=action['claim'], focus=sorted(action['focus']), effect=action['effect'], scope=scope)
            if old and all(old[k] == v for k, v in semantic.items()):
                touched.add(hid)
                review.update(accepted=True, outcome='unchanged_version_preserved', version=old['version'])
                reviews.append(review)
                continue
            if any(n['id'] != hid and all(n[k] == semantic[k] for k in ('focus', 'effect', 'scope')) for n in result):
                raise ValueError('Structural duplicate; revise the existing ID')
            live = sum(n['recommendation']['permission'] != 'withdrawn' for n in result)
            if live - int(old is not None and old['recommendation']['permission'] != 'withdrawn') >= 4:
                raise ValueError('At most four live notes')
            note = control.new_note(hid, old['version'] + 1 if old else 1, action['effect'], scope,
                                    semantic['focus'], created_query=query, predecessors=result)
            note.update(claim=action['claim'], reference_id=ref['candidate_id'], reference_objective=ref['raw_value'],
                        birth_history_sha256=control.digest(state['history']), edit_reason=action['reason'])
            if old:
                result[result.index(old)] = control.withdraw(old, query, 'superseded by explicit revision')
            result.append(note)
            touched.add(hid)
            review.update(accepted=True, outcome=op, version=note['version'],
                          carried_fact_count=len(note['recommendation']['risk_facts']))
        except (ValueError, KeyError, TypeError) as error:
            review['reason'] = str(error)
        reviews.append(review)
    return dict(notes=result, reviews=reviews)
