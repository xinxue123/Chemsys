"""Frozen-prefix diagnosis only: no LLM calls, GP fits, or new trajectories.

Counterfactual scores reuse existing GP checkpoints and MC seeds. Hidden targets
are read only after all alternative choices at a round have been fixed. These
one-step comparisons are not replacement trajectories or causal AUC ablations.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from chem.benchmarks import continuous_benchmark_engine as base
from chem.benchmarks import revised_continuous_engine as r1
from chem.benchmarks.suzuki_domain import digest

OUT = ROOT / 'research/continuous_r1_diagnosis_20261030'
R1 = ROOT / 'data/runs/continuous_revised_20261030'
OLD = ROOT / 'data/runs/continuous_20261029'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def filehash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def refuse(*args, **kwargs):
    raise RuntimeError('Diagnosis refuses new GP fitting or experiment outcome calls.')


def prefix(state, t):
    prev = state['decisions'][:t-1]
    banks = {d['bank_key']: deepcopy(d['bank_after']) for d in prev if d['bank_key'] and d['bank_after']}
    call = next((c for c in state['calls'] if c['iteration'] == t), None)
    return {**deepcopy(state), 'history': deepcopy(state['history'][:t+3]),
            'decisions': deepcopy(prev), 'banks': banks,
            'knowledge': deepcopy(call['knowledge_before'] if call else prev[-1]['knowledge'] if prev else state['initial_knowledge']),
            'region_trust': deepcopy(prev[-1]['region_trust_after']) if prev else {},
            'calls': [deepcopy(c) for c in state['calls'] if c['iteration'] < t],
            'interaction_tests': [deepcopy(c) for c in state['interaction_tests'] if c['query'] < t]}


class FrozenRoundCache(base.Cache):
    """Read existing checkpoint and arrays without locks, writes, or fit fallback."""
    def __init__(self, pool, commit, arrays):
        self.pool = pool
        self.ids = pool['ids']
        self.ix = {k: i for i, k in enumerate(self.ids)}
        self.X = torch.tensor(pool['z'], dtype=torch.double)
        self.commit = commit
        self.arrays = arrays
        self.fields = {}
        self.value = None
        self.counters = {'fits': 0, 'posteriors': 0, 'acquisitions': 0}

    def model(self, history):
        assert digest(history) == self.commit['history_sha256']
        if self.value is not None:
            return self.value
        ordered = sorted(history, key=lambda h: tuple(self.pool['z'][self.ix[h['candidate_id']]]))
        x = self.X[[self.ix[h['candidate_id']] for h in ordered]]
        y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
        mean = self.commit['GP_mean_y']
        std = self.commit['GP_scale_y']
        base.seed_all(self.commit['base_meta']['fit_seed'])
        model = base.SingleTaskGP(x, (y-mean)/std)
        key = self.commit['base_meta']['fit_key']
        saved = torch.load(R1 / 'cache' / self.pool['task'] / (key+'.pt'), map_location='cpu', weights_only=False)
        assert saved['key'] == key
        model.load_state_dict(saved['state'])
        model.eval()
        self.value = {'model': model, 'mean': mean, 'std': std,
                      'best_std': self.commit['best_std'], 'fit_key': key,
                      'fit_seed': self.commit['base_meta']['fit_seed'],
                      **{name: self.arrays[name] for name in ['mu_std', 'sd_std', 'mu_y', 'sd_y', 'sd_predictive_y']}}
        return self.value

    def score(self, history, shift=None):
        v = self.model(history)
        self.counters['acquisitions'] += 1
        scores = self.arrays['base_qLogEI'] if shift is None or not np.any(shift) else base.acquisition(
            v['mu_std'], v['sd_std'], shift, v['best_std'], self.commit['base_meta']['mc_seed'])
        return scores, self.commit['base_meta']


def audit_task(task):
    base.fit_gpytorch_mll = refuse
    base.data.reveal = refuse
    base.warm()
    pool = read(ROOT / f'data/benchmarks/continuous_20261029/{task}/pool.json')
    state = read(R1 / 'runs' / (task+'_New_Agent_R1') / 'state.json')
    folder = R1 / 'runs' / state['run_id'] / 'rounds'
    stages, wide, refreshes, calibration, rankings, interactions = [], [], [], [], [], []
    ix = {key: j for j, key in enumerate(pool['ids'])}
    scale = pool['objective_scale']
    max_error = 0.
    for t, event in enumerate(state['decisions'], 1):
        commit = read(folder / f'{t:02d}_committed.json')
        assert filehash(folder / f'{t:02d}_committed.json') == event['commit_sha256']
        assert filehash(folder / f'{t:02d}_candidates.npz') == commit['candidate_table_sha256']
        with np.load(folder / f'{t:02d}_candidates.npz') as file:
            arrays = {k: file[k].copy() for k in file.files}
        cache = FrozenRoundCache(pool, commit, arrays)
        own = prefix(state, t)
        before = r1.setup(cache, own, t)
        call = next((c for c in state['calls'] if c['iteration'] == t), None)
        if call:
            own['knowledge'] = deepcopy(call['answer']['plan']['knowledge'])
            own['calls'].append(deepcopy(call))
            after = r1.setup(cache, own, t)
        else:
            after = before
        for name, value in [('base_qLogEI', after['base']), ('guided_qLogEI', after['score']),
                            ('raw_shift_std', after['raw_shift_std']), ('deployed_shift_std', after['shift_std'])]:
            error = float(np.max(abs(value-arrays[name])))
            assert error < 1e-8, (task, t, name, error)
            max_error = max(max_error, error)
        assert pool['ids'][after['selected']] == commit['normal_proposal_id']
        assert after['amplitude'] == commit['amplitude']
        v = after['stats']
        def choose(scores, mask):
            js = np.flatnonzero(mask)
            assert len(js)
            js = js[scores[js] == scores[js].max()]
            js = js[after['information_proxy'][js] == after['information_proxy'][js].max()]
            js = js[v['sd_y'][js] == v['sd_y'][js].max()]
            return int(js[np.argmin(after['tie_priority'][js])])
        rows = []
        def record(stage, j, scores, shift=None, alpha=None):
            rows.append({'task': task, 'query_round': t, 'stage': stage, 'candidate_id': pool['ids'][j],
                         'GP_mean': float(v['mu_y'][j])*scale, 'GP_latent_SD': float(v['sd_y'][j])*scale,
                         'base_qLogEI': float(after['base'][j]), 'stage_qLogEI': float(scores[j]),
                         'normal_gate_eligible': bool(after['eligible'][j]), 'alpha': alpha,
                         'shift_raw_units': float(shift[j]*v['std']*scale) if shift is not None else 0.,
                         'incumbent_before': event['incumbent_before']*scale})
        def field_choice(stage, knowledge, retained, alpha):
            raw, _ = cache.field(own['history'], base.region(pool, knowledge, retained))
            scores, _ = cache.score(own['history'], alpha*raw)
            record(stage, choose(scores, after['eligible']), scores, alpha*raw, alpha)
        record('base_BO', after['reference'], after['base'])
        field_choice('full_region_a1', own['knowledge'], [h['id'] for h in own['knowledge']['clauses']], 1.)
        field_choice('B4_geometry_a1', own['knowledge'], after['retained'], 1.)
        a = after['prior']['amplitude'] if after['prior'] else 0.
        field_choice('KF_before_conflict_gate', own['knowledge'], after['retained'], a)
        conflict = bool(after['diagnosis'].revoked_ids) or after['diagnosis'].conflict_kind != 'none'
        field_choice('B4_conflict_gate', own['knowledge'], after['retained'], 0. if conflict else a)
        record('normal_after_trust', after['selected'], after['score'], after['shift_std'], after['amplitude'])
        record('before_refresh_normal', before['selected'], before['score'], before['shift_std'], before['amplitude'])
        if call:
            previous_a = before['prior']['amplitude'] if before['prior'] else 0.
            deployed_a = 0. if conflict or after['trust']['trust_blocked'] else previous_a
            field_choice('new_geometry_previous_amplitude', own['knowledge'], after['retained'], deployed_a)
        full_spec = base.region(pool, own['knowledge'], [h['id'] for h in own['knowledge']['clauses']])
        full_key = base.trust.geometry_key(full_spec)
        full_a = r1.forecast(own['banks'].get(full_key, r1.new_bank(t)), t)['amplitude'] if full_key else 0.
        if own['region_trust'].get(full_key, {}).get('blocked'):
            full_a = 0.
        field_choice('no_B4_full_calibrated', own['knowledge'], [h['id'] for h in own['knowledge']['clauses']], full_a)
        record('normal_without_budget_mask', choose(after['score'], after['available']), after['score'], after['shift_std'], after['amplitude'])
        selected = ix[commit['selected_id']]
        record('executed', selected, after['score'], after['shift_std'], after['amplitude'])
        # Independent KF algebra check: the measured feature is the deployed one.
        m = event['measurement']; prior = event['bank_prior']
        H = m['raw_region_shift_y'] if event['gate_open'] else 0.
        innovation = event['y']-m['prequery_mu_y']-prior['mean']*H
        R = m['prequery_predictive_sd_y']**2 + .0009
        K = prior['variance']*H/(H*H*prior['variance']+R)
        expected = float(np.clip(prior['mean']+K*innovation, 0, 1))
        assert abs(expected-event['bank_after']['mean']) < 1e-12
        if m['actual_paired_SE_gain'] < -1e-12:
            assert expected <= prior['mean']+1e-12
        calibration.append({'task': task, 'query_round': t, 'bank_key': event['bank_key'],
                            'bank_updates_before': event['bank_before']['updates'], 'gate_open': event['gate_open'],
                            'coefficient_before': prior['mean'], 'coefficient_after': expected,
                            'GP_mean': m['prequery_mu_y']*scale,
                            'deployed_correction': m['actual_deployed_correction_y']*scale,
                            'deployed_mean': m['actual_deployed_mean_y']*scale, 'observed': event['raw_value'],
                            'GP_absolute_error': abs(event['y']-m['prequery_mu_y'])*scale,
                            'deployed_absolute_error': abs(event['y']-m['actual_deployed_mean_y'])*scale,
                            'paired_SE_gain_raw_units_squared': m['actual_paired_SE_gain']*scale*scale})
        # All choices above are fixed before the first hidden target access here.
        with np.load(ROOT / f'data/benchmarks/continuous_20261029/{task}/oracle.npz') as file:
            truth = file['raw_values'].copy()
        for row in rows:
            value = float(truth[ix[row['candidate_id']]])
            row['objective_analysis_only'] = value
            row['incumbent_after_one_step_analysis_only'] = max(row['incumbent_before'], value)
        stages.extend(rows)
        by_stage = {r['stage']: r for r in rows}
        def delta(a, b, field='objective_analysis_only'):
            return by_stage[a][field]-by_stage[b][field]
        parts = [delta('full_region_a1', 'base_BO'), delta('B4_geometry_a1', 'full_region_a1'),
                 delta('KF_before_conflict_gate', 'B4_geometry_a1'), delta('B4_conflict_gate', 'KF_before_conflict_gate'),
                 delta('normal_after_trust', 'B4_conflict_gate'), delta('executed', 'normal_after_trust')]
        assert abs(sum(parts)-delta('executed', 'base_BO')) < 1e-8
        full_membership = base.membership(pool, own['knowledge'])
        violated = [h['id'] for k, h in enumerate(own['knowledge']['clauses']) if not full_membership[k, selected]]
        wide.append({'task': task, 'query_round': t, 'selected_id': commit['selected_id'],
                     'selected_objective': float(truth[selected]), 'same_prefix_BO_id': commit['reference_id'],
                     'same_prefix_BO_objective': float(truth[ix[commit['reference_id']]]),
                     'knowledge_a1_delta': parts[0], 'B4_geometry_delta': parts[1], 'KF_scaling_delta': parts[2],
                     'B4_conflict_fallback_delta': parts[3], 'persistent_trust_delta': parts[4],
                     'interaction_override_delta': parts[5], 'actual_vs_same_prefix_BO_delta': sum(parts),
                     'actual_vs_same_prefix_BO_incumbent_delta': delta('executed','base_BO','incumbent_after_one_step_analysis_only'),
                     'refresh_choice_delta': delta('normal_after_trust','before_refresh_normal') if call else 0.,
                     'unqueried_knowledge_witnesses': commit['admission']['unqueried_witness_count'],
                     'unqueried_knowledge_witnesses_inside_gate': int(np.sum(arrays['full_region_unqueried'] & after['eligible'])),
                     'knowledge_admitted': commit['admission']['admitted'], 'selected_violated_preferences': ','.join(violated),
                     'KF_changed_choice': by_stage['KF_before_conflict_gate']['candidate_id'] != by_stage['B4_geometry_a1']['candidate_id'],
                     'gate_open': event['gate_open'], 'B4_revoked_ids': ','.join(event['diagnosis']['revoked_ids']),
                     'normal_eligible_count': event['eligible_count'], 'base_opportunity_loss': event['base_opportunity_loss'],
                     'epsilon': event['budget'], 'BO_in_interaction_pool': bool(arrays['interaction_pool'][ix[commit['reference_id']]]),
                     'executed_in_interaction_pool': bool(arrays['interaction_pool'][selected])})
        if commit['interaction']:
            joint = commit['interaction']; prepared = joint['prepared']; plan = joint['answer']['plan']
            anchor_id = prepared['anchor_candidate_id']; anchor_j = ix[anchor_id]
            record_joint = {'task':task, 'query_round':t, 'geometry':prepared['geometry'],
                            'anchor_id':anchor_id, 'selected_id':commit['selected_id'],
                            'focus_axes':plan['focus_axes'], 'all_changed_axes':plan['all_changed_axes'],
                            'BO_candidate_in_joint_pool':bool(arrays['interaction_pool'][after['reference']]),
                            'reason':plan['reason'], 'single_axis_controls':[]}
            if prepared['geometry'] == 'exact_two_axes':
                z = np.asarray(pool['z']); controls = []
                for axis in plan['focus_axes']:
                    k = pool['axes'].index(axis); point = z[anchor_j].copy(); point[k] = z[selected,k]
                    matches = np.flatnonzero(np.all(np.isclose(z,point[None,:],atol=1e-12,rtol=0),axis=1))
                    assert len(matches)==1
                    j = int(matches[0])
                    controls.append({'changed_axis':axis,'candidate_id':pool['ids'][j],
                                     'observed_before':not bool(after['available'][j]),
                                     'inside_gate_if_unqueried':bool(after['eligible'][j]),
                                     'objective_analysis_only':float(truth[j])})
                record_joint['single_axis_controls'] = controls
                record_joint['complete_two_by_two_observed_before_joint'] = all(c['observed_before'] for c in controls)
            else:
                record_joint['complete_two_by_two_observed_before_joint'] = False
                record_joint['control_scope'] = 'Scattered joint-condition challenge; no matched factorial contrast claimed.'
            interactions.append(record_joint)
        if call:
            refreshes.append({'task': task, 'query_round': t, 'trigger': call['reasons'],
                              'knowledge_before': call['knowledge_before'], 'knowledge_after': own['knowledge'],
                              'prior_alpha_before_refresh': before['prior']['amplitude'] if before['prior'] else 0.,
                              'alpha_after_refresh': after['amplitude'], 'new_bank': after['bank']['updates'] == 0,
                              'before_choice': by_stage['before_refresh_normal'], 'after_choice': by_stage['normal_after_trust'],
                              'new_geometry_previous_amplitude': by_stage['new_geometry_previous_amplitude'],
                              'summary': call['answer']['plan']['summary']})
        target = int(truth.argmax())
        for label, j in [('pool_best', target), ('selected', selected), ('same_prefix_BO', ix[commit['reference_id']])]:
            rankings.append({'task': task, 'query_round': t, 'point_role': label, 'candidate_id': pool['ids'][j],
                             'objective_analysis_only': float(truth[j]), 'GP_mean': float(v['mu_y'][j])*scale,
                             'GP_latent_SD': float(v['sd_y'][j])*scale, 'base_qLogEI': float(after['base'][j]),
                             'guided_qLogEI': float(after['score'][j]), 'available': bool(after['available'][j]),
                             'inside_gate': bool(after['eligible'][j]),
                             'base_rank': int(1+np.sum(after['base'][after['available']]>after['base'][j])),
                             'violated_preferences': ','.join(h['id'] for k,h in enumerate(own['knowledge']['clauses']) if not full_membership[k,j]),
                             **pool['points'][j]})
        assert cache.counters['fits'] == 0
    return {'task': task, 'stages': stages, 'module_effects': wide, 'refreshes': refreshes,
            'KF_updates': calibration, 'rankings': rankings, 'interactions':interactions, 'max_replay_error': max_error,
            'refresh_checks': state['refresh_checks'], 'interaction_checks': state['interaction_checks']}


def main():
    protected = read(ROOT / 'research/continuous_revised_20261030/protected_snapshot.json')
    protected.update(read(ROOT / 'research/continuous_revised_20261030/release_inventory.json'))
    protected['research/continuous_revised_20261030/release_inventory.json'] = filehash(ROOT/'research/continuous_revised_20261030/release_inventory.json')
    protected['research/continuous_revised_20261030/release_receipt.json'] = filehash(ROOT/'research/continuous_revised_20261030/release_receipt.json')
    for name, expected in protected.items():
        assert filehash(ROOT/name) == expected, name
    with patch.object(base, 'fit_gpytorch_mll', refuse), patch.object(base.data, 'reveal', refuse):
        # Independent tasks use separate warmed CPU processes; no training jobs.
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(audit_task, ['SnAr', 'HPLC']))
    for name, expected in protected.items():
        assert filehash(ROOT/name) == expected, name
    gaps, segments, metrics, trust_rows, old_refreshes = [], [], [], [], []
    for task in ['SnAr', 'HPLC']:
        states = {'R1': read(R1/'runs'/(task+'_New_Agent_R1')/'state.json')}
        states.update({method: read(OLD/'runs'/(task+'_'+method)/'state.json') for method in ['New_Agent','Pure_BO']})
        for event in states['R1']['decisions']:
            gate = event['trust']; before = gate.get('entry_before') or {}; after = gate.get('entry_after') or {}
            pending = before.get('pending_evidence'); feedback = event.get('trust_feedback') or {}
            cleared = bool(pending and not after.get('pending_evidence') and gate['b4_fallback'] and after.get('blocked'))
            if cleared:
                assert pending['positive_evidence'] and pending['query'] < event['iteration']
            trust_rows.append({'task':task, 'query_round':event['iteration'], 'geometry_key':event['bank_key'],
                               'B4_conflict_fallback':gate['b4_fallback'], 'blocked':gate['trust_blocked'],
                               'pending_positive_evidence_before':bool(pending and pending['positive_evidence']),
                               'pending_evidence_query':pending['query'] if pending else None,
                               'positive_evidence_cleared_by_reblock':cleared,
                               'new_shadow_positive':feedback.get('positive_evidence'),
                               'shadow_squared_error_gain_y_squared':feedback.get('squared_error_gain_y_squared')})
        old_refreshes.extend({'task':task,'query_round':c['iteration'],'reasons':c['reasons'],
                             'knowledge_after':c['answer']['plan']['knowledge'],
                             'summary':c['answer']['plan']['summary']} for c in states['New_Agent']['calls'])
        for method, state in states.items():
            y = np.array([h['raw_value'] for h in state['history']])
            best = np.maximum.accumulate(y)
            metrics.append({'task': task, 'method': method, 'SearchAUC': float(best[4:].mean()),
                            'final_best': float(best[-1]), 'first_best_query': int(np.flatnonzero(y == best[-1])[0])-3})
        for method in ['New_Agent','Pure_BO']:
            a = states['R1']['decisions']; b = states[method]['decisions']
            scale = 13000. if task == 'SnAr' else 10000.
            values = []
            for t, (x, y) in enumerate(zip(a, b), 1):
                gap = (x['incumbent_after']-y['incumbent_after'])*scale
                values.append(gap/16)
                gaps.append({'task': task, 'query_round': t, 'comparator': method,
                             'R1_selected': x['selected_id'], 'comparator_selected': y['selected_id'],
                             'R1_objective': x['raw_value'], 'comparator_objective': y['raw_value'],
                             'R1_cumulative_best': x['incumbent_after']*scale,
                             'comparator_cumulative_best': y['incumbent_after']*scale,
                             'incumbent_gap': gap, 'SearchAUC_gap_contribution': gap/16})
            for start in [0,4,8,12]:
                segments.append({'task': task, 'comparator': method, 'first_query': start+1,
                                 'last_query': start+4, 'SearchAUC_gap_contribution': sum(values[start:start+4])})
            pair = {m['method']:m for m in metrics if m['task']==task}
            assert abs(sum(values)-(pair['R1']['SearchAUC']-pair[method]['SearchAUC'])) < 1e-8
    for field in ['stages','module_effects','KF_updates','rankings']:
        csv_write(OUT/(field+'.csv'), [r for result in results for r in result[field]])
    csv_write(OUT/'trajectory_gaps.csv', gaps)
    csv_write(OUT/'SearchAUC_gap_segments.csv', segments)
    csv_write(OUT/'trust_evidence_continuity.csv', trust_rows)
    write(OUT/'old_new_agent_refreshes.json', old_refreshes)
    write(OUT/'refresh_analysis.json', [r for result in results for r in result['refreshes']])
    write(OUT/'interaction_control_coverage.json', [r for result in results for r in result['interactions']])
    write(OUT/'findings.json', {'metrics': metrics, 'gap_segments': segments,
          'refresh_checks': {result['task']:result['refresh_checks'] for result in results},
          'interaction_checks': {result['task']:result['interaction_checks'] for result in results},
          'interpretation': 'Frozen-prefix one-step effects and accounting decomposition only. No replacement trajectories; no causal module-level AUC attribution.'})
    write(OUT/'receipt.json', {'status':'PASS', 'frozen_R1_rounds_checked':32,
          'new_trajectories':0, 'LLM_calls':0, 'GP_fits':0, 'selected_outcome_calls':0,
          'protected_files_unchanged':len(protected),
          'max_array_replay_error':max(r['max_replay_error'] for r in results),
          'module_decomposition_exact':True, 'SearchAUC_gap_reconciliation_exact':True,
          'script_sha256':filehash(__file__), 'operator_compiles_in_loop':0})
    inventory = {p.relative_to(ROOT).as_posix():filehash(p) for p in OUT.rglob('*') if p.is_file() and p.name!='inventory.json'}
    write(OUT/'inventory.json', inventory)
    print(json.dumps({'receipt':read(OUT/'receipt.json'),'metrics':metrics,'gap_segments':segments},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
