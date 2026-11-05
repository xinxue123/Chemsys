"""Recalculate the registered confirmation from immutable decisions and labels."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/experiment_tools'))
import run_program_kf_confirmation as run
from audit_program_kf_weight import independent_rank, private_paths, normalize_public
from cold_start_v13 import control
from reasoning_memory_v13 import calibration
from knowledge_rank_v9 import engine

OUT = ROOT/'experiments/reports'/run.VARIANT


def assert_hash(packet, field):
    assert packet[field] == control.digest({k:v for k,v in packet.items() if k != field})


def output(name, value):
    if isinstance(value, list):
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(value[0]))
            w.writeheader()
            w.writerows(value)
    else:
        (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def audit_response(response, cells):
    req = ROOT/response['request_relpath']
    meta, ctx = run.read(req), run.read(req.parent/'context.json')
    assert_hash(meta, 'request_sha256')
    assert meta['context_sha256'] == control.digest(ctx)
    assert not private_paths(normalize_public(ctx['public']))
    entry_path = ROOT/meta['response_map_entry']
    assert run.sha(entry_path) == meta['entry_sha256']
    entry = run.read(entry_path)
    assert entry['context_sha256'] == meta['context_sha256']
    raw = ROOT/entry['raw_path']
    assert run.sha(raw) == entry['raw_sha256'] == meta['raw_sha256']
    assert run.read(raw) == response['raw']
    request_path = ROOT/entry['request_path']
    request = run.read(request_path)
    assignment_path = request_path.parent/'assignment.json'
    assignment = run.read(assignment_path)
    receipt = run.read(ROOT/entry['receipt_path'])
    assert_hash(request, 'request_sha256')
    assert run.sha(ROOT/request['input']) == request['input_sha256']
    assert receipt['request_sha256'] == assignment['request_sha256'] == request['request_sha256']
    assert receipt['assignment_sha256'] == run.sha(assignment_path)
    assert receipt['raw_sha256'] == entry['raw_sha256']
    inv = assignment['spawn_request']
    assert receipt['agent_id'] == assignment['agent_id'] == run.PARENT+'/'+inv['task_name']
    assert (inv['model'], inv['reasoning_effort'], inv['fork_turns']) == (run.MODEL, run.EFFORT, 'none')
    cells[meta['cell']].add(meta['response_map_entry'])


def truth(task):
    pool = run.pool_for(task)
    with np.load(pool.source_path.with_name('oracle.npz'), allow_pickle=False) as z:
        raw = z['raw_values'] if 'raw_values' in z else z['values']*pool.scale
        return pool, dict(zip(pool.ids, map(float, raw)))


def prefix_audit(p, entries):
    lock = run.read(run.BASE/'prefix_lock.json')
    assert lock['protocol'] == p['protocol_sha256']
    assert len(lock['decisions_sha256']) == 27
    assert all(run.sha(ROOT/path) == h for path,h in lock['decisions_sha256'].items())
    evals = run.read(run.BASE/'prefix_complete.json')
    assert evals['prefixes'] == len(evals['rows']) == 27
    rows = {(x['cell'],x['query']):x for x in evals['rows']}
    summary, candidates = {}, []
    for task, short, seed in run.CELLS:
        cell = f'{short}{seed}'
        pool, oracle = truth(task)
        path = ROOT/p['references'][cell+'_v13']
        old = run.read(path)
        kf = calibration.initial()
        mine = []
        for q in p['prefix_rounds']:
            r = rows[cell,q]
            folder = run.BASE/'prefix'/cell
            c = run.read(folder/f'{q:02}_commit.json')
            assert_hash(c, 'commit_sha256')
            history = old['history'][:q-1]
            ctx = run.previous.saved_context(path.parent,q)
            assert c['GP_sha256'] == engine.gp_hash(ctx)
            audit_response(c['response'], entries)
            f,d = c['fixed'],c['dynamic']
            for weight, plan in ((.4,f),(kf['reliability'],d)):
                assert_hash(plan, 'commit_sha256')
                assert plan['ranker_before'] == kf
                assert plan['history_sha256'] == control.digest(history) == c['history_sha256']
                independent_rank(pool,ctx,plan,history,weight)
                assert plan['probabilities'] == run.policy.predictions(c['response']['raw'],pool,
                    plan['ranking']['ordered_ids'],history)
            assert f['selected_id'] == r['fixed_id'] and d['selected_id'] == r['dynamic_id']
            assert r['gp_all_tied'] == f['ranking']['GP_all_tied']
            assert r['llm_all_tied'] == f['ranking']['LLM_all_tied']
            for i,cid in enumerate(f['ranking']['ordered_ids']):
                j = pool.ix[cid]
                candidates.append(dict(cell=cell,query=q,candidate_id=cid,p_LLM=f['probabilities'][cid],
                    qLogEI=ctx['base_logei'][j], GP_mean_raw=ctx['stats']['mu_y'][j]*pool.scale,
                    GP_observation_SD_raw=ctx['stats']['sd_predictive_y'][j]*pool.scale,
                    fixed_score=f['ranking']['combined_scores'][i],dynamic_score=d['ranking']['combined_scores'][i],
                    fixed_selected=cid==r['fixed_id'],dynamic_selected=cid==r['dynamic_id']))
            inc = max(x['raw_value'] for x in history)
            assert r['incumbent'] == inc
            if f['selected_id'] != d['selected_id']:
                fy,dy = oracle[f['selected_id']],oracle[d['selected_id']]
                assert r['switched'] and r['offline_lookups']==2
                assert (r['fixed_outcome'],r['dynamic_outcome'])==(fy,dy)
                assert r['delta_improvement']==max(0.,dy-inc)-max(0.,fy-inc)
            else:
                assert not r['switched'] and r['offline_lookups']==0 and r['delta_improvement']==0
            obs = old['history'][q-1]
            pg = engine.gp_probability(ctx,pool.ix[obs['candidate_id']],inc/pool.scale)
            kf, event = calibration.update(kf,f['probabilities'][obs['candidate_id']],pg,obs['raw_value']>inc)
            saved = run.read(folder/f'{q:02}_archived_feedback.json')
            assert saved == dict(archived_observation=obs,feedback=event)
            mine.append(r)
        summary[cell] = dict(prefixes=len(mine), all_GP_tied=sum(x['gp_all_tied'] for x in mine),
            all_LLM_tied=sum(x['llm_all_tied'] for x in mine),
            mathematical_opportunities=sum(x['envelope_switch_possible'] for x in mine),
            switched=[x['query'] for x in mine if x['switched']],
            positive=[x['query'] for x in mine if x['delta_improvement']>0],
            negative=[x['query'] for x in mine if x['delta_improvement']<0],
            neutral=[x['query'] for x in mine if x['switched'] and x['delta_improvement']==0],
            net_immediate_uplift=sum(x['delta_improvement'] for x in mine))
    output('prefix_pairs.csv', evals['rows'])
    output('prefix_candidates.csv', candidates)
    return summary


def trace_audit(p, entries):
    metrics, rounds, effects, divergences = [], [], [], []
    for task, short, seed in run.CELLS:
        cell = f'{short}{seed}'
        pool, oracle = truth(task)
        threshold = .95*max(oracle.values())
        state = {}
        commits = {}
        for arm in ('bo','reasoning_bo','v13','fixed','dynamic'):
            new = arm in ('fixed','dynamic')
            folder = (run.BASE/'runs'/f'traces_{cell}_{arm}' if new
                      else (ROOT/p['references'][cell+'_'+arm]).parent)
            s = state[arm] = run.read(folder/'state.json')
            h = s['history']
            assert s['status']=='COMPLETE' and len(h)==20 and len({o['candidate_id'] for o in h})==20
            assert all(o['raw_value']==oracle[o['candidate_id']] for o in h)
            y = [o['raw_value'] for o in h]
            best = np.maximum.accumulate(y)
            metrics.append(dict(cell=cell,arm=arm,source='new' if new else 'frozen',
                queries=len(y),SearchAUC=float(best.mean()),final_best=float(best[-1]),
                first_final_best=y.index(max(y))+1,target_95=threshold,
                first_target=next((i for i,v in enumerate(y,1) if v>=threshold),None)))
            commits[arm] = []
            if new:
                counters = run.read(folder/'runtime_counters.json')
                assert counters['operator_compiles_in_loop']==0
                initial = run.read(folder/'initial_commit.json')
                assert_hash(initial,'commit_sha256')
                replay = run.previous.frozen.initial_state(task,arm,p,initial)
                replay.update(seed=seed,starting_seed=seed,numerical_seed=0,ranker_KF=calibration.initial())
            for q,o in enumerate(h,1):
                row = dict(cell=cell,arm=arm,query=q,candidate_id=o['candidate_id'],
                           objective=y[q-1],best=float(best[q-1]),weight=None,p_LLM=None,p_GP=None)
                if new:
                    c = run.read(folder/'rounds'/f'{q:02}_commit.json')
                    commits[arm].append(c)
                    audit_response(c['reasoning'],entries)
                    if c['memory']:
                        audit_response(c['memory'],entries)
                    ctx = run.previous.saved_context(folder,q)
                    assert c['GP_sha256']==engine.gp_hash(ctx)
                    plan = c['decision']
                    independent_rank(pool,ctx,plan,h[:q-1],.4 if arm=='fixed' else replay['ranker_KF']['reliability'])
                    assert plan['probabilities']==run.policy.predictions(c['reasoning']['raw'],pool,c['display_ids'],h[:q-1])
                    replay = run.policy.feedback(replay,c,o)
                    row.update(weight=plan['ranking']['weight'],p_LLM=plan['actual_p_LLM'],p_GP=plan['actual_p_GP'])
                rounds.append(row)
            if new:
                assert replay==s
        m = {x['arm']:x for x in metrics if x['cell']==cell}
        f,d = m['fixed'],m['dynamic']
        fy,dy = ([x['candidate_id'] for x in state[a]['history']] for a in ('fixed','dynamic'))
        divergence = next((i for i,(a,b) in enumerate(zip(fy,dy),1) if a!=b),None)
        for fc,dc in zip(commits['fixed'][:divergence],commits['dynamic'][:divergence]):
            for field in ('GP_sha256','display_ids','ordered_ids'):
                assert fc[field]==dc[field]
            for field in ('context_sha256','raw'):
                assert fc['reasoning'][field]==dc['reasoning'][field]
            assert fc['decision']['ranker_before']==dc['decision']['ranker_before']
        if divergence is not None:
            fc, dc = (commits[a][divergence-1] for a in ('fixed','dynamic'))
            assert fc['decision']['probabilities']==dc['decision']['probabilities']
            assert divergence>1
            shared = state['fixed']['history'][:divergence-1]
            assert shared==state['dynamic']['history'][:divergence-1]
            incumbent = max(o['raw_value'] for o in shared)
            fy, dy = (state[a]['history'][divergence-1]['raw_value'] for a in ('fixed','dynamic'))
            divergences.append(dict(cell=cell,query=divergence,incumbent=incumbent,
                fixed_id=fc['selected_id'],dynamic_id=dc['selected_id'],fixed_value=fy,dynamic_value=dy,
                fixed_weight=fc['decision']['ranking']['weight'],
                dynamic_weight=dc['decision']['ranking']['weight'],
                immediate_improvement_delta=max(0.,dy-incumbent)-max(0.,fy-incumbent),
                shared_response_sha256=fc['reasoning']['response_receipt']['raw_sha256']))
        delta = d['SearchAUC']-f['SearchAUC']
        effects.append(dict(cell=cell,SearchAUC_delta=delta,SearchAUC_relative_delta=delta/f['SearchAUC'],
            final_best_delta=d['final_best']-f['final_best'],fixed_first_target=f['first_target'],
            dynamic_first_target=d['first_target'],first_divergence=divergence))
    output('metrics.csv',metrics)
    output('rounds.csv',rounds)
    output('paired_effects.csv',effects)
    if divergences:
        output('first_divergences.csv',divergences)
    return effects


if __name__=='__main__':
    p = run.verify()
    OUT.mkdir(parents=True,exist_ok=True)
    entries = defaultdict(set)
    result = dict(protocol_sha256=p['protocol_sha256'], prefix=prefix_audit(p,entries))
    if '--prefix-only' in sys.argv:
        result.update(status='PASS_PREFIX',complete_online_trajectories=0)
    else:
        result['effects'] = trace_audit(p,entries)
        wins = sum(r['SearchAUC_relative_delta']>=.01 for r in result['effects'])
        losses = sum(r['SearchAUC_relative_delta']<=-.01 for r in result['effects'])
        positive_cells = sum(bool(x['positive']) for x in result['prefix'].values())
        result.update(status='PASS_COMPLETE', complete_online_trajectories=6, online_queries=120,
            material_gain_cells=wins,material_loss_cells=losses,positive_switch_cells=positive_cells,
            mechanism_confirmed=positive_cells>=2 and losses==0,
            cross_task_performance_gate=positive_cells>=2 and losses==0 and wins>=2)
    result.update(accepted_unique_response_contexts=sum(map(len,entries.values())),
                  new_LLM_calls_in_audit=0,new_online_queries_in_audit=0)
    output('prefix_audit.json' if '--prefix-only' in sys.argv else 'audit.json',result)
    print(json.dumps(result,ensure_ascii=False))
