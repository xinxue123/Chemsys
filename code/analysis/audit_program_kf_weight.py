"""Audit prompts, independent scores, commitments and observed feedback only."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from math import erfc,sqrt,isclose
import argparse,csv,json,sys

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/analysis'),str(ROOT/'code/tools')]
from program_kf_weight import study,policy,prompts,VARIANT,CELLS,ARMS
from cold_start_v13 import control,policy as note_policy
from reasoning_memory_v13 import calibration
from knowledge_rank_v9 import engine
from v13_origin_ablation import queue
from audit_v13_origin_ablation import normalize_public

OUT=ROOT/'experiments/reports'/VARIANT


def close(a,b):
    if not isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10):raise ValueError((a,b))


def csv_write(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def private_paths(value,path=''):
    forbidden=prompts.PRIVATE_KEYS|{'ranker_KF','arm','combined_scores','base_logei','ranking'}
    if isinstance(value,dict):
        return [path+'/'+k for k in value if k in forbidden]+[
            x for k,v in value.items() for x in private_paths(v,path+'/'+k)]
    if isinstance(value,(list,tuple)):
        return [x for i,v in enumerate(value) for x in private_paths(v,path+'/'+str(i))]
    return []


def check_call(response,system,public,cell,counts):
    req=ROOT/response['request_relpath'];meta=queue.read(req);ctx=queue.read(req.parent/'context.json')
    assert meta['request_sha256']==control.digest({k:v for k,v in meta.items() if k!='request_sha256'})
    assert ctx['system']==system and normalize_public(ctx['public'])==normalize_public(public)
    assert not private_paths(normalize_public(ctx['public']))
    assert meta['context_sha256']==control.digest(ctx)
    entry_path=ROOT/meta['response_map_entry'];entry=queue.read(entry_path)
    assert entry['cell']==cell and entry['context_sha256']==control.digest(ctx)
    assert queue.file_sha(entry_path)==meta['entry_sha256']
    raw=ROOT/entry['raw_path'];assert queue.file_sha(raw)==entry['raw_sha256']
    assert queue.read(raw)==response['raw']
    rpath=ROOT/entry['request_path'];r=queue.read(rpath);assignment=queue.read(rpath.parent/'assignment.json')
    receipt=queue.read(ROOT/entry['receipt_path'])
    assert receipt['request_sha256']==r['request_sha256'] and receipt['raw_sha256']==queue.file_sha(raw)
    assert receipt['assignment_sha256']==queue.file_sha(rpath.parent/'assignment.json')
    assert queue.file_sha(ROOT/r['input'])==r['input_sha256']
    actual=assignment['spawn_result'].get('task_name',assignment['spawn_result'].get('agent_id'))
    assert actual==assignment['agent_id']=='/root/program_dispatch/'+assignment['spawn_request']['task_name']
    assert (assignment['spawn_request']['model'],assignment['spawn_request']['reasoning_effort'],assignment['spawn_request']['fork_turns'])==('gpt-6.1-terra','medium','none')
    packet=queue.read(ROOT/r['input']);assert packet['system']==ctx['system']
    actual_public=packet['public'] if r['attempt']==1 else packet['public']['original']
    assert actual_public==ctx['public']
    if entry.get('schema_exception_path'):
        ep=ROOT/entry['schema_exception_path'];exception=queue.read(ep)
        assert r['schema_exception_sha256']==queue.file_sha(ep)
        initial=queue.read(rpath.parent.parent/'a1/response.raw.json')
        accepted=queue.read(raw)
        assert accepted['assessments']==[]
        assert all(accepted[k]==v for k,v in initial.items() if k!='assessments')
        assert accepted['decision_card']['predictions']==exception['candidate_predictions_locked']
        counts['metadata_only_locked_forecast_exceptions']+=1
    counts['logical_calls']+=1;counts['attempts_in_accepted_calls']+=entry['attempts']
    return meta['response_map_entry']


def independent_rank(pool,ctx,packet,history,weight):
    r=packet['ranking'];assert r['GP_sha256']==engine.gp_hash(ctx)
    close(r['weight'],weight)
    ordered,_=engine.candidates(pool,ctx,len(history)+1);ids=[pool.ids[int(i)] for i in ordered]
    assert r['ordered_ids']==ids
    if not history:
        assert r['GP_rank_scores'] is None and r['selected_id']==packet['proposed_id'];return
    b=(rankdata(ctx['base_logei'][ordered],method='average')-1)/4
    p=np.array([packet['probabilities'][cid] for cid in ids])
    l=(rankdata(p,method='average')-1)/4
    no_info=bool(np.all(p==p[0]));score=b if no_info else (1-weight)*b+weight*l
    np.testing.assert_array_equal(b,r['GP_rank_scores']);np.testing.assert_array_equal(l,r['LLM_rank_scores'])
    np.testing.assert_allclose(score,r['combined_scores'],rtol=0,atol=1e-14)
    best=np.flatnonzero(score==score.max())
    selected=int(ordered[best[np.argmin(ctx['tie_priority'][ordered[best]])]])
    if no_info:selected=int(ordered[0])
    assert r['selected_id']==pool.ids[selected]


def prefix_audit(p,counts):
    summaries=[];scores=[];entries=set()
    for task,seed in CELLS:
        cell=f'{study.SHORT[task]}{seed}';folder=study.base()/'prefix'/cell
        path=study.reference(p,task,seed);old=study.read(path);pool=study.frozen.pool_for(ROOT,task)
        kf=calibration.initial();different=[];flat=[];uninformative=[]
        for q in range(1,21):
            c=study.read(folder/'rounds'/f'{q:02}_commit.json');assert c['commit_sha256']==control.digest({k:v for k,v in c.items() if k!='commit_sha256'})
            original=study.read(path.parent/'rounds'/f'{q:02}_commit.json');ctx=study.saved_context(path.parent,q)
            assert c['source_commit_sha256']==queue.file_sha(ROOT/c['source_commit_path'])
            assert c['GP_sha256']==original['GP_sha256']==engine.gp_hash(ctx)
            assert c['display_ids']==original['display_ids']
            state=study.held_state(old,q);state['ranker_KF']=kf;assert c['ranker_before']==kf
            notes=original['edited_notes'];ids=c['display_ids']
            entries.add(check_call(c['reasoning'],prompts.REASONING_SYSTEM,
                prompts.reasoning_public(ROOT,pool,state,notes,ids,q),cell,counts))
            for arm in ARMS:
                plan=policy.plan(ROOT,pool,notes,state,ctx,ids,q,c['reasoning']['raw'],arm)
                assert plan==c[arm]
                independent_rank(pool,ctx,c[arm],state['history'],.4 if arm=='fixed' else kf['reliability'])
            f,d=c['fixed'],c['dynamic'];different.append(q) if f['selected_id']!=d['selected_id'] else None
            if f['ranking']['GP_all_tied']:flat.append(q)
            if f['ranking']['LLM_all_tied']:uninformative.append(q)
            for i,cid in enumerate(f['ranking']['ordered_ids']):
                j=pool.ix[cid]
                scores.append(dict(task=task,seed=seed,query=q,candidate_id=cid,
                    p_LLM=f['probabilities'][cid],GP_Borda=f['ranking']['GP_rank_scores'][i] if q>1 else None,
                    base_qLogEI=float(ctx['base_logei'][j]) if q>1 else None,
                    GP_mean_raw=float(ctx['stats']['mu_y'][j])*pool.scale if q>1 else None,
                    GP_predictive_sd_raw=float(ctx['stats']['sd_predictive_y'][j])*pool.scale if q>1 else None,
                    LLM_Borda=f['ranking']['LLM_rank_scores'][i] if q>1 else None,
                    fixed_weight=.4,dynamic_weight=kf['reliability'],
                    fixed_score=f['ranking']['combined_scores'][i] if q>1 else None,
                    dynamic_score=d['ranking']['combined_scores'][i] if q>1 else None,
                    fixed_selected=cid==f['selected_id'],dynamic_selected=cid==d['selected_id'],
                    GP_all_tied=f['ranking']['GP_all_tied'],LLM_all_tied=f['ranking']['LLM_all_tied']))
            feedback=study.read(folder/'rounds'/f'{q:02}_feedback.json')
            obs=old['history'][q-1];assert feedback['observed_archived_point']==obs
            if q>1:
                assert (obs['y']>max(h['y'] for h in state['history']))==(obs['raw_value']>max(h['raw_value'] for h in state['history']))
                cid=obs['candidate_id'];pl=f['probabilities'][cid]
                pg=engine.gp_probability(ctx,pool.ix[cid],max(h['y'] for h in state['history']))
                after,event=calibration.update(kf,pl,pg,obs['y']>max(h['y'] for h in state['history']))
                assert feedback['calibration']==event;close(event['Brier_gain'],(float(obs['y']>max(h['y'] for h in state['history']))-pg)**2-(float(obs['y']>max(h['y'] for h in state['history']))-pl)**2)
                kf=after
            assert feedback['after']==kf
        assert study.read(folder/'state.json')==dict(query=20,ranker_KF=kf,status='COMPLETE')
        summaries.append(dict(task=task,seed=seed,historical_prefixes=20,selection_disagreement_queries=different,
            GP_all_tied_queries=flat,LLM_uniform_queries=uninformative,private_KF_updates=kf['updates'],new_queries=0))
    csv_write('prefix_scores.csv',scores)
    value=dict(status='PASS',cells=2,historical_prefixes=40,new_queries=0,new_LLM_calls_in_audit=0,
        all_five_predictions_shared_between_weights=True,independent_rank_formula_verified=True,
        KF_and_arm_hidden_from_all_prompts=True,summaries=summaries)
    (OUT/'prefix_audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return value,entries


def truth_tables(p):
    result={}
    with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['task'] not in ('perera','hplc'):continue
            pool=queue.read(ROOT/r['pool_path']);path=ROOT/r['oracle_path']
            assert queue.file_sha(path)==p['inputs_sha256'][study.rel(path)]
            with np.load(path,allow_pickle=False) as z:
                y=z['raw_values'] if 'raw_values' in z else z['values']*100
                result[r['task']]=dict(zip(pool['ids'],map(float,y)))
    return result


def full_audit(p,counts):
    truth=truth_tables(p);entries=set();total=0;states={};scores=[]
    for task,seed in CELLS:
        pool=study.frozen.pool_for(ROOT,task);cell=f'{study.SHORT[task]}{seed}'
        for arm in ARMS:
            folder=study.base()/'runs'/f'trace_{cell}_{arm}';final=study.read(folder/'state.json')
            assert final['status']=='COMPLETE' and len(final['history'])==20
            initial=study.read(folder/'initial_commit.json')
            assert initial['initial']==0 and initial['initial_ids']==[] and initial['queries']==20
            assert final['protocol_sha256']==p['protocol_sha256']
            assert initial['commit_sha256']==control.digest({k:v for k,v in initial.items() if k!='commit_sha256'})
            state=deepcopy(final);state.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[],ranker_KF=calibration.initial())
            for q,obs in enumerate(final['history'],1):
                c=study.read(folder/'rounds'/f'{q:02}_commit.json');ctx=study.saved_context(folder,q)
                close(obs['raw_value'],truth[study.SHORT[task]][obs['candidate_id']])
                assert c['prestate_sha256']==control.digest(state) and c['GP_sha256']==engine.gp_hash(ctx)
                np.testing.assert_array_equal(ctx['available'],~np.isin(pool.ids,[h['candidate_id'] for h in state['history']]))
                np.testing.assert_array_equal(ctx['tie_priority'],study.PRIORITIES(pool,q,seed if q==1 else 0))
                ordered,display=engine.candidates(pool,ctx,q);ids=[pool.ids[int(j)] for j in display]
                assert c['display_ids']==ids and c['ordered_ids']==[pool.ids[int(j)] for j in ordered]
                assert c['refresh']==note_policy.edit_event(pool,state,ordered,q)
                notes=state['notes']
                if c['memory']:
                    entries.add(check_call(c['memory'],prompts.MEMORY_SYSTEM,
                        prompts.memory_public(pool,state,ids,q,c['refresh']),cell,counts))
                    m=note_policy.apply_edits(c['memory']['raw'],pool,state,q)
                    assert m==c['memory']['plan'];notes=m['notes']
                assert notes==c['edited_notes']
                entries.add(check_call(c['reasoning'],prompts.REASONING_SYSTEM,
                    prompts.reasoning_public(ROOT,pool,state,notes,ids,q),cell,counts))
                packet=policy.plan(ROOT,pool,notes,state,ctx,ids,q,c['reasoning']['raw'],arm)
                assert packet==c['decision']
                independent_rank(pool,ctx,packet,state['history'],.4 if arm=='fixed' else state['ranker_KF']['reliability'])
                before=deepcopy(state['ranker_KF']);state=policy.feedback(state,c,obs)
                assert state['ranker_KF']['updates']==q-1
                if q>1:
                    ref=max(h['y'] for h in state['history'][:-1]);j=pool.ix[obs['candidate_id']]
                    assert (obs['y']>ref)==(obs['raw_value']>max(h['raw_value'] for h in state['history'][:-1]))
                    mu=float(ctx['stats']['mu_y'][j]);sd=float(ctx['stats']['sd_predictive_y'][j])
                    expected=.5*erfc((ref-mu)/(sqrt(2)*sd)) if sd>0 else float(mu>ref)
                    close(expected,packet['actual_p_GP'])
                    z=(float(obs['y']>ref)-expected)**2-(float(obs['y']>ref)-packet['actual_p_LLM'])**2
                    close(z,state['decisions'][-1]['ranker_feedback']['Brier_gain'])
                for i,cid in enumerate(packet['ranking']['ordered_ids']):
                    r=packet['ranking'];shadow=packet['fixed_weight_shadow']
                    j=pool.ix[cid]
                    scores.append(dict(task=task,seed=seed,arm=arm,query=q,candidate_id=cid,
                        p_LLM=packet['probabilities'][cid],weight=r['weight'],
                        base_qLogEI=float(ctx['base_logei'][j]) if q>1 else None,
                        GP_mean_raw=float(ctx['stats']['mu_y'][j])*pool.scale if q>1 else None,
                        GP_predictive_sd_raw=float(ctx['stats']['sd_predictive_y'][j])*pool.scale if q>1 else None,
                        GP_Borda=r['GP_rank_scores'][i] if q>1 else None,
                        LLM_Borda=r['LLM_rank_scores'][i] if q>1 else None,
                        program_score=r['combined_scores'][i] if q>1 else None,
                        fixed_weight_shadow_score=shadow['combined_scores'][i] if q>1 else None,
                        selected=cid==packet['selected_id'],fixed_weight_shadow_selected=cid==shadow['selected_id'],
                        observed_objective=obs['raw_value'] if cid==packet['selected_id'] else None))
                total+=1
            assert state==final;assert study.read(folder/'runtime_counters.json')['operator_compiles_in_loop']==0
            states[task,seed,arm]=state
    matched=[]
    for task,seed in CELLS:
        f=states[task,seed,'fixed'];d=states[task,seed,'dynamic'];cell=f'{study.SHORT[task]}{seed}'
        first=next((q for q,(a,b) in enumerate(zip(f['history'],d['history']),1) if a!=b),None)
        for q in range(1,(first or 20)+1):
            a=study.read(study.base()/'runs'/f'trace_{cell}_fixed/rounds/{q:02}_commit.json')
            b=study.read(study.base()/'runs'/f'trace_{cell}_dynamic/rounds/{q:02}_commit.json')
            assert a['GP_sha256']==b['GP_sha256'] and a['display_ids']==b['display_ids']
            assert a['reasoning']['context_sha256']==b['reasoning']['context_sha256']
            assert a['reasoning']['raw']==b['reasoning']['raw']
            assert a['decision']['probabilities']==b['decision']['probabilities']
        assert f['history'][:2]==d['history'][:2]
        matched.append(dict(task=task,seed=seed,first_physical_divergence=first,
            identical_forecasts_and_GP_through_first_divergence=True))
    csv_write('trajectory_scores.csv',scores)
    return dict(status='PASS_COMPLETE',trajectories=4,observed_queries=total,
        all_selected_values_match_frozen_oracle=True,only_executed_points_receive_feedback=True,
        full_trajectories_start_from_prior_not_offline_KF=True,
        KF_hidden=True,operator_compiles_in_loop=0,matched_prefixes=matched),entries


def main():
    a=argparse.ArgumentParser();a.add_argument('--prefix-only',action='store_true');args=a.parse_args()
    p=study.check(True);OUT.mkdir(parents=True,exist_ok=True);counts=Counter()
    import program_kf_operations
    program_kf_operations.check()
    alias=study.base()/'alias_recovery_registration.json'
    if alias.exists():
        reg=study.read(alias)
        assert reg['helper_sha256']==queue.file_sha(ROOT/'code/tools/program_kf_alias_recovery.py')
        assert reg['scientific_protocol_sha256']==p['protocol_sha256']
    prefix,entries=prefix_audit(p,counts)
    if args.prefix_only:result=prefix
    else:
        result,more=full_audit(p,counts);entries|=more
        result.update(prefix_status=prefix['status'],unique_accepted_response_inputs=len(entries),
            call_counts=dict(counts),no_new_LLM_or_oracle_calls_in_audit=True,
            raw_bytes_and_actual_spawn_receipts_verified=True,old_results_unchanged=True,
            protocol_sha256=p['protocol_sha256'])
        (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    study.check(True);print(json.dumps(result))


if __name__=='__main__':main()
