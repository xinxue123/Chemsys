"""Replay continuations and audit public-response/model custody without refits.

No new experiment or LLM call is requested. Archived labels are read only for
offline checking that the saved observations match their frozen table entries.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter,defaultdict
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study,queue
from knowledge_rank_v9 import engine as native
from reasoning_memory_v12.llm import validate_insight

OUT=ROOT/'experiments/reports/v13_late_validation_20261103'


def read(p):return queue.read(p)
def write_json(name,x):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name,x):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)


def core(raw):
    keys={'comment','keywords','hypotheses'}
    if isinstance(raw,dict) and keys<=set(raw) and set(raw)-keys<={'knowledge_uses','assessments'} and all(raw[k]==[] for k in set(raw)-keys):
        return {k:raw[k] for k in raw if k in keys}
    return raw


def context_from(folder,q,c):
    with np.load(folder/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    return dict(stats={k:a[k] for k in ('mu_y','sd_y','sd_predictive_y')},
        base_logei=a['base_logei'],available=a['available'],tie_priority=a['tie_priority'])


def verify_request(result,state,ids,arm,agents):
    path=ROOT/result['request_relpath'];req=read(path)
    if req['request_sha256']!=queue.digest({k:v for k,v in req.items() if k!='request_sha256'}):raise ValueError('Request changed')
    af=path.parent;a=read(af/'assignment.json');receipt=read(af/'response_receipt.json')
    if a['request_sha256']!=req['request_sha256'] or receipt['raw_sha256']!=queue.file_sha(af/'response.raw.json'):
        raise ValueError('Raw response custody failed')
    invocation=read(af/'invocation.json')
    if a['spawn_request']!=invocation or a['agent_id']!=a['spawn_result'].get('task_name',a['spawn_result'].get('agent_id')):
        raise ValueError('Actual fresh agent assignment mismatch')
    if not a['agent_id'].endswith('/'+invocation['task_name']) or invocation['fork_turns']!='none' or invocation['model']!='gpt-6.1-terra' or invocation['reasoning_effort']!='medium':
        raise ValueError('Wrong agent/model/context')
    if a['agent_id'] in agents:raise ValueError('Decision agent reused across accepted requests')
    agents.add(a['agent_id'])
    if receipt['assignment_sha256']!=queue.file_sha(af/'assignment.json'):raise ValueError('Assignment receipt changed')
    if queue.file_sha(ROOT/req['input'])!=req['input_sha256']:raise ValueError('Input packet changed')
    packet=read(ROOT/req['input']);public=packet['public']
    if 'original' in public:public=public['original']
    if [x['candidate_id'] for x in public['own_observations']]!=[x['candidate_id'] for x in state['history']]:raise ValueError('Other/future observations exposed')
    for exposed,known in zip(public['own_observations'],state['history']):
        if exposed['objective']!=known['raw_value']:raise ValueError('Wrong public outcome')
    if {x['candidate_id'] for x in public['candidate_menu']}!=set(ids):raise ValueError('Wrong public menu')
    if read(af/'response.raw.json')!=result['raw']:raise ValueError('Parent modified a response')
    if result['context_sha256']!=queue.digest(dict(system=packet['system'],public=public)):
        raise ValueError('Public context does not match result commitment')


def check_observations(pool,history):
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        raw=z['raw_values'] if 'raw_values' in z else z['values']*pool.scale
    for h in history:
        if not math.isclose(float(raw[pool.ix[h['candidate_id']]]),h['raw_value'],rel_tol=1e-12,abs_tol=1e-8):
            raise ValueError('Saved outcome differs from frozen table')
        if not math.isclose(h['y'],h['raw_value']/pool.scale,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Wrong normalized observation')


def audit_job(job,p,agents):
    folder=study.base()/job['id']/'runs'/job['id'];final=read(folder/'state.json')
    if final['status']!='COMPLETE' or len(final['history'])!=20:raise ValueError('Incomplete job')
    if len({x['candidate_id'] for x in final['history']})!=20:raise ValueError('Repeated experiment')
    initial=read(folder/'initial_commit.json');pool=study.pool_for(job['task'])
    check_observations(pool,final['history'])
    arm='bo' if job['arm']=='bo' else 'reasoning_bo' if job['arm']=='rbo' else 'v13'
    if job['kind']=='prefix':
        state=read(study.base()/'prefixes'/f'{study.SHORT[job["task"]]}{job["seed"]}.json')
        state.update(arm=arm,protocol_sha256=p['protocol_sha256'],initial_commit_sha256=initial['commit_sha256'])
        if job['arm']=='coupled':
            for note in state['notes']:note['_coupled_filter']=study.coupled_initial(note)
        if state['history']!=final['history'][:10]:raise ValueError('Different common prefix')
    else:state=study.frozen.initial_state(job['task'],arm,p,initial)
    records=[]
    with study.redirected(job,p):
        for q in range(job['prefix_queries']+1,21):
            c=read(folder/'rounds'/f'{q:02}_commit.json')
            if c['commit_sha256']!=queue.digest({k:v for k,v in c.items() if k!='commit_sha256'}) or c['prestate_sha256']!=queue.digest(state):
                raise ValueError('Non-prospective or altered commitment')
            ctx=context_from(folder,q,c)
            if native.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('GP arrays changed')
            ordered,display=native.candidates(pool,ctx,q)
            expected=[pool.ids[int(j)] for j in ordered];ids=[pool.ids[int(j)] for j in display]
            if expected!=c['ordered_ids'] or ids!=c['display_ids'] or c['selected_id'] not in ids:raise ValueError('Candidate/tie/display mismatch')
            if job['arm']=='bo' and c['selected_id']!=expected[0]:raise ValueError('BO did not maximize qLogEI')
            for result in (c['memory'],c['reasoning']):
                if result is not None:verify_request(result,state,ids,job['arm'],agents)
            if c['memory']:
                if study.policy.apply_edits(c['memory']['raw'],pool,state,q)!=c['memory']['plan']:
                    raise ValueError('Editor output was not processed by the native validator')
                if c['memory']['plan']['notes']!=c['edited_notes']:raise ValueError('Committed Notes differ from editor plan')
            elif c['edited_notes']!=state['notes']:raise ValueError('Notes changed without editor')
            if c['reasoning']:
                raw=c['reasoning']['raw'];insight=core(raw) if job['arm']=='rbo' else raw['insight']
                if validate_insight(ROOT,insight,ids)!=c['selected_id']:raise ValueError('Official selection mismatch')
                if arm=='v13':
                    plan=study.policy.validate_decision(ROOT,pool,c['edited_notes'],state,ctx,ids,q,raw)
                    if plan!=c['decision'] or plan!=c['reasoning']['plan']:raise ValueError('Prospective decision validation changed')
            if job['arm']=='no_refresh' and c['memory'] is not None:raise ValueError('Refresh ablation called editor')
            before_best=max((h['raw_value'] for h in state['history']),default=None)
            state=study.NATIVE_FEEDBACK(state,c,final['history'][q-1])
            feedback=state['decisions'][-1]['feedback']
            for e in feedback:
                if job['arm']=='kf_fixed':
                    for k in ('mean','variance','reliability','probability_positive'):
                        if e['calibration']['before'][k]!=e['calibration']['after'][k]:raise ValueError('Frozen KF adapted')
                if job['arm']!='coupled' and e['accurate_failure_forecast'] and e['recommendation_after']=='allowed':raise ValueError('Failure restored positive authority')
            records.append(dict(job=job['id'],kind=job['kind'],task=study.SHORT[job['task']],prefix_seed=job['seed'],
                repetition=job['rep'],arm=job['arm'],query=q,selected_id=c['selected_id'],objective=final['history'][q-1]['raw_value'],
                incumbent_before='' if before_best is None else before_best,incumbent_after=max(h['raw_value'] for h in state['history']),
                forecast_events=len(feedback),editor_call=int(c['memory'] is not None),
                selected_is_GP_top=int(c['selected_id']==expected[0]),commit_sha256=c['commit_sha256']))
    if state!=final:raise ValueError('Final state differs from complete replay')
    return records


def main(partial=False):
    p=study.check(True);agents=set();metrics=[];rounds=[];missing=[];failures=[]
    resume_proof=None
    rp=study.base()/'runtime_resume_registration.json'
    if rp.exists():
        reg=read(rp)
        if reg['original_protocol_sha256']!=p['protocol_sha256'] or queue.file_sha(ROOT/'code/tools/resume_v13_late.py')!=reg['adapter_sha256']:
            raise ValueError('Operational resume registration/source changed')
        for name,value in reg['immutable_existing_files_sha256'].items():
            if queue.file_sha(ROOT/name)!=value:raise ValueError('Pre-reset evidence changed: '+name)
        if not partial:
            done=read(study.base()/'runtime_resume_completion.json')
            if done['status']!='COMPLETE' or done['errors'] or done['registration_sha256']!=queue.file_sha(rp):
                raise ValueError('Operational resume incomplete')
        resume_proof=dict(completed_states_unchanged=len(reg['completed_trajectories_preserved']),
            immutable_pre_reset_files_verified=len(reg['immutable_existing_files_sha256']),
            registration_sha256=queue.file_sha(rp),scientific_protocol_unchanged=True)
    with (ROOT/'paper/v13/data/datasets.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            for k in ('pool','oracle'):
                if queue.file_sha(ROOT/row[k+'_path'])!=row[k+'_sha256']:raise ValueError('Frozen benchmark table changed')
    for name,k in (('pool.json','pool_sha256'),('oracle.npz','oracle_sha256')):
        if queue.file_sha(study.base()/'holdout'/name)!=p['holdout'][k]:raise ValueError('Holdout table changed')
    for job in p['jobs']:
        sf=study.base()/job['id']/'runs'/job['id']/'state.json'
        if not sf.exists() or read(sf)['status']!='COMPLETE':missing.append(job['id']);continue
        try:rr=audit_job(job,p,agents)
        except Exception as e:failures.append(dict(job=job['id'],error=type(e).__name__+': '+str(e)));continue
        rounds.extend(rr);s=read(sf);ys=[h['raw_value'] for h in s['history']];M=[max(ys[:q]) for q in range(1,21)]
        tau=job['prefix_queries'];M0=max(ys[:tau]) if tau else 0.
        metrics.append(dict(job=job['id'],kind=job['kind'],task=study.SHORT[job['task']],prefix_seed=job['seed'],
            repetition=job['rep'],arm=job['arm'],prefix_queries=tau,new_queries=20-tau,
            prefix_best=M0,final_best=M[-1],SearchAUC20=statistics.mean(M),
            continuation_SearchAUC=statistics.mean(M[tau:]),continuation_improvement=M[-1]-M0,
            continuation_incremental_SearchAUC=statistics.mean(x-M0 for x in M[tau:]),
            first_final_best_query=next(q for q,b in enumerate(M,1) if b==M[-1]),
            forecast_events=sum(x['forecast_events'] for x in rr),editor_calls=sum(x['editor_call'] for x in rr),
            state_path=sf.resolve().relative_to(ROOT).as_posix(),state_sha256=queue.file_sha(sf)))
    status='PASS_COMPLETE' if not missing and not failures else 'PARTIAL' if not failures else 'AUDIT_FAILURE'
    result=dict(status=status,registered_trajectories=75,complete_verified=len(metrics),missing=missing,failures=failures,
        newly_selected_queries_verified=len(rounds),distinct_accepted_fresh_agents=len(agents),
        original_results_and_sources_unchanged=True,GP_refits_in_audit=0,new_LLM_calls_in_audit=0,
        new_experimental_queries_in_audit=0,archived_label_access='offline observation validation only',
        protocol_sha256=p['protocol_sha256'],runtime_resume_verification=resume_proof)
    name='partial' if partial else 'final'
    write_json(name+'_audit.json',result)
    if metrics:write_csv(name+'_metrics.csv',metrics);write_csv(name+'_rounds.csv',rounds)
    print(json.dumps(result))
    if failures or (missing and not partial):raise SystemExit(1)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args();main(args.partial)
