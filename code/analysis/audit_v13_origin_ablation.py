"""Replay every existing ablation commitment without fitting or querying."""
from copy import deepcopy
from pathlib import Path
from collections import Counter
from functools import lru_cache
import json
import re
import string
import sys
import csv

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_origin_ablation import study,queue
from cold_start_v13 import policy,prompts,control


def read(p):return queue.read(p)
def close(a,b):
    if not np.isclose(a,b,atol=1e-10,rtol=1e-10):raise ValueError((a,b))


@lru_cache(maxsize=1)
def official_pattern():
    path=ROOT/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json'
    template=read(path)['optimization_loop']['en'];pieces=[];names=[]
    for literal,name,spec,conversion in string.Formatter().parse(template):
        pieces.append(re.escape(literal))
        if name is not None:
            names.append(name);pieces.append('(.*?)')
    return re.compile(''.join(pieces),re.DOTALL),names


def normalize_public(public):
    result=deepcopy(public)
    if 'official_optimization_prompt' in result:
        pattern,names=official_pattern();match=pattern.fullmatch(result['official_optimization_prompt'])
        if not match:raise ValueError('Official template text changed')
        values=[]
        for name,value in zip(names,match.groups()):
            try:value=json.loads(value)
            except json.JSONDecodeError:pass
            values.append((name,value))
        result['official_optimization_prompt']=values
    return result


def context(folder,q):
    with np.load(folder/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as z:
        return dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
                    **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})


def check_call(result,expected,cell):
    req=ROOT/result['request_relpath'];meta=read(req)
    if meta['request_sha256']!=control.digest({k:v for k,v in meta.items() if k!='request_sha256'}):
        raise ValueError('Call receipt changed')
    ctx=read(req.parent/'context.json')
    # Commits are canonically serialized. Reconstructing their objects can
    # reorder keys inside JSON embedded in the official prompt string. Compare
    # those JSON fields structurally, while checking actual packet hashes below.
    if ctx['system']!=expected['system'] or normalize_public(ctx['public'])!=normalize_public(expected['public']) or meta['context_sha256']!=control.digest(ctx):
        raise ValueError('Chooser/editor received unexpected scientific input')
    entry=read(ROOT/meta['response_map_entry'])
    if entry['cell']!=cell or entry['context_sha256']!=control.digest(ctx):
        raise ValueError('Response leaked across starts or inputs')
    if queue.file_sha(ROOT/meta['response_map_entry'])!=meta['entry_sha256']:
        raise ValueError('Response-map entry changed')
    raw=ROOT/entry['raw_path']
    if queue.file_sha(raw)!=entry['raw_sha256'] or read(raw)!=result['raw']:
        raise ValueError('Raw response changed')
    if entry['mode']=='historical_frozen_response':
        origin=entry['provenance']
        for k in ('request','context','raw','commit'):
            if queue.file_sha(ROOT/origin['source_'+k])!=origin['source_'+k+'_sha256']:
                raise ValueError('Historical language evidence changed')
        if queue.file_sha(raw)!=origin['source_raw_sha256']:raise ValueError('Seeded bytes differ from original')
    else:
        rp=ROOT/entry['request_path'];r=read(rp);assignment=read(rp.parent/'assignment.json')
        receipt=read(ROOT/entry['receipt_path'])
        if receipt['request_sha256']!=r['request_sha256'] or receipt['raw_sha256']!=queue.file_sha(raw):
            raise ValueError('Fresh response receipt mismatch')
        if receipt['assignment_sha256']!=queue.file_sha(rp.parent/'assignment.json'):
            raise ValueError('Actual agent receipt changed')
        actual=assignment['spawn_result'].get('task_name',assignment['spawn_result'].get('agent_id'))
        suffix=assignment['spawn_request']['task_name']
        allowed={'/root/'+suffix}
        registration=study.base()/'operational_dispatch_registration.json'
        if registration.exists():
            reg=read(registration)
            if queue.file_sha(ROOT/reg['helper_path'])!=reg['helper_sha256'] or reg['scientific_protocol_sha256']!=read(study.base()/'protocol.json')['protocol_sha256']:
                raise ValueError('Operational dispatch registration changed')
            if assignment.get('operational_registration_sha256')==queue.file_sha(registration):
                allowed.add(reg['allowed_agent_parent']+'/'+suffix)
        if actual!=assignment['agent_id'] or actual not in allowed:
            raise ValueError('Actual model agent differs')
        if assignment['spawn_request']['model']!='gpt-6.1-terra' or assignment['spawn_request']['reasoning_effort']!='medium' or assignment['spawn_request']['fork_turns']!='none':
            raise ValueError('Requested experimental model changed')
    return entry['mode']


def main(partial=False):
    p=study.check(True);counts=Counter();q2={};outcomes=0
    truth={}
    with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            pool=read(ROOT/row['pool_path']);path=ROOT/row['oracle_path']
            if queue.file_sha(path)!=p['inputs_sha256'][queue.rel(ROOT,path)]:raise ValueError('Evaluation input changed')
            with np.load(path,allow_pickle=False) as z:
                values=z['raw_values'] if 'raw_values' in z else z['values']*100.
                truth[row['task']]=dict(zip(pool['ids'],map(float,values)))
    jobs=p['jobs']
    if partial:jobs=[j for j in jobs if (study.base()/'runs'/j['id']/'state.json').exists()
        and read(study.base()/'runs'/j['id']/'state.json')['status']=='COMPLETE']
    for job in jobs:
        folder=study.base()/'runs'/job['id'];final=read(folder/'state.json')
        if final['status']!='COMPLETE' or len(final['history'])!=20:raise ValueError('Incomplete job: '+job['id'])
        state=deepcopy(final);state.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
        pool=study.POOL(ROOT,job['task'])
        with study.redirected(job,p):
            queue.CURRENT=None  # numerical replay must not write diagnostic sidecars
            for q,h in enumerate(final['history'],1):
                close(h['raw_value'],truth[study.SHORT[job['task']]][h['candidate_id']])
                c=read(folder/'rounds'/f'{q:02}_commit.json');ctx=context(folder,q)
                if c['prestate_sha256']!=control.digest(state):raise ValueError('Pre-outcome state mismatch')
                if study.native.native.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('GP arrays changed')
                original,display=study.native.native.candidates(pool,ctx,q)
                if c['ordered_ids']!=[pool.ids[int(j)] for j in original] or c['display_ids']!=[pool.ids[int(j)] for j in display]:
                    raise ValueError('Native acquisition/menu changed')
                available=~np.isin(pool.ids,[x['candidate_id'] for x in state['history']])
                np.testing.assert_array_equal(available,ctx['available'])
                if c['selected_id'] not in c['display_ids']:raise ValueError('Off-menu candidate')
                if q==1:
                    if c['qLogEI_defined'] or not np.isnan(ctx['stats']['mu_y']).all():raise ValueError('Free bootstrap GP')
                else:
                    if not np.isfinite(ctx['base_logei'][available]).all():raise ValueError('Nonfinite acquisition')
                expected=policy.edit_event(pool,state,original,q)
                if c['refresh']!=expected:raise ValueError('Wrong refresh intervention')
                if c['memory']:
                    memory=dict(system=prompts.MEMORY_SYSTEM,public=prompts.memory_public(pool,state,c['display_ids'],q,expected))
                    counts[check_call(c['memory'],memory,job['cell'])]+=1
                    plan=policy.apply_edits(c['memory']['raw'],pool,state,q)
                    if plan!=c['memory']['plan'] or plan['notes']!=c['edited_notes']:raise ValueError('Editor plan changed')
                elif c['edited_notes']!=state['notes']:raise ValueError('Uncommitted knowledge edit')
                chooser=dict(system=prompts.REASONING_SYSTEM,
                    public=prompts.reasoning_public(ROOT,pool,state,c['edited_notes'],c['display_ids'],q))
                origin=check_call(c['reasoning'],chooser,job['cell']);counts[origin]+=1
                if job['arm']=='full' and origin!='historical_frozen_response':raise ValueError('Full replay used fresh language')
                plan=policy.validate_decision(ROOT,pool,c['edited_notes'],state,ctx,c['display_ids'],q,c['reasoning']['raw'])
                if plan!=c['decision']:raise ValueError('Native selection/forecast validation differs')
                if q==2:q2[job['cell'],job['arm']]=control.digest(c['edited_notes'])
                if job['arm']=='full':
                    sf=ROOT/study.original(job['task'],job['seed'])['state_path']
                    old=read(sf.parent/'rounds'/f'{q:02}_commit.json')
                    if c['GP_sha256']!=old['GP_sha256'] or c['selected_id']!=old['selected_id']:
                        raise ValueError('Full baseline numerical/choice replay differs')
                state=study.native.feedback(state,c,h);outcomes+=1
                for note in state['notes']:
                    if job['arm'] in ('fixed','neither'):
                        for k in ('mean','variance','probability_positive','reliability'):
                            close(note['calibration'][k],control.calibration.initial()[k])
                for e in state['decisions'][-1]['feedback']:
                    f=next(f for f in c['decision']['forecasts'] if f['commit_sha256']==e['commit_sha256'])
                    s=int(e['improved']);close(e['calibration']['Brier_gain'],(s-f['p_GP'])**2-(s-f['p_LLM'])**2)
        if state!=final:raise ValueError('Final native state replay differs: '+job['id'])
        if job['arm'] in ('no_refresh','neither') and final['edit_calls']!=[2]:raise ValueError('Unexpected refresh')
        if job['arm']=='no_notes' and (final['notes'] or final['edit_calls']):raise ValueError('Note ablation failed')
    for cell in {j['cell'] for j in jobs}:
        values=[q2[cell,a] for a in ('full','fixed','no_refresh','neither') if (cell,a) in q2]
        if len(set(values))!=1:raise ValueError('Initial knowledge differs between factorial arms')
    probes=0
    for job in p['probes']:
        folder=study.base()/'probes'/job['id'];c=read(folder/'commit.json');o=read(folder/'outcome.json')
        if c['selected_id']!=o['observation']['candidate_id'] or c['selected_id'] not in c['menu']:
            raise ValueError('Probe outcome mismatch')
        req=ROOT/c['response']['request_relpath'];ctx=read(req.parent/'context.json')
        sf=ROOT/study.original(job['task'],job['seed'])['state_path'];old=read(sf)
        expected_state=deepcopy(old)
        expected_state.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
        for q in range(1,job['query']):
            cp=read(sf.parent/'rounds'/f'{q:02}_commit.json')
            expected_state=study.native.feedback(expected_state,cp,old['history'][q-1])
        expected_state['notes']=[];pool=study.POOL(ROOT,job['task'])
        public=prompts.reasoning_public(ROOT,pool,expected_state,[],c['menu'],job['query']) if job['template']=='v13' else prompts.rbo_public(ROOT,pool,expected_state,c['menu'],job['query'])
        expected=dict(system=prompts.REASONING_SYSTEM if job['template']=='v13' else prompts.RBO_SYSTEM,public=public)
        if c['observed_prefix']!=expected_state['history'] or c['shared_Insight_history']!=expected_state['insights']:
            raise ValueError('Prompt probe prefix or Insight history changed')
        original=read(sf.parent/'rounds'/f"{job['query']:02}_commit.json")
        if c['menu']!=original['display_ids'] or c['GP_sha256']!=original['GP_sha256']:
            raise ValueError('Prompt probe numerical context changed')
        close(o['observation']['raw_value'],truth['doyle'][c['selected_id']])
        if check_call(c['response'],expected,job['cell'])!='fresh_subagent':raise ValueError('Probe reused original language')
        probes+=1
    study.check(True)
    receipt=dict(status='PASS_PARTIAL' if partial else 'PASS_COMPLETE',factorial_trajectories=len(jobs),
        full_replay=sum(j['arm']=='full' for j in jobs),controls=sum(j['arm']!='full' for j in jobs),
        replayed_committed_outcomes=outcomes,new_control_budgeted_queries=20*sum(j['arm']!='full' for j in jobs),targeted_one_step_probes=probes,
        call_origin_counts=dict(counts),all_initial_notes_matched=True,all_GP_menus_and_transitions_replayed=True,
        old_results_and_sources_unchanged=True,new_GP_fits_in_audit=0,new_LLM_calls_in_audit=0,
        actual_recorded_packet_hashes_verified=True,embedded_JSON_reconstruction='semantic key-order normalization only',
        response_pairing='same scientific input within a start shares response; altered inputs are not common-random-number pairs')
    destination=ROOT/'experiments/reports'/study.VARIANT;destination.mkdir(parents=True,exist_ok=True)
    (destination/('audit_partial.json' if partial else 'audit.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main('--partial' in sys.argv)
