"""Replay the reduced, first-valid-Note factorial without any new decisions."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/analysis')]
import v13_targeted_ablation as study
from v13_origin_ablation import queue
from cold_start_v13 import policy,prompts,control
from audit_v13_origin_ablation import context,check_call,close


def main(partial=False):
    p=study.check(True);source=study.original.check();counts=Counter();first_notes={};paid=0
    truth={}
    with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            path=ROOT/row['oracle_path']
            if queue.file_sha(path)!=source['inputs_sha256'][queue.rel(ROOT,path)]:raise ValueError('Frozen evaluation data changed')
            ids=queue.read(ROOT/row['pool_path'])['ids']
            with np.load(path,allow_pickle=False) as z:
                y=z['raw_values'] if 'raw_values' in z else z['values']*100.
                truth[row['task']]=dict(zip(ids,map(float,y)))
    jobs=p['jobs']
    if partial:jobs=[j for j in jobs if (study.base()/'runs'/j['id']/'state.json').exists()
        and queue.read(study.base()/'runs'/j['id']/'state.json')['status']=='COMPLETE']
    for job in jobs:
        folder=study.base()/'runs'/job['id'];final=queue.read(folder/'state.json')
        if final['status']!='COMPLETE' or len(final['history'])!=20:raise ValueError('Incomplete cell: '+job['id'])
        state=deepcopy(final);state.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
        pool=study.original.POOL(ROOT,job['task']);first=None
        with study.redirected(job,p):
            queue.CURRENT=None
            for q,h in enumerate(final['history'],1):
                close(h['raw_value'],truth[study.original.SHORT[job['task']]][h['candidate_id']])
                c=queue.read(folder/'rounds'/f'{q:02}_commit.json');ctx=context(folder,q)
                if c['prestate_sha256']!=control.digest(state):raise ValueError('Pre-outcome state differs')
                if study.original.native.native.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('Saved GP arrays differ')
                order,display=study.original.native.native.candidates(pool,ctx,q)
                if c['ordered_ids']!=[pool.ids[int(i)] for i in order] or c['display_ids']!=[pool.ids[int(i)] for i in display]:
                    raise ValueError('Menu differs from native acquisition')
                np.testing.assert_array_equal(ctx['available'],~np.isin(pool.ids,[x['candidate_id'] for x in state['history']]))
                np.testing.assert_array_equal(ctx['tie_priority'],study.original.PRIORITIES(pool,q,job['seed'] if q==1 else 0))
                if c['selected_id'] not in c['display_ids']:raise ValueError('Off-menu point')
                if q==1 and (c['qLogEI_defined'] or not np.isnan(ctx['stats']['mu_y']).all()):raise ValueError('Free GP bootstrap')
                expected=policy.edit_event(pool,state,order,q)
                if c['refresh']!=expected:raise ValueError('First-valid-Note editing rule differs')
                if c['memory']:
                    public=prompts.memory_public(pool,state,c['display_ids'],q,expected)
                    counts[check_call(c['memory'],dict(system=prompts.MEMORY_SYSTEM,public=public),job['cell'])]+=1
                    plan=policy.apply_edits(c['memory']['raw'],pool,state,q)
                    if plan!=c['memory']['plan'] or plan['notes']!=c['edited_notes']:raise ValueError('Knowledge editor differs')
                elif c['edited_notes']!=state['notes']:raise ValueError('Uncommitted knowledge change')
                public=prompts.reasoning_public(ROOT,pool,state,c['edited_notes'],c['display_ids'],q)
                origin=check_call(c['reasoning'],dict(system=prompts.REASONING_SYSTEM,public=public),job['cell'])
                counts[origin]+=1
                if job['arm']=='full' and origin!='historical_frozen_response':raise ValueError('Full replay used fresh language')
                plan=policy.validate_decision(ROOT,pool,c['edited_notes'],state,ctx,c['display_ids'],q,c['reasoning']['raw'])
                if plan!=c['decision']:raise ValueError('Selected point or prospective forecast differs')
                if first is None and c['edited_notes']:
                    first=(q,control.digest(c['edited_notes']))
                if job['arm']=='full':
                    sf=ROOT/study.original.original(job['task'],job['seed'])['state_path']
                    old=queue.read(sf.parent/'rounds'/f'{q:02}_commit.json')
                    if c['selected_id']!=old['selected_id'] or c['GP_sha256']!=old['GP_sha256']:raise ValueError('Archived full path changed')
                state=study.original.native.feedback(state,c,h);paid+=job['arm']!='full'
                if job['arm'] in ('fixed','neither'):
                    for n in state['notes']:
                        for k in ('mean','variance','probability_positive','reliability'):
                            close(n['calibration'][k],control.calibration.initial()[k])
                for e in state['decisions'][-1]['feedback']:
                    f=next(f for f in c['decision']['forecasts'] if f['commit_sha256']==e['commit_sha256'])
                    s=int(e['improved']);close(e['calibration']['Brier_gain'],(s-f['p_GP'])**2-(s-f['p_LLM'])**2)
        if state!=final:raise ValueError('Final state differs: '+job['id'])
        first_notes[job['cell'],job['arm']]=first
        if job['arm'] in ('no_refresh','neither') and first is not None and any(q>first[0] for q in final['edit_calls']):
            raise ValueError('Semantic editing continued after accepted knowledge')
    for cell in {j['cell'] for j in jobs}:
        values=[first_notes[cell,a] for a in study.ARMS if (cell,a) in first_notes]
        if len(set(values))!=1:raise ValueError('Initial accepted knowledge differs among factorial arms')
    study.check(True)
    result=dict(status='PASS_PARTIAL' if partial else 'PASS_COMPLETE',cells=4,trajectories=len(jobs),
        full_replays=sum(j['arm']=='full' for j in jobs),controls=sum(j['arm']!='full' for j in jobs),
        control_queries=paid,initial_accepted_Note_identical=True,call_origins=dict(counts),
        all_GP_arrays_menus_raw_responses_and_feedback_replayed=True,new_GP_fits=0,new_LLM_calls=0,
        new_oracle_queries=0,old_results_and_cancelled_broad_phase_unchanged=True,
        byte_packets_verified=True,embedded_JSON_semantic_reconstruction_only=True,
        targeted_protocol_sha256=p['protocol_sha256'])
    out=ROOT/'experiments/reports'/study.VARIANT;out.mkdir(parents=True,exist_ok=True)
    (out/('audit_partial.json' if partial else 'audit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main('--partial' in sys.argv)
