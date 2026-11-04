"""Read-only full-prefix replay and fixed-budget comparison for V13."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from state_split_v13_experiment import study,policy,transport,prompts
from reasoning_memory_v13 import control
from knowledge_rank_v9 import engine as native
from reasoning_memory_v12.llm import validate_insight
from chem.benchmarks import knowledge_gp as gpmod


def metrics(state):
    values=[r['raw_value'] for r in state['history']]
    if len(values)!=20 or len(state['decisions'])!=16 or len({r['candidate_id'] for r in state['history']})!=20:
        raise ValueError('A complete, unrepeated 4+16 trajectory is required')
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite objective')
    best=np.maximum.accumulate(values)
    return dict(SearchAUC=float(best[4:].mean()),final_best=float(best[-1]),
                first_final_best_query=max(0,values.index(float(best[-1]))-3),
                initial_best=max(values[:4]))


def dump_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def replay(task,state,protocol):
    pool=gpmod.load_task(ROOT,task,seed=0)
    run=transport.base(ROOT)/'runs'/study.SHORT[task]
    current={k:deepcopy(state[k]) for k in ('task','arm','seed','protocol_sha256','initial_commit_sha256')}
    current.update(status='READY',history=deepcopy(state['history'][:4]),notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
    bo=transport.read(ROOT/protocol['controls'][task]['bo']['path'])
    if current['history']!=bo['history'][:4]:raise ValueError('Initial mismatch')
    if transport.read(run/'initial_GP_check.json')['status']!='PASS_EXACT':raise ValueError('Initial GP mismatch')
    audit_rows=[]
    for q,observation in enumerate(state['history'][4:],1):
        commit=transport.read(run/'rounds'/f'{q:02d}_commit.json')
        with np.load(run/'rounds'/f'{q:02d}_GP.npz',allow_pickle=False) as z:
            ctx=dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
                     **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})
        if native.gp_hash(ctx)!=commit['GP_sha256']:raise ValueError('GP archive drift')
        expected=~np.isin(pool.ids,[r['candidate_id'] for r in current['history']])
        np.testing.assert_array_equal(ctx['available'],expected)
        ordered,display=native.candidates(pool,ctx,q)
        ids=[pool.ids[int(j)] for j in display]
        if commit['ordered_ids']!=[pool.ids[int(j)] for j in ordered] or commit['display_ids']!=ids:raise ValueError('Menu drift')
        if commit['refresh']!=policy.edit_event(pool,current,ordered,q):raise ValueError('Refresh drift')
        memory=commit['memory']
        memory_raw=transport.read((ROOT/memory['request_relpath']).parent/'response.raw.json') if memory else None
        if memory and memory_raw!=memory['raw']:raise ValueError('Editor response drift')
        edited=policy.apply_edits(memory_raw,pool,current,q) if memory else None
        notes=edited['notes'] if memory else current['notes']
        if notes!=commit['edited_notes'] or (memory and memory['plan']!=edited):raise ValueError('Edit replay mismatch')
        public=prompts.reasoning_public(ROOT,pool,current,notes,ids,q)
        context=transport.read(transport.base(ROOT)/'requests'/study.SHORT[task]/commit['reasoning']['context_sha256']/'context.json')
        if control.digest(public)!=control.digest(context['public']) or context['system']!=prompts.REASONING_SYSTEM:raise ValueError('Public context mismatch')
        if memory:
            context=transport.read(transport.base(ROOT)/'requests'/study.SHORT[task]/memory['context_sha256']/'context.json')
            if control.digest(context['public'])!=control.digest(prompts.memory_public(pool,current,ids,q,commit['refresh'])):raise ValueError('Editor context mismatch')
        raw=transport.read((ROOT/commit['reasoning']['request_relpath']).parent/'response.raw.json')
        if raw!=commit['reasoning']['raw']:raise ValueError('Reasoning response drift')
        packet=policy.validate_decision(ROOT,pool,notes,current,ctx,ids,q,raw)
        if packet!=commit['decision']:raise ValueError('Decision replay mismatch')
        # JSON commitments sort keys; exact official prompt strings depend on
        # original in-memory insertion order. Reconstruct it from authored bytes
        # and replayed states, while requiring canonically identical commitments.
        commit['decision']=packet
        commit['edited_notes']=notes
        current=study.feedback(current,commit,observation)
        events=current['decisions'][-1]['feedback']
        audit_rows.append(dict(task=study.SHORT[task],query=q,candidate_id=observation['candidate_id'],
            objective=observation['raw_value'],incumbent=max(r['raw_value'] for r in current['history']),
            GP_top1=commit['BO_reference_id'],changed_from_GP=observation['candidate_id']!=commit['BO_reference_id'],
            forecast_count=len(packet['forecasts']),uses=json.dumps(packet['knowledge_uses']),
            permissions=json.dumps([{**{k:n[k] for k in ('id','version')},'permission':n['recommendation']['permission'],
                'reliability':n['calibration']['reliability']} for n in current['notes']]),
            refresh=bool(memory),refresh_reasons=';'.join(commit['refresh']['reasons']),
            accurate_failure_events=sum(e['accurate_failure_forecast'] for e in events),
            new_suppressions=sum(e['recommendation_before']!='suppressed' and e['recommendation_after']=='suppressed' for e in events),
            newly_allowed=sum(e['recommendation_before']!='allowed' and e['recommendation_after']=='allowed' for e in events)))
    if current!=state:raise ValueError('Full final state mismatch')
    return audit_rows


def main():
    protocol=study.check(ROOT,True)
    folder=ROOT/'experiments/reports/v13_state_split_experiment_20261102';folder.mkdir(parents=True,exist_ok=True)
    rows,mechanisms,comparisons,missing=[],[],[],[]
    note_counts=Counter();use_counts=Counter();review_rounds={};calibration_counts=Counter();edit_counts=Counter()
    for task,short in study.SHORT.items():
        sf=transport.base(ROOT)/'runs'/short/'state.json'
        state=transport.read(sf) if sf.exists() else {}
        if state.get('status')!='COMPLETE':missing.append(short);continue
        counters=transport.read(sf.parent/'runtime_counters.json')
        if counters['operator_compiles_in_loop']!=0:raise ValueError('Operator recompiled in loop')
        if len(state['review_queries'])>policy.PARAMETERS['trajectory_review_query_budget']:raise ValueError('Review budget exceeded')
        review_rounds[short]=state['review_queries']
        note_counts.update(n['recommendation']['permission'] for n in state['notes'])
        for d in state['decisions']:
            use_counts.update(r['use'] for r in d['knowledge_uses'])
            for e in d['feedback']:
                a,b=e['calibration']['before']['reliability'],e['calibration']['after']['reliability']
                calibration_counts['increased' if b>a+1e-12 else 'decreased' if b<a-1e-12 else 'unchanged']+=1
                if e['recommendation_after']=='allowed' and e['recommendation_before']!='allowed' and not e['improved']:
                    raise ValueError('Failure feedback granted positive permission')
                if e['recommendation_before']=='reviewing' and e['recommendation_after']=='allowed':
                    calibration_counts['restored_after_review']+=1
            c=transport.read(sf.parent/'rounds'/f"{d['query']:02d}_commit.json")
            if c['memory']:
                for edit in c['memory']['plan']['reviews']:
                    edit_counts[edit.get('outcome','rejected') if edit['accepted'] else 'rejected']+=1
        mechanisms.extend(replay(task,state,protocol))
        new=metrics(state)
        rows.append(dict(task=short,method='V13',backend='fresh_subagent',**new))
        comp=dict(task=short,V13_SearchAUC=new['SearchAUC'],V13_final_best=new['final_best'],V13_first_best_query=new['first_final_best_query'])
        for arm in ('bo','reasoning_bo','v12_subagent'):
            reference=transport.read(ROOT/protocol['controls'][task][arm]['path'])
            if reference['history'][:4]!=state['history'][:4]:raise ValueError('Control initialization mismatch')
            m=metrics(reference)
            rows.append(dict(task=short,method=arm,backend='none' if arm=='bo' else 'fresh_subagent' if arm=='v12_subagent' else 'historical_API',**m))
            comp[arm+'_SearchAUC']=m['SearchAUC']
            comp['delta_vs_'+arm+'_pct']=100*(new['SearchAUC']/m['SearchAUC']-1)
        comparisons.append(comp)
        if 'v12_api' in protocol['controls'][task]:
            m=metrics(transport.read(ROOT/protocol['controls'][task]['v12_api']['path']))
            rows.append(dict(task=short,method='v12_api',backend='historical_API',**m))
    ids=[];attempts=Counter();locked_repairs=0
    for path in (transport.base(ROOT)/'requests').glob('*/*/attempt*/response_receipt.json'):
        receipt=transport.read(path);af=path.parent;req=transport.verify_request(af/'request.json')
        assignment=transport.read(af/'assignment.json');inv=transport.read(af/'invocation.json')
        if (receipt['request_sha256']!=req['request_sha256'] or receipt['raw_sha256']!=transport.file_sha(af/'response.raw.json') or
            receipt['assignment_sha256']!=transport.file_sha(af/'assignment.json') or assignment['spawn_request']!=inv or
            assignment['spawn_result']!={'task_name':assignment['agent_id']} or assignment['agent_id']!='/root/'+inv['task_name'] or
            inv['model']!='gpt-6.1-terra' or inv['reasoning_effort']!='medium' or inv['fork_turns']!='none'):
            raise ValueError('Response or actual spawn receipt mismatch')
        export=transport.read(af/'export.json')
        if transport.file_sha(ROOT/export['input'])!=export['input_sha256']:raise ValueError('Export drift')
        if transport.read(ROOT/export['input'])!={'system':req['system'],'public':req['public']}:raise ValueError('Export mismatch')
        validation=transport.read(af/'validation.json')
        if req['kind']=='choose' and req['attempt']>1 and validation['status']=='VALIDATED':
            original=req['public']['original'];locked=req['public']['selected_id_must_remain']
            chosen=validate_insight(ROOT,transport.read(af/'response.raw.json')['insight'],
                                    [r['candidate_id'] for r in original['candidate_menu']])
            if locked is not None:
                if chosen!=locked:raise ValueError('Schema repair changed the official point')
                locked_repairs+=1
        ids.append(receipt['agent_id']);attempts[transport.read(af/'validation.json')['status']]+=1
    if len(ids)!=len(set(ids)):raise ValueError('Agent reused across requests')
    status='PASS_COMPLETE' if not missing else 'PARTIAL_NOT_A_PERFORMANCE_CONCLUSION'
    audit=dict(status=status,complete_tasks=len(comparisons),missing=missing,queries_replayed=len(mechanisms),
               unique_fresh_agents=len(ids),schema_attempts=dict(attempts),protocol_sha256=protocol['protocol_sha256'],
               source_and_controls_unchanged=True,oracle_calls_in_this_audit=0,
               locked_point_repairs_verified=locked_repairs,operator_compiles_in_loop=0,
               final_note_permissions=dict(note_counts),knowledge_use_counts=dict(use_counts),
               calibration_changes=dict(calibration_counts),review_queries_by_task=review_rounds,
               memory_edit_counts=dict(edit_counts),failure_based_positive_restorations=0,
               trajectory_counts={k:sum(r[k] for r in mechanisms) for k in ('forecast_count','accurate_failure_events','new_suppressions','newly_allowed','changed_from_GP')})
    dump_csv(folder/'metrics.csv',rows);dump_csv(folder/'comparison.csv',comparisons);dump_csv(folder/'rounds.csv',mechanisms)
    (folder/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    lines=['V13 prediction/recommendation split: seed0 experiment','',status,
           'Same pools, native GP, four initial experiments and sixteen queries. Fresh gpt-6.1-terra medium child agents.',
           'SearchAUC = mean of the 16 cumulative best objectives after each new query. All results count.',
           'RBO is the frozen API cohort; V12 is the frozen fresh-subagent cohort. API/child differences are not a causal module effect.','',
           'Task | V13 AUC | Pure BO AUC | RBO API AUC | V12 child AUC | V13 vs BO | V13 vs RBO | V13 vs V12']
    for r in comparisons:
        lines.append(f"{r['task']} | {r['V13_SearchAUC']:.4f} | {r['bo_SearchAUC']:.4f} | {r['reasoning_bo_SearchAUC']:.4f} | {r['v12_subagent_SearchAUC']:.4f} | {r['delta_vs_bo_pct']:+.2f}% | {r['delta_vs_reasoning_bo_pct']:+.2f}% | {r['delta_vs_v12_subagent_pct']:+.2f}%")
    if not missing:
        for arm in ('bo','reasoning_bo','v12_subagent'):
            ds=[r['V13_SearchAUC']-r[arm+'_SearchAUC'] for r in comparisons]
            lines.append(f"vs {arm}: {sum(d>1e-8 for d in ds)} wins, {sum(abs(d)<=1e-8 for d in ds)} ties, {sum(d< -1e-8 for d in ds)} losses.")
    lines+=['','Mechanism diagnostics: '+json.dumps(audit['trajectory_counts']),
            'Single seed and one LLM realization per task cannot establish stability. Fresh initial knowledge differs from controls.',
            'Declared citations are audited; hidden semantic dependence of free-text reasoning cannot be fully enforced.',
            'Detailed objectives, first-best query and cohorts: metrics.csv; full per-round public decisions/state changes: rounds.csv.']
    (folder/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False));print('\n'.join(lines[7:]))

if __name__=='__main__':main()
