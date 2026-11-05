"""Independent saved-evidence audit; never fits a GP or requests an experiment.

Checks every available committed choice, then requires all preregistered jobs
for --final. Native input files are read only for post-hoc label reconciliation.
Scientific requests, controls, and experimental state files are never edited.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
from statistics import NormalDist
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import sys
from functools import lru_cache
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
OUT=ROOT/'experiments/reports/grounded_baselines_20261105'
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp as native
from grounded_first_start import proposal


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def require(value,label):
    if not value:raise AssertionError(label)


def official_extractor():
    path=ROOT/'code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py'
    cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='BaseReasoner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_candidates_from_insight')
    ns={'json':json,'re':re}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    return ns[method.name]


def argmax(scores,available,ties):
    choices=np.flatnonzero(available)
    winners=choices[scores[choices]==max(scores[choices])]
    return int(winners[np.argmin(ties[winners])])


def region(pool,scope,background):
    require(isinstance(scope,dict) and bool(scope) and set(scope)<=set(pool.axes),'physical scope axes')
    domain={r['name']:r for r in background['inputs']};mask=np.ones(len(pool.ids),bool)
    for name,spec in scope.items():
        if pool.kinds[pool.axes.index(name)]=='categorical':
            require(set(spec)=={'allowed'} and bool(spec['allowed']) and set(spec['allowed'])<=set(domain[name]['levels']),'exact category membership')
            mask&=np.array([p[name] in spec['allowed'] for p in pool.points])
        else:
            require(set(spec)=={'bounds'},'physical bounds schema');lo,hi=spec['bounds'];lower,upper=domain[name]['declared_input_bounds']
            require(math.isfinite(lo) and math.isfinite(hi) and lower-1e-10<=lo<=hi<=upper+1e-10,'physical bounds')
            mask&=np.array([lo-1e-10<=p[name]<=hi+1e-10 for p in pool.points])
    return np.flatnonzero(mask).tolist()


def lgbo_plan(pool,raw,available,ties,background):
    require(set(raw)=={'summary','mode','confidence','conditions','scope','rationale'},'LGBO envelope')
    c=raw['confidence'];require(0<c<1 and math.isfinite(c),'LGBO finite confidence')
    mode=raw['mode']
    if mode=='none':
        require(raw['conditions'] is None and raw['scope']=={},'LGBO abstention')
        return dict(mode='none',confidence=float(c),indices=[])
    if mode=='region':
        require(raw['conditions'] is None,'LGBO region conditions')
        js=region(pool,raw['scope'],background)
        require(available[js].any(),'executable regional candidate')
        return dict(mode='region',confidence=float(c),indices=js,scope=raw['scope'])
    require(mode=='point' and raw['scope']=={},'LGBO mode')
    conditions=raw['conditions'];require(set(conditions)==set(pool.axes),'full point conditions')
    valid=available.copy();squared=np.zeros(len(pool.ids));dims=0
    domain={r['name']:r for r in background['inputs']}
    for axis,kind in zip(pool.axes,pool.kinds):
        v=conditions[axis]
        if kind=='categorical':
            require(v in domain[axis]['levels'],'point category');valid&=np.array([p[axis]==v for p in pool.points])
        else:
            lo,hi=domain[axis]['declared_input_bounds'];require(lo<=v<=hi and math.isfinite(v),'point physical input')
            squared+=(np.array([p[axis] for p in pool.points])-v)**2/(hi-lo)**2;dims+=1
    distances=np.sqrt(squared/max(1,dims));js=np.flatnonzero(valid)
    require(len(js)>0,'executable point');j=int(js[np.lexsort((ties[js],distances[js]))[0]])
    return dict(mode='point',confidence=float(c),indices=[j],scope={},projected_id=pool.ids[j],input_distance=float(distances[j]))


@lru_cache(maxsize=1)
def system_constants():
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse((ROOT/'code/experiment_tools/grounded_baselines/prompts.py').read_text(encoding='utf-8')).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('LGBO_SYSTEM','RBO_SYSTEM')}


def check_llm(c,pool,state,prior,background,examples,extract):
    saved=c['LLM'];request_path=ROOT/saved['request'];af=request_path.parent
    req=read(request_path);receipt=read(af/'receipt.json');assignment=read(af/'assignment.json');completion=read(af/'completion.json')
    require(receipt==saved['receipt'],'committed LLM receipt')
    require(req['request_sha256']==digest({k:v for k,v in req.items() if k!='request_sha256'}),'request digest')
    require(receipt['request_sha256']==req['request_sha256'],'receipt request')
    require(receipt['assignment_sha256']==sha(af/'assignment.json') and receipt['completion_sha256']==sha(af/'completion.json'),'actual assignment/completion custody')
    require(receipt['raw_sha256']==sha(af/'response.raw.json') and saved['raw']==read(af/'response.raw.json'),'raw response custody')
    inv=read(af/'invocation.json');require(assignment['spawn_request']==inv,'immutable isolated invocation')
    require(inv['fork_turns']=='none' and inv['model']=='gpt-6.1-terra' and inv['reasoning_effort']=='medium','frozen LLM backend')
    agent=assignment['agent_id'];require(agent==assignment['spawn_result']['task_name'] and agent==receipt['agent_id']==completion['agent_id'],'actual canonical child identity')
    require(agent in ('/root/'+inv['task_name'],'/root/grounded_experimental_llm_transport/'+inv['task_name']),'per-request fresh child')
    export=read(af/'export.json');require(sha(ROOT/export['input'])==export['input_sha256'],'experimental input custody')
    require(sha(ROOT/export['output'])==completion['output_sha256'],'completed output unchanged')
    ctx=read(af.parent/'context.json');public=ctx['public']
    require(digest(ctx)==saved['context_sha256']==req['context_sha256'],'exact scientific context key')
    constants=system_constants()
    require(ctx['system']==req['system']==constants[state['arm'].upper()+'_SYSTEM'],'frozen method system instructions')
    payload=public if req['attempt']==1 else dict(original=public,schema_repair_only=read(af.parent/'attempt1/error.json'))
    require(req['public']==payload and read(ROOT/export['input'])==dict(system=req['system'],public=payload),'child exact input')
    observations=[dict(query=i+1,candidate_id=h['candidate_id'],conditions=pool.points[pool.ix[h['candidate_id']]],objective_raw=h['raw_value']) for i,h in enumerate(state['history'])]
    require(public['own_observations']==observations,'only complete own prefix')
    require(public['background']==background and public['blinded_initial_hypotheses']==prior,'same grounded chemistry and initial knowledge')
    require(public['experimental_budget']==dict(initial=0,total_paid_queries=20,observed=len(observations),remaining=20-len(observations)),'paid query budget')
    require(public['incumbent_raw']==max(h['raw_value'] for h in state['history']) and not public['current_GP_predictions_or_ranking_supplied'],'no numeric GP ranking exposed')
    if state['arm']=='lgbo':
        common_keys={'background','objective_unit','experimental_budget','own_observations','blinded_initial_hypotheses','incumbent_raw','current_GP_predictions_or_ranking_supplied'}
        require(set(public)==common_keys|{'input_only_examples','previous_preferences','all_unqueried_pool_inputs_allowed'},'only authorized LGBO context fields')
        require(public['previous_preferences']==state['preferences'],'own preference history')
        require(public['all_unqueried_pool_inputs_allowed'] is True,'LGBO full input domain')
        seen={h['candidate_id'] for h in state['history']}
        require(public['input_only_examples']==[r for r in examples if r['candidate_id'] not in seen],'input-only unscored archetypes')
    else:
        common_keys={'background','objective_unit','experimental_budget','own_observations','blinded_initial_hypotheses','incumbent_raw','current_GP_predictions_or_ranking_supplied'}
        require(set(public)==common_keys|{'candidate_menu','axes','candidate_source','official_optimization_prompt'},'only authorized RBO context fields')
        ids=c['decision']['display_ids'];require([p['candidate_id'] for p in public['candidate_menu']]==ids,'RBO public top5 membership')
        for row in public['candidate_menu']:require(row['conditions']==pool.points[pool.ix[row['candidate_id']]],'RBO full chemical conditions')
        template=read(ROOT/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json')['optimization_loop']['en']
        expected=template.format(iteration=len(state['history']),target=background['objective'],retrieved_context=json.dumps(prior),
            insight_history=json.dumps(state['insights'],ensure_ascii=False),trial_data=json.dumps(observations,ensure_ascii=False),
            bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
        require(public['official_optimization_prompt']==expected,'official RBO prompt, own Insights only')
        with redirect_stdout(io.StringIO()):selected=extract(None,json.dumps(saved['raw']),n=1)
        require(selected==[{'candidate_id':c['decision']['selected_id']}],'unchanged official confidence extraction')
    return agent


def audit(final=False):
    p=read(BASE/'protocol.json');require(p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'}),'protocol digest')
    source_groups=[p['sources_sha256'],p['inputs_sha256'],read(BASE/'execution_compatibility.json')['new_source_sha256'],read(BASE/'delegated_transport_registration.json')['source_sha256']]
    if final:source_groups.append(p['protected_sha256'])
    protected={k:v for group in source_groups for k,v in group.items()}
    for path,h in protected.items():require(sha(ROOT/path)==h,'frozen source/control '+path)
    extract=official_extractor();rows=[];agents=[];checked=0;complete=0;pool_data={}
    for task,seed,arm in p['jobs']:
        short=dict(zip(native.TASKS,('c1','c2','c3','c4','perera','doyle','snar','hplc')))[task]
        folder=BASE/'runs'/f's{seed}'/(short+'_'+arm);sf=folder/'state.json'
        if not sf.exists():continue
        saved=read(sf);n=len(saved['history'])
        if task not in pool_data:
            pool=native.load_task(ROOT,task)
            pool_data[task]=(pool,proposal.background(pool),proposal.examples(pool))
        pool,background,examples=pool_data[task]
        state=dict(task=task,seed=seed,arm=arm,protocol_sha256=p['protocol_sha256'],initial_ids=[],initial=0,queries=20,history=[],decisions=[],preferences=[],insights=[],calls=[],status='RUNNING')
        with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
            y=z['values'].copy();raw=z['raw_values'].copy() if 'raw_values' in z else y*pool.scale
        previous_best=-math.inf;integral=0.
        for q in range(1,n+1):
            cp=folder/'rounds'/f'{q:02d}_commit.json';fp=folder/'rounds'/f'{q:02d}_feedback.json'
            c=read(cp);feedback=read(fp);d=c['decision'];j=pool.ix[d['selected_id']]
            require(c['query']==q and c['prestate_sha256']==digest(state),'exact prestate chain')
            require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'}),'committed decision digest')
            require(feedback['commit_sha256']==c['commit_sha256'] and cp.stat().st_mtime_ns<=fp.stat().st_mtime_ns,'commit before feedback receipt')
            observed=feedback['observation'];require(observed==saved['history'][q-1],'feedback preserved')
            require(observed['candidate_id']==pool.ids[j] and observed['y']==float(y[j]) and observed['raw_value']==float(raw[j]),'oracle row/scale reconciliation')
            require(d['conditions']==pool.points[j] and pool.ids[j] not in [h['candidate_id'] for h in state['history']],'unique executable physical input')
            if q==1:
                require(pool.ids[j]==p['shared_first'][task]['candidate_id'] and not d['GP_fitted'] and not d['qLogEI_defined'] and c['LLM'] is None,'common paid first input, no zero-data GP')
            else:
                ap=folder/'rounds'/f'{q:02d}_GP.npz';require(sha(ap)==c['GP_arrays_sha256'],'immutable candidate arrays')
                with np.load(ap,allow_pickle=False) as z:a={k:z[k] for k in z.files}
                require(all(v.shape==(len(pool.ids),) for v in a.values()),'all pool candidates scored')
                available=np.ones(len(pool.ids),bool);available[[pool.ix[h['candidate_id']] for h in state['history']]]=False
                require(np.array_equal(a['available'],available),'no previously queried candidate')
                ties=native.semantic_priorities(pool,q,seed);require(np.array_equal(a['tie_priority'],ties),'registered all-query feature tie stream')
                bo=argmax(a['base_logei'],available,ties);require(d['BO_reference_id']==pool.ids[bo],'own-prefix pure BO reference')
                require(d['changed_from_BO']==(j!=bo) and d['native_opportunity_gap']==float(a['base_logei'][bo]-a['base_logei'][j]),'opportunity gap')
                require(d['base_logEI']==a['base_logei'][j] and d['final_logEI']==a['final_logei'][j],'saved acquisition used')
                require(d['GP_mean_raw']==float(a['mu_y'][j]*pool.scale) and d['GP_SD_raw']==float(a['sd_y'][j]*pool.scale)
                    and d['GP_predictive_SD_raw']==float(a['sd_predictive_y'][j]*pool.scale),'reported GP output units')
                require(all(np.isfinite(a[k]).all() for k in a),'finite full-pool values')
                agent=check_llm(c,pool,state,p['shared_blinded_priors'][task],background,examples,extract);agents.append(agent)
                if arm=='rbo':
                    js=np.flatnonzero(available);ordered=js[np.lexsort((ties[js],-a['base_logei'][js]))][:5]
                    physical=sorted(map(int,ordered),key=lambda i:pool.features[i]);display=np.array(physical)[np.random.default_rng(np.random.SeedSequence([0,q,991])).permutation(5)]
                    require(d['ordered_ids']==[pool.ids[int(i)] for i in ordered] and d['display_ids']==[pool.ids[int(i)] for i in display],'exact GP top5 and neutral display')
                    require(j in ordered,'actual RBO selection in GP top5')
                    require(all(point=={'candidate_id':point['candidate_id']} and point['candidate_id'] in d['display_ids']
                        for h in c['LLM']['raw']['hypotheses'] for point in h['parameter_sets']),'all official Insight proposals in displayed menu')
                    require(c['LLM']['plan']==dict(selected_id=pool.ids[j]),'committed RBO extraction plan')
                    require(np.array_equal(a['final_logei'],a['base_logei']) and not np.any(a['mean_shift_y']),'RBO does not change GP/acquisition')
                else:
                    plan=lgbo_plan(pool,c['LLM']['raw'],available,ties,background);require(plan==c['LLM']['plan'],'independent LGBO physical plan')
                    require(j==argmax(a['final_logei'],available,ties),'LGBO full-pool qLogEI exact maximum')
                    meta=d['lgbo'];require(meta['native_replay_error']<=2e-8,'native acquisition numerical guard')
                    if plan['mode']=='none':require(np.array_equal(a['base_logei'],a['final_logei']) and not np.any(a['mean_shift_y']),'exact native abstention')
                    else:
                        # Vendor _norm_ppf evaluates erfinv in float32. Compare
                        # with an independent double-precision definition at
                        # its actual precision, without altering the vendor.
                        confidence=min(max(plan['confidence'],1e-12),1-1e-12)
                        delta=3*NormalDist().inv_cdf(confidence)
                        tolerance=2e-6*max(1.,abs(delta))
                        require(abs(meta['delta']-delta)<tolerance,'official confidence to delta (vendor float32 tolerance)')
                        indices=sorted(plan['indices'],key=lambda i:pool.features[i])
                        if len(indices)>512:
                            key=digest(dict(mode=plan['mode'],features=[pool.features[i] for i in indices]))
                            pos=np.random.default_rng(int(key[:12],16)).choice(len(indices),512,replace=False)
                            indices=[indices[int(i)] for i in sorted(pos)]
                        require(meta['quadrature_ids']==[pool.ids[i] for i in indices],'input-only finite region quadrature')
                        require(meta['covariance_unchanged'] and meta['output_covariance_units_corrected'],'qualified official covariance implementation')
                        require(abs(d['shifted_mean_raw']-float((a['mu_y'][j]+a['mean_shift_y'][j])*pool.scale))<1e-9,'LGBO mean shift units')
                require(c['acquisition']['observed_count']==q-1,'GP uses all and only previous paid observations')
            state['history'].append(observed);previous_best=max(previous_best,observed['raw_value']);integral+=previous_best
            state['decisions'].append(dict(query=q,**observed,decision=d,best=previous_best))
            if c['LLM'] is not None:
                state['calls'].append(dict(query=q,request=c['LLM']['request']))
                state['preferences' if arm=='lgbo' else 'insights'].append(deepcopy(c['LLM']['raw']))
            state['status']='COMPLETE' if q==20 else 'RUNNING';checked+=1
        require(state==saved,'full state exactly replays committed feedback')
        if n>1:require(read(folder/'runtime_counters.json')['operator_compiles_in_loop']==0,'operators compiled/warmed outside rounds')
        complete+=n==20
        rows.append(dict(task=short,seed=seed,method=arm,queries=n,status=state['status'],SearchAUC_independent=integral/n if n else None,final_best=previous_best if n else None))
    require(len(agents)==len(set(agents)),'fresh isolated child for every committed LLM request')
    if final:
        require(complete==64 and checked==1280 and len(agents)==1216,'entire preregistered new population complete')
        require(not list(BASE.glob('runs/*/*/execution_error.json')),'no unresolved execution failures')
        with (OUT/'per_seed_results.csv').open(encoding='utf-8-sig',newline='') as f:reported={(r['task'],int(r['seed']),r['method']):r for r in csv.DictReader(f)}
        for r in rows:
            v=reported[(r['task'],r['seed'],r['method'])]
            require(abs(float(v['SearchAUC'])-r['SearchAUC_independent'])<1e-9 and float(v['final_best'])==r['final_best'],'independent metric/report reconciliation')
        for control in p['controls']:
            v=read(ROOT/control['path']);best=-math.inf;integral=0.
            for h in v['history']:best=max(best,h['raw_value']);integral+=best
            require(len(v['history'])==len(v['decisions'])==20 and v['status']=='COMPLETE'
                and len({h['candidate_id'] for h in v['history']})==20,'complete frozen control, twenty paid decisions')
            task=dict(zip(('c1','c2','c3','c4','perera','doyle','snar','hplc'),native.TASKS))[control['task']]
            pool=pool_data[task][0]
            expected=pool.ids[int(np.argmin(native.semantic_priorities(pool,1,control['seed'])))] if control['method']=='bo' else p['shared_first'][task]['candidate_id']
            require(v['history'][0]['candidate_id']==expected,'frozen control initial-input semantics')
            if control['method']=='bo':
                head=v['decisions'][0];head=head.get('decision',head)
                require(head.get('stage')=='no_data' or head.get('GP_fitted') is False,'pure BO zero-data first decision')
            r=reported[(control['task'],control['seed'],control['method'])]
            require(abs(float(r['SearchAUC'])-integral/20)<1e-9 and float(r['final_best'])==best,'frozen control metric reconciliation')
        # Include schema-only failed attempts in custody checks, rather than
        # silently counting only successful responses committed to a decision.
        all_attempt_agents=[];attempts=0;repairs=0
        for vp in BASE.glob('requests/*/*/validated.json'):
            validated=read(vp);successful=Path(validated['request']).parent.name
            require(successful in ('attempt1','attempt2'),'maximum two fresh schema attempts')
            for af in sorted(vp.parent.glob('attempt*')):
                if not af.is_dir():continue
                req=read(af/'request.json');receipt=read(af/'receipt.json');assignment=read(af/'assignment.json');completion=read(af/'completion.json');export=read(af/'export.json')
                require(receipt['request_sha256']==req['request_sha256'] and receipt['assignment_sha256']==sha(af/'assignment.json')
                    and receipt['completion_sha256']==sha(af/'completion.json') and receipt['raw_sha256']==sha(af/'response.raw.json'),'all attempts custody')
                require(completion['output_sha256']==sha(ROOT/export['output']) and export['input_sha256']==sha(ROOT/export['input']),'all attempts immutable exported I/O')
                require(assignment['agent_id']==receipt['agent_id']==completion['agent_id'],'all attempts actual agent identity')
                if af.name!=successful:
                    require(af.name=='attempt1' and (af/'error.json').exists(),'only documented schema repair can replace a response');repairs+=1
                all_attempt_agents.append(assignment['agent_id']);attempts+=1
        require(len(all_attempt_agents)==len(set(all_attempt_agents)),'every schema attempt uses its own child')
        require(len(list(BASE.glob('requests/*/*/validated.json')))==1216 and attempts==1216+repairs,'all logical requests accounted for')
        require(len(list(BASE.glob('requests/*/*/attempt*/receipt.json')))==attempts,'no unaccounted responses')
    value=dict(status='PASS_COMPLETE_SAVED_EVIDENCE_AUDIT' if final else 'PASS_PARTIAL_SAVED_EVIDENCE_AUDIT',
        completed_new=complete,planned_new=64,checked_committed_experiments=checked,checked_fresh_LLM_requests=len(agents),
        hashed_protected_inputs_sources_controls=len(protected),all_selected_inputs_and_observed_labels_reconciled=True,
        full_decision_state_replay=True,commit_before_feedback_receipt=True,own_history_and_grounded_background_verified=True,
        official_confidence_extractor_precompiled_once=True,all_candidate_saved_acquisition_argmax_and_ties_verified=True,
        GP_fitting_or_new_oracle_calls_in_audit=0,numerical_covariance_check='Registered native compatibility/preflight qualification; no new GP fits in this audit',
        independent_metric_report_reconciliation=bool(final),protocol_sha256=p['protocol_sha256'],rows=rows)
    if final:value.update(actual_fresh_LLM_attempts=attempts,schema_only_repairs=repairs)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/('audit.json' if final else 'audit_partial.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return {k:v for k,v in value.items() if k!='rows'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    print(json.dumps(audit(args.final)))
