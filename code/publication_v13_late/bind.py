"""Bind fresh GP checkpoints, raw-response costs and immutable experiment inputs."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study,queue
from reasoning_memory_v12.llm import official_extract,validate_insight

P=ROOT/'paper/v13_late'
R=ROOT/'experiments/reports/v13_late_validation_20261103'


def rel(x):return Path(x).resolve().relative_to(ROOT).as_posix()
def put(name,value):
    path=P/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(name,rows):
    with (P/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def core(raw):
    req={'comment','keywords','hypotheses'}
    if isinstance(raw,dict) and req<=set(raw) and set(raw)-req<={'knowledge_uses','assessments'} and all(raw[k]==[] for k in set(raw)-req):return {k:raw[k] for k in raw if k in req}
    return raw


def main():
    p=study.check(True);audit=queue.read(R/'final_audit.json')
    if audit['status']!='PASS_COMPLETE':raise ValueError('Full native audit required')
    models=[];requests=[];operational=[];agent_ids=set();cost=Counter();armcost=defaultdict(Counter)
    accepted={};counter_totals=Counter()
    for job in p['jobs']:
        folder=study.base()/job['id']/'runs'/job['id']
        for q in range(job['prefix_queries']+1,21):
            c=queue.read(folder/'rounds'/f'{q:02}_commit.json');empty=c['stage']=='no_data'
            checkpoint=study.base()/'cache'/job['task']/(c['GP_fit_key']+'.pt')
            if not empty and not checkpoint.is_file():raise ValueError('Missing GP checkpoint')
            row=dict(job=job['id'],kind=job['kind'],task=study.SHORT[job['task']],prefix_seed=job['seed'],
                repetition=job['rep'],arm=job['arm'],query=q,stage=c['stage'],fit_key=c['GP_fit_key'],
                checkpoint='' if empty else rel(checkpoint),checkpoint_sha256='' if empty else queue.file_sha(checkpoint))
            for label,suffix in (('fit_record','fit.json'),('commit','commit.json'),('GP_arrays','GP.npz')):
                f=folder/'rounds'/f'{q:02}_{suffix}'
                row[label]=rel(f);row[label+'_sha256']=queue.file_sha(f)
            models.append(row)
            for key in ('memory','reasoning'):
                if c[key]:accepted[rel(ROOT/c[key]['request_relpath'])]=dict(job=job,selected=c['selected_id'],kind=key)
        counters=queue.read(folder/'runtime_counters.json')
        if counters['operator_compiles_in_loop']!=0:raise ValueError('Unexpected operator recompilation')
        counter_totals.update(counters)
        for name in ('execution_error.json','schema_recovery_completion.json'):
            if (folder/name).exists():operational.append(dict(job=job['id'],receipt=rel(folder/name),content=queue.read(folder/name)))
    if len(models)!=840 or sum(x['stage']=='no_data' for x in models)!=9:raise ValueError('Wrong numerical-prefix coverage')
    for f in sorted((study.base()/'requests').glob('*/*/a*/request.json')):
        req=queue.read(f);folder=f.parent
        if not (folder/'response_receipt.json').is_file():raise ValueError('Unsealed request remains')
        a=queue.read(folder/'assignment.json');receipt=queue.read(folder/'response_receipt.json')
        if a['agent_id'] in agent_ids:raise ValueError('Experimental agent reused')
        agent_ids.add(a['agent_id'])
        if receipt['assignment_sha256']!=queue.file_sha(folder/'assignment.json') or receipt['raw_sha256']!=queue.file_sha(folder/'response.raw.json'):raise ValueError('Response bytes changed')
        if req['input_sha256']!=queue.file_sha(ROOT/req['input']):raise ValueError('Input changed')
        validations={name:queue.read(folder/name) for name in ('validation.json','core_projection_validation.json') if (folder/name).exists()}
        used=rel(f) in accepted
        arm=next(j['arm'] for j in p['jobs'] if j['id']==req['job'])
        raw=queue.read(folder/'response.raw.json')
        projected=used and arm=='rbo' and core(raw)!=raw
        row=dict(job=req['job'],kind=req['kind'],query=req['query'],attempt=req['attempt'],accepted_for_paid_query=int(used),
            public_context_sha256=req['context_sha256'],
            request=rel(f),request_sha256=queue.file_sha(f),assignment=rel(folder/'assignment.json'),assignment_sha256=queue.file_sha(folder/'assignment.json'),
            input=req['input'],input_sha256=req['input_sha256'],raw_response=rel(folder/'response.raw.json'),raw_response_sha256=receipt['raw_sha256'],
            agent_id=a['agent_id'],requested_model=req['model'],requested_effort=req['effort'],fork_turns=req['fork_turns'],
            schema_status=validations.get('validation.json',{}).get('status',''),empty_sidecar_projection=int('core_projection_validation.json' in validations),
            schema_error=validations.get('validation.json',{}).get('error',''),
            deterministic_core_projection_applied=int(projected),
            usage_available=False)
        requests.append(row);cost['raw_requests']+=1;cost[req['kind']+'_raw_requests']+=1
        armcost[arm]['raw_requests']+=1
        if used:cost['accepted_requests']+=1;cost[req['kind']+'_accepted_requests']+=1;armcost[arm]['accepted_requests']+=1
        if row['schema_status']=='SCHEMA_REJECTED':cost['original_schema_rejections']+=1
        if arm=='coupled' and req['kind']=='edit' and row['schema_error']=='Invalid memory editor envelope':cost['coupled_editor_envelope_rejections']+=1
        if row['empty_sidecar_projection']:cost['recovered_empty_sidecar_rejections']+=1
        if projected:cost['accepted_deterministic_empty_sidecar_projections']+=1
    if cost['choose_accepted_requests']!=720 or cost['accepted_requests']!=len(accepted):
        raise ValueError('Committed request paths were not fully matched')
    # Every unaccepted attempt is retained. Detect earlier official-core choices
    # different from the ultimately native-accepted choice, without changing it.
    unlocked=[]
    for path,used in accepted.items():
        if used['job']['arm']!='rbo' or used['kind']!='reasoning':continue
        af=ROOT/path;selected=used['selected'];parent=af.parent.parent
        for earlier in sorted(parent.glob('a*/response.raw.json')):
            if earlier.parent.name>=af.parent.name:continue
            raw=queue.read(earlier);input_packet=queue.read(ROOT/queue.read(earlier.parent/'request.json')['input'])
            public=input_packet['public'];public=public.get('original',public)
            ids=[x['candidate_id'] for x in public['candidate_menu']]
            try:old_selected=validate_insight(ROOT,core(raw),ids)
            except (ValueError,KeyError,TypeError):continue
            if old_selected!=selected:unlocked.append(dict(accepted_request=path,earlier_response=rel(earlier),
                earlier_core_choice=old_selected,accepted_choice=selected,outcome_access_before_repair=False))
    csvwrite('data/new_fit_inventory.csv',models);csvwrite('data/LLM_request_inventory.csv',requests)
    put('data/operational_incidents.json',operational)
    put('data/RBO_prior_core_choice_differences.json',unlocked)
    put('data/new_costs.json',dict(total_trajectories=75,new_paid_queries=840,numerical_prefixes=840,
        fitted_prefixes=831,no_data_prefixes=9,unique_bound_checkpoints=len({x['checkpoint'] for x in models if x['checkpoint']}),
        request_counts=dict(cost),arm_counts={k:dict(v) for k,v in armcost.items()},distinct_fresh_agents=len(agent_ids),
        runtime_counters=dict(counter_totals),counter_scope='Native per-job final counters; interrupted workers can have earlier cache counters outside the final resume snapshot',
        tokens=None,monetary_cost=None,token_and_monetary_reason='Not exposed by child-agent backend',
        earlier_core_choice_differs_from_native_accepted=len(unlocked),recorded_operational_receipts=len(operational)))
    put('audit/new_model_binding.json',dict(status='PASS',decisions=840,fitted_checkpoints_bound=831,no_data_decisions=9,
        every_fitted_decision_has_checkpoint=True,operator_compiles_in_loop=0,
        all_original_sources_and_results_unchanged=True,new_GP_refits_in_binding=0,new_LLM_calls_in_binding=0))
    for src,name in ((study.base()/'protocol.json','protocols/v13_late_validation_20261103.json'),
                     (study.base()/'runtime_resume_registration.json','audit/runtime_resume_registration.json'),
                     (study.base()/'runtime_resume_completion.json','audit/runtime_resume_completion.json'),
                     (study.base()/'preflight.json','audit/new_preflight.json'),(R/'final_audit.json','audit/new_native_replay.json')):
        target=P/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(src.read_bytes())
    print(json.dumps(dict(status='PASS',model_prefixes=len(models),fresh_requests=len(requests),prior_core_choice_differences=len(unlocked))))


if __name__=='__main__':main()
