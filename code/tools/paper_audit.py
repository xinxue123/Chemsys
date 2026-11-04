"""Audit completed publication trajectories and report blocked cells explicitly."""
from pathlib import Path
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12 import study
from audit_v12_initialization_seeds import audit_bo
from audit_knowledge_rank_v9 import require,independent_kf
import audit_knowledge_rank_v9 as v9audit
import audit_reasoning_memory_v12 as v12audit
from audit_v12_llm_repeats import validate_response_ids,validate_shared_prefixes

def independent_fixed_kf(before,prob,g,success,t,prior=False):
    """Removal rule reconstructed independently of the production wrapper."""
    _,measurement=independent_kf(before,prob,g,success,t,prior)
    after=deepcopy(before)
    after['updates']+=1
    after['local_updates']+=int(not prior)
    measurement['reason']='fixed_calibration_ablation'
    return after,measurement

def independent_initial_edit(pool,pre,original,t):
    return dict(allowed=not pre['edit_calls'],reasons=['initial_knowledge'] if not pre['edit_calls'] else ['refresh_disabled_ablation'])

def control_run(root,task,seed):
    if seed==0:return study.v9.directory(root)/'runs'/f'{task}_bo'
    if task in study.FOCUS:return study.seeds.pair_base(root,task,seed)/'runs'/f'{task}_bo'
    return study.run_path(root,task,seed,'bo')

def audit_llm(root,run,p,pool,control,arm):
    """Reuse native numeric replay with only file/input and removal-rule adapters.

    This function never enters the production runner or GP fitting adapter.
    Paths are explicit so it can also be exercised on frozen historical runs.
    """
    require(arm in ('memory','no_kf','no_refresh','reasoning_bo'),'known audit intervention')
    base=run.parent.parent;reader=study.old.read
    legacy=study.old.paths(root,'v8')/'runs'/f'{pool.task}_bo_agent_initial_terra/state.json'
    def load(r,task,*,seed=0):
        require(Path(r).resolve()==Path(root).resolve() and task==pool.task and seed==0,'auditor pool identity')
        return pool
    def read(path):
        return reader(control/'state.json' if Path(path)==legacy else path)
    with ExitStack() as stack:
        stack.enter_context(patch.object(study.v12.gpmod,'load_task',side_effect=load))
        stack.enter_context(patch.object(study.v12,'check',return_value=deepcopy(p)))
        stack.enter_context(patch.object(study.v12,'directory',return_value=base))
        if arm=='reasoning_bo':
            stack.enter_context(patch.object(study.v9,'directory',return_value=base))
            stack.enter_context(patch.object(study.old,'read',side_effect=read))
            result=v9audit.run_audit(root,run/'state.json',p)
        else:
            stack.enter_context(patch.object(study.v9,'directory',return_value=control.parent.parent))
            if arm=='no_kf':stack.enter_context(patch.object(v12audit,'independent_kf',side_effect=independent_fixed_kf))
            if arm=='no_refresh':stack.enter_context(patch.object(v12audit,'independent_event',side_effect=independent_initial_edit))
            result=v12audit.audit_run(root,pool.task,'memory')
    return dict(result,arm=arm)

def request_and_prefix_evidence(root,run,arm):
    state=study.old.read(run/'state.json');base=run.parent.parent
    contexts=[];prefixes=[];responses=[];keys=[]
    for t in range(1,17):
        c=study.old.read(run/'rounds'/f'{t:02d}_commit.json')
        prefix=sorted([dict(candidate_id=r['candidate_id'],y=r['y']) for r in c['prestate']['history']],key=lambda r:r['candidate_id'])
        prefixes.append(dict(task=state['task'],query=t,prefix_sha256=study.digest(prefix),GP_sha256=c['GP_sha256'],ordered_ids=c['ordered_ids'],display_ids=c['display_ids']))
        if arm=='bo':continue
        for kind,key in [('memory',c.get('memory_context_sha256')),('selection',c['LLM_context_sha256'])]:
            if key is None:continue
            keys.append(key);folder=base/'requests'/key
            ctx=study.old.read(folder/'context.json');valid=study.old.read(folder/'validated.json')
            require(study.old.read(folder/'attempt1/request.json')==dict(system=ctx['system'],public=ctx['public']),'exact first prompt, no replicate salt')
            require(not any(k in ctx['public'] for k in ('initialization_seed','protocol_sha256','replicate')),'no bookkeeping tags in LLM prompt')
            physical={study.old.read(path)['id'] for path in folder.glob('attempt*/response.json')}
            require(valid['response_id'] in physical,'validated response physically present')
            contexts.append(dict(task=state['task'],arm=arm,query=t,kind=kind,context_sha256=key,response_id=valid['response_id']))
    if arm!='bo':
        require(len(keys)==len(set(keys))==16+len(state['edit_calls']),'one committed context per decision/edit')
        require({p.parent.name for p in (base/'requests').glob('*/validated.json')}==set(keys),'no discarded validated requests')
    if arm=='bo':return contexts,prefixes,responses
    for path in (base/'requests').glob('*/attempt*/response.json'):
        require(path.parent.parent.name in keys,'response belongs to committed request')
        r=study.old.read(path)
        responses.append(dict(path=path.relative_to(root).as_posix(),response_id=r['id'],model=r.get('resolved_model',r.get('model')),usage=r.get('usage',{})))
    return contexts,prefixes,responses

def main():
    p=study.check(ROOT,True);rows=[];pending=[];failures=[];contexts=[];prefixes=[];responses=[]
    for task,seed,arm in study.JOBS:
        run=study.run_path(ROOT,task,seed,arm);state=study.old.read(run/'state.json') if (run/'state.json').exists() else None
        if not state or state['status']!='COMPLETE':
            pending.append(dict(task=task,initialization_seed=seed,arm=arm,queries=len(state['decisions']) if state else 0));continue
        try:
            pool=study.pool_for(ROOT,task,seed,p)
            launch=study.old.read(run.parent.parent/'launch.json')
            require((launch['task'],launch['initialization_seed'],launch['arm'],launch['native_arm'])==(task,seed,arm,study.native_arm(arm)),'registered launch identity')
            require(launch['protocol_sha256']==p['protocol_sha256'] and launch['started_utc']>=p['created_utc'],'registration before launch')
            require(launch['LLM_cache_initially_empty'] and launch['response_files_copied']==0,'isolated fresh response namespace')
            r=audit_bo(run,pool,p) if arm=='bo' else audit_llm(ROOT,run,p,pool,control_run(ROOT,task,seed),arm)
            pf=study.directory(ROOT)/'preflight'/f's{seed}'/study.SHORT[task]
            require(state['history'][:4]==study.old.read(pf/'receipt.json')['initial_history'],'registered initial observations')
            with np.load(pf/'01_GP.npz') as a,np.load(run/'rounds/01_GP.npz') as b:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
            cc,pp,rr=request_and_prefix_evidence(ROOT,run,arm)
            rows.append(dict(r,initialization_seed=seed))
            contexts.extend(dict(x,initialization_seed=seed) for x in cc)
            prefixes.extend(dict(x,initialization_seed=seed,arm=arm) for x in pp)
            responses.extend(rr)
        except Exception as e:failures.append(dict(task=task,seed=seed,arm=arm,kind=type(e).__name__,message=str(e)))
    historical_ids=set();historical_prefixes=[];seen_controls={study.run_path(ROOT,r['task'],r['initialization_seed'],r['arm']) for r in rows};shared=None
    try:
        for name in p['protected_controls_sha256']:
            if '/requests/' in name and '/attempt' in name and name.endswith('/response.json'):
                value=study.old.read(ROOT/name);key=value.get('id',value.get('response_id'))
                if isinstance(key,str):historical_ids.add(key)
        validate_response_ids(responses,historical_ids)
        for row in rows:
            control=control_run(ROOT,row['task'],row['initialization_seed'])
            if control in seen_controls:continue
            seen_controls.add(control)
            _,pp,_=request_and_prefix_evidence(ROOT,control,'bo');historical_prefixes.extend(pp)
        shared=validate_shared_prefixes(prefixes+historical_prefixes)
    except Exception as e:failures.append(dict(kind=type(e).__name__,message=str(e),stage='cross_run_custody'))
    r=dict(status='PASS_COMPLETED_TRAJECTORIES_WITH_PENDING_CELLS' if not failures and pending else ('PASS' if not failures else 'FAIL'),
        completed=rows,pending=pending,failures=failures,completed_trajectories=len(rows),audited_queries=sum(x['rounds'] for x in rows),
        protected_files_checked=len(p['protected_controls_sha256']),protocol_sha256=p['protocol_sha256'],
        new_oracle_queries=0,new_API_calls=0,new_GP_fits=0,auditor_sha256=study.old.sha(__file__),
        unique_new_HTTP_response_ids=len(responses),historical_response_ids_checked=len(historical_ids),shared_history_groups_checked=shared,
        boundary='Completed runs only: native BO/V12/RBO independent numeric replay; separately reconstructed registered removal rules; response custody and shared-prefix equality. Schema replay reuses production; no independent GP refit or external timestamp attestation. Pending cells are not zero outcomes or completed comparisons.')
    study.write(ROOT/'experiments/reports/paper_v12/request_evidence.json',dict(contexts=contexts,prefixes=prefixes,responses=responses))
    study.write(ROOT/'experiments/reports/paper_v12/audit.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('completed','pending')},indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
