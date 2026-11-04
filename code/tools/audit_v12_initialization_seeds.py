"""Independent paired BO audit and reuse of the frozen V12 numerical auditor."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks.knowledge_llm import digest
from v12_initialization_seeds import study
from audit_knowledge_rank_v9 import require,gp_hash,close
from audit_reasoning_memory_v12 import audit_run as frozen_v12_audit
from audit_reasoning_memory_v11 import without_cache
from audit_v12_llm_repeats import validate_response_ids,validate_shared_prefixes


def audit_bo(run,pool,protocol):
    state=study.old.read(run/'state.json');require(state['task']==pool.task and state['arm']=='bo' and state['seed']==0,'native BO identity')
    require(state['protocol_sha256']==protocol['protocol_sha256'] and state['status']=='COMPLETE','BO protocol and completion')
    require(len(state['history'])==20 and len(state['decisions'])==16 and len({r['candidate_id'] for r in state['history']})==20,'BO4+16 unique')
    require(not state['hypotheses'] and not state['insights'] and not state['edit_calls'],'pure BO has no LLM state')
    init=study.old.read(run/'initial_commit.json')
    require(init['commit_sha256']==digest({k:v for k,v in init.items() if k!='commit_sha256'})==state['initial_commit_sha256'],'BO initial custody')
    require(init['initial_ids']==list(pool.initial_ids)==[r['candidate_id'] for r in state['history'][:4]],'BO registered initialization')
    previous=None
    for t,d in enumerate(state['decisions'],1):
        c=study.old.read(run/'rounds'/f'{t:02d}_commit.json');pre=c['prestate'];obs=state['history'][t+3]
        require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})==d['commit_sha256'],'BO decision custody')
        require(c['query']==t and c['arm']=='bo' and c['protocol_sha256']==protocol['protocol_sha256'],'BO query identity')
        require(c['prestate_sha256']==digest(pre) and c['history_sha256']==digest(pre['history']),'BO own history hash')
        require(pre['history']==state['history'][:t+3] and len(pre['decisions'])==t-1,'BO chronology')
        if previous is not None:require(without_cache(previous)==without_cache(pre),'BO full feedback reconstruction')
        with np.load(run/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        require(gp_hash(a)==c['GP_sha256']==c['GP_sha256_after_ranking'],'BO original GP arrays')
        available=np.array([key not in {r['candidate_id'] for r in pre['history']} for key in pool.ids])
        np.testing.assert_array_equal(a['available'],available)
        ties=study.v12.gpmod.semantic_priorities(pool,t,0);np.testing.assert_array_equal(a['tie_priority'],ties)
        js=np.flatnonzero(available);ordered=js[np.lexsort((ties[js],-a['base_logei'][js]))][:5];selected=int(ordered[0])
        physical=sorted(map(int,ordered),key=lambda j:pool.features[j]);perm=np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)
        require(c['ordered_ids']==[pool.ids[int(j)] for j in ordered] and c['display_ids']==[pool.ids[physical[j]] for j in perm],'BO native top5/display')
        require(c['selected_id']==c['BO_reference_id']==pool.ids[selected]==obs['candidate_id'],'BO independent acquisition argmax')
        require(c['mode']=='base_bo' and c['opportunity_gap']==0 and not c['outside_V8_gate'],'BO unrestricted acquisition winner')
        close(c['raw_qLogEI'],a['base_logei'][ordered],'BO native acquisition values')
        require(c['LLM_raw'] is None and c['LLM_context_sha256'] is None and c['refresh'] is None,'no BO LLM request')
        require(not c['hypotheses_at_commit'] and not c['predictions'] and not c['reviews'],'no knowledge in BO')
        require(c['ranking'] is None and c['fixed_initial_weight_selected_id'] is None and not c['KF_changes_selection_from_fixed_weights'],'no BO rank fusion')
        before=max(r['y'] for r in pre['history'])
        expected=dict(query=t,selected_id=pool.ids[selected],mode='base_bo',BO_reference_id=pool.ids[selected],
            raw_value=obs['raw_value'],y=obs['y'],incumbent_before_y=before,incumbent_after_y=max(before,obs['y']),
            KF_feedback=[],commit_sha256=c['commit_sha256'],KF_changes_selection_from_fixed_weights=False)
        require(d==expected,'BO entire feedback record')
        post=deepcopy(pre);post['history'].append(obs);post['decisions'].append(expected);post['status']='COMPLETE' if t==16 else 'RUNNING';previous=post
    require(without_cache(previous)==without_cache(state),'BO final state reconstruction')
    require(state['cache_counters']['operator_compiles_in_loop']==0,'BO loop compilation')
    return dict(task=pool.task,arm='bo',rounds=16,forecasts=0)


def audit_pair(root,task,seed):
    root=Path(root);p=study.check(root);base=study.pair_base(root,task,seed);launch=study.old.read(base/'launch.json')
    require(launch['task']==task and launch['initialization_seed']==seed and launch['protocol_sha256']==p['protocol_sha256'],'pair launch identity')
    require(launch['LLM_cache_initially_empty'] and launch['original_response_files_copied']==0 and not launch['public_seed_tag'],'fresh response namespace')
    require(launch['started_utc']>=p['created_utc'],'registered before launch')
    with study.redirected(root,task,seed,p,redirect_gp=False) as pool:
        bo=study.run_path(root,task,seed,'bo');mem=study.run_path(root,task,seed,'memory')
        rows=[audit_bo(bo,pool,p),frozen_v12_audit(root,task,'memory')]
        bstate=study.old.read(bo/'state.json');mstate=study.old.read(mem/'state.json')
        require(bstate['history'][:4]==mstate['history'][:4],'paired initial observations')
        pf=study.old.read(study.directory(root)/'preflight'/task/f'seed_{seed}/receipt.json')
        require(bstate['history'][:4]==pf['initial_history'] and list(pool.initial_ids)==launch['initial_ids']==pf['initial_ids'],'paired registry/preflight inputs')
        paths=[bo/'rounds/01_GP.npz',mem/'rounds/01_GP.npz',study.directory(root)/'preflight'/task/f'seed_{seed}/01_GP.npz']
        with np.load(paths[0],allow_pickle=False) as z:reference={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority')}
        for path in paths[1:]:
            with np.load(path,allow_pickle=False) as z:
                for key,value in reference.items():np.testing.assert_array_equal(value,z[key])
        require(gp_hash(reference)==pf['GP_sha256'],'paired first GP matches preflight')
    contexts=[];responses=[];prefixes=[];expected_keys=[]
    for arm in study.ARMS:
        run=study.run_path(root,task,seed,arm)
        for t in range(1,17):
            c=study.old.read(run/'rounds'/f'{t:02d}_commit.json')
            prefix=sorted([dict(candidate_id=r['candidate_id'],y=r['y']) for r in c['prestate']['history']],key=lambda r:r['candidate_id'])
            prefixes.append(dict(task=task,initialization_seed=seed,arm=arm,query=t,prefix_sha256=digest(prefix),GP_sha256=c['GP_sha256'],
                ordered_ids=c['ordered_ids'],display_ids=c['display_ids']))
            if arm=='bo':continue
            for kind,key in [('memory',c['memory_context_sha256']),('selection',c['LLM_context_sha256'])]:
                if key is None:continue
                expected_keys.append(key);folder=base/'requests'/key
                ctx=study.old.read(folder/'context.json');valid=study.old.read(folder/'validated.json')
                require(study.old.read(folder/'attempt1/request.json')==dict(system=ctx['system'],public=ctx['public']),'exact unsalted initial request')
                require('initialization_seed' not in ctx['public'] and 'protocol_sha256' not in ctx['public'],'seed not injected in prompt')
                physical={study.old.read(path)['id'] for path in folder.glob('attempt*/response.json')}
                require(valid['response_id'] in physical,'validated response physically present')
                contexts.append(dict(task=task,initialization_seed=seed,kind=kind,query=t,context_sha256=key,response_id=valid['response_id'],
                    raw_sha256=digest(valid['raw']),selected_id=valid['plan']['selected_id'] if kind=='selection' else None,attempts=valid['attempts']))
    require(len(expected_keys)==len(set(expected_keys))==16+len(mstate['edit_calls']),'one committed context per decision/edit')
    require({x.parent.name for x in (base/'requests').glob('*/validated.json')}==set(expected_keys),'no discarded validated requests')
    for path in (base/'requests').glob('*/attempt*/response.json'):
        r=study.old.read(path);require(path.parent.parent.name in expected_keys,'HTTP request belongs to committed context')
        responses.append(dict(task=task,initialization_seed=seed,path=path.relative_to(root).as_posix(),response_id=r['id'],
            model=r.get('resolved_model',r.get('model')),usage=r.get('usage',{}),attempt=path.parent.name))
    return dict(rows=[dict(r,initialization_seed=seed) for r in rows],contexts=contexts,responses=responses,prefixes=prefixes)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=9);args=ap.parse_args()
    p=study.check(ROOT,verify_controls=True);rows=[];contexts=[];responses=[];prefixes=[];failures=[];historical_ids=set()
    for name in p['protected_controls_sha256']:
        if '/requests/' in name and '/attempt' in name and name.endswith('/response.json'):
            r=study.old.read(ROOT/name);key=r.get('id',r.get('response_id'))
            if isinstance(key,str):historical_ids.add(key)
    with ProcessPoolExecutor(max_workers=min(args.workers,9)) as ex:
        futures={ex.submit(audit_pair,str(ROOT),task,seed):(task,seed) for task,seed in study.PAIRS}
        for future,key in futures.items():
            try:
                r=future.result();rows+=r['rows'];contexts+=r['contexts'];responses+=r['responses'];prefixes+=r['prefixes']
            except Exception as e:failures.append(dict(task=key[0],initialization_seed=key[1],kind=type(e).__name__,message=str(e)))
    unique=None;shared=None
    try:
        unique=validate_response_ids(responses,historical_ids);shared=validate_shared_prefixes(prefixes)
        require(len(contexts)<=180,'logical LLM budget')
    except Exception as e:failures.append(dict(kind=type(e).__name__,message=str(e)))
    out=ROOT/'experiments/reports/knowledge_loop_20261101'
    study.write(out/'v12_seeds_request_evidence.json',dict(contexts=contexts,responses=responses,prefixes=prefixes,historical_response_ids_checked=len(historical_ids)))
    result=dict(status='PASS_LISTED_CHECKS' if not failures and len(rows)==18 else 'FAIL',rows=rows,failures=failures,
        protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],
        auditor_sha256=study.old.sha(__file__),reused_V12_auditor_sha256=study.old.sha(ROOT/'code/tools/audit_reasoning_memory_v12.py'),
        rounds=sum(r['rounds'] for r in rows),forecasts=sum(r['forecasts'] for r in rows),paired_initializations_checked=len(rows)//2,
        unique_new_HTTP_response_ids=unique,identical_history_GP_menu_groups_checked=shared,protected_control_files=len(p['protected_controls_sha256']),
        new_oracle_queries=0,new_API_calls=0,new_GP_fits=0,
        boundary='Independent BO argmax and full state; frozen independent V12 numeric/chronology auditor; paired first arrays, registered initial inputs, isolated request identities. No independent GP refit or service-side/cache/weight attestation.')
    study.write(out/'v12_seeds_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='PASS_LISTED_CHECKS':raise SystemExit(1)


if __name__=='__main__':main()
