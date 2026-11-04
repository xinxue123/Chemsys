"""Frozen V12 per-round numerical audit plus independent-response custody."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks.knowledge_llm import digest
from v12_llm_repeats import study
from audit_reasoning_memory_v12 import audit_run as frozen_audit


def audit_one(root,task,rep):
    root=Path(root);p=study.check(root);base=study.replicate_base(root,task,rep)
    launch=study.old.read(base/'launch.json')
    assert launch['protocol_sha256']==p['protocol_sha256'] and launch['task']==task and launch['replicate']==rep
    assert launch['LLM_cache_initially_empty'] and launch['original_V12_response_files_copied']==0 and not launch['public_replicate_tag']
    assert launch['started_utc']>=p['created_utc']
    with study.redirected_v12(root,task,rep,p,redirect_gp=False):row=frozen_audit(root,task,'memory')
    assert row['rounds']==16 and row['category_insertions']==0
    state=study.old.read(study.run_path(root,task,rep)/'state.json')
    expected_keys=[];contexts=[];prefixes=[]
    for t in range(1,17):
        c=study.old.read(study.run_path(root,task,rep)/'rounds'/f'{t:02d}_commit.json')
        prefix=sorted([dict(candidate_id=r['candidate_id'],y=r['y']) for r in c['prestate']['history']],key=lambda r:r['candidate_id'])
        prefixes.append(dict(task=task,replicate=rep,query=t,prefix_sha256=digest(prefix),GP_sha256=c['GP_sha256'],
            ordered_ids=c['ordered_ids'],display_ids=c['display_ids']))
        for kind,key in [('memory',c['memory_context_sha256']),('selection',c['LLM_context_sha256'])]:
            if key is None:continue
            expected_keys.append(key);folder=base/'requests'/key
            ctx=study.old.read(folder/'context.json');valid=study.old.read(folder/'validated.json')
            first_request=study.old.read(folder/'attempt1/request.json')
            assert first_request==dict(system=ctx['system'],public=ctx['public'])
            assert 'replicate' not in ctx['public'] and 'protocol_sha256' not in ctx['public']
            selected=(valid['plan']['selected_id'] if kind=='selection' else None)
            contexts.append(dict(task=task,replicate=rep,query=t,kind=kind,context_sha256=key,response_id=valid['response_id'],
                raw_sha256=digest(valid['raw']),selected_id=selected,attempts=valid['attempts']))
    assert len(expected_keys)==len(set(expected_keys))==16+len(state['edit_calls'])
    actual_keys={path.parent.name for path in (base/'requests').glob('*/validated.json')}
    assert actual_keys==set(expected_keys)
    responses=[]
    for path in (base/'requests').glob('*/attempt*/response.json'):
        response=study.old.read(path)
        assert path.parent.parent.name in actual_keys
        responses.append(dict(task=task,replicate=rep,path=path.relative_to(root).as_posix(),response_id=response['id'],
            context_sha256=path.parent.parent.name,model=response.get('resolved_model',response.get('model')),
            usage=response.get('usage',{}),attempt=path.parent.name))
    return dict(row=dict(row,replicate=rep,logical_calls=len(expected_keys),HTTP_responses=len(responses)),contexts=contexts,prefixes=prefixes,responses=responses)


def validate_response_ids(responses,historical_ids):
    ids=[r['response_id'] for r in responses]
    if any(not isinstance(key,str) or not key for key in ids):raise AssertionError('Missing response identity')
    if len(ids)!=len(set(ids)):raise AssertionError('A response ID was reused across independent HTTP records')
    if set(ids)&historical_ids:raise AssertionError('Historical response ID reused')
    return len(ids)


def validate_shared_prefixes(prefixes):
    groups=defaultdict(list)
    for r in prefixes:groups[(r['task'],r['prefix_sha256'])].append(r)
    checked=0
    for rows in groups.values():
        if len(rows)<2:continue
        assert len({r['GP_sha256'] for r in rows})==1,'Same measured set produced different GP/acquisition'
        assert len({tuple(r['ordered_ids']) for r in rows})==1,'Same measured set produced different menu'
        assert len({tuple(r['display_ids']) for r in rows})==1,'Same measured set produced different display order'
        checked+=1
    return checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=9);args=ap.parse_args()
    p=study.check(ROOT);rows=[];contexts=[];prefixes=[];responses=[];failures=[];history_ids=set()
    for name,sha in p['protected_controls_sha256'].items():
        path=ROOT/name
        if study.old.sha(path)!=sha:failures.append('Protected artifact drift: '+name)
        if '/requests/' in name and '/attempt' in name and name.endswith('/response.json'):
            value=study.old.read(path);key=value.get('id',value.get('response_id'))
            if isinstance(key,str):history_ids.add(key)
    with ProcessPoolExecutor(max_workers=min(args.workers,9)) as ex:
        futures={ex.submit(audit_one,str(ROOT),task,rep):(task,rep) for task,rep in study.RUNS}
        for future,key in futures.items():
            try:
                result=future.result();rows.append(result['row']);contexts+=result['contexts'];prefixes+=result['prefixes'];responses+=result['responses']
            except Exception as error:failures.append(str(key)+': '+type(error).__name__+': '+str(error))
    unique_ids=None;shared=None
    try:
        unique_ids=validate_response_ids(responses,history_ids)
        # Include original V12 only for same-prefix numerical checks, not the new n=3 sample.
        for task in study.TASKS:
            for t in range(1,17):
                c=study.old.read(study.v12.directory(ROOT)/'runs'/f'{task}_memory/rounds'/f'{t:02d}_commit.json')
                pre=sorted([dict(candidate_id=r['candidate_id'],y=r['y']) for r in c['prestate']['history']],key=lambda r:r['candidate_id'])
                prefixes.append(dict(task=task,replicate='original',query=t,prefix_sha256=digest(pre),GP_sha256=c['GP_sha256'],ordered_ids=c['ordered_ids'],display_ids=c['display_ids']))
        shared=validate_shared_prefixes(prefixes)
        for task in study.TASKS:
            first=[r for r in contexts if r['task']==task and r['query']==1 and r['kind']=='memory']
            assert len(first)==3 and len({r['context_sha256'] for r in first})==1
            assert len({r['response_id'] for r in first})==3
        assert sum(r['logical_calls'] for r in rows)<=180
    except Exception as error:failures.append('Cross-replica audit: '+type(error).__name__+': '+str(error))
    out=ROOT/'experiments/reports/knowledge_loop_20261101'
    evidence=dict(contexts=contexts,responses=responses,prefixes=prefixes,historical_response_ids_checked=len(history_ids))
    study.write(out/'v12_repeats_request_evidence.json',evidence)
    result=dict(status='PASS_LISTED_CHECKS' if not failures and len(rows)==9 else 'FAIL',rows=rows,failures=failures,
        protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],
        auditor_sha256=study.old.sha(__file__),reused_numeric_auditor_sha256=study.old.sha(ROOT/'code/tools/audit_reasoning_memory_v12.py'),
        rounds=sum(r['rounds'] for r in rows),forecasts=sum(r['forecasts'] for r in rows),
        protected_control_files=len(p['protected_controls_sha256']),unique_new_HTTP_response_ids=unique_ids,
        identical_history_GP_menu_groups_checked=shared,isolated_response_namespaces=len(rows),
        new_oracle_queries=0,new_API_calls=0,new_GP_fits=0,
        boundary='Original independent V12 numerical/chronology auditor plus unique HTTP IDs, empty launch namespaces, exact prompts, and shared-prefix GP/menu identity. No independent GP refits, server-side cache/weights attestation, or external trusted timestamp.')
    study.write(out/'v12_repeats_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
