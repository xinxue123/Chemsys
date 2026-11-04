"""Registered IO recovery only; keep all frozen scientific rules unchanged."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import argparse,json,sys,time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/analysis'),str(ROOT/'code/tools')]
from program_kf_weight import study
from v13_origin_ablation import queue
from audit_v13_origin_ablation import normalize_public
import program_kf_schema_repair as repair

REQUEST=queue.request


def register():
    p=study.check(True)
    value=dict(kind='IO_recovery_only',registered_utc=datetime.now(timezone.utc).isoformat(),
        scientific_protocol_sha256=p['protocol_sha256'],new_full_trace_queries_at_registration=0,
        helper_sources={study.rel(Path(__file__)):queue.file_sha(__file__),
            study.rel(Path(repair.__file__)):queue.file_sha(repair.__file__)},
        changes=['Reuse an already prepared exact input after checkpoint reload when JSON key order alone changes.',
            'Wait for a locked-forecast annotation repair, then resume the same job from its last committed observation.'],
        scoring_GP_weights_parameters_and_budget_unchanged=True)
    for job in p['jobs']:
        if job['phase']=='traces' and (study.base()/'runs'/job['id']/'state.json').exists():
            raise ValueError('Register IO adapter before full trajectories')
    queue.write_once(study.base()/'IO_recovery_registration.json',value)
    return dict(status='REGISTERED_IO_RECOVERY',scientific_source_changed=False)


def check():
    r=queue.read(study.base()/'IO_recovery_registration.json')
    for name,h in r['helper_sources'].items():
        if queue.file_sha(ROOT/name)!=h:raise ValueError('IO helper changed')
    return r


def exact_prepared_request(root,tag,kind,q,system,public,validator,**kwargs):
    job=queue.CURRENT
    path=study.base()/'calls'/job['id']/f'{q:02}_{kind}'/'context.json'
    if path.exists():
        original=queue.read(path)
        if original['system']!=system or normalize_public(original['public'])!=normalize_public(public):
            raise ValueError('Checkpoint reload changed scientific input values')
        # Actual original bytes/embedded JSON are retained, not regenerated.
        public=original['public']
    return REQUEST(root,tag,kind,q,system,public,validator,**kwargs)


def repair_pending():
    check();results=[]
    for folder in (study.base()/'responses').glob('*/*'):
        if (folder/'entry.json').exists() or not (folder/'a3/validation.json').exists():continue
        try:
            if not (folder/'schema_exception.json').exists():
                value=repair.prepare(folder)
            elif (folder/'a4/response_receipt.json').exists():
                value=repair.finalize(folder)
            else:continue
            meta=queue.read(folder/'a1/request.json')
            results.append(dict(job=meta['job'],query=meta['query'],folder=str(folder),**value))
        except ValueError as error:
            results.append(dict(status='NEEDS_ROOT_INSPECTION',folder=str(folder),reason=str(error)))
    return dict(metadata_only=True,results=results)


def resilient_job(job):
    check()
    with patch.object(queue,'request',side_effect=exact_prepared_request):
        while True:
            try:return study.run_job(job)
            except ValueError as error:
                if str(error)!='Three schema attempts exhausted; no new benchmark outcome':raise
                waiting=[]
                folder=study.base()/('prefix/'+job['cell'] if job['phase']=='prefix' else 'runs/'+job['id'])
                state=queue.read(folder/'state.json') if (folder/'state.json').exists() else {}
                q=state.get('query',len(state.get('history',[])))+1
                for kind in ('edit','choose'):
                    cp=study.base()/'calls'/job['id']/f'{q:02}_{kind}'/'context.json'
                    if not cp.exists():continue
                    ctx=queue.read(cp)
                    with study.transport_scope(job):entry=queue.response_dir(ROOT,job['cell'],kind,ctx)/'entry.json'
                    if not entry.exists():waiting.append(entry)
                if not waiting:raise ValueError('No recorded input for failed schema attempt')
                while not all(path.exists() for path in waiting):time.sleep(.2)
                queue.write_once(study.base()/'IO_resumes'/f"{job['id']}_{q:02}.json",dict(job=job['id'],query=q,
                    reason='locked_forecast_metadata_repair',entries=[study.rel(path) for path in waiting],
                    extra_experiment_queries=0,scientific_source_changed=False))


def batch(phase):
    check();p=study.check(True)
    assert queue.read(study.base()/'preflight.json')['status']=='PASS'
    if phase=='traces':assert queue.read(study.base()/'prefix_complete.json')['status']=='COMPLETE'
    jobs=[j for j in p['jobs'] if j['phase']==phase];results=[];errors=[]
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures={executor.submit(resilient_job,j):j for j in jobs}
        for f in as_completed(futures):
            try:results.append(f.result())
            except Exception as error:
                e=dict(job=futures[f]['id'],type=type(error).__name__,message=str(error));errors.append(e)
                print('JOB_ERROR '+json.dumps(e),flush=True)
    value=dict(status='COMPLETE' if not errors else 'FAILED',phase=phase,results=results,errors=errors,
               IO_recovery_registration_sha256=queue.file_sha(study.base()/'IO_recovery_registration.json'))
    study.check(True);queue.write_once(study.base()/f'{phase}_complete.json',value)
    if errors:raise RuntimeError(str(errors))
    return value


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('register','repair','prefix','traces'))
    args=a.parse_args()
    print(json.dumps(register() if args.action=='register' else repair_pending() if args.action=='repair' else batch(args.action)))
