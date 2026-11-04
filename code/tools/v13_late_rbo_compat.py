"""Metadata-preserving recovery for bare RBO responses with EMPTY V13 sidecars.

No response bytes, candidate sets, ranking or outcomes are changed. Original
schema rejections remain. The first core-valid response is used, never selected
by performance. This external I/O compatibility receipt supplements registration.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
from datetime import datetime,timezone
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study,queue

NATIVE_REQUEST=queue.request


def project(raw):
    if not isinstance(raw,dict):return raw
    required={'comment','keywords','hypotheses'};extras=set(raw)-required
    if required<=set(raw) and extras<= {'knowledge_uses','assessments'} and all(raw[k]==[] for k in extras):
        return {k:raw[k] for k in raw if k in required}
    return raw


def request(root,tag,kind,q,system,public,validator,*,selected_validator=None):
    if 'rbo' not in queue.CURRENT or kind!='choose':
        return NATIVE_REQUEST(root,tag,kind,q,system,public,validator,selected_validator=selected_validator)
    folder=study.base()/'requests'/queue.CURRENT/f'{q:02}_{kind}'
    # Recover a previous rejection without overwriting validation.json or asking
    # another decision model. No oracle result for this query exists yet.
    for attempt in range(1,4):
        af=folder/f'a{attempt}';vp=af/'validation.json'
        if not (af/'response_receipt.json').exists():continue
        if vp.exists() and queue.read(vp)['status']=='SCHEMA_REJECTED':
            raw=queue.read(af/'response.raw.json');core=project(raw)
            if core==raw:continue
            selected=selected_validator(core);plan=validator(core)
            receipt=dict(status='EMPTY_SIDECARS_IGNORED',selected_id_locked=selected,
                request_sha256=queue.read(af/'request.json')['request_sha256'],raw_sha256=queue.file_sha(af/'response.raw.json'),
                core_sha256=queue.digest(core),adapter_sha256=queue.file_sha(__file__),
                original_rejection_retained=True,original_response_bytes_modified=False,new_LLM_call=False,
                selection_rule='first core-valid original response; no outcome consulted')
            queue.write_once(af/'core_projection_validation.json',receipt)
            return dict(raw=raw,plan=plan,context_sha256=queue.digest(dict(system=system,public=public)),
                attempts=attempt,response_receipt=queue.read(af/'response_receipt.json'),
                request_relpath=(af/'request.json').relative_to(ROOT).as_posix(),core_projection=receipt)
    return NATIVE_REQUEST(root,tag,kind,q,system,public,lambda raw:validator(project(raw)),
        selected_validator=(lambda raw:selected_validator(project(raw))) if selected_validator else None)


def recover(job,p):
    folder=study.base()/job['id']/'runs'/job['id']
    while not (folder/'execution_error.json').exists():
        if (folder/'state.json').exists() and queue.read(folder/'state.json')['status']=='COMPLETE':return dict(job=job['id'],status='NATIVE_COMPLETE')
        time.sleep(1)
    with patch.object(queue,'request',side_effect=request):
        result=study.run_job(job,p)
    queue.write_once(folder/'schema_recovery_completion.json',dict(status='COMPLETE',adapter_sha256=queue.file_sha(__file__),
        retained_original_error=True,GP_policy_changed=False,original_responses_modified=False))
    return result


def main():
    p=study.check(True)
    queue.write_once(study.base()/'RBO_empty_sidecar_compat_registration.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),adapter_sha256=queue.file_sha(__file__),
        original_protocol_sha256=p['protocol_sha256'],scope='RBO official-core JSON with empty V13-only side fields',
        rule='deterministic projection; first original core-valid response; raw bytes and schema rejections retained',
        no_GP_candidate_model_or_budget_change=True,no_outcome_based_response_selection=True))
    js=[j for j in p['jobs'] if j['arm']=='rbo'];results=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        fs={ex.submit(recover,j,p):j for j in js}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:results.append(dict(job=fs[f]['id'],status='FAILED',error=type(e).__name__+': '+str(e)))
    queue.write_once(study.base()/'RBO_compat_batch_receipt.json',dict(results=results))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
