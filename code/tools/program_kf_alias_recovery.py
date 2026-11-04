"""Recover an alias caller without rewriting another job's owned request."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import json,sys,time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/tools'),str(ROOT/'code/analysis'),str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import program_kf_operations as operations
from program_kf_weight import study
from v13_origin_ablation import queue
from audit_v13_origin_ablation import normalize_public

NATIVE_REQUEST=operations.REQUEST


def synchronized_request(root,tag,kind,q,system,public,validator,**kwargs):
    job=queue.CURRENT
    if job['id']!='trace_perera2_fixed':raise ValueError('Recovery is scoped to the failed alias job')
    other=study.base()/'runs/trace_perera2_dynamic/state.json'
    own_history=public['own_observations']
    while True:
        peer=queue.read(other)
        if len(peer['history'])>=q-1:break
        time.sleep(.2)
    same_prefix=[h['candidate_id'] for h in peer['history'][:q-1]]==[h['candidate_id'] for h in own_history]
    if same_prefix:
        # Schedule behind the existing owner while the physical prefix agrees.
        # No peer outcome or future observation is supplied to the forecaster.
        cp=study.base()/'calls/trace_perera2_dynamic'/f'{q:02}_{kind}'/'context.json'
        while not cp.exists():
            peer=queue.read(other)
            if len(peer['history'])>=q and kind=='edit':
                raise ValueError('Matched-prefix editor schedule differs')
            time.sleep(.2)
        original=queue.read(cp)
        if original['system']!=system or normalize_public(original['public'])!=normalize_public(public):
            raise ValueError('Matched physical prefix has different scientific values')
        public=original['public']
    ctx=queue.context(system,public)
    folder=queue.response_dir(ROOT,job['cell'],kind,ctx)
    owner=folder/'a1/request.json'
    if owner.exists() and queue.read(owner)['job']!=job['id']:
        while not (folder/'entry.json').exists():time.sleep(.2)
    return NATIVE_REQUEST(root,tag,kind,q,system,public,validator,**kwargs)


def main():
    p=study.check(True);operations.check()
    state=queue.read(study.base()/'runs/trace_perera2_fixed/state.json')
    reg=dict(kind='shared_response_alias_ownership_IO_only',registered_utc=datetime.now(timezone.utc).isoformat(),
        helper_sha256=queue.file_sha(__file__),scientific_protocol_sha256=p['protocol_sha256'],
        resumed_job='trace_perera2_fixed',committed_queries_before_recovery=len(state['history']),
        cause='Immutable shared a1 request belongs to the other job; never overwrite it.',
        scientific_rules_and_budget_unchanged=True,only_reuse_same_scientific_values_and_exact_prepared_input=True)
    rp=study.base()/'alias_recovery_registration.json'
    if rp.exists():
        old=queue.read(rp)
        if old['helper_sha256']!=queue.file_sha(__file__):raise ValueError('Alias recovery helper changed')
    else:queue.write_once(rp,reg)
    job=next(j for j in p['jobs'] if j['id']=='trace_perera2_fixed')
    with patch.object(operations,'REQUEST',side_effect=synchronized_request):
        result=operations.resilient_job(job)
    study.check(True);print(json.dumps(result))


if __name__=='__main__':main()
