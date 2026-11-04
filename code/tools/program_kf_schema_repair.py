"""Document one metadata-only repair without changing a valid rank forecast."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from program_kf_weight import study,policy,MODEL,EFFORT,VARIANT
from v13_origin_ablation import queue
from reasoning_memory_v12.llm import validate_insight


def prepare(folder):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(study.base().resolve()/'responses'):raise ValueError('Wrong namespace')
    if (folder/'entry.json').exists():raise ValueError('Accepted responses cannot be repaired')
    ctx=queue.read(folder/'context.json');original=queue.read(folder/'a1/response.raw.json')
    r=queue.read(folder/'a1/request.json');ids=[x['candidate_id'] for x in ctx['public']['candidate_menu']]
    pool=study.frozen.pool_for(ROOT,ctx['public']['task'])
    first=validate_insight(ROOT,original['insight'],ids)
    policy.predictions(original,pool,ids,ctx['public']['own_observations'])
    for a in (1,2,3):
        validation=queue.read(folder/f'a{a}/validation.json')
        if validation['status']!='SCHEMA_REJECTED' or validation['error']!='Invalid or duplicate probability':
            raise ValueError('Exception only covers unmatched Note probability metadata')
        if validate_insight(ROOT,queue.read(folder/f'a{a}/response.raw.json')['insight'],ids)!=first:
            raise ValueError('Proposal lock changed')
    expected=[dict(id=n['id'],version=n['version']) for n in ctx['public']['retrieved_notes']['knowledge']
        if first in n['covered_candidate_ids'] and n['recommendation']['permission']!='withdrawn']
    if expected:raise ValueError('This exception only removes out-of-scope assessments; expected list must be empty')
    exception=dict(kind='schema_only_transport_exception',created_utc=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=queue.read(study.base()/'protocol.json')['protocol_sha256'],
        reason='Three generic repairs failed to remove an out-of-scope Note assessment.',
        no_algorithm_parameter_or_budget_change=True,new_oracle_queries=0,maximum_additional_attempts=1,
        original_a1_response_sha256=queue.file_sha(folder/'a1/response.raw.json'),selected_id_locked=first,
        candidate_predictions_locked=original['decision_card']['predictions'],expected_applicable_notes=expected,
        preserve_all_other_fields=True,helper_sha256=queue.file_sha(__file__))
    # The exception is append-only and registered before a new responder is called.
    ep=folder/'schema_exception.json'
    if ep.exists():exception=queue.read(ep)
    else:queue.write_once(ep,exception)
    public=dict(original=ctx['public'],original_response=original,
        schema_repair_only='The official proposed point is outside EVERY supplied Note scope. The assessments list must therefore be EMPTY. Preserve every other original response field, all five probabilities, candidate order and the official proposed point exactly. Remove only the invalid out-of-scope assessments. No new forecast or decision is requested.',
        selected_id_must_remain=first,candidate_predictions_must_remain=original['decision_card']['predictions'],
        expected_applicable_note_versions=[])
    opaque=study.control.digest(dict(exception=queue.file_sha(ep),attempt=4))[:24]
    io=ROOT/'experiments/agent_io'/VARIANT/opaque;af=folder/'a4'
    queue.write_once(io/'input.json',dict(system=ctx['system'],public=public))
    meta={k:r[k] for k in ('job','cell','kind','query','context_sha256')}
    meta.update(attempt=4,input=study.rel(io/'input.json'),output=study.rel(io/'output.json'),
        input_sha256=queue.file_sha(io/'input.json'),model=MODEL,effort=EFFORT,fork_turns='none',
        schema_exception_sha256=queue.file_sha(ep))
    meta['request_sha256']=study.control.digest(meta);queue.write_once(af/'request.json',meta)
    queue.write_once(af/'invocation.json',dict(task_name=f'oa_{opaque}_choose_4',fork_turns='none',model=MODEL,
        reasoning_effort=EFFORT,message=queue.INSTRUCTIONS.format(input=(io/'input.json').as_posix(),output=(io/'output.json').as_posix())))
    return dict(status='SCHEMA_REPAIR_PREPARED',request=str(af/'request.json'))


def finalize(folder):
    folder=Path(folder).resolve();exception=queue.read(folder/'schema_exception.json')
    if exception['helper_sha256']!=queue.file_sha(__file__):raise ValueError('Repair helper changed')
    raw=queue.read(folder/'a4/response.raw.json');ctx=queue.read(folder/'context.json')
    ids=[x['candidate_id'] for x in ctx['public']['candidate_menu']]
    if validate_insight(ROOT,raw['insight'],ids)!=exception['selected_id_locked']:raise ValueError('Proposal changed')
    original=queue.read(folder/'a1/response.raw.json')
    for key in original:
        if key!='assessments' and raw.get(key)!=original[key]:raise ValueError('Repair changed valid field: '+key)
    if raw['assessments']!=[]:raise ValueError('Out-of-scope assessments persist')
    rp=folder/'a4/request.json';r=queue.read(rp);receipt=queue.read(folder/'a4/response_receipt.json')
    if receipt['request_sha256']!=r['request_sha256'] or receipt['raw_sha256']!=queue.file_sha(folder/'a4/response.raw.json'):
        raise ValueError('Repair receipt mismatch')
    queue.write_once(folder/'a4/validation.json',dict(status='VALIDATED_SCHEMA_ONLY',
        first_forecast_vector_preserved=True,proposal_preserved=True,new_outcome_read=False))
    queue.write_once(folder/'entry.json',dict(mode='fresh_subagent',cell=r['cell'],kind=r['kind'],
        context_sha256=r['context_sha256'],raw_path=study.rel(folder/'a4/response.raw.json'),
        raw_sha256=queue.file_sha(folder/'a4/response.raw.json'),request_path=study.rel(rp),
        receipt_path=study.rel(folder/'a4/response_receipt.json'),attempts=4,
        schema_exception_path=study.rel(folder/'schema_exception.json')))
    return dict(status='SCHEMA_ONLY_REPAIR_ACCEPTED',all_valid_forecasts_and_proposal_unchanged=True)


if __name__=='__main__':
    print(json.dumps((prepare if sys.argv[1]=='prepare' else finalize)(sys.argv[2])))
