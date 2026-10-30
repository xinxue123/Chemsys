"""Immutable first-response cards for identical public inputs and actual prefixes."""
from copy import deepcopy
from pathlib import Path
from filelock import FileLock
from . import stable_followup_study as previous
from . import stable_card_policy as policy
from . import pointwise_hte_llm as llm
from . import reasoning_bo_data as data
from .suzuki_domain import digest


def semantic(value):
    if isinstance(value,list):return [semantic(v) for v in value]
    if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in ('context_sha256','protocol_sha256','variant','run_id')}
    return value


def request_for(study,variant,cache):
    root,_=study.folders(variant);legacy=previous.request_for(variant,cache)
    recipe=digest({'request_source':data.filehash(__file__),'study_source':data.filehash(study.__file__),
        'policy_source':data.filehash(policy.__file__),'rule':'immutable first validated reply; no outcome selection'})
    def request(client,public,state,system,validator,phase):
        if state.get('fit_recovery_seen',False):return legacy(client,public,state,system,validator,phase)
        public=deepcopy(public)
        if phase=='refresh':
            public['GP_summary']=policy.grouped_summary(cache,state)
            public['GP_summary_schema']='Complete exact-score classes, stopping after at least16 represented points; no truncated tie class.'
            if state['validations']:public['controlled_validation_results']=deepcopy(state['validations'])
        # These IDs merely witness equal acquisition scores, not chemical evidence.
        public.pop('normal_BO_reference_id',None);public.pop('normal_R2_proposal_id',None)
        diagnosis=public.get('B4_diagnosis')
        if diagnosis is not None:
            diagnosis.pop('candidate_id',None);diagnosis.pop('candidate_index',None)
            diagnosis['witness_scope']='Maximum feasible acquisition value retained; arbitrary tied witness identity omitted.'
        public=semantic(public);system_hash=digest(system)
        context={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
            'phase':phase,'seed':0,'variant':variant,'protocol_sha256':state['protocol_sha256'],'system_sha256':system_hash}
        context['context_sha256']=digest(context);key=context['context_sha256'];folder=root/'requests'/key
        data.write(folder/'context.json',context)
        if (folder/'validated.json').exists():
            answer=data.read(folder/'validated.json');assert answer['plan']==validator(answer['raw']);return answer,key
        shared={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
            'phase':phase,'seed':0,'advisor_recipe':recipe,'system_sha256':system_hash}
        shared['context_sha256']=digest(shared)
        directory=study.ROOT/'card_shared'/shared['context_sha256'];directory.mkdir(parents=True,exist_ok=True)
        with FileLock(str(directory/'card.lock'),timeout=900):
            if (directory/'context.json').exists():assert data.read(directory/'context.json')==shared
            else:
                data.write(directory/'context.json',shared)
                data.write(directory/'owner.json',{'own_context_sha256':key,'variant':variant})
            answer=llm.nominal.get(client,shared,directory,system,validator)
            source=directory/'validated.json';owner=data.read(directory/'owner.json')
            info={'source':source.as_posix(),'source_sha256':data.filehash(source),'system_sha256':system_hash,
                'source_context_sha256':shared['context_sha256'],'new_api':owner['own_context_sha256']==key}
            data.write(folder/'advisor_source.json',info)
            if not info['new_api']:data.write(folder/'reused.json',{'source':source.as_posix(),'source_sha256':info['source_sha256']})
            answer={**answer,'context_sha256':key};data.write(folder/'validated.json',answer)
        return answer,key
    return request
