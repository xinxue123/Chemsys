"""Canonical public inputs and immutable raw Agent replies, not seed-specific plans."""
from copy import deepcopy
import json
from pathlib import Path
from filelock import FileLock
from . import pointwise_hte_llm as llm,reasoning_bo_data as data
from . import stable_complete_policy as policy
from .suzuki_domain import digest


def semantic(value):
    if isinstance(value,(list,tuple)):return [semantic(v) for v in value]
    if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in ('context_sha256','protocol_sha256','variant','run_id')}
    return value


def normalize(public,pool):
    value=semantic(public);ix={k:i for i,k in enumerate(pool['ids'])}
    def visit(v):
        if isinstance(v,dict):
            for k,child in v.items():
                if k=='candidates' and isinstance(child,list) and child and all(isinstance(r,dict) and 'candidate_id' in r for r in child):
                    child.sort(key=lambda r:tuple(pool['points'][ix[r['candidate_id']]][a] for a in pool['axes']))
                visit(child)
        elif isinstance(v,list):
            for child in v:visit(child)
    visit(value)
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def find_archive(root,context,exclude):
    """Earliest valid response for exactly the same system/public/own prefix.

    Reads no objective after the requested prefix; never ranks by achieved yields.
    """
    matches=[]
    for p in (root/'card_shared').glob('*/context.json'):
        if p.parent==exclude or not (p.parent/'validated.json').exists():continue
        old=data.read(p)
        if any(old.get(k)!=context.get(k) for k in ('task','phase','system_sha256','observed_prefix','public')):continue
        answer=data.read(p.parent/'validated.json')
        attempt=p.parent/f"attempt{answer.get('accepted_attempt',1):02d}"/'response.json'
        # Only a real original response can seed the cache, not an import of it.
        if not attempt.exists():continue
        timestamp=data.read(attempt).get('created')
        if type(timestamp) not in (int,float):continue
        matches.append((timestamp,p.parent.name,p.parent/'validated.json'))
    return min(matches)[2] if matches else None


def request_for(study,variant,cache):
    root,_=study.folders(variant)
    cfg={k:v for k,v in study.config(variant).items() if k not in ('id','tie_seed')}
    recipe=digest({'request_source':data.filehash(__file__),'study_source':data.filehash(study.__file__),
        'policy_source':data.filehash(study.policy.__file__),'bridge_source':data.filehash(study.bridge.__file__),
        'engine_source':data.filehash(study.engine.__file__),'pool':digest(cache.pool),'config':cfg})
    def request(client,public,state,system,validator,phase):
        public=deepcopy(public)
        if phase=='refresh':
            if state.get('fit_recovery_seen',False):public['GP_summary']=study.engine.summary(cache,state)
            else:
                public['GP_summary']=policy.grouped_summary(cache,state)
                public['GP_summary_schema']='Complete exact-score classes, stopping after at least16 represented points; no truncated tie class.'
            if state['validations']:public['controlled_validation_results']=deepcopy(state['validations'])
            if state['recombination_calls']:
                public['model_challenge_results']=deepcopy(state['recombination_calls'])
                public['model_challenge_scope']='Joint context counterexamples; do not attribute a multi-factor result to one axis alone.'
        public.pop('normal_BO_reference_id',None);public.pop('normal_R2_proposal_id',None)
        if public.get('B4_diagnosis') is not None:
            public['B4_diagnosis'].pop('candidate_id',None);public['B4_diagnosis'].pop('candidate_index',None)
            public['B4_diagnosis']['witness_scope']='Maximum feasible acquisition value retained; arbitrary tied witness identity omitted.'
        public=normalize(public,cache.pool);system_hash=digest(system)
        context={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
            'phase':phase,'seed':0,'variant':variant,'protocol_sha256':state['protocol_sha256'],'system_sha256':system_hash}
        context['context_sha256']=digest(context);key=context['context_sha256'];folder=root/'requests'/key
        data.write(folder/'context.json',context)
        if (folder/'validated.json').exists():
            answer=data.read(folder/'validated.json');assert answer['plan']==validator(answer['raw']);return answer,key
        shared={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
            'phase':phase,'seed':0,'advisor_recipe':recipe,'system_sha256':system_hash}
        shared['context_sha256']=digest(shared);directory=study.ROOT/'card_shared'/shared['context_sha256'];directory.mkdir(parents=True,exist_ok=True)
        with FileLock(str(directory/'card.lock'),timeout=900):
            if (directory/'context.json').exists():assert data.read(directory/'context.json')==shared
            else:
                data.write(directory/'context.json',shared)
                imported=find_archive(study.ROOT,shared,directory)
                if imported is not None:
                    saved=data.read(imported);plan=validator(saved['raw'])
                    data.write(directory/'validated.json',{**saved,'plan':plan,'context_sha256':shared['context_sha256']})
                    data.write(directory/'seed_response_source.json',{'source':imported.as_posix(),'source_sha256':data.filehash(imported),
                        'rule':'earliest identical system/public/own prefix response, independent of results'})
                data.write(directory/'owner.json',{'own_context_sha256':key,'variant':variant,'new_api':imported is None})
            source=directory/'validated.json'
            if source.exists():
                saved=data.read(source);assert saved['context_sha256']==shared['context_sha256']
                answer={**saved,'plan':validator(saved['raw'])}
            else:answer=llm.nominal.get(client,shared,directory,system,validator)
            owner=data.read(directory/'owner.json')
            info={'source':source.as_posix(),'source_sha256':data.filehash(source),'system_sha256':system_hash,
                'source_context_sha256':shared['context_sha256'],'new_api':owner['own_context_sha256']==key and owner['new_api'],
                'raw_reply_revalidated_for_own_decision':True}
            data.write(folder/'advisor_source.json',info)
            if not info['new_api']:data.write(folder/'reused.json',{'source':source.as_posix(),'source_sha256':info['source_sha256']})
            answer={**answer,'context_sha256':key};data.write(folder/'validated.json',answer)
        return answer,key
    return request
