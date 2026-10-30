"""Audit actual provider inputs, first-valid acceptance and typed observations."""
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import csv
import io
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_llm as llm
from chem.benchmarks import reasoning_bo_study as study
from chem.benchmarks.suzuki_domain import digest


def native_prompt(pool,state,context,t):
    comp=data.read(data.ROOT/pool['task_id']/'compass.json')
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['observed_index',*pool['axes'],'yield_percent'])
    lookup={key:i for i,key in enumerate(pool['ids'])}
    for i,h in enumerate(state['history'][:3*t],1):
        point=pool['points'][lookup[h['candidate_id']]]
        writer.writerow([i,*[point[a] for a in pool['axes']],round(h['y']*100,3)])
    history=[{'trial_index':0,'insight':json.dumps(state['initial_insight'],ensure_ascii=False)}]
    history += [{'trial_index':d['iteration'],'insight':json.dumps(d['insight'],ensure_ascii=False)}
        for d in state['decisions'][:t-1]]
    with redirect_stdout(io.StringIO()):concatenated=llm.NATIVE['concatenate_jsonl'](history)
    template=data.read(data.UPSTREAM/'src/prompts/test_prompts.json')['optimization_loop']['en']
    return template.format(**{**comp,'iteration':t-1,'trial_data':stream.getvalue(),
        'insight_history':concatenated,'bo_recommendations':[pool['points'][lookup[key]] for key in context['shortlist_ids']],
        'retrieved_context':None})


def main():
    p=study.check_protocol();counts=Counter();fingerprints=set();models=set()
    pool_cache={t:data.load_pool(t) for t in data.SPECS}
    for folder in (study.ROOT/'requests').iterdir():
        if not (folder/'validated.json').exists():continue
        context=data.read(folder/'context.json');validated=data.read(folder/'validated.json')
        raw=dict(context);expected=raw.pop('context_sha256')
        assert digest(raw)==expected==folder.name==validated['context_sha256']
        assert context['protocol_sha256']==p['protocol_sha256']
        mode=context.get('mode','region');public=context['public']
        task=context['task'] if mode!='region' else public['task'];pool=pool_cache[task]
        seed=public['replicate_seed'] if mode!='region' else context['seed']
        if mode.startswith('optimization_'):
            t=int(mode.split('_')[-1]);state=data.read(study.ROOT/'runs'/f'{task}_s{seed:02d}_ReasoningBO'/'state.json')
            assert len(context['observed_prefix'])==3*t
            assert context['observed_prefix']==state['history'][:3*t]
            assert public['official_prompt']==native_prompt(pool,state,context,t)
            expected_plan=llm.validate_insight(validated['raw'],pool,context['observed_prefix'],context['shortlist_ids'])
            assert expected_plan==validated['plan']
            counts['native_full_prompt_reconstructions']+=1
            counts['native_observed_rows_checked']+=3*t
        elif mode=='initial_sampling':
            assert context['observed_prefix']==[] and context['shortlist_ids'] is None
            template=data.read(data.UPSTREAM/'src/prompts/test_prompts.json')['initial_sampling']['en']
            assert public['official_prompt']==template.format(**data.read(data.ROOT/task/'compass.json'))
            assert llm.validate_insight(validated['raw'],pool,[],None)==validated['plan']
            counts['native_initializations_checked']+=1
        elif mode=='overview':
            assert context['observed_prefix']==[]
            assert set(validated['raw'])=={'overview'}
            template=data.read(data.UPSTREAM/'src/prompts/test_prompts.json')['generate_overview']['en']
            assert public['official_prompt']==template.format(**data.read(data.ROOT/task/'compass.json'))
            counts['native_overviews_checked']+=1
        else:
            observed=public['observations'];assert len(observed)==3*public['iteration']
            if public['phase']=='initial_knowledge':
                state=data.read(study.ROOT/'initialization'/f'{task}_s{seed:02d}'/'ready.json')
            else:state=data.read(study.ROOT/'runs'/f'{task}_s{seed:02d}_New_R2'/'state.json')
            assert len(state['history'])>=len(observed)
            assert [r['index'] for r in observed]==list(range(1,len(observed)+1))
            assert public['domain']==llm.nominal.domain(pool)
            lookup={key:i for i,key in enumerate(pool['ids'])}
            for row,h in zip(observed,state['history']):
                assert row['point']==llm.nominal.public_point(pool,pool['points'][lookup[h['candidate_id']]])
                assert row['yield_percent']==round(h['y']*100,3)
            _,reverse=llm.nominal.token_maps(pool)
            old=public['current_knowledge']
            knowledge={'confidence':old['confidence'],'clauses':[
                {'id':h['id'],'axis':h['axis'],'allowed':sorted(reverse[h['axis']][v] for v in h['allowed'])}
                for h in old['clauses']]}
            checked=llm.nominal.validate_plan(validated['raw'],pool,knowledge,observed,grounded=True)
            assert checked==validated['plan'] and len(checked['knowledge']['clauses'])<=4
            counts['typed_facts_checked']+=len(checked['checked_facts'])
            counts['regional_plans_checked']+=1
        accepted=validated['accepted_attempt']
        for attempt in range(1,accepted):
            q=folder/f'attempt{attempt:02d}'
            assert (q/'validation_error.json').exists() or (q/'transport_error.json').exists()
        response=data.read(folder/f'attempt{accepted:02d}'/'response.json')
        assert response['model']=='deepseek-flash' and len(response['choices'])==1
        assert response['choices'][0]['finish_reason']=='stop'
        assert response['retained_raw_reasoning'] is False
        assert 'reasoning_content' not in response['choices'][0]['message']
        models.add(response['model']);fingerprints.add(response.get('system_fingerprint'))
        counts['first_valid_acceptances_checked']+=1
    complete=counts['native_full_prompt_reconstructions']==600 and counts['native_initializations_checked']==20
    result={'status':'PASS_COMPLETE_STUDY_INPUTS' if complete else 'PASS_AVAILABLE_INPUTS','protocol_sha256':p['protocol_sha256'],
        'auditor_sha256':data.filehash(__file__),'counts':dict(counts),'response_models':sorted(models),
        'response_system_fingerprints':list(fingerprints),
        'boundary':'Checks supplied data and typed factual evidence; untyped chemical rationales are hypotheses, not verified causal truth. Cannot establish absence of pretraining contamination.'}
    data.write(study.OUT/'input_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
