"""Read-only V11 audit: independent selection, KF, scopes and chronological state.

Prompt generation and note action schema replay reuse production functions.
This audit never calls a model, reads unqueried outcomes, or refits a GP.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import erfc,sqrt
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks.knowledge_llm import digest
from reasoning_memory_v11 import study,llm
from audit_knowledge_rank_v9 import independent_kf,gp_hash,event_allowed,close,require,scope


def official_winner(insight):
    priority={'high':0,'medium':1,'low':2}
    # Deliberately do not call or compile the official production extractor.
    winning=min(enumerate(insight['hypotheses']),key=lambda item:(priority[item[1]['confidence'].lower()],item[0]))
    return winning[1]['parameter_sets'][0]['candidate_id'],winning[0]


def without_cache(state):return {k:v for k,v in state.items() if k!='cache_counters'}


def check_request(base,key,system,public,raw,protocol):
    context=study.old.read(base/'requests'/key/'context.json');valid=study.old.read(base/'requests'/key/'validated.json')
    require(context==dict(system=system,public=public,model=protocol['model'],reasoning_effort=protocol['reasoning_effort']),
        'exact own-prefix public prompt')
    require(digest(context)==key==valid['context_sha256'] and valid['raw']==raw,'request response custody')
    return valid


def audit_birth(pool,pre,hs,t):
    known={(h['id'],h['version']) for h in pre['hypotheses']};counts=0
    incumbent=min(pre['history'],key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))
    for h in hs:
        if (h['id'],h['version']) in known:continue
        require(h['created_round']==t and h['birth_history_sha256']==digest(pre['history']),'version birth prefix')
        require(h['reference_id']==incumbent['candidate_id'] and h['reference_y']==incumbent['y'],'known incumbent reference')
        rows={}
        for old in pre['hypotheses']:
            if old['focus']!=h['focus'] or old['effect']!=h['effect']:continue
            for e in old['events']+old['inherited_negative']:
                if e['Brier_gain']>=-1e-12 or not all(scope(pool,h,pool.ix[e[k]]) for k in ('point_id','reference_id')):continue
                key=e['query'],e['point_id']
                if key not in rows or e['Brier_gain']<rows[key]['Brier_gain']:rows[key]=e
        negatives=[rows[k] for k in sorted(rows)];require(negatives==h['inherited_negative'],'all applicable negative evidence retained once')
        kf=dict(mean=0.,variance=.25,weight=.4,probability_positive=.5,bad_streak=0,good_streak=0,updates=0,
            local_updates=0,status='active',paused_round=None,retracted_round=None)
        for e in negatives:kf,_=independent_kf(kf,e['p_LLM'],e['p_GP'],e['success'],e['query'],prior=True)
        if negatives:kf.update(status='paused',paused_round=t,good_streak=0,local_updates=0)
        for key,value in kf.items():
            if isinstance(value,float):close(h['kf'][key],value,'inherited KF initial state')
            else:require(h['kf'][key]==value,'inherited lifecycle')
        counts+=len(negatives)
    return counts


def audit_task(root,task):
    root=Path(root);protocol=study.check(root);base=study.directory(root);path=base/'runs'/f'{task}_rbo_memory'
    state=study.old.read(path/'state.json');pool=study.gpmod.load_task(root,task,seed=0)
    require(state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16,'complete4+16')
    require(state['protocol_sha256']==protocol['protocol_sha256'] and state['seed']==0 and state['arm']=='rbo_memory','registered run')
    require(len({r['candidate_id'] for r in state['history']})==20,'20 unique observations')
    control=study.old.read(study.v9.directory(root)/'runs'/f'{task}_bo/state.json')
    require(state['history'][:4]==control['history'][:4],'shared initial observations')
    init=study.old.read(path/'initial_commit.json')
    require(init['commit_sha256']==digest({k:v for k,v in init.items() if k!='commit_sha256'})==state['initial_commit_sha256'],'initial commitment')
    require(init['initial_ids']==list(pool.initial_ids),'input-only initial IDs')
    counts=dict(task=task,rounds=0,forecasts=0,memory_calls=0,independent_official_choices=0,note_citations=0,
        citations_outside_retrieval=0,inherited_negative=0)
    previous=None
    for t,d in enumerate(state['decisions'],1):
        c=study.old.read(path/'rounds'/f'{t:02d}_commit.json');pre=c['prestate'];obs=state['history'][t+3]
        require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})==d['commit_sha256'],'commit hash')
        require(c['query']==t and c['protocol_sha256']==protocol['protocol_sha256'],'registered query')
        require(c['prestate_sha256']==digest(pre) and c['history_sha256']==digest(pre['history']),'prestate hash')
        require(pre['history']==state['history'][:t+3] and len(pre['decisions'])==t-1,'own chronological prefix')
        if previous is not None:require(digest(without_cache(previous))==digest(without_cache(pre)),'feedback to next prestate')
        with np.load(path/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        require(gp_hash(a)==c['GP_sha256']==c['GP_sha256_after_ranking'],'unaltered original GP arrays')
        available=np.array([key not in {r['candidate_id'] for r in pre['history']} for key in pool.ids]);np.testing.assert_array_equal(a['available'],available)
        ties=study.gpmod.semantic_priorities(pool,t,0);np.testing.assert_array_equal(a['tie_priority'],ties)
        js=np.flatnonzero(available);ordered=js[np.lexsort((ties[js],-a['base_logei'][js]))][:5]
        physical=sorted(map(int,ordered),key=lambda j:pool.features[j]);permutation=np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)
        ids=[pool.ids[physical[j]] for j in permutation]
        require(ids==c['display_ids'] and c['ordered_ids']==[pool.ids[j] for j in ordered],'native top5 and neutral display')
        require(c['BO_reference_id']==pool.ids[ordered[0]],'native BO reference');close(c['raw_qLogEI'],a['base_logei'][ordered],'native qLogEI')
        refresh=event_allowed(pool,pre,ordered,t);require(refresh==c['refresh'],'independent event schedule')
        hs=deepcopy(pre['hypotheses'])
        if refresh['allowed']:
            public=llm.memory_public(pool,pre,ids,t,refresh)
            valid=check_request(base,c['memory_context_sha256'],llm.MEMORY_SYSTEM,public,c['memory_raw'],protocol)
            result=llm.validate_memory(c['memory_raw'],pool,pre,ids,t)
            require(result==valid['plan'] and result['reviews']==c['reviews'],'note action schema replay')
            hs=result['hypotheses'];counts['memory_calls']+=1
        else:require(c['memory_raw'] is None and c['memory_context_sha256'] is None and not c['reviews'],'no non-event edit')
        require(hs==c['hypotheses_at_commit'],'frozen note versions');counts['inherited_negative']+=audit_birth(pool,pre,hs,t)
        public=llm.reasoning_public(root,pool,pre,hs,ids,t)
        valid=check_request(base,c['LLM_context_sha256'],llm.REASONING_SYSTEM,public,c['reasoning_raw'],protocol)
        require(public['own_observations']==[dict(llm.previous.point(pool,r['candidate_id']),objective=r['raw_value']) for r in pre['history']],
            'all and only own observed results')
        require(not any(key in public for key in ('mu_y','sd_y','base_logei')),'no current numerical GP fields exposed')
        result=llm.validate_reasoning(root,c['reasoning_raw'],hs,ids,public['retrieved_notes'])
        require(result==valid['plan'] and result['probabilities']==c['forecast_probabilities'],'complete forecast sidecar')
        require(c['reasoning_raw']['insight']==c['LLM_raw'],'official Insight history content')
        key,index=official_winner(c['LLM_raw']);j=pool.ix[key]
        require(key==c['selected_id']==d['selected_id']==obs['candidate_id'] and index==c['winning_hypothesis_index'],'independent official winner')
        require(key in ids and available[j],'all chosen points in native BO menu')
        require(c['mode']==d['mode']==('base_bo' if j==ordered[0] else 'ReasoningBO_memory'),'mode indicates BO agreement only')
        require(c['memory_references']==result['memory_references']==d['memory_references'],'retrieved note citations')
        retrieved={(h['id'],h['version']) for group in ('supporting_notes','cautionary_notes') for h in public['retrieved_notes'][group]}
        visible=retrieved|{(h['id'],h['version']) for h in public['forecast_notes']}
        require(all((r['id'],r['version']) in visible for r in c['memory_references']),'all cited evidence visible before query')
        counts['citations_outside_retrieval']+=sum((r['id'],r['version']) not in retrieved for r in c['memory_references'])
        require(not c['KF_changes_selection_from_fixed_weights'],'no mechanical weight reordering')
        close(c['opportunity_gap'],a['base_logei'][ordered[0]]-a['base_logei'][j],'logged qLogEI gap only')
        expected={(h['id'],h['version']) for h in hs if h['kf']['status']!='retracted' and scope(pool,h,j)}
        actual=[(s['prediction']['id'],s['prediction']['version']) for s in c['predictions']]
        require(set(actual)==expected and len(actual)==len(expected)==len(d['KF_feedback']),'exactly once scoped forecast feedback')
        for sh in c['predictions']:
            pred=sh['prediction'];h=next(h for h in hs if (h['id'],h['version'])==(pred['id'],pred['version']))
            require(digest(pred)==sh['prediction_sha256'] and pred['semantic_sha256']==h['semantic_sha256'],'frozen prospective prediction')
            require(pred['history_sha256']==digest(pre['history']) and pred['GP_sha256']==c['GP_sha256'] and pred['query']==t and pred['point_id']==key,'prospective prefix')
            ref=next(r for r in pre['history'] if r['candidate_id']==h['reference_id'])
            require(ref['y']==h['reference_y']==pred['reference_y'] and ref['candidate_id']==pred['reference_id'],'fixed known reference')
            sd=float(a['sd_predictive_y'][j]);g=.5*erfc(-(a['mu_y'][j]-ref['y'])/(sqrt(2)*sd)) if sd else float(a['mu_y'][j]>ref['y'])
            close(g,pred['p_GP'],'original GP event probability');p=result['probabilities'][h['id']][key]
            require(p==pred['p_LLM'],'prequery forecast');post,measurement=independent_kf(h['kf'],p,g,bool(obs['y']>ref['y']),t)
            e=next(e for e in d['KF_feedback'] if (e['id'],e['version'])==(h['id'],h['version']))
            require(e['before']==h['kf'] and e['event_id']==digest({k:v for k,v in e.items() if k!='event_id'}),'KF custody')
            require(e['prediction_sha256']==sh['prediction_sha256'] and e['point_id']==key and e['query']==t,'feedback identity')
            for name,value in post.items():
                if isinstance(value,float):close(e['after'][name],value,'independent KF posterior')
                else:require(e['after'][name]==value,'independent lifecycle')
            for name,value in measurement.items():
                if isinstance(value,float):close(e[name],value,'independent KF measurement')
                else:require(e[name]==value,'independent feedback reason')
            h['kf']=deepcopy(e['after']);h['events'].append(deepcopy(e));counts['forecasts']+=1
        post=deepcopy(pre);post['hypotheses']=hs;post['history'].append(obs);post['insights'].append(dict(query=t,insight=c['LLM_raw']))
        if refresh['allowed']:post['edit_calls'].append(t)
        close(d['incumbent_before_y'],max(r['y'] for r in pre['history']),'previous incumbent')
        close(d['incumbent_after_y'],max(r['y'] for r in post['history']),'next incumbent')
        post['decisions'].append(d);post['status']='COMPLETE' if t==16 else 'RUNNING';previous=post
        counts['rounds']+=1;counts['independent_official_choices']+=1;counts['note_citations']+=len(c['memory_references'])
    require(digest(without_cache(previous))==digest(without_cache(state)),'full final state reconstructed')
    require(len(state['edit_calls'])==counts['memory_calls']<=4 and len(state['insights'])==16,'call budget')
    require(state['cache_counters']['operator_compiles_in_loop']==0,'no repeated operator compilation')
    return counts


def audit_amendment(root):
    root=Path(root);base=study.directory(root);a=study.old.read(base/'technical_amendment_01.json')
    snapshot=root/a['snapshot'];require(study.old.read(snapshot/'technical_amendment_01.json')==a,'amendment receipt snapshot')
    for name,sha in a['pre_amendment_artifacts_sha256'].items():
        require(study.old.sha(snapshot/'pre_amendment_evidence'/name)==sha,'pre-amendment evidence custody')
        if name.endswith('/state.json'):
            prior=study.old.read(snapshot/'pre_amendment_evidence'/name);now=study.old.read(base/name)
            require(now['history'][:len(prior['history'])]==prior['history'] and now['decisions'][:len(prior['decisions'])]==prior['decisions'],
                'all pre-amendment decisions retained')
        else:require(study.old.sha(base/name)==sha,'pre-amendment immutable records unchanged')
    return dict(amendment_sha256=a['amendment_sha256'],original_queries_preserved=sum(s['completed_queries'] for s in a['pre_amendment_states'].values()),
        pre_amendment_records=len(a['pre_amendment_artifacts_sha256']),strategy_or_prompt_changed=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    p=study.check(ROOT);rows=[];failures=[];amendment=None
    try:amendment=audit_amendment(ROOT)
    except Exception as error:failures.append('Amendment audit: '+type(error).__name__+': '+str(error))
    for name,sha in p['protected_controls_sha256'].items():
        if study.old.sha(ROOT/name)!=sha:failures.append('Historical control drift: '+name)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(audit_task,str(ROOT),task):task for task in p['tasks']}
        for f,task in fs.items():
            try:rows.append(f.result())
            except Exception as error:failures.append(task+': '+type(error).__name__+': '+str(error))
    result=dict(status='PASS_LISTED_CHECKS' if not failures and len(rows)==8 else 'FAIL',rows=rows,failures=failures,
        protocol_sha256=p['protocol_sha256'],auditor_sha256=study.old.sha(__file__),rounds=sum(r['rounds'] for r in rows),
        forecasts=sum(r['forecasts'] for r in rows),independent_official_choices=sum(r['independent_official_choices'] for r in rows),
        boundary='Independent confidence selection, event schedule, GP CDF, Brier/KF/lifecycle, inherited negative priors and full state. Prompt and action schema replay reuse production. No independent GP refit or external timestamp attestation.',
        technical_amendment=amendment,protected_control_files=len(p['protected_controls_sha256']),new_oracle_queries=0,new_API_calls=0,new_GP_fits=0)
    study.write(ROOT/'experiments/reports/knowledge_loop_20261101/v11_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
