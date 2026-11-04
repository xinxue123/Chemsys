"""Independent numeric/decision audit; never refits GP or reads unused oracle rows."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import erfc,sqrt
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks.knowledge_llm import digest
from reasoning_memory_v12 import study,llm
from audit_knowledge_rank_v9 import independent_kf,gp_hash,close,require,scope
from audit_reasoning_memory_v11 import official_winner,without_cache


def alternatives(pool,pre,original):
    js=set();details=[]
    if len(pre['decisions'])<3:return [],[]
    for k,kind in enumerate(pool.kinds):
        if kind!='categorical':continue
        last=[pool.features[pool.ix[d['selected_id']]][k] for d in pre['decisions'][-3:]]
        if len(set(last))!=1 or {pool.features[int(j)][k] for j in original}!={last[0]}:continue
        counts={v:sum(pool.features[pool.ix[r['candidate_id']]][k]==v for r in pre['history']) for v in {f[k] for f in pool.features}}
        other={v for v,n in counts.items() if v!=last[0] and n<=1}
        if not other:continue
        js.update(j for j,row in enumerate(pool.features) if row[k] in other)
        details.append(dict(axis=pool.axes[k],repeated_level=last[0],alternative_observation_counts={v:counts[v] for v in sorted(other)}))
    return sorted(js),details


def independent_menu(pool,pre,a,t):
    js=np.flatnonzero(a['available']);original=js[np.lexsort((a['tie_priority'][js],-a['base_logei'][js]))][:5];ordered=original.copy()
    cat=dict(enabled=pre['arm']=='category',inserted=False,details=[],challenger_id=None,replaced_id=None)
    if pre['arm']=='category' and t<=8 and pre['category_menu_insertions']<2:
        choices,details=alternatives(pool,pre,original);js=np.array([j for j in choices if a['available'][j] and j not in original],int)
        if len(js):
            best=int(js[np.lexsort((a['tie_priority'][js],-a['base_logei'][js]))[0]])
            cat.update(inserted=True,details=details,challenger_id=pool.ids[best],replaced_id=pool.ids[int(ordered[-1])]);ordered[-1]=best
    physical=sorted(map(int,ordered),key=lambda j:pool.features[j]);perm=np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)
    return original,ordered,[pool.ids[physical[i]] for i in perm],cat


def independent_event(pool,pre,original,t):
    calls=pre['edit_calls']
    if not calls:return dict(allowed=True,reasons=['initial_knowledge'])
    if len(calls)>=4:return dict(allowed=False,reasons=['edit_budget_exhausted'])
    if t-calls[-1]<3:return dict(allowed=False,reasons=['minimum_edit_gap'])
    if t<11 and len(calls)>=3:return dict(allowed=False,reasons=['reserved_late_edit'])
    reasons=[];last=pre['decisions'][-3:];feedback=[e for d in pre['decisions'][-2:] for e in d['KF_feedback']]
    if len(last)==3 and last[-1]['incumbent_after_y']<=last[0]['incumbent_before_y']+1e-12:reasons.append('stagnation')
    if any(e['Brier_gain']<-.05 for e in feedback):reasons.append('negative_prediction')
    if any(e['reason'] in ('automatic_pause','automatic_retract') for e in feedback):reasons.append('automatic_lifecycle_event')
    if any(d['incumbent_after_y']>d['incumbent_before_y']+1e-12 for d in pre['decisions'][-2:]):reasons.append('new_incumbent_evidence')
    if alternatives(pool,pre,original)[1]:reasons.append('category_concentration_without_contrasts')
    recent=pre['decisions'][-2:]
    if len(recent)==2 and all(any(r['requested_use']=='support' and r['use']!='support' for r in d['memory_references']) for d in recent):reasons.append('repeated_unsupported_positive_citation')
    active=[h for h in pre['hypotheses'] if h['kf']['status']=='active']
    for i,h in enumerate(active):
        for other in active[i+1:]:
            if h['focus']==other['focus'] and h['effect']!=other['effect'] and any(scope(pool,h,int(j)) and scope(pool,other,int(j)) for j in original):reasons.append('opposed_overlapping_rules')
    return dict(allowed=bool(reasons),reasons=sorted(set(reasons)))


def check_request(base,key,system,public,raw,p):
    context=study.old.read(base/'requests'/key/'context.json');valid=study.old.read(base/'requests'/key/'validated.json')
    require(context==dict(system=system,public=public,model=p['model'],reasoning_effort=p['reasoning_effort']),'own-prefix prompt identity')
    require(digest(context)==key==valid['context_sha256'] and valid['raw']==raw,'response custody')
    return valid


def audit_birth(pool,pre,hs,t):
    existing={(h['id'],h['version']) for h in pre['hypotheses']};carried=0
    ref=min(pre['history'],key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))
    initial=dict(mean=0.,variance=.25,weight=.4,probability_positive=.5,bad_streak=0,good_streak=0,updates=0,local_updates=0,status='active',paused_round=None,retracted_round=None)
    facts=[]
    for h in pre['hypotheses']:
        for e in h['events']:
            if e['Brier_gain']<-1e-12 or (not e['success'] and e['p_LLM']>=.5):facts.append(dict(e,source_id=h['id'],source_version=h['version']))
    facts.sort(key=lambda e:(e['query'],e['source_id'],e['source_version']))
    for h in hs:
        if (h['id'],h['version']) in existing:continue
        require(h['created_round']==t and h['birth_history_sha256']==digest(pre['history']),'version birth prefix')
        require(h['reference_id']==ref['candidate_id'] and h['reference_y']==ref['y'],'birth current incumbent')
        require(h['kf']==initial and not h['events'] and not h['inherited_negative'],'no numeric prior replay')
        expected=[e['event_id'] for e in facts if e['source_id']==h['id'] or scope(pool,h,pool.ix[e['point_id']])]
        require(h['carried_fact_ids']==expected,'cross-version and cross-focus facts retained')
        carried+=len(expected)
    return carried


def audit_run(root,task,arm):
    root=Path(root);p=study.check(root);base=study.directory(root);run=base/'runs'/f'{task}_{arm}'
    state=study.old.read(run/'state.json');pool=study.gpmod.load_task(root,task,seed=0)
    require(state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16,'complete4+16')
    require(state['protocol_sha256']==p['protocol_sha256'] and state['arm']==arm and state['seed']==0,'registered run')
    control=study.old.read(study.v9.directory(root)/'runs'/f'{task}_bo/state.json')
    require(state['history'][:4]==control['history'][:4] and len({r['candidate_id'] for r in state['history']})==20,'shared init and unique queries')
    init=study.old.read(run/'initial_commit.json');require(init['commit_sha256']==digest({k:v for k,v in init.items() if k!='commit_sha256'})==state['initial_commit_sha256'],'initial custody')
    require(init['initial_ids']==list(pool.initial_ids),'input-only initial points')
    counts=dict(task=task,arm=arm,rounds=0,forecasts=0,expected_forecasts=0,carried_facts=0,reference_changed_from_birth=0,
        normalized_support_citations=0,valid_support_citations=0,sidecar_warnings=0,category_insertions=0,category_selected=0)
    previous=None
    for t,d in enumerate(state['decisions'],1):
        c=study.old.read(run/'rounds'/f'{t:02d}_commit.json');pre=c['prestate'];obs=state['history'][t+3]
        require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})==d['commit_sha256'],'decision custody')
        require(c['query']==t and c['arm']==arm and c['protocol_sha256']==p['protocol_sha256'],'query registration')
        require(c['prestate_sha256']==digest(pre) and c['history_sha256']==digest(pre['history']),'prestate hash')
        require(pre['history']==state['history'][:t+3] and len(pre['decisions'])==t-1,'own chronological prefix')
        if previous is not None:require(without_cache(previous)==without_cache(pre),'exact feedback to next state')
        with np.load(run/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        require(gp_hash(a)==c['GP_sha256']==c['GP_sha256_after_ranking'],'unmodified GP arrays')
        available=np.array([key not in {r['candidate_id'] for r in pre['history']} for key in pool.ids]);np.testing.assert_array_equal(a['available'],available)
        np.testing.assert_array_equal(a['tie_priority'],study.gpmod.semantic_priorities(pool,t,0))
        original,ordered,ids,category=independent_menu(pool,pre,a,t)
        require(c['original_ordered_ids']==[pool.ids[j] for j in original] and c['ordered_ids']==[pool.ids[j] for j in ordered] and c['display_ids']==ids,'native/ablation menu and display')
        require(c['category']==category and c['BO_reference_id']==pool.ids[int(original[0])],'category gate and native reference')
        close(c['raw_qLogEI'],a['base_logei'][ordered],'native acquisition values')
        refresh=independent_event(pool,pre,original,t);require(refresh==c['refresh'],'event schedule')
        hs=deepcopy(pre['hypotheses'])
        if refresh['allowed']:
            pub=llm.memory_public(pool,pre,ids,t,refresh);valid=check_request(base,c['memory_context_sha256'],llm.MEMORY_SYSTEM,pub,c['memory_raw'],p)
            replay=llm.validate_memory(c['memory_raw'],pool,pre,ids,t)
            require(replay==valid['plan'] and replay['reviews']==c['reviews'],'action schema replay');hs=replay['hypotheses']
        else:require(c['memory_raw'] is None and not c['reviews'],'no non-event memory changes')
        require(hs==c['hypotheses_at_commit'],'prospective immutable notes');counts['carried_facts']+=audit_birth(pool,pre,hs,t)
        pub=llm.reasoning_public(root,pool,pre,hs,ids,t)
        require('forecast_notes' not in pub and not any(k in pub for k in ('mu_y','sd_y','base_logei')),'no global-note spill or GP numeric exposure')
        relevant={(h['id'],h['version']) for h in hs if any(scope(pool,h,int(j)) for j in ordered)}
        require({(h['id'],h['version']) for h in pub['retrieved_notes']['notes']}==relevant,'scoped relevant notes')
        valid=check_request(base,c['LLM_context_sha256'],llm.REASONING_SYSTEM,pub,c['reasoning_raw'],p)
        replay=llm.validate_reasoning(root,c['reasoning_raw'],pool,hs,ids,pub['retrieved_notes'])
        require(replay==valid['plan'] and replay['forecasts']==c['forecasts'] and replay['memory_references']==c['memory_references'],'nonblocking annotation replay')
        chosen,index=official_winner(c['LLM_raw']);j=pool.ix[chosen]
        require(chosen==c['selected_id']==d['selected_id']==obs['candidate_id'] and chosen in ids and a['available'][j],'original confidence selection')
        require(index==c['winning_hypothesis_index'] and c['LLM_raw']==c['reasoning_raw']['insight'],'official raw Insight retained')
        require(c['mode']==d['mode']==('base_bo' if j==original[0] else 'ReasoningBO_memory'),'BO agreement mode')
        close(c['opportunity_gap'],a['base_logei'][original[0]]-a['base_logei'][j],'opportunity gap is annotation only')
        ref=min(pre['history'],key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))
        require(c['current_incumbent']==dict(candidate_id=ref['candidate_id'],y=ref['y']),'current incumbent before oracle')
        for r in c['memory_references']:
            h=next((h for h in hs if (h['id'],h['version'])==(r['id'],r['version'])),None)
            if r['use']=='support':
                require((r['id'],r['version']) in relevant and scope(pool,h,j) and h['kf']['status']=='active' and h['kf']['weight']>0,'support state/scope')
                require(sum(e['success'] and e['Brier_gain']>1e-12 for e in h['events'])>=2,'mature positive improvement evidence')
                counts['valid_support_citations']+=1
            counts['normalized_support_citations']+=int(r['requested_use']=='support' and r['use']!='support')
        expected={(h['id'],h['version']) for h in hs if h['kf']['status']!='retracted' and scope(pool,h,j)}
        actual=[(s['prediction']['id'],s['prediction']['version']) for s in c['predictions']]
        require(len(actual)==len(set(actual))==len(d['KF_feedback']) and set(actual)<=expected,'valid scoped forecast subset exactly once')
        require(len(expected)==c['expected_forecasts'] and len(actual)==len(c['forecasts']),'forecast coverage accounting')
        counts['expected_forecasts']+=len(expected)
        for sh in c['predictions']:
            pred=sh['prediction'];h=next(h for h in hs if (h['id'],h['version'])==(pred['id'],pred['version']))
            require(digest(pred)==sh['prediction_sha256'] and pred['semantic_sha256']==h['semantic_sha256'],'immutable prediction')
            require(pred['history_sha256']==c['history_sha256'] and pred['GP_sha256']==c['GP_sha256'] and pred['query']==t and pred['point_id']==chosen,'prospective forecast identity')
            require(pred['reference_id']==ref['candidate_id'] and pred['reference_y']==ref['y'],'each KF target is current incumbent')
            sd=float(a['sd_predictive_y'][j]);g=.5*erfc(-(a['mu_y'][j]-ref['y'])/(sqrt(2)*sd)) if sd else float(a['mu_y'][j]>ref['y'])
            close(g,pred['p_GP'],'Gaussian observation probability');prob=c['forecasts'][h['id']]['p'];require(prob==pred['p_LLM'],'prospective LLM probability')
            after,measurement=independent_kf(h['kf'],prob,g,bool(obs['y']>ref['y']),t)
            e=next(e for e in d['KF_feedback'] if (e['id'],e['version'])==(h['id'],h['version']))
            require(e['event_id']==digest({k:v for k,v in e.items() if k!='event_id'}) and e['before']==h['kf'],'KF event custody')
            require(e['prediction_sha256']==sh['prediction_sha256'] and e['reference_y']==ref['y'],'feedback target')
            for name,value in after.items():
                if isinstance(value,float):close(e['after'][name],value,'independent KF posterior')
                else:require(e['after'][name]==value,'independent lifecycle')
            for name,value in measurement.items():
                if isinstance(value,float):close(e[name],value,'independent Brier/KF')
                else:require(e[name]==value,'independent update metadata')
            require(e['positive_improvement_evidence']==bool(obs['y']>ref['y'] and e['Brier_gain']>1e-12),'failure calibration not counted as improvement evidence')
            h['kf']=deepcopy(e['after']);h['events'].append(deepcopy(e));counts['forecasts']+=1
            counts['reference_changed_from_birth']+=int(pred['reference_id']!=h['reference_id'])
        post=deepcopy(pre);post['hypotheses']=hs;post['history'].append(obs);post['insights'].append(dict(query=t,insight=c['LLM_raw']))
        if refresh['allowed']:post['edit_calls'].append(t)
        post['category_menu_insertions']+=int(category['inserted']);post['category_selected_queries']+=int(category['challenger_id']==chosen)
        close(d['incumbent_before_y'],ref['y'],'previous incumbent');close(d['incumbent_after_y'],max(ref['y'],obs['y']),'next incumbent')
        require(d['memory_references']==c['memory_references'] and d['sidecar_warnings']==c['sidecar_warnings'] and d['decision_purpose']==c['decision_purpose'],'feedback sidecars')
        post['decisions'].append(d);post['status']='COMPLETE' if t==16 else 'RUNNING';previous=post
        counts['rounds']+=1;counts['sidecar_warnings']+=len(c['sidecar_warnings']);counts['category_insertions']+=int(category['inserted']);counts['category_selected']+=int(category['challenger_id']==chosen)
    require(without_cache(previous)==without_cache(state),'full state reconstruction')
    require(len(state['edit_calls'])<=4 and len(state['insights'])==16 and counts['category_insertions']<=2,'fixed budgets')
    require(state['cache_counters']['operator_compiles_in_loop']==0,'no repeated operator compilation')
    return counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    p=study.check(ROOT);rows=[];failures=[]
    for name,sha in p['protected_controls_sha256'].items():
        if study.old.sha(ROOT/name)!=sha:failures.append('Historical control drift: '+name)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(audit_run,str(ROOT),task,arm):(task,arm) for task,arm in study.RUNS}
        for future,key in futures.items():
            try:rows.append(future.result())
            except Exception as error:failures.append(str(key)+': '+type(error).__name__+': '+str(error))
    result=dict(status='PASS_LISTED_CHECKS' if not failures and len(rows)==5 else 'FAIL',rows=rows,failures=failures,
        protocol_sha256=p['protocol_sha256'],auditor_sha256=study.old.sha(__file__),rounds=sum(r['rounds'] for r in rows),
        forecasts=sum(r['forecasts'] for r in rows),expected_forecasts=sum(r['expected_forecasts'] for r in rows),
        protected_control_files=len(p['protected_controls_sha256']),new_oracle_queries=0,new_API_calls=0,new_GP_fits=0,
        boundary='Independent confidence selection, menu/category gate, event schedule, current-incumbent CDF, Brier/KF/lifecycle, fact inheritance and final state. Prompt/action/annotation schema replay reuses production. No independent GP refit or external timestamp attestation.')
    study.write(ROOT/'experiments/reports/knowledge_loop_20261101/v12_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
