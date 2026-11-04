"""Read-only V9 numerical/decision audit. No model fitting or oracle access.

KF, lifecycle, probabilities and rankings are independently reconstructed.
The production action validator is additionally reused for schema/scope replay;
that portion is a consistency check, not an independent scientific review.
"""
from copy import deepcopy
from hashlib import sha256
from math import erfc,isfinite,sqrt
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks import knowledge_gp as gpmod,knowledge_study as old
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import study,llm


def require(ok,label):
    if not ok:raise AssertionError(label)


def close(a,b,label):np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12,err_msg=label)


def independent_kf(before,p,g,success,t,prior=False):
    z=(float(success)-g)**2-(float(success)-p)**2
    r=max(.0025,4*(p-g)**2*g*(1-g));pp=before['variance']+.0025;k=pp/(pp+r)
    m=before['mean']+k*(z-before['mean']);v=(1-k)**2*pp+k*k*r
    probability=.5*erfc(-m/sqrt(2*v));raw=.8*probability
    weight=min(raw,before['weight']) if z<=1e-12 else raw
    out=deepcopy(before);bad=before['bad_streak']+1 if z<-.02 else 0;good=before['good_streak']+1 if z>.02 else 0
    out.update(mean=m,variance=v,weight=weight,probability_positive=probability,bad_streak=bad,good_streak=good,
        updates=before['updates']+1,local_updates=before['local_updates']+int(not prior))
    reason='negative_feedback_downgrade' if z<-1e-12 else 'prospective_update'
    if not prior:
        if bad>=4 or out['local_updates']>=4 and probability<=.05:
            out.update(status='retracted',retracted_round=t,weight=0.);reason='automatic_retract'
        elif before['status']=='active' and (bad>=2 or out['local_updates']>=2 and probability<=.2):
            out.update(status='paused',paused_round=t);reason='automatic_pause'
        elif before['status']=='paused' and t-before['paused_round']>=2 and good>=2 and probability>=.6:
            out.update(status='active',paused_round=None);reason='automatic_resume'
    return out,dict(Brier_gain=z,measurement_variance=r,predicted_variance=pp,kalman_gain=k,reason=reason)


def scope(pool,h,j):
    for axis,spec in h['scope'].items():
        x=pool.features[j][pool.axes.index(axis)]
        if 'allowed' in spec:
            if x not in spec['allowed']:return False
        elif not spec['bounds'][0]-1e-12<=float(x)<=spec['bounds'][1]+1e-12:return False
    return True


def select(scores,indices,ties):
    best=np.flatnonzero(scores==max(scores));return int(indices[best[np.argmin(ties[indices[best]])]])


def rank(pool,ordered,ties,hs,probs,fixed=False):
    b=np.linspace(1.,0.,5);score=b.copy();terms=[]
    for h in hs:
        status=h['kf']['status']
        if status=='retracted':continue
        inside=np.array([scope(pool,h,int(j)) for j in ordered])
        p=np.array([probs[h['id']][pool.ids[j]] for j in ordered])
        tied=bool(np.all(p==p[0]));l=b.copy()
        if not tied:
            ranked=np.argsort(p,kind='stable');ranks=np.empty(5)
            for value in np.unique(p):ranks[p==value]=np.flatnonzero(p[ranked]==value).mean()/4
            l=np.where(inside,ranks,b)
        w=(.4 if fixed else h['kf']['weight']) if status=='active' and inside.any() else 0.
        terms.append(dict(id=h['id'],version=h['version'],weight=w,inside=inside.tolist(),probabilities=p.tolist(),
            rank_scores=l.tolist(),no_rank_information=tied))
    total=sum(t['weight'] for t in terms);factor=min(1.,.8/total) if total else 1.
    for term in terms:
        term['effective_weight']=term['weight']*factor
        score+=term['effective_weight']*(np.array(term['rank_scores'])-b)
    return select(score,ordered,ties),dict(base_rank_scores=b.tolist(),combined_rank_scores=score.tolist(),terms=terms)


def event_allowed(pool,state,shortlist,t):
    calls=state['edit_calls'];reasons=[]
    if not calls:return dict(allowed=True,reasons=['initial_knowledge'])
    if len(calls)>=4:return dict(allowed=False,reasons=['edit_budget_exhausted'])
    if t-calls[-1]<3:return dict(allowed=False,reasons=['minimum_edit_gap'])
    if t<11 and len(calls)>=3:return dict(allowed=False,reasons=['reserved_late_edit'])
    if t>=11 and max(calls)<11:reasons.append('reserved_late_edit')
    active=[h for h in state['hypotheses'] if h['kf']['status']=='active' and h['kf']['weight']>0]
    if not any(scope(pool,h,int(j)) for h in active for j in shortlist):reasons.append('no_active_coverage')
    for i,h in enumerate(active):
        for other in active[i+1:]:
            if h['focus']==other['focus'] and h['effect']!=other['effect'] and any(scope(pool,h,int(j)) and scope(pool,other,int(j)) for j in shortlist):
                reasons.append('opposed_overlapping_rules')
    recent=state['decisions'][-3:]
    if len(recent)==3 and recent[-1]['incumbent_after_y']<=recent[0]['incumbent_before_y']+1e-12:reasons.append('stagnation')
    events=[e for d in state['decisions'][-2:] for e in d['KF_feedback']]
    if any(e['Brier_gain']<-.05 for e in events):reasons.append('negative_prediction')
    if any(e['reason'] in ('automatic_pause','automatic_retract') for e in events):reasons.append('automatic_lifecycle_event')
    return dict(allowed=bool(reasons),reasons=sorted(set(reasons)))


def gp_hash(arrays):
    keys={'mu':'mu_y','sd':'sd_y','observation_sd':'sd_predictive_y','base':'base_logei','available':'available','tie_priority':'tie_priority'}
    return digest({key:dict(shape=list(arrays[name].shape),dtype=str(arrays[name].dtype),
        sha256=sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()) for key,name in keys.items()})


def run_audit(root,path,protocol):
    state=old.read(path);task,arm=state['task'],state['arm'];pool=gpmod.load_task(root,task,seed=0)
    require(path.parent.name==f'{task}_{arm}' and task in protocol['tasks'] and arm in study.ARMS,'canonical run')
    require(state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16,'complete4+16')
    require(len({r['candidate_id'] for r in state['history']})==20,'unique observations')
    require(state['protocol_sha256']==protocol['protocol_sha256'] and state['seed']==0,'registered state')
    init=old.read(path.parent/'initial_commit.json')
    require(init['commit_sha256']==digest({k:v for k,v in init.items() if k!='commit_sha256'})==state['initial_commit_sha256'],'initial commitment SHA')
    require(init['initial_ids']==list(pool.initial_ids)==[r['candidate_id'] for r in state['history'][:4]],'input-only initial IDs')
    baseline=old.read(old.paths(root,'v8')/'runs'/f'{task}_bo_agent_initial_terra/state.json')
    require(state['history'][:4]==baseline['history'][:4],'shared initial outcomes')
    if arm=='bo':require(state['history']==baseline['history'],'full20point BO reproduction')
    previous=None;counts=dict(rounds=0,forecasts=0)
    for t,d in enumerate(state['decisions'],1):
        c=old.read(path.parent/'rounds'/f'{t:02d}_commit.json');pre=c['prestate'];obs=state['history'][t+3]
        require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})==d['commit_sha256'],'commit SHA')
        require(c['prestate_sha256']==digest(pre) and pre['history']==state['history'][:t+3] and len(pre['decisions'])==t-1,'chronological prestate')
        if previous is not None:
            a={k:v for k,v in previous.items() if k!='cache_counters'};b={k:v for k,v in pre.items() if k!='cache_counters'}
            require(digest(a)==digest(b),'complete feedback-to-next-state replay')
        with np.load(path.parent/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        require(gp_hash(arrays)==c['GP_sha256']==c['GP_sha256_after_ranking'],'unchanged original GP arrays')
        available=np.array([key not in {r['candidate_id'] for r in pre['history']} for key in pool.ids])
        require(np.array_equal(available,arrays['available']),'unused candidate mask')
        ties=gpmod.semantic_priorities(pool,t,0);require(np.array_equal(ties,arrays['tie_priority']),'physical tie stream')
        js=np.flatnonzero(available);ordered=js[np.lexsort((ties[js],-arrays['base_logei'][js]))][:5]
        physical=sorted(map(int,ordered),key=lambda j:pool.features[j]);order=np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)
        display=[pool.ids[physical[j]] for j in order];require(display==c['display_ids'],'neutral physical display order')
        require(c['ordered_ids']==[pool.ids[j] for j in ordered] and c['BO_reference_id']==pool.ids[ordered[0]],'shared exact5candidate interface')
        close(c['raw_qLogEI'],arrays['base_logei'][ordered],'unaltered qLogEI')
        require(c['selected_id']==d['selected_id']==obs['candidate_id'] and c['mode']==d['mode'],'observed selection')
        require(c['selected_id'] in c['ordered_ids'],'selected within5')
        hs=deepcopy(c['hypotheses_at_commit']);selected=ordered[0]
        if arm!='bo':
            req=study.directory(root)/'requests'/c['LLM_context_sha256'];ctx=old.read(req/'context.json');validated=old.read(req/'validated.json')
            require(digest(ctx)==c['LLM_context_sha256']==validated['context_sha256'],'exact model context')
            require(ctx['model']==protocol['model'] and ctx['reasoning_effort']==protocol['reasoning_effort'],'matched LLM')
            require(validated['raw']==c['LLM_raw'],'committed first validated response')
            public=ctx['public']
            expected_obs=[dict(llm.point(pool,r['candidate_id']),objective=r['raw_value']) for r in pre['history']]
            require(public['own_observations']==expected_obs and public['candidate_menu']==[llm.point(pool,key) for key in display],'no future observations / correct physical conditions')
            if arm=='reasoning_bo':
                require(ctx['system']==llm.RBO_SYSTEM,'RBO prompt')
                priority={'high':0,'medium':1,'low':2}
                hypotheses=sorted(c['LLM_raw']['hypotheses'],key=lambda h:priority[h['confidence'].lower()])
                selected=pool.ix[hypotheses[0]['parameter_sets'][0]['candidate_id']]
                require(f'We now have data from {len(pre["history"])} completed experiments' in public['official_optimization_prompt'],'accurate RBO observation count')
                require(json.dumps(pre['insights'],ensure_ascii=False) in public['official_optimization_prompt'],'complete own Insight prefix')
            else:
                refresh=event_allowed(pool,pre,ordered,t);require(refresh==c['refresh']==public['refresh'],'independent event schedule')
                checked=llm.validate_kf(c['LLM_raw'],pool,pre,display,t,refresh)
                require(digest(checked['hypotheses'])==digest(hs) and checked['reviews']==c['reviews'],'action/schema/scope consistency replay')
                selected,ranks=rank(pool,ordered,ties,hs,checked['probabilities']);close(ranks['combined_rank_scores'],c['ranking']['combined_rank_scores'],'independent full rank fusion')
                require(digest(ranks)==digest(c['ranking']),'all rule ranks and deployment weights')
                fixed,_=rank(pool,ordered,ties,hs,checked['probabilities'],fixed=True)
                require(pool.ids[fixed]==c['fixed_initial_weight_selected_id'] and (fixed!=selected)==c['KF_changes_selection_from_fixed_weights'],'fixed-status weight shadow')
        require(pool.ids[selected]==c['selected_id'],'independent final argmax')
        events=[]
        for sh in c['predictions']:
            pred=sh['prediction'];require(digest(pred)==sh['prediction_sha256'],'frozen prediction SHA')
            h=next(h for h in hs if (h['id'],h['version'])==(pred['id'],pred['version']))
            require(scope(pool,h,int(selected)) and h['kf']['status']!='retracted','prospective scope admission')
            require(pred['history_sha256']==digest(pre['history']) and pred['point_id']==obs['candidate_id'] and pred['query']==t,'prospective prefix and point')
            ref=next(r for r in pre['history'] if r['candidate_id']==h['reference_id']);require(ref['y']==pred['reference_y']==h['reference_y'],'fixed known reference')
            sd=float(arrays['sd_predictive_y'][selected]);mu=float(arrays['mu_y'][selected])
            g=.5*erfc(-(mu-ref['y'])/(sqrt(2)*sd)) if sd>0 else float(mu>ref['y'])
            close(g,pred['p_GP'],'unchanged GP event probability')
            p=checked['probabilities'][h['id']][obs['candidate_id']];require(p==pred['p_LLM'],'prequery LLM probability')
            after,expected=independent_kf(h['kf'],p,g,bool(obs['y']>ref['y']),t)
            e=next(e for e in d['KF_feedback'] if (e['id'],e['version'])==(h['id'],h['version']))
            require(e['event_id']==digest({k:v for k,v in e.items() if k!='event_id'}) and e['before']==h['kf'],'KF receipt custody')
            for key,value in expected.items():
                if key=='reason':require(e[key]==value,'automatic lifecycle transition')
                else:close(e[key],value,'KF measurement '+key)
            for key,value in after.items():
                if type(value) is float:close(e['after'][key],value,'KF posterior '+key)
                else:require(e['after'][key]==value,'KF lifecycle state '+key)
            require(e['prediction_sha256']==sh['prediction_sha256'] and e['query']==t and e['point_id']==obs['candidate_id'],'scored forecast identity')
            h['events'].append(deepcopy(e));h['kf']=deepcopy(e['after']);events.append(e);counts['forecasts']+=1
        require(len(events)==len(d['KF_feedback']),'exactly once feedback per prospective rule')
        post=deepcopy(pre);post['hypotheses']=hs;post['history'].append(obs)
        if c['LLM_raw'] is not None:post['insights'].append(dict(query=t,insight=c['LLM_raw']))
        if arm=='kf_rank' and c['refresh']['allowed']:post['edit_calls'].append(t)
        close(d['incumbent_before_y'],max(r['y'] for r in pre['history']),'before incumbent')
        close(d['incumbent_after_y'],max(r['y'] for r in post['history']),'after incumbent')
        post['decisions'].append(d);post['status']='COMPLETE' if t==16 else 'RUNNING';previous=post;counts['rounds']+=1
    require(digest({k:v for k,v in previous.items() if k!='cache_counters'})==digest({k:v for k,v in state.items() if k!='cache_counters'}),'full final-state replay')
    require(state['cache_counters']['operator_compiles_in_loop']==0,'no loop compilation')
    return dict(task=task,arm=arm,**counts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bo-only',action='store_true');args=ap.parse_args()
    p=study.check(ROOT);failures=[];rows=[]
    for name,expected in p['protected_V8_sha256'].items():
        require(old.sha(ROOT/name)==expected,'protected V8 changed: '+name)
    for task in p['tasks']:
        for arm in ('bo',) if args.bo_only else study.ARMS:
            path=study.directory(ROOT)/'runs'/f'{task}_{arm}/state.json'
            try:rows.append(run_audit(ROOT,path,p))
            except Exception as error:failures.append(dict(task=task,arm=arm,kind=type(error).__name__,message=str(error)))
    totals=dict(runs=len(rows),rounds=sum(r['rounds'] for r in rows),forecasts=sum(r['forecasts'] for r in rows))
    result=dict(status='FAIL' if failures else 'PASS_BO_ONLY' if args.bo_only else 'PASS_LISTED_CHECKS',
        protocol_sha256=p['protocol_sha256'],auditor_sha256=old.sha(__file__),totals=totals,rows=rows,failures=failures,
        GP_refits=0,LLM_calls=0,new_oracle_queries=0,protected_V8_files=len(p['protected_V8_sha256']),
        boundaries=['Independent formulas for rank/KF/lifecycle/CDF/physical tie/feedback chronology.',
            'Production action validator additionally reused for syntax/scope/version consistency; not independent causal validation.',
            'Original GP adapter/source and pre/post arrays authenticated; no independent refit or external timestamp attestation.'])
    name='v9_bo_audit.json' if args.bo_only else 'v9_audit.json'
    study.write(ROOT/'experiments/reports/knowledge_loop_20261101'/name,result)
    print(json.dumps(dict(status=result['status'],totals=totals,failures=failures),ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
