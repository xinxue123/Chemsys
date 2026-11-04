"""V9 rank fallback plus bounded prospective knowledge-guided information queries."""
from copy import deepcopy
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import engine as previous,kf,policy
from . import proposals,information

PARAMETERS=dict(previous.PARAMETERS,proposal_examples=24,information_maximum=3,information_last_query=8,
    dissent_maximum=1,dissent_probability_gap=.20,probe_qLogEI_gap=float(np.log(4.)),
    minimum_KG_increment_absolute=1e-12,minimum_KG_increment_relative=1e-10,
    max_proposal_calls=4,max_forecast_calls=16,reference_nonfocus_continuous_tolerance=.25,
    GP_modified=False,knowledge_scope='full_pool_targets_plus_V9_rank_fallback')


def mapped(pool,state,hs,context):
    rows=[]
    for h in hs:
        proposals.validate(h)
        if h['kf']['status']=='retracted':continue
        point=proposals.project(pool,h,context['available'],context['tie_priority'])
        if point:rows.append(dict(id=h['id'],version=h['version'],probe_signature=h['probe_signature'],**point))
    return rows


def menu(pool,state,hs,context,t):
    ordered,_=previous.candidates(pool,context,t);rows=mapped(pool,state,hs,context)
    union=set(map(int,ordered))|{r['index'] for r in rows}
    physical=sorted(union,key=lambda j:pool.features[j]);shuffle=np.random.default_rng(np.random.SeedSequence([0,t,1010])).permutation(len(physical))
    return ordered,np.asarray(physical,dtype=int)[shuffle],rows


def probe_rows(pool,state,hs,context,t,probabilities,fallback,rows,kg):
    answer=[];bo=int(previous.candidates(pool,context,t)[0][0])
    for point in rows:
        h=next(h for h in hs if (h['id'],h['version'])==(point['id'],point['version']))
        j=point['index'];w=kf.deployment_weight(h['kf']);g=previous.gp_probability(context,j,h['reference_y'])
        p=probabilities[h['id']][pool.ids[j]];gap=float(context['base_logei'][bo]-context['base_logei'][j])
        reasons=[]
        if t>8:reasons.append('early_window_closed')
        if state['information_used']>=3:reasons.append('information_budget_exhausted')
        if w<=0:reasons.append('no_deployment_credit')
        if h['probe_signature'] in state['tested_probe_signatures']:reasons.append('contrast_already_tested')
        if j==fallback:reasons.append('already_fallback')
        if gap>PARAMETERS['probe_qLogEI_gap']+1e-12:reasons.append('opportunity_cost_exceeds_factor4')
        value=kg.get(j);base=kg.get(fallback);delta=value-base if value is not None and base is not None else None
        tolerance=max(1e-12,1e-10*max(abs(value or 0),abs(base or 0)))
        answer.append(dict(point,weight=w,p_LLM=p,p_GP=g,probability_gap=abs(p-g),qLogEI_gap=gap,
            KG=value,fallback_KG=base,KG_increment=delta,KG_tolerance=tolerance,
            priority=w*max(0.,delta) if delta is not None else 0.,
            KG_admitted=not reasons and delta is not None and delta>tolerance,
            dissent_priority=w*abs(p-g),dissent_admitted=not reasons and not state['dissent_used'] and abs(p-g)>=.20,
            blocked_reasons=reasons))
    return answer


def choose(rows,context,fallback):
    eligible=[r for r in rows if r['KG_admitted']]
    key='priority';mode='information_KG'
    if not eligible:eligible=[r for r in rows if r['dissent_admitted']];key='dissent_priority';mode='information_disagreement'
    if not eligible:return fallback,None,None
    best=min(eligible,key=lambda r:(-r[key],-float(context['base_logei'][r['index']]),
        int(context['tie_priority'][r['index']]),r['probe_signature']))
    return best['index'],mode,best


def plan(pool,state,context,t,proposal_receipt,forecast_receipt,refresh,gp):
    original=previous.gp_hash(context)
    hs=proposal_receipt['plan']['hypotheses'] if proposal_receipt else deepcopy(state['hypotheses'])
    reviews=proposal_receipt['plan']['reviews'] if proposal_receipt else []
    probabilities=forecast_receipt['plan']['probabilities'];ordered,display,projected=menu(pool,state,hs,context,t)
    receipt=dict(forecast_receipt,plan=dict(hypotheses=hs,reviews=reviews,probabilities=probabilities))
    commit=previous.plan(pool,state,context,t,'kf_rank',receipt,refresh)
    fallback=pool.ix[commit['selected_id']];kg={};arrays=None
    if t<=8 and state['information_used']<3 and projected:
        columns={fallback,int(ordered[0])}|{r['index'] for r in projected}
        kg,arrays=information.evaluate(gp,context,columns)
    rows=probe_rows(pool,state,hs,context,t,probabilities,fallback,projected,kg)
    selected,mode,source=choose(rows,context,fallback)
    forecasts=[]
    for h in hs:
        proposals.validate(h)
        if h['kf']['status']=='retracted' or selected not in policy.scope_indices(pool,h,[selected]):continue
        p=dict(id=h['id'],version=h['version'],query=t,point_id=pool.ids[selected],semantic_sha256=h['semantic_sha256'],
            reference_id=h['reference_id'],reference_y=h['reference_y'],p_LLM=probabilities[h['id']][pool.ids[selected]],
            p_GP=previous.gp_probability(context,selected,h['reference_y']),history_sha256=digest(state['history']),GP_sha256=original)
        forecasts.append(dict(prediction=p,prediction_sha256=digest(p)))
    commit.update(arm='information',selected_id=pool.ids[selected],mode=mode or commit['mode'],
        predictions=forecasts,display_ids=[pool.ids[j] for j in display],fallback_id=pool.ids[fallback],
        information_override=mode is not None,information_source=deepcopy(source),probe_reviews=rows,
        KG_values={pool.ids[j]:v for j,v in kg.items()},
        opportunity_gap=float(context['base_logei'][ordered[0]]-context['base_logei'][selected]),
        outside_V8_gate=not bool(context['eligible'][selected]),
        KF_changes_selection_from_fixed_weights=commit['fixed_initial_weight_selected_id']!=pool.ids[selected],
        weight_shadow_limitation='V9 rank fallback shadow only; information override not a no-KF ablation',
        proposal_context_sha256=proposal_receipt['context_sha256'] if proposal_receipt else None,
        proposal_raw=proposal_receipt['raw'] if proposal_receipt else None,
        forecast_probabilities=probabilities,GP_sha256_after_ranking=previous.gp_hash(context))
    if original!=previous.gp_hash(context):raise ValueError('Read-only information calculation mutated GP arrays')
    commit['commit_sha256']=digest({k:v for k,v in commit.items() if k!='commit_sha256'})
    return commit,arrays


def feedback(state,commit,observation):
    for h in commit['hypotheses_at_commit']:proposals.validate(h)
    updated=deepcopy(state);previous.feedback(updated,commit,observation)
    if commit['refresh']['allowed']:updated['edit_calls'].append(commit['query'])
    if commit['information_override']:
        updated['information_used']+=1
        signature=commit['information_source']['probe_signature']
        if signature in updated['tested_probe_signatures']:raise ValueError('Repeated prospective contrast budget')
        updated['tested_probe_signatures'].append(signature)
        if commit['mode']=='information_disagreement':
            if updated['dissent_used']:raise ValueError('Duplicate dissent pilot')
            updated['dissent_used']=True
    if updated['information_used']>3 or commit['information_override'] and commit['query']>8:raise ValueError('Probe budget exceeded')
    updated['decisions'][-1].update(fallback_id=commit['fallback_id'],information_override=commit['information_override'])
    state.clear();state.update(updated)
