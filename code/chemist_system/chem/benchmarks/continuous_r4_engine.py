"""R4: blind physical proposals, scheduled local validation and shared evidence.

The fixed pool, exact GP, acquisition, seeds and normal loss gate are reused.
This module has no LLM client or experimental outcome provider.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from . import continuous_r3_engine as old
from . import continuous_benchmark_engine as base
from . import revised_continuous_engine as kf
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_r4_20261030')
METHOD='New_Agent_R4'
BOOTSTRAP={'SnAr':4,'HPLC':3}
PARAMETERS={
    'version':'Continuous_R4','seed':0,'initial_queries':4,'additional_queries':16,
    'protected_R2_prefix':BOOTSTRAP,'GP_acquisition_loss_gate_ties':'unchanged original continuous adapter',
    'blind_proposal':{'maximum_calls_each':4,'minimum_gap':2,'maximum_before_round9':3,'last_round':14,
        'inputs':'physical domain, own observed physical points/objectives and own declared hypotheses only',
        'order':'LLM proposal committed before current GP computation/review; no GP table or BO reference',
        'projection':'nearest unqueried fixed-pool point in normalized Euclidean distance, original independent tie priority'},
    'local_model':{'scope':'original 32nd-nearest normalized RMS compact support',
        'focus':'affine contrast from one own observed reference to the projected proposal, along declared axes',
        'extrapolation_coordinate':[-1.,.5],'background_max_normalized_change':.25,
        'background_kernel':'(1-(maximum absolute off-focus change/0.25)^2)^2',
        'unidentified_contrast':'explicit local constant fallback, not a claimed slope',
        'feature':'compact kernel*(local forecast-GP mean), clipped to one GP predictive SD'},
    'validation':{'probe_amplitude':.25,'strong_signal_ratio':.05,'recent_query_horizon':6,'window_checks':4,
        'weak_weight':'min(1,(prequery probe/effective_SD/0.05)^2), positive and negative retained',
        'admission':'two distinct own prospective informative positives, latest informative positive and positive weighted score',
        'caps':{'two_positives':.25,'three_positives':.5},
        'after_failure_or_pause':'two new own informative positives after latest applicable actual failure/pause',
        'scheduled_tests_maximum':3,'scheduled_tests_per_model':2,'last_scheduled_round':14,
        'schedule':'top two positive pending models; nonreference point inside original gate, informative and distinguishes from BO reference',
        'test_priority':'maximum expected scalar information gain, contrast, lower base loss, original tie priority',
        'contrast_uncertainty':'sum of marginal effective variances; planning proxy, not a calibrated causal probability'},
    'shared_evidence':{'synchronize':'all existing and new hypotheses every round and after feedback',
        'related':'support overlap >=0.5, failed observation inside target support and target frozen probe also has negative gain',
        'actual_failure':'fresh-positive recovery clock applies to all related models, independent of creation time',
        'shadow_failure':'negative context score transfers; never transfers retrospective positive validation',
        'deduplication':'one inherited penalty per observed candidate/query; own check takes precedence',
        'LLM_pause':'deployment off, shadow monitoring continues, automatic evidence-based recovery; no destructive withdrawal'},
    'KF':{'initial_mean':.25,'initial_variance':.25,'process_variance':.0025,'measurement_variance_floor':.0009,
        'shadow':'separate bank; weak checks use R/weight','actual':'actual deployed correction only; negative gain cannot increase coefficient'},
    'arbitration':'validated strict local ranking improvement, then scheduled validation, then GP-approved blind proposal, then BO',
    'interpretation':'two protected-prefix continuations; no retuning or comparator reruns; experiment budgets matched, LLM budgets differ'}

pick=old.pick
setup=old.setup


def seed_cache(pool):
    target=ROOT/'cache'/pool['task'];target.mkdir(parents=True,exist_ok=True);count=0
    for root in [Path('data/runs/continuous_20261029'),kf.ROOT,Path('data/runs/continuous_r2_20261030'),old.ROOT]:
        for source in sorted((root/'cache'/pool['task']).glob('*.pt')):
            dest=target/source.name
            if dest.exists():assert old.data.filehash(source)==old.data.filehash(dest)
            else:shutil.copyfile(source,dest);count+=1
    return count


class Cache(base.Cache):
    def __init__(self,pool):
        super().__init__(pool);self.directory=ROOT/'cache'/pool['task'];self.directory.mkdir(parents=True,exist_ok=True)


def scope_weight(pool,h):
    weight=old.scope_weight(pool,h);z=np.asarray(pool['z']);focus=h['basis']['focus_indices']
    background=[j for j in range(z.shape[1]) if j not in focus]
    if background:
        delta=np.max(abs(z[:,background]-np.asarray(h['scope']['center'])[background]),axis=1)
        weight*=np.maximum(0.,1-(delta/.25)**2)**2
        weight[delta>=.25]=0.
    return weight


def local_forecast(pool,h):
    z=np.asarray(pool['z']);b=h['basis'];focus=b['focus_indices'];v=np.asarray(b['focus_displacement'])
    den=float(np.dot(v,v));center=np.asarray(h['scope']['center'])
    coordinate=(z[:,focus]-center[focus])@v/den if den>1e-12 else np.zeros(len(z))
    coordinate=np.clip(coordinate,-1.,.5)
    forecast=h['prediction_y']+(h['prediction_y']-b['reference_y'])*coordinate
    return np.clip(forecast,0.,2.)


def features(pool,h,stats):
    w=scope_weight(pool,h);raw=w*(local_forecast(pool,h)-stats['mu_y'])
    return np.clip(raw,-stats['sd_predictive_y'],stats['sd_predictive_y']),w


def projection(pool,state,t,point):
    z=np.asarray(pool['z']);target=np.array([(point[a]-pool['bounds'][a][0])/(pool['bounds'][a][1]-pool['bounds'][a][0]) for a in pool['axes']])
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]);distance=np.sum((z-target)**2,axis=1)
    js=np.flatnonzero(available);js=js[distance[js]==distance[js].min()]
    priority=base.priority(len(z),t,0);j=int(js[np.argmin(priority[js])])
    return {'candidate_id':pool['ids'][j],'index':j,'point':pool['points'][j],
        'proposed_physical_point':deepcopy(point),'normalized_Euclidean_distance':float(np.sqrt(distance[j])),
        'exact_distance_ties':len(js),'uses_GP':False}


def register(pool,state,t,plan,key):
    for name in plan['pause_ids']:
        h=state['hypotheses'][name];h['last_pause_query']=t;h['paused_by_LLM']=True;h['active_cap']=0.
        h['pause_records'].append({'query':t,'context_sha256':key,'reason':plan['reason']})
    if plan['action']=='abstain':return None,None
    p=projection(pool,state,t,plan['point']);j=p['index'];spec,_=old.scope(pool,j)
    r=plan['hypothesis']['reference_index'];ref=state['history'][r-1]
    refj=pool['ids'].index(ref['candidate_id']);focus=sorted(pool['axes'].index(a) for a in plan['hypothesis']['focus_axes'])
    v=np.asarray(pool['z'])[j,focus]-np.asarray(pool['z'])[refj,focus]
    basis={'reference_index':r,'reference_candidate_id':ref['candidate_id'],'reference_y':ref['y'],
        'focus_indices':focus,'focus_displacement':v.tolist(),
        'kind':'anchored_local_affine_contrast' if float(v@v)>1e-12 else 'unidentified_contrast_constant_fallback'}
    canonical=digest({'scope':spec,'basis':basis,'prediction_raw':float(plan['prediction'])});name='H_'+canonical[:16]
    if name in state['hypotheses']:
        state['hypotheses'][name]['proposal_contexts'].append(key);return name,p
    state['hypotheses'][name]={'id':name,'canonical_sha256':canonical,'order':len(state['hypotheses']),
        'created_query':t,'candidate_id':p['candidate_id'],'prediction_raw':float(plan['prediction']),
        'prediction_y':float(plan['prediction'])/pool['objective_scale'],'scope':spec,'basis':basis,
        'statement':plan['hypothesis']['statement'],'focus_axes':list(plan['hypothesis']['focus_axes']),
        'falsification':plan['hypothesis']['falsification'],'support':list(plan['support']),
        'proposal_contexts':[key],'shadow_records':[],'actual_records':[],
        'shadow_bank':old.new_bank(t),'actual_bank':old.new_bank(t),'last_actual_failure_query':-1,
        'inherited_actual_failure_query':-1,'inherited_failure_ids':[],'inherited_evidence':[],
        'paused_by_LLM':False,'last_pause_query':-1,'pause_records':[],'status':'unconfirmed','active_cap':0.}
    return name,p


def synchronize(pool,state,t):
    """Counterevidence is contextual, not retrospective positive validation."""
    changes=[]
    for key,h in state['hypotheses'].items():
        w=scope_weight(pool,h);f=local_forecast(pool,h);own={(r['query'],r['candidate_id']) for r in h['shadow_records']}
        records={};failed=[];cut=-1
        for event in state['evidence_ledger']:
            if event['query']>=t or event['source_hypothesis_id']==key:continue
            j=pool['ids'].index(event['candidate_id']);source_w=scope_weight(pool,event['source_model'])
            den=min(int(np.sum(w>1e-12)),int(np.sum(source_w>1e-12)))
            overlap=float(np.sum((w>1e-12)&(source_w>1e-12))/den) if den else 0.
            if overlap<.5 or w[j]<=1e-12:continue
            sd=event['GP_predictive_SD_y'];H=float(np.clip(w[j]*(f[j]-event['GP_mean_y']),-sd,sd));delta=.25*H
            R=sd**2+.0009;residual=event['observed_y']-event['GP_mean_y'];gain=float(2*residual*delta-delta**2)
            if gain>=-1e-12:continue
            ratio=abs(delta)/np.sqrt(R);weight=min(1.,(ratio/.05)**2)
            if event['kind']=='actual_failure':cut=max(cut,event['query']);failed.append(event['event_id'])
            token=(event['query'],event['candidate_id'])
            if token in own:continue
            row={'query':event['query'],'candidate_id':event['candidate_id'],'source_event_id':event['event_id'],
                'source_kind':event['kind'],'support_overlap':overlap,'target_probe_y':delta,
                'paired_SE_gain':gain,'weighted_standardized_gain':float(weight*gain/(2*R)),
                'kind':'inherited counterevidence only, never prospective positive validation'}
            if token not in records or row['weighted_standardized_gain']<records[token]['weighted_standardized_gain']:records[token]=row
        if sorted(failed)!=h['inherited_failure_ids']:
            changes.append({'query':t,'hypothesis_id':key,'before':h['inherited_failure_ids'],'after':sorted(failed)})
        h['inherited_failure_ids']=sorted(failed);h['inherited_actual_failure_query']=cut
        h['inherited_evidence']=sorted(records.values(),key=lambda r:(r['query'],r['candidate_id']))
    return changes


def evidence(h,t):
    assert all(r['query']<t for r in h['shadow_records'])
    cut=max(h['last_actual_failure_query'],h['inherited_actual_failure_query'],h['last_pause_query'])
    rows=[r for r in h['shadow_records'] if r['query']>cut and r['query']>=t-6][-4:]
    inherited=[r for r in h['inherited_evidence'] if r['query']>=t-6 and r['query']>cut]
    strong=[r for r in rows if r['informative']];positive={r['candidate_id'] for r in strong if r['paired_SE_gain']>1e-12}
    score=float(sum(r['weighted_standardized_gain'] for r in rows)+sum(r['weighted_standardized_gain'] for r in inherited))
    eligible=bool(len(positive)>=2 and score>0 and strong[-1]['paired_SE_gain']>1e-12)
    return {'eligible':eligible,'positive_count':len(positive),'strong_count':len(strong),'score':score,
        'cutoff_query':cut,'own_evidence_queries':[r['query'] for r in rows],
        'inherited_negative_queries':[r['query'] for r in inherited],'cap':.5 if len(positive)>=3 else .25,
        'latest_informative_gain':strong[-1]['paired_SE_gain'] if strong else None}


def authorize(pool,state,t):
    changes=synchronize(pool,state,t);events=[]
    for key,h in state['hypotheses'].items():
        proof=evidence(h,t);before=h['status'];eligible=proof['eligible']
        if eligible:
            coefficient=min(proof['cap'],max(0.,kf.forecast(h['shadow_bank'],t)['mean']))
            eligible=coefficient>1e-12
            if eligible and before not in ('probation','supported'):
                h['actual_bank']['mean']=coefficient;h['paused_by_LLM']=False
                events.append({'query':t,'hypothesis_id':key,'from':before,
                    'to':'supported' if proof['positive_count']>=3 else 'probation',
                    'authorized_coefficient':coefficient,'evidence':deepcopy(proof),
                    'kind':'two-point prospective shadow authorization; not actual KF gain'})
        if eligible:status='supported' if proof['positive_count']>=3 else 'probation'
        elif h['paused_by_LLM']:status='paused'
        elif max(h['last_actual_failure_query'],h['inherited_actual_failure_query'])>=0:status='quarantined'
        elif proof['score']<0:status='counterevidence'
        else:status='unconfirmed'
        h['status']=status;h['active_cap']=proof['cap'] if eligible else 0.;h['evidence']=proof
    state['authorization_events'].extend(deepcopy(events));state['inheritance_events'].extend(deepcopy(changes))
    return events,changes


def call_permission(state,t):
    calls=state['calls'];reasons=[];last=calls[-1]['query'] if calls else -100
    if len(calls)>=4:return {'allowed':False,'reason':'maximum_calls'}
    if t>14:return {'allowed':False,'reason':'leave_two_feedback_rounds'}
    if t-last<2:return {'allowed':False,'reason':'call_cooldown'}
    if t<9 and len(calls)>=3:return {'allowed':False,'reason':'reserve_late_call'}
    waiting=[h for h in state['hypotheses'].values() if not h['paused_by_LLM'] and not h['evidence']['eligible']
        and h['evidence']['positive_count']>=1 and h['evidence']['score']>0
        and any(r['informative'] and r['paired_SE_gain']>0 and r['query']>=t-2 for r in h['shadow_records'])]
    if waiting:return {'allowed':False,'reason':'prioritize_pending_positive_validation'}
    if not calls:reasons.append('first_fresh_blind_proposal')
    if not any(h['active_cap']>0 for h in state['hypotheses'].values()):reasons.append('no_deployable_model')
    history=state['history']
    if len(history)>=7 and max(h['y'] for h in history)-max(h['y'] for h in history[:-3])<.005:reasons.append('own_stagnation_3')
    return {'allowed':bool(reasons),'reason':','.join(reasons) if reasons else 'no_observed_event',
        'current_GP_consulted':False}


def guidance(cache,state,d,t):
    n=len(cache.ids);keys=list(state['hypotheses']);matrix=[];weights=[];priors=[]
    for key in keys:
        h=state['hypotheses'][key];f,w=features(cache.pool,h,d['stats']);matrix.append(f);weights.append(w)
        priors.append({'hypothesis_id':key,'shadow':kf.forecast(h['shadow_bank'],t),'actual':kf.forecast(h['actual_bank'],t),
            'status':h['status'],'active_cap':h['active_cap'],'evidence':deepcopy(h.get('evidence',evidence(h,t)))})
    matrix=np.asarray(matrix).reshape(len(keys),n);weights=np.asarray(weights).reshape(len(keys),n)
    owner=np.full(n,-1,dtype=np.int64);shift=np.zeros(n);coefficient=np.zeros(n)
    active=[k for k,key in enumerate(keys) if state['hypotheses'][key]['active_cap']>0]
    active.sort(key=lambda k:(-priors[k]['evidence']['score'],-priors[k]['evidence']['positive_count'],state['hypotheses'][keys[k]]['order']))
    for k in active:
        alpha=min(priors[k]['active_cap'],priors[k]['actual']['mean'])
        mask=(owner<0)&(weights[k]>1e-12)&(abs(matrix[k])>1e-12)&(alpha>0)
        owner[mask]=k;shift[mask]=alpha*matrix[k,mask];coefficient[mask]=alpha
    score,meta=cache.score(state['history'],shift/d['stats']['std']) if np.any(shift) else (d['base'],d['base_meta'])
    return {'keys':keys,'features_y':matrix,'weights':weights,'priors':priors,'owner':owner,
        'shift_y':shift,'coefficient':coefficient,'score':score,'meta':meta,'guided':pick(d,score)}


def validation_schedule(state,d,g,t):
    result={'selected_index':None,'hypothesis_id':None,'reason':'no_supported_validation_candidate','candidates':[]}
    if t>14 or int(d['eligible'].sum())<2 or len(state['scheduled_validations'])>=3:
        result['reason']='validation_window_or_budget_closed';return result
    waiting=[]
    for k,key in enumerate(g['keys']):
        h=state['hypotheses'][key];proof=evidence(h,t)
        if h['created_query']>=t or h['paused_by_LLM'] or proof['eligible'] or proof['positive_count']<1 or proof['score']<=0:continue
        if sum(v['hypothesis_id']==key for v in state['scheduled_validations'])>=2:continue
        waiting.append((k,key,proof))
    waiting.sort(key=lambda v:(-v[2]['positive_count'],-v[2]['score'],state['hypotheses'][v[1]]['order']))
    ref=d['reference'];R=d['stats']['sd_predictive_y']**2+.0009
    for k,key,proof in waiting[:2]:
        H=g['features_y'][k];probe=.25*H;ratio=abs(probe)/np.sqrt(R)
        contrast=abs(probe-probe[ref])/np.sqrt(R+R[ref]);mask=d['eligible']&(ratio>=.05)&(contrast>=.05)
        mask[ref]=False
        for j in np.flatnonzero(mask):
            info=.5*np.log1p(g['priors'][k]['shadow']['variance']*H[j]**2/R[j])
            result['candidates'].append({'index':int(j),'hypothesis_id':key,'expected_information_gain':float(info),
                'contrast_proxy':float(contrast[j]),'probe_ratio':float(ratio[j]),
                'base_loss':float(d['base'][ref]-d['base'][j]),'tie_priority':int(d['tie_priority'][j])})
    if result['candidates']:
        winner=min(result['candidates'],key=lambda r:(-r['expected_information_gain'],-r['contrast_proxy'],r['base_loss'],r['tie_priority']))
        result.update(selected_index=winner['index'],hypothesis_id=winner['hypothesis_id'],reason='informative_nonreference_inside_original_gate')
    return result


def decide(cache,state,d,g,t,projected):
    ref=d['reference'];candidate=projected['index'] if projected else None;reviewed=bool(candidate is not None and d['eligible'][candidate])
    meaningful=bool(g['guided']!=ref and g['score'][g['guided']]>g['score'][ref]+1e-12)
    validation=validation_schedule(state,d,g,t)
    if meaningful:selected=g['guided'];mode='validated_local_guidance'
    elif validation['selected_index'] is not None:
        selected=validation['selected_index'];mode='scheduled_local_validation'
        state['scheduled_validations'].append({'query':t,'hypothesis_id':validation['hypothesis_id'],'selected_index':selected})
    elif reviewed:selected=candidate;mode='GP_reviewed_blind_LLM_trial'
    else:selected=ref;mode='BO_rejected_blind_proposal' if candidate is not None else 'BO_no_new_proposal'
    assert d['eligible'][selected] and d['available'][selected]
    return {'selected':int(selected),'mode':mode,'proposed_index':candidate,'GP_review_accepted':reviewed,
        'GP_review_loss':float(d['base'][ref]-d['base'][candidate]) if candidate is not None else None,
        'local_ranking_changed':meaningful,'proposal_selected':candidate==selected if candidate is not None else False,
        'validation':validation}


def update_bank(bank,prior,t,H,mu,sd,y,cap,weight=1.):
    a=float(prior['mean']);P=float(prior['variance']);R=float((sd**2+.0009)/weight)
    delta=a*H;residual=y-mu;innovation=residual-delta;S=P*H*H+R;K=P*H/S
    raw=a+K*innovation;value=float(np.clip(raw,0.,cap));variance=P*R/S;gain=residual**2-(residual-delta)**2
    if gain < -1e-12:assert value<=a+1e-12
    after={'mean':value,'variance':float(variance),'last_round':t,'updates':bank['updates']+1,
        'informative_updates':bank['informative_updates']+int(abs(H)>1e-12)}
    m={'feature':H,'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,'observed_y':y,
        'actual_deployed_mean_y':mu+delta,'actual_deployed_correction_y':delta,'actual_paired_SE_gain':gain,
        'innovation':innovation,'innovation_z':float(innovation/np.sqrt(S)),'measurement_variance':R,'signal_weight':weight,
        'gain':K,'coefficient_before':a,'coefficient_after_unprojected':raw,'coefficient_after':value,'cap':cap}
    return after,m


def feedback(pool,state,d,g,decision,t,y):
    j=decision['selected'];cid=pool['ids'][j];mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j]);R=sd**2+.0009
    shadows=[];actual=None;events=[]
    for k,key in enumerate(g['keys']):
        h=state['hypotheses'][key]
        if g['weights'][k,j]<=1e-12:continue
        H=float(g['features_y'][k,j]);delta=.25*H;ratio=abs(delta)/np.sqrt(R);weight=float(min(1.,(ratio/.05)**2))
        gain=float((y-mu)**2-(y-mu-delta)**2);strong=bool(ratio>=.05 and abs(H)>1e-12)
        row={'query':t,'candidate_id':cid,'GP_mean_y':mu,'GP_predictive_SD_y':sd,'feature_y':H,
            'kernel_weight':float(g['weights'][k,j]),'probe_amplitude':.25,'probe_prediction_y':mu+delta,
            'observed_y':y,'paired_SE_gain':gain,'standardized_gain':gain/(2*R),'signal_weight':weight,
            'weighted_standardized_gain':weight*gain/(2*R),'probe_over_effective_SD':float(ratio),'informative':strong,
            'kind':'frozen prospective shadow; never actual deployment'}
        if weight>1e-12 and abs(H)>1e-12:
            h['shadow_bank'],m=update_bank(h['shadow_bank'],g['priors'][k]['shadow'],t,H,mu,sd,y,.5,weight)
            for a,b in [('actual_paired_SE_gain','hypothetical_paired_SE_gain'),('actual_deployed_mean_y','hypothetical_mean_y'),('actual_deployed_correction_y','hypothetical_correction_y')]:m[b]=m.pop(a)
            row['shadow_KF']=m
        h['shadow_records'].append(deepcopy(row));shadows.append({'hypothesis_id':key,**deepcopy(row)})
        if strong and gain < -1e-12:events.append((key,'shadow_failure',delta))
    owner=int(g['owner'][j])
    if owner>=0:
        key=g['keys'][owner];h=state['hypotheses'][key];prior=deepcopy(g['priors'][owner]['actual'])
        prior['mean']=min(prior['mean'],g['priors'][owner]['active_cap']);H=float(g['features_y'][owner,j])
        h['actual_bank'],m=update_bank(h['actual_bank'],prior,t,H,mu,sd,y,g['priors'][owner]['active_cap'])
        actual={'hypothesis_id':key,'query':t,'candidate_id':cid,'prior':prior,'measurement':m,'bank_after':deepcopy(h['actual_bank'])}
        h['actual_records'].append(deepcopy(actual))
        if m['actual_paired_SE_gain'] < -1e-12:
            h['last_actual_failure_query']=t;h['active_cap']=0.;h['status']='quarantined';events.append((key,'actual_failure',m['actual_deployed_correction_y']))
    for key,kind,delta in events:
        h=state['hypotheses'][key];model={k:deepcopy(h[k]) for k in ['scope','basis','prediction_y']}
        event={'query':t,'candidate_id':cid,'source_hypothesis_id':key,'kind':kind,'source_model':model,
            'source_deployed_or_probe_shift_y':delta,'GP_mean_y':mu,'GP_predictive_SD_y':sd,'observed_y':y}
        event['event_id']=digest(event);state['evidence_ledger'].append(event)
    changes=synchronize(pool,state,t+1);state['inheritance_events'].extend(deepcopy(changes))
    for h in state['hypotheses'].values():h['evidence_after_feedback']=evidence(h,t+1)
    deployed=mu+float(g['shift_y'][j]);gain=float((y-mu)**2-(y-deployed)**2)
    if actual:assert abs(actual['measurement']['actual_deployed_mean_y']-deployed)<1e-12
    else:assert gain==0. and g['shift_y'][j]==0.
    return {'shadow_checks':shadows,'actual_local_update':actual,'GP_mean_y':mu,'GP_predictive_SD_y':sd,
        'deployed_mean_y':deployed,'deployed_shift_y':float(g['shift_y'][j]),'actual_paired_SE_gain':gain,
        'actual_owner_hypothesis':g['keys'][owner] if owner>=0 else None,'post_feedback_inheritance_changes':changes}
