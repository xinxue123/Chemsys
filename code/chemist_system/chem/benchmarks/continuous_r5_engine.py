"""R5: blind physical proposals, scheduled local validation and shared evidence.

The fixed pool, exact GP, acquisition, seeds and normal loss gate are reused.
This module has no LLM client or experimental outcome provider.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from . import continuous_r3_engine as old
from . import continuous_r4_engine as previous
from . import continuous_benchmark_engine as base
from . import revised_continuous_engine as kf
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_r5_20261030')
METHOD='New_Agent_R5'
BOOTSTRAP={'SnAr':4,'HPLC':3}
PARAMETERS={
    'version':'Continuous_R5','seed':0,'initial_queries':4,'additional_queries':16,
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

PARAMETERS['blind_proposal']={'maximum_calls_each':4,'minimum_gap':2,'maximum_before_round11':2,'last_round':14,
    'menu':'input-only actual unqueried points: nearest12 per high reference plus12 greedy maximin points',
    'references':'up to3 observed references, objective >=0.75*incumbent and normalized RMS separation >=0.1',
    'inputs':'observations, initial uncertain preferences, physical menu, past declarations; no GP information',
    'order':'controller computes GP feasibility first; blind proposer receives no GP information; review follows proposal',
    'projection':'exact displayed candidate identity; explicit focus direction and background contract checked'}
PARAMETERS['shared_evidence']={'synchronize':'all existing/new models each round and after feedback',
    'point_context':'failed observation inside target contract, target frozen probe negative; no global support-overlap cutoff',
    'region_failure':'only own actual deployment failure resets hard recovery clock; shared failures are point-context penalties',
    'weak_negative':'retain nonzero weak negative checks with prequery signal weights',
    'deduplication':'one penalty per candidate/query; own check takes precedence; never transfer positive validation',
    'pause':'idempotent while paused; retain shadow monitoring and original recovery epoch'}
PARAMETERS['validation'].update(last_scheduled_round=15,initial_probe_last_round=14,maximum_initial_probes=1,
    schedule='confirmations before at most one unconfirmed initial probe; original signal, contrast and loss gate unchanged',
    waiting='defer new LLM only when validation is executable now')
PARAMETERS['local_model']['semantic_contract']='explicit directions on actual displayed inputs; scope preserves focus directions and background change <=0.25 from reference'
pick=old.pick
setup=old.setup


def seed_cache(pool):
    target=ROOT/'cache'/pool['task'];target.mkdir(parents=True,exist_ok=True);count=0
    for root in [Path('data/runs/continuous_20261029'),kf.ROOT,Path('data/runs/continuous_r2_20261030'),old.ROOT,previous.ROOT]:
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
    ref=np.asarray(pool['z'][pool['ids'].index(h['basis']['reference_candidate_id'])])
    for axis,direction in h['semantic_contract']['directions'].items():
        a=pool['axes'].index(axis);sign=1 if direction=='increase' else -1
        weight[(z[:,a]-ref[a])*sign<=1e-12]=0.
    if background:weight[np.max(abs(z[:,background]-ref[background]),axis=1)>.25+1e-12]=0.
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


def high_references(pool,state):
    z=np.asarray(pool['z']);ix={k:j for j,k in enumerate(pool['ids'])};best=max(h['y'] for h in state['history'])
    order=sorted(range(len(state['history'])),key=lambda i:(-state['history'][i]['y'],tuple(z[ix[state['history'][i]['candidate_id']]])))
    selected=[]
    for i in order:
        h=state['history'][i];j=ix[h['candidate_id']]
        if h['y']<.75*best:continue
        if any(np.sqrt(np.mean((z[j]-z[ix[state['history'][v]['candidate_id']]])**2))<.1 for v in selected):continue
        selected.append(i)
        if len(selected)==3:break
    return [{'index':i+1,'candidate_id':state['history'][i]['candidate_id'],
        'point':deepcopy(pool['points'][ix[state['history'][i]['candidate_id']]]),
        'objective':state['history'][i]['raw_value']} for i in selected]


def candidate_menu(pool,state,t):
    z=np.asarray(pool['z']);ix={key:j for j,key in enumerate(pool['ids'])};ranks=base.priority(len(z),t,0)
    queried=[ix[h['candidate_id']] for h in state['history']];available=~np.isin(np.arange(len(z)),queried)
    js=np.flatnonzero(available);refs=high_references(pool,state);chosen=set()
    for ref in refs:
        distance=np.mean((z-z[ix[ref['candidate_id']]])**2,axis=1)
        near=js[np.lexsort((ranks[js],distance[js]))[:12]];chosen.update(map(int,near))
    distance=np.min(np.mean((z[:,None,:]-z[queried][None,:,:])**2,axis=2),axis=1)
    for j in chosen:distance=np.minimum(distance,np.mean((z-z[j])**2,axis=1))
    for unused in range(min(12,len(js))):
        eligible=np.array([j for j in js if int(j) not in chosen],dtype=int)
        if not len(eligible):break
        far=eligible[distance[eligible]==distance[eligible].max()];j=int(far[np.argmin(ranks[far])]);chosen.add(j)
        distance=np.minimum(distance,np.mean((z-z[j])**2,axis=1))
    rows=[]
    for j in sorted(chosen,key=lambda k:ranks[k]):
        ref=min(refs,key=lambda r:float(np.mean((z[j]-z[ix[r['candidate_id']]])**2)))
        delta=abs(z[j]-z[ix[ref['candidate_id']]]);focus=np.argsort(-delta,kind='stable')[:min(2,z.shape[1])]
        background=[a for a in range(z.shape[1]) if a not in focus]
        rows.append({'candidate_id':pool['ids'][j],'point':deepcopy(pool['points'][j]),
            'nearest_reference_index':ref['index'],'largest_changed_axes':[pool['axes'][a] for a in focus],
            'background_max_change_after_two_axes':float(delta[background].max()) if background else 0.})
    return rows


def direction_check(pool,reference,point,focus,directions):
    violations=[]
    for axis in focus:
        change=point[axis]-reference[axis];sign=1 if directions[axis]=='increase' else -1
        if change*sign<=1e-12:violations.append('focus_direction_mismatch:'+axis)
    background=[a for a in pool['axes'] if a not in focus]
    for axis in background:
        lo,hi=pool['bounds'][axis]
        if abs(point[axis]-reference[axis])/(hi-lo)>.25+1e-12:violations.append('background_outside_contract:'+axis)
    return {'admitted':not violations,'violations':violations}


def projection(pool,state,t,plan):
    key=plan['candidate_id'];j=pool['ids'].index(key)
    assert key not in {h['candidate_id'] for h in state['history']}
    assert key in {r['candidate_id'] for r in candidate_menu(pool,state,t)}
    reference=state['history'][plan['hypothesis']['reference_index']-1]
    reference_point=pool['points'][pool['ids'].index(reference['candidate_id'])]
    point=pool['points'][j];guard=direction_check(pool,reference_point,point,plan['hypothesis']['focus_axes'],plan['hypothesis']['directions'])
    return {'candidate_id':key,'index':j,'point':deepcopy(point),'proposed_physical_point':deepcopy(point),
        'normalized_Euclidean_distance':0.,'exact_distance_ties':1,'uses_GP':False,'mode':'exact_input_identity',
        'semantic_review':guard}


def register(pool,state,t,plan,key):
    for name in plan['pause_ids']:
        h=state['hypotheses'][name];started=not h['paused_by_LLM'];before=h['last_pause_query']
        if started:h['last_pause_query']=t
        h['paused_by_LLM']=True;h['active_cap']=0.
        h['pause_records'].append({'query':t,'context_sha256':key,'reason':plan['reason'],
            'new_pause_epoch':started,'epoch_before':before,'epoch_after':h['last_pause_query']})
    if plan['action']=='abstain':return None,None
    p=projection(pool,state,t,plan)
    if not p['semantic_review']['admitted']:return None,p
    j=p['index'];spec,_=old.scope(pool,j)
    r=plan['hypothesis']['reference_index'];ref=state['history'][r-1]
    refj=pool['ids'].index(ref['candidate_id']);focus=sorted(pool['axes'].index(a) for a in plan['hypothesis']['focus_axes'])
    v=np.asarray(pool['z'])[j,focus]-np.asarray(pool['z'])[refj,focus]
    basis={'reference_index':r,'reference_candidate_id':ref['candidate_id'],'reference_y':ref['y'],
        'focus_indices':focus,'focus_displacement':v.tolist(),
        'kind':'anchored_local_affine_contrast' if float(v@v)>1e-12 else 'unidentified_contrast_constant_fallback'}
    contract={'directions':deepcopy(plan['hypothesis']['directions']),'reference_index':r,
        'actual_candidate_id':p['candidate_id'],'background_max_change':.25,'projection_distance':0.}
    canonical=digest({'scope':spec,'basis':basis,'contract':contract,'prediction_raw':float(plan['prediction'])});name='H_'+canonical[:16]
    if name in state['hypotheses']:
        state['hypotheses'][name]['proposal_contexts'].append(key);return name,p
    state['hypotheses'][name]={'id':name,'canonical_sha256':canonical,'order':len(state['hypotheses']),
        'created_query':t,'candidate_id':p['candidate_id'],'prediction_raw':float(plan['prediction']),
        'prediction_y':float(plan['prediction'])/pool['objective_scale'],'scope':spec,'basis':basis,'semantic_contract':contract,
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
        records={};failed=[]
        for event in state['evidence_ledger']:
            if event['query']>=t or event['source_hypothesis_id']==key:continue
            j=pool['ids'].index(event['candidate_id']);source_w=scope_weight(pool,event['source_model'])
            den=min(int(np.sum(w>1e-12)),int(np.sum(source_w>1e-12)))
            overlap=float(np.sum((w>1e-12)&(source_w>1e-12))/den) if den else 0.
            if w[j]<=1e-12:continue
            sd=event['GP_predictive_SD_y'];H=float(np.clip(w[j]*(f[j]-event['GP_mean_y']),-sd,sd));delta=.25*H
            R=sd**2+.0009;residual=event['observed_y']-event['GP_mean_y'];gain=float(2*residual*delta-delta**2)
            if gain>=-1e-12:continue
            ratio=abs(delta)/np.sqrt(R);weight=min(1.,(ratio/.05)**2)
            if event['kind']=='actual_failure':failed.append(event['event_id'])
            token=(event['query'],event['candidate_id'])
            if token in own:continue
            row={'query':event['query'],'candidate_id':event['candidate_id'],'source_event_id':event['event_id'],
                'source_kind':event['kind'],'support_overlap':overlap,'target_probe_y':delta,
                'paired_SE_gain':gain,'weighted_standardized_gain':float(weight*gain/(2*R)),
                'kind':'inherited counterevidence only, never prospective positive validation'}
            if token not in records or row['weighted_standardized_gain']<records[token]['weighted_standardized_gain']:records[token]=row
        updated=sorted(records.values(),key=lambda r:(r['query'],r['candidate_id']))
        if digest(updated)!=digest(h['inherited_evidence']):
            changes.append({'query':t,'hypothesis_id':key,'kind':'point_context_synchronized',
                'before_event_ids':[r['source_event_id'] for r in h['inherited_evidence']],
                'after_event_ids':[r['source_event_id'] for r in updated]})
        h['inherited_failure_ids']=sorted(failed);h['inherited_actual_failure_query']=-1
        h['inherited_evidence']=updated
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


def call_permission(pool,state,d,g,validation,t):
    calls=state['calls'];reasons=[];last=calls[-1]['query'] if calls else -100
    if int(d['eligible'].sum())<2:return {'allowed':False,'reason':'singleton_budget_no_choice'}
    ref=d['reference'];strict=bool(g['guided']!=ref and g['score'][g['guided']]>g['score'][ref]+1e-12)
    if strict:return {'allowed':False,'reason':'validated_guidance_executable'}
    if validation['selected_index'] is not None:
        return {'allowed':False,'reason':'validation_executable_now','validation_index':validation['selected_index']}
    if len(calls)>=4:return {'allowed':False,'reason':'maximum_calls'}
    if t>14:return {'allowed':False,'reason':'leave_two_feedback_rounds'}
    if t-last<2:return {'allowed':False,'reason':'call_cooldown'}
    if t<11 and len(calls)>=2:return {'allowed':False,'reason':'reserve_two_late_calls'}
    if not calls:reasons.append('first_fresh_blind_proposal')
    history=state['history']
    if len(history)>=7 and max(h['y'] for h in history)-max(h['y'] for h in history[:-3])<.005:reasons.append('own_stagnation_3')
    recent_high=[r for r in high_references(pool,state) if r['index']>last+3]
    if calls and recent_high:reasons.append('new_high_reference_region')
    latest=[r for h in state['hypotheses'].values() for r in h['shadow_records'] if r['query']==t-1 and r['informative'] and r['paired_SE_gain']<0]
    if latest:reasons.append('new_local_counterevidence')
    if not state['hypotheses']:reasons.append('no_admissible_hypothesis')
    untestable=[key for key,h in state['hypotheses'].items() if h.get('evidence',{}).get('positive_count',0)>0 and h['active_cap']==0]
    return {'allowed':bool(reasons),'reason':','.join(reasons) if reasons else 'no_observed_event',
        'untestable_pending_models_released':untestable,'controller_GP_feasibility_used':True,'GP_information_given_to_LLM':False}


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
    result={'selected_index':None,'hypothesis_id':None,'stage':None,'reason':'no_executable_validation_candidate','candidates':[]}
    if t>15 or int(d['eligible'].sum())<2 or len(state['scheduled_validations'])>=3:
        result['reason']='validation_window_or_budget_closed';return result
    waiting=[]
    for k,key in enumerate(g['keys']):
        h=state['hypotheses'][key];proof=evidence(h,t)
        if h['created_query']>=t or h['paused_by_LLM'] or proof['eligible']:continue
        if sum(v['hypothesis_id']==key for v in state['scheduled_validations'])>=2:continue
        if proof['positive_count']>=1 and proof['score']>0 and proof['latest_informative_gain']>0:stage='confirmation'
        elif t<=14 and proof['strong_count']==0 and proof['score']>=0 and not any(v['stage']=='initial_probe' for v in state['scheduled_validations']):stage='initial_probe'
        else:continue
        waiting.append((k,key,proof,stage))
    waiting.sort(key=lambda v:(v[3]!='confirmation',-v[2]['positive_count'],-v[2]['score'],state['hypotheses'][v[1]]['order']))
    ref=d['reference'];R=d['stats']['sd_predictive_y']**2+.0009
    for k,key,proof,stage in waiting[:2]:
        H=g['features_y'][k];probe=.25*H;ratio=abs(probe)/np.sqrt(R)
        contrast=abs(probe-probe[ref])/np.sqrt(R+R[ref]);mask=d['eligible']&(ratio>=.05)&(contrast>=.05)
        mask[ref]=False
        for j in np.flatnonzero(mask):
            info=.5*np.log1p(g['priors'][k]['shadow']['variance']*H[j]**2/R[j])
            result['candidates'].append({'index':int(j),'hypothesis_id':key,'stage':stage,'expected_information_gain':float(info),
                'contrast_proxy':float(contrast[j]),'probe_ratio':float(ratio[j]),
                'base_loss':float(d['base'][ref]-d['base'][j]),'tie_priority':int(d['tie_priority'][j])})
    if result['candidates']:
        winner=min(result['candidates'],key=lambda r:(r['stage']!='confirmation',-r['expected_information_gain'],-r['contrast_proxy'],r['base_loss'],r['tie_priority']))
        result.update(selected_index=winner['index'],hypothesis_id=winner['hypothesis_id'],stage=winner['stage'],reason='informative_nonreference_inside_original_gate')
    return result


def decide(cache,state,d,g,t,projected):
    ref=d['reference'];candidate=projected['index'] if projected else None
    semantic=bool(projected is not None and projected['semantic_review']['admitted'])
    reviewed=bool(semantic and candidate is not None and d['eligible'][candidate])
    meaningful=bool(g['guided']!=ref and g['score'][g['guided']]>g['score'][ref]+1e-12)
    validation=validation_schedule(state,d,g,t)
    if meaningful:selected=g['guided'];mode='validated_local_guidance'
    elif validation['selected_index'] is not None:
        selected=validation['selected_index'];mode='scheduled_local_validation'
        state['scheduled_validations'].append({'query':t,'hypothesis_id':validation['hypothesis_id'],'selected_index':selected,'stage':validation['stage']})
    elif reviewed:selected=candidate;mode='GP_reviewed_blind_LLM_trial'
    else:selected=ref;mode='BO_semantically_rejected_proposal' if candidate is not None and not semantic else 'BO_rejected_blind_proposal' if candidate is not None else 'BO_no_new_proposal'
    assert d['eligible'][selected] and d['available'][selected]
    return {'selected':int(selected),'mode':mode,'proposed_index':candidate,'semantic_review_accepted':semantic,
        'GP_gate_contains_nominee':bool(candidate is not None and d['eligible'][candidate]),'GP_review_accepted':reviewed,
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
        if weight>1e-12 and gain < -1e-12:events.append((key,'shadow_failure',delta))
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
        h=state['hypotheses'][key];model={k:deepcopy(h[k]) for k in ['scope','basis','prediction_y','semantic_contract']}
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
