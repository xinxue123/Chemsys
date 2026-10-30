"""R3: GP-reviewed proposals and independently validated local forecasts.

Input-only policy code. No outcome provider or LLM is called here. The base GP,
MC acquisition, normal budget, fit seeds and exact tie rule are unchanged.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from . import continuous_benchmark_engine as base
from . import continuous_benchmark_data as data
from . import revised_continuous_engine as kf
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_r3_20261030')
METHOD='New_Agent_R3'
BOOTSTRAP={'SnAr':4,'HPLC':3}
PARAMETERS={
    'version':'Continuous_R3','seed':0,'initial_queries':4,'additional_queries':16,
    'GP_acquisition_budget_ties':'unchanged original continuous adapter, MC512 qLogEI and prefix seeds',
    'protected_R2_prefix':BOOTSTRAP,'LLM':'one proposal review per remaining round; may abstain or withdraw models',
    'proposal_card':{'base_top':16,'incumbent_neighbors':16,'unconfirmed_models':2,'neighbors_each':8,
                     'order':'independent existing tie priority; full base GP still scores every fixed-pool candidate'},
    'local_forecast':{'model':'local constant LLM response-level forecast, not an inferred causal slope',
        'scope':'normalized RMS distance to the 32nd nearest fixed-pool input; include distance ties',
        'neighbors':32,'kernel':'(1-(distance/radius)^2)^2 inside radius, zero outside',
        'feature':'kernel*(LLM_forecast_y-GP_mean_y), clipped to one GP predictive SD',
        'maximum_prediction_raw':'2*objective_scale'},
    'validation':{'shadow_probe':.25,'window':4,'minimum_probe_over_effective_SD':.05,
        'score':'sum of prequery paired squared-error gain / (2*(GP_predictive_SD^2+0.0009))',
        'probation':'at least one informative positive, positive window score and latest informative gain positive',
        'supported':'at least two positives under the same criteria',
        'after_actual_failure':'at least two new informative positives after failure; shadow evaluation continues',
        'overlapping_failed_model':'overlap of geometric support / smaller support >=0.5 requires two fresh positives'},
    'local_amplitude':{'probation_cap':.25,'supported_cap':.5,'initial_mean':.25,'initial_variance':.25,
                       'process_variance':.0025,'measurement_variance_floor':.0009},
    'arbitration':'strict ranking improvement from validated local mean, else GP-approved LLM trial, else base BO',
    'owners':'one eligible local model per candidate, highest historical validation score; no summing duplicate claims',
    'recovery':'per model, no global pause; withdrawals do not need a positive prediction gain',
    'legacy_basis':'R2 covariance-field coefficients are archived, not numerically transferred to the new local-forecast basis',
    'interpretation':'single frozen version, only two new trajectories, protected-prefix continuation; no post-result tuning',
}


def seed_cache(pool):
    target=ROOT/'cache'/pool['task'];target.mkdir(parents=True,exist_ok=True);count=0
    for root in [Path('data/runs/continuous_20261029'),kf.ROOT,Path('data/runs/continuous_r2_20261030')]:
        for source in sorted((root/'cache'/pool['task']).glob('*.pt')):
            destination=target/source.name
            if not destination.exists():shutil.copyfile(source,destination);count+=1
            else:assert data.filehash(source)==data.filehash(destination)
    return count


class Cache(base.Cache):
    def __init__(self,pool):
        super().__init__(pool);self.directory=ROOT/'cache'/pool['task'];self.directory.mkdir(parents=True,exist_ok=True)


def pick(d,scores,mask=None):
    js=np.flatnonzero(d['eligible'] if mask is None else mask);assert len(js)
    js=js[scores[js]==scores[js].max()]
    js=js[d['information_proxy'][js]==d['information_proxy'][js].max()]
    js=js[d['stats']['sd_y'][js]==d['stats']['sd_y'][js].max()]
    return int(js[np.argmin(d['tie_priority'][js])])


def setup(cache,state,t):
    scores,meta=cache.score(state['history']);stats=cache.model(state['history'])
    available=~np.isin(cache.pool['ids'],[h['candidate_id'] for h in state['history']])
    proxy,pairs,unseen=base.information_proxy(cache.X.numpy(),[cache.ix[h['candidate_id']] for h in state['history']])
    d={'base':scores,'base_meta':meta,'stats':stats,'available':available,
       'information_proxy':proxy,'tie_priority':base.priority(len(scores),t,0),
       'budget':float(np.log(4)*(16-t)/15),'isolated_pairs':pairs,'unseen_values':unseen}
    ref=pick(d,scores,available);d['reference']=ref
    d['eligible']=available&(scores[ref]-scores<=d['budget'])
    return d


def scope(pool,j):
    z=np.asarray(pool['z']);distance=np.sqrt(np.mean((z-z[j])**2,axis=1))
    radius=float(np.sort(distance)[min(32,len(distance))-1])
    assert radius>0.
    inside=distance<=radius+1e-12
    weight=np.maximum(0.,1-(distance/radius)**2)**2;weight[~inside]=0.
    weight[distance>=radius]=0.
    return {'anchor_candidate_id':pool['ids'][j],'center':pool['z'][j],
            'radius_RMS_normalized':radius,'target_neighbors':min(32,len(distance)),
            'membership_count_with_ties':int(inside.sum()),'positive_weight_count':int(np.sum(weight>1e-12))},weight


def scope_weight(pool,hypothesis):
    z=np.asarray(pool['z']);center=np.asarray(hypothesis['scope']['center']);r=hypothesis['scope']['radius_RMS_normalized']
    distance=np.sqrt(np.mean((z-center)**2,axis=1))
    weight=np.maximum(0.,1-(distance/r)**2)**2;weight[distance>=r]=0.
    return weight


def features(pool,hypothesis,stats):
    weight=scope_weight(pool,hypothesis)
    raw=weight*(hypothesis['prediction_y']-stats['mu_y'])
    feature=np.clip(raw,-stats['sd_predictive_y'],stats['sd_predictive_y'])
    return feature,weight


def new_bank(t):
    bank=kf.new_bank(t);bank['mean']=.25;return bank


def update_bank(bank,prior,t,feature,mu,sd,y,cap):
    after,measurement=kf.update_bank(bank,prior,t,feature,mu,sd,y,True)
    after['mean']=float(min(cap,after['mean']))
    measurement['coefficient_after']=after['mean'];measurement['cap']=cap
    return after,measurement


def evidence(hypothesis):
    since=hypothesis.get('last_actual_failure_query',-1)
    rows=[row for row in hypothesis['shadow_records'] if row['informative'] and row['query']>since][-4:]
    gains=[row['paired_SE_gain'] for row in rows];score=float(sum(row['standardized_gain'] for row in rows))
    positive=sum(gain>1e-12 for gain in gains)
    required=2 if since>=0 or hypothesis['inherited_actual_failure_ids'] else 1
    eligible=bool(len(rows)>=required and positive>=required and score>0. and gains[-1]>1e-12)
    return {'eligible':eligible,'informative_count':len(rows),'positive_count':positive,'required_positives':required,
            'score':score,'latest_gain':gains[-1] if gains else None,
            'evidence_queries':[row['query'] for row in rows],
            'cap':.5 if positive>=2 else .25}


def authorize(state,t):
    changes=[]
    for key,h in state['hypotheses'].items():
        assert all(row['query']<t for row in h['shadow_records']),'Authorization requires strictly earlier experimental feedback.'
        proof=evidence(h);before=h['status'];eligible=proof['eligible'] and not h['withdrawn']
        if h['withdrawn']:status='withdrawn'
        elif eligible:status='supported' if proof['positive_count']>=2 else 'probation'
        elif h['last_actual_failure_query']>=0:status='quarantined'
        else:status='unconfirmed'
        if eligible and before not in ('probation','supported'):
            prior=kf.forecast(h['shadow_bank'],t)
            value=min(proof['cap'],max(0.,prior['mean']))
            if value<=1e-12:
                eligible=False;status='unconfirmed'
            else:
                h['actual_bank']['mean']=value
                changes.append({'query':t,'hypothesis_id':key,'from':before,'to':status,
                    'authorized_coefficient':value,'evidence':deepcopy(proof),
                    'kind':'shadow-evidence authorization, not actual-deployment KF gain'})
        h['status']=status;h['evidence']=proof
        h['active_cap']=proof['cap'] if eligible else 0.
    state['authorization_events'].extend(deepcopy(changes));return changes


def candidate_card(cache,state,d):
    n=len(cache.pool['ids']);z=cache.X.numpy();available=d['available'];roles={}
    def add(js,role):
        for j in js:roles.setdefault(int(j),[]).append(role)
    js=np.flatnonzero(available)
    top=sorted(js,key=lambda j:(-d['base'][j],-d['information_proxy'][j],-d['stats']['sd_y'][j],d['tie_priority'][j]))[:16];add(top,'base_top16')
    add([d['reference']],'BO_reference')
    anchor=max(state['history'],key=lambda h:h['y']);ja=cache.ix[anchor['candidate_id']]
    near=sorted(js,key=lambda j:(float(np.mean((z[j]-z[ja])**2)),d['tie_priority'][j]))[:16]
    add(near,'incumbent_neighbors16')
    waiting=[h for h in state['hypotheses'].values() if h['status'] in ('unconfirmed','quarantined') and not h['withdrawn']]
    waiting.sort(key=lambda h:(len(h['shadow_records']),h['created_query'],h['order']))
    for h in waiting[:2]:
        center=np.asarray(h['scope']['center'])
        near=sorted(js,key=lambda j:(float(np.mean((z[j]-center)**2)),d['tie_priority'][j]))[:8]
        add(near,'local_validation_opportunity')
    rows=[];scale=cache.pool['objective_scale']
    for j in sorted(roles,key=lambda j:d['tie_priority'][j]):
        local,_=scope(cache.pool,j)
        rows.append({'candidate_id':cache.pool['ids'][j],'point':cache.pool['points'][j],
            'GP_mean':float(d['stats']['mu_y'][j])*scale,'GP_predictive_SD':float(d['stats']['sd_predictive_y'][j])*scale,
            'base_qLogEI':float(d['base'][j]),'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),
            'inside_review_gate':bool(d['eligible'][j]),'roles':roles[j],
            'local_scope_radius_RMS':local['radius_RMS_normalized']})
    return rows


def register(pool,state,t,plan,context_key):
    for key in plan['withdraw_ids']:
        h=state['hypotheses'][key];h['withdrawn']=True;h['status']='withdrawn'
        h['withdrawal_records'].append({'query':t,'context_sha256':context_key,'reason':plan['reason']})
    if plan['action']=='abstain':return None
    j=pool['ids'].index(plan['candidate_id']);local,_=scope(pool,j)
    canonical=digest({'center':local['center'],'radius':local['radius_RMS_normalized'],
                      'prediction_raw':float(plan['prediction'])})
    key='H_'+canonical[:16]
    if key in state['hypotheses']:
        assert state['hypotheses'][key]['canonical_sha256']==canonical
        state['hypotheses'][key]['proposal_contexts'].append(context_key)
        return key
    inherited=[];new_weight=scope_weight(pool,{'scope':local})>1e-12
    for source,h in state['hypotheses'].items():
        if h['last_actual_failure_query']<0:continue
        old_weight=scope_weight(pool,h)>1e-12;den=min(int(new_weight.sum()),int(old_weight.sum()))
        overlap=float(np.sum(new_weight&old_weight)/den) if den else 0.
        if overlap>=.5:inherited.append(source)
    state['hypotheses'][key]={'id':key,'canonical_sha256':canonical,'order':len(state['hypotheses']),
        'created_query':t,'candidate_id':plan['candidate_id'],'prediction_raw':float(plan['prediction']),
        'prediction_y':float(plan['prediction'])/pool['objective_scale'],'scope':local,
        'statement':plan['hypothesis']['statement'],'focus_axes':list(plan['hypothesis']['focus_axes']),
        'falsification':plan['hypothesis']['falsification'],'support':list(plan['support']),
        'proposal_contexts':[context_key],'shadow_records':[],'actual_records':[],
        'shadow_bank':new_bank(t),'actual_bank':new_bank(t),'last_actual_failure_query':-1,
        'inherited_actual_failure_ids':inherited,'withdrawn':False,'withdrawal_records':[],
        'status':'unconfirmed','active_cap':0.,'evidence':evidence({'shadow_records':[],
            'last_actual_failure_query':-1,'inherited_actual_failure_ids':inherited})}
    return key


def guidance(cache,state,d,t):
    n=len(cache.pool['ids']);keys=list(state['hypotheses']);feature=[];weight=[];prior=[]
    for key in keys:
        h=state['hypotheses'][key];f,w=features(cache.pool,h,d['stats'])
        feature.append(f);weight.append(w)
        prior.append({'hypothesis_id':key,'shadow':kf.forecast(h['shadow_bank'],t),
                      'actual':kf.forecast(h['actual_bank'],t),'status':h['status'],
                      'active_cap':h['active_cap'],'evidence':deepcopy(h['evidence'])})
    matrix=np.asarray(feature).reshape(len(keys),n);weights=np.asarray(weight).reshape(len(keys),n)
    owner=np.full(n,-1,dtype=np.int64);shift=np.zeros(n);coefficient=np.zeros(n)
    active=[k for k,key in enumerate(keys) if state['hypotheses'][key]['status'] in ('probation','supported') and not state['hypotheses'][key]['withdrawn']]
    active.sort(key=lambda k:(-state['hypotheses'][keys[k]]['evidence']['score'],
                             -state['hypotheses'][keys[k]]['evidence']['positive_count'],
                             state['hypotheses'][keys[k]]['created_query'],state['hypotheses'][keys[k]]['order']))
    for k in active:
        alpha=min(prior[k]['active_cap'],prior[k]['actual']['mean'])
        mask=(owner<0)&(weights[k]>1e-12)&(np.abs(matrix[k])>1e-12)&(alpha>0.)
        owner[mask]=k;shift[mask]=alpha*matrix[k,mask];coefficient[mask]=alpha
    score,meta=cache.score(state['history'],shift/d['stats']['std']) if np.any(shift) else (d['base'],d['base_meta'])
    guided=pick(d,score)
    return {'keys':keys,'features_y':matrix,'weights':weights,'priors':prior,'owner':owner,
            'shift_y':shift,'coefficient':coefficient,'score':score,'meta':meta,'guided':guided}


def decide(cache,state,d,t,answer,context_key):
    plan=answer['plan'];registered=register(cache.pool,state,t,plan,context_key)
    g=guidance(cache,state,d,t);ref=d['reference'];candidate=cache.ix[plan['candidate_id']] if plan['action']=='propose' else None
    reviewed=bool(candidate is not None and d['eligible'][candidate])
    meaningful=g['guided']!=ref and g['score'][g['guided']]>g['score'][ref]+1e-12
    if meaningful:selected=g['guided'];mode='validated_local_guidance'
    elif reviewed:selected=candidate;mode='GP_reviewed_LLM_trial'
    else:selected=ref;mode='BO_after_LLM_abstention' if candidate is None else 'BO_rejected_LLM_proposal'
    assert d['available'][selected] and d['eligible'][selected]
    return {'guidance':g,'selected':int(selected),'mode':mode,'registered_hypothesis':registered,
        'proposed_index':candidate,'GP_review_accepted':reviewed,
        'GP_review_loss':float(d['base'][ref]-d['base'][candidate]) if candidate is not None else None,
        'local_ranking_changed':meaningful,'proposal_selected':candidate==selected if candidate is not None else False}


def feedback(state,d,decision,t,y,candidate_id):
    g=decision['guidance'];j=decision['selected'];mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j])
    R=sd**2+.0009;owner=int(g['owner'][j]);actual=None;shadows=[]
    for k,key in enumerate(g['keys']):
        h=state['hypotheses'][key]
        if h['withdrawn'] or g['weights'][k,j]<=1e-12:continue
        H=float(g['features_y'][k,j]);probe=.25
        shift=probe*H;ratio=abs(shift)/np.sqrt(R);informative=bool(ratio>=.05 and abs(H)>1e-12)
        gain=float((y-mu)**2-(y-mu-shift)**2)
        row={'query':t,'candidate_id':candidate_id,'GP_mean_y':mu,'GP_predictive_SD_y':sd,
            'feature_y':H,'kernel_weight':float(g['weights'][k,j]),'probe_amplitude':probe,
            'probe_prediction_y':mu+shift,'observed_y':float(y),'paired_SE_gain':gain,
            'standardized_gain':gain/(2*R),'probe_over_effective_SD':float(ratio),
            'informative':informative,'kind':'frozen local shadow, not actual-deployment gain'}
        h['shadow_records'].append(row)
        if informative:
            h['shadow_bank'],measurement=update_bank(h['shadow_bank'],g['priors'][k]['shadow'],t,H,mu,sd,y,.5)
            measurement['hypothetical_paired_SE_gain']=measurement.pop('actual_paired_SE_gain')
            measurement['hypothetical_mean_y']=measurement.pop('actual_deployed_mean_y')
            measurement['hypothetical_correction_y']=measurement.pop('actual_deployed_correction_y')
            measurement.pop('gate_open',None)
            measurement['calibration_target']='adaptive shadow coefficient; separate from fixed .25 validation probe and actual deployment'
            row['shadow_KF']=measurement
        shadows.append({'hypothesis_id':key,**deepcopy(row)})
    if owner>=0:
        key=g['keys'][owner];h=state['hypotheses'][key];prior=deepcopy(g['priors'][owner]['actual'])
        prior['mean']=min(prior['mean'],g['priors'][owner]['active_cap']);prior['amplitude']=prior['mean']
        H=float(g['features_y'][owner,j]);bank,measurement=update_bank(h['actual_bank'],prior,t,H,mu,sd,y,g['priors'][owner]['active_cap'])
        h['actual_bank']=bank
        actual={'hypothesis_id':key,'query':t,'candidate_id':candidate_id,'prior':prior,'measurement':measurement,'bank_after':bank}
        h['actual_records'].append(deepcopy(actual))
        if measurement['actual_paired_SE_gain'] < -1e-12:
            h['last_actual_failure_query']=t;h['status']='quarantined';h['active_cap']=0.
    deployed_mu=mu+float(g['shift_y'][j]);deployed_gain=float((y-mu)**2-(y-deployed_mu)**2)
    if actual:
        assert abs(actual['measurement']['actual_deployed_mean_y']-deployed_mu)<1e-12
        assert abs(actual['measurement']['actual_paired_SE_gain']-deployed_gain)<1e-12
    else:assert g['shift_y'][j]==0. and deployed_gain==0.
    return {'shadow_checks':shadows,'actual_local_update':actual,'GP_mean_y':mu,'GP_predictive_SD_y':sd,
            'deployed_mean_y':deployed_mu,'deployed_shift_y':float(g['shift_y'][j]),
            'actual_paired_SE_gain':deployed_gain,'actual_owner_hypothesis':g['keys'][owner] if owner>=0 else None}
