"""Continuous R1: executable knowledge, deployed-gain KF, reserved refreshes.

The frozen continuous GP, MC acquisition, LGBO field, B4 subset policy, pools
and tie rule are reused. No outcomes are read by this decision module.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import numpy as np
from . import continuous_benchmark_engine as old
from . import continuous_benchmark_data as data
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_revised_20261030')
PARAMETERS={'version':'Continuous_R1','initial_queries':4,'additional_queries':16,'seed':0,'tie_seed':0,
    'GP_acquisition_LGBO_B4':'unchanged frozen continuous adapter and exact prefix GP recipe',
    'knowledge_admission':'nonempty preferred region must have at least one UNQUERIED fixed-pool witness',
    'scalar_KF':{'initial_mean':.5,'initial_variance':.25,'process_variance':.0025,
        'measurement_variance_floor':.0009,'project_mean':[0.,1.],
        'observation_model':'y-mu_GP = a * (gate * raw_region_field) + noise; no nuisance intercept',
        'innovation':'observed minus frozen ACTUALLY DEPLOYED predicted mean',
        'negative_paired_gain':'must not increase scalar posterior mean; record paired SSE gain relative to base GP'},
    'refresh':{'maximum':3,'early_last_round':8,'maximum_early':2,'reserved_from_round':9,'last_round':14,'minimum_gap':2},
    'interaction_queries':[2,6],'maximum_interaction_tests':2,'interaction_acquisition_gate':'strict original epsilon, no exceptions',
    'interaction_geometry':{'exact':'two physical input axes change, all others identical to own incumbent',
        'scattered':'two largest normalized changes each >=0.1, RMS changes of remaining axes <=0.2; observational joint test, not an isolated causal interaction',
        'minimum_focus_change':.1,'maximum_background_RMS':.2},
    'trust_recovery':'unchanged geometry-specific latch with FROZEN shadow-prediction gain; shadow evidence is separate from actual-deployment KF updates'}

class Cache(old.Cache):
    """Same GP recipe; new cache destination, frozen old fits read via copies."""
    def __init__(self,pool):
        super().__init__(pool)
        self.frozen_directory=self.directory
        self.directory=ROOT/'cache'/pool['task'];self.directory.mkdir(parents=True,exist_ok=True)
        self.copied=0
        # Small model checkpoints are copied byte-for-byte, never moved/edited.
        for source in self.frozen_directory.glob('*.pt'):
            destination=self.directory/source.name
            if not destination.exists():shutil.copyfile(source,destination);self.copied+=1

def admission(pool,knowledge,history):
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in history])
    mask=old.membership(pool,knowledge).all(0)
    js=np.flatnonzero(available&mask)
    ranks=old.priority(len(pool['ids']),max(1,len(history)-3),0)
    witness=int(js[np.argmin(ranks[js])]) if len(js) else None
    return {'admitted':bool(len(js)),'has_preferences':bool(knowledge['clauses']),
        'pool_witness_count':int(mask.sum()),'unqueried_witness_count':len(js),
        'unqueried_witness_id':pool['ids'][witness] if witness is not None else None,
        'rule':'at least one unqueried fixed-pool witness; no target values consulted'}

def new_bank(t):
    return {'mean':.5,'variance':.25,'last_round':t-1,'updates':0,'informative_updates':0}

def forecast(bank,t):
    elapsed=t-bank['last_round'];assert elapsed>=1
    mean=float(np.clip(bank['mean'],0.,1.))
    return {'mean':mean,'variance':float(bank['variance']+.0025*elapsed),'elapsed_rounds':elapsed,'amplitude':mean}

def update_bank(bank,prior,t,raw_y,mu_y,sd_predictive_y,y,gate_open):
    """Projected scalar KF whose innovation equals actual deployed error.

    gate=0 gives H=0: observing a BO-only query does not pretend that knowledge
    was deployed. Shadow predictions may inform the separate trust latch only.
    """
    H=float(raw_y) if gate_open else 0.;a=float(prior['mean']);P=float(prior['variance'])
    R=float(sd_predictive_y**2+.0009);correction=a*H;r=float(y-mu_y)
    innovation=r-correction;S=H*H*P+R;K=P*H/S
    unprojected=a+K*innovation;after=float(np.clip(unprojected,0.,1.))
    variance=float((1-K*H)**2*P+K*K*R)
    gain=float(r*r-(r-correction)**2)
    if gain < -1e-12:assert after<=a+1e-12,'negative deployed gain cannot increase amplitude'
    result={'mean':after,'variance':variance,'last_round':t,'updates':bank['updates']+1,
        'informative_updates':bank.get('informative_updates',0)+int(abs(H)>1e-12)}
    measurement={'feature':H,'gate_open':bool(gate_open),'prequery_mu_y':float(mu_y),'prequery_predictive_sd_y':float(sd_predictive_y),
        'raw_region_shift_y':float(raw_y),'actual_deployed_correction_y':correction,
        'actual_deployed_mean_y':float(mu_y+correction),'observed_y':float(y),'residual':r,
        'innovation':innovation,'innovation_variance':S,'measurement_variance':R,'innovation_z':float(innovation/np.sqrt(S)),
        'gain':K,'coefficient_before':a,'coefficient_after_unprojected':unprojected,'coefficient_after':after,
        'paired_SE_GP':r*r,'paired_SE_deployed':(r-correction)**2,'actual_paired_SE_gain':gain,
        'calibration_target':'actual deployed mean relative to base GP; no b offset or hidden unbounded a'}
    return result,measurement

def trust_feedback(state,d,t,y,updated_bank):
    """Transport the existing latch's frozen shadow test to a scalar bank."""
    key=d['trust']['geometry_key'];entry=state.setdefault('region_trust',{}).get(key)
    if entry is None or not entry['blocked']:return None
    j=d['selected'];mu=float(d['stats']['mu_y'][j]);raw=float(d['_measurement']['raw_region_shift_y'])
    alpha=float(d['prior']['amplitude']);probe=alpha*raw;gain=(y-mu)**2-(y-mu-probe)**2;n=len(state['history'])+1
    valid=bool(n>entry['blocked_observation_count'] and abs(probe)>1e-12 and gain>1e-12 and updated_bank['mean']>0.)
    receipt={'query':t,'geometry_key':key,'before_observation_count':len(state['history']),'after_observation_count':n,
        'GP_mean_y':mu,'raw_region_shift_y':raw,'frozen_probe_amplitude':alpha,'frozen_probe_shift_y':probe,
        'observed_y':float(y),'squared_error_gain_y_squared':float(gain),'positive_evidence':valid,
        'KF_coefficient_after_feedback':float(updated_bank['mean']),
        'evidence_kind':'shadow prediction for trust recovery ONLY; not counted as an actual-deployment KF gain'}
    entry['last_feedback']=deepcopy(receipt);entry['pending_evidence']=deepcopy(receipt) if valid else None
    return receipt

def setup(cache,state,t):
    pool=cache.pool;history=state['history'];knowledge=state['knowledge'];base,bmeta=cache.score(history);v=cache.model(history)
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in history]);budget=float(np.log(4)*(16-t)/15)
    proxy,pairs,unseen=old.information_proxy(cache.X.numpy(),[cache.ix[h['candidate_id']] for h in history]);ranks=old.priority(len(base),t,0)
    def choose(scores,mask):
        js=np.flatnonzero(mask);assert len(js)
        js=js[scores[js]==scores[js].max()];js=js[proxy[js]==proxy[js].max()];js=js[v['sd_y'][js]==v['sd_y'][js].max()]
        return int(js[np.argmin(ranks[js])])
    ref=choose(base,available);eligible=available&(base[ref]-base<=budget)
    admitted=admission(pool,knowledge,history)
    mem=old.membership(pool,knowledge);dg=old.select_candidate(base,mem,budget,candidate_ids=pool['ids'],
        hypothesis_ids=[h['id'] for h in knowledge['clauses']],queried_mask=~available)
    kept=list(dg.retained_ids);positions=[k for k,h in enumerate(knowledge['clauses']) if h['id'] in kept]
    witness=choose(base,eligible&(mem[positions].all(0) if positions else True));dg=replace(dg,candidate_index=witness,candidate_id=pool['ids'][witness])
    spec=old.region(pool,knowledge,kept);raw,fmeta=cache.field(history,spec);key=old.trust.geometry_key(spec)
    bank=deepcopy(state['banks'].get(key,new_bank(t))) if key else None;prior=forecast(bank,t) if bank else None
    alpha=prior['amplitude'] if prior else 0.;conflict=bool(dg.revoked_ids) or dg.conflict_kind!='none'
    fallback=bool(conflict or not admitted['admitted'])
    d={'region':spec,'amplitude':0. if fallback else alpha,'fallback':fallback}
    gate=old.trust.permission(state,d,t);amplitude=gate['amplitude_after_trust']
    gate_open=bool(key and not fallback and not gate['trust_blocked'])
    score,gmeta=cache.score(history,amplitude*raw) if amplitude else (base,bmeta);j=choose(score,eligible)
    assert gate_open or j==ref
    d.update(base=base,score=score,base_meta=bmeta,guided_meta=gmeta,field_meta=fmeta,stats=v,
        available=available,eligible=eligible,reference=ref,selected=j,normal_proposal=j,budget=budget,
        diagnosis=dg,retained=kept,admission=admitted,raw_shift_std=raw,shift_std=amplitude*raw,
        amplitude=amplitude,gate_open=gate_open,fallback=not gate_open,bank_key=key,bank=bank,prior=prior,trust=gate,
        information_proxy=proxy,tie_priority=ranks,isolated_pairs=pairs,unseen_values=unseen)
    return d

def refresh_permission(state,t):
    calls=state['calls'];early=sum(c['iteration']<=8 for c in calls)
    reason='allowed'
    if len(calls)>=3:reason='total_refresh_limit'
    elif t>14:reason='after_last_refresh_round'
    elif calls and t-calls[-1]['iteration']<2:reason='refresh_cooldown'
    elif t<=8 and early>=2:reason='preserve_at_least_one_late_refresh'
    return {'allowed':reason=='allowed','reason':reason,'used':len(calls),'remaining':3-len(calls),'early_used':early,
        'late_window':[9,14],'late_reserve_intact':early<=2}

def event_reasons(state,d,t):
    reasons=[]
    if not d['admission']['admitted']:reasons.append('no_unqueried_knowledge_witness')
    if not state['knowledge']['clauses']:reasons.append('empty_knowledge')
    if d['diagnosis'].conflict_kind!='none' or d['diagnosis'].revoked_ids:reasons.append('B4_conflict')
    if len(state['decisions'])>=3 and max(h['y'] for h in state['history'])-max(h['y'] for h in state['history'][:-3])<.005:reasons.append('stagnation_3')
    tail=state['decisions'][-2:]
    if len(tail)==2 and all(x['measurement'].get('actual_paired_SE_gain',0.) < -1e-12 for x in tail):reasons.append('two_deployed_predictions_worse_than_GP')
    return reasons

def interaction_candidates(cache,state,d,t):
    pool=cache.pool;history=state['history'];anchor=max(history,key=lambda h:h['y']);ja=cache.ix[anchor['candidate_id']]
    X=cache.X.numpy();diff=np.abs(X-X[ja]);changed=diff>1e-12;count=changed.sum(1)
    exact_domain=bool(np.any(d['available']&(count==2)));geometry='exact_two_axes' if exact_domain else 'matched_background_joint_conditions'
    mem=old.membership(pool,state['knowledge']);rows=[]
    for j in np.flatnonzero(d['eligible']):
        axes=np.flatnonzero(changed[j]);background=0.
        if exact_domain:
            if count[j]!=2:continue
            focus=axes
        else:
            focus=np.argsort(-diff[j],kind='stable')[:2]
            if diff[j,focus].min()<.1:continue
            other=np.setdiff1d(np.arange(X.shape[1]),focus);background=float(np.sqrt(np.mean(diff[j,other]**2)))
            if background>.2:continue
        focus=sorted(int(a) for a in focus);violated=[h['id'] for k,h in enumerate(state['knowledge']['clauses']) if not mem[k,j]]
        rows.append({'candidate_id':pool['ids'][j],'point':pool['points'][j],'focus_axes':[pool['axes'][a] for a in focus],
            'all_changed_axes':[pool['axes'][a] for a in axes],'background_RMS_normalized':background,
            'geometry':geometry,'violated_preferences':violated,
            'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
            'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'],
            'deployed_guided_mean':float(d['stats']['mu_y'][j]+d['shift_std'][j]*d['stats']['std'])*pool['objective_scale'],
            'base_qLogEI':float(d['base'][j]),'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),
            'normal_gate_eligible':True})
    rows.sort(key=lambda r:d['tie_priority'][cache.ix[r['candidate_id']]])
    return {'anchor_candidate_id':anchor['candidate_id'],'anchor_observation_index':history.index(anchor)+1,
        'anchor_point':pool['points'][ja],'anchor_objective':anchor['y']*pool['objective_scale'],
        'geometry':geometry,'candidates':rows,'complete_pool':True,'candidate_count':len(rows),
        'skip_reason':None if rows else 'no qualifying joint candidate inside unchanged acquisition gate',
        'scope':'Exact two-axis joint test' if exact_domain else 'Matched-background observational joint condition test; not an isolated causal interaction',
        'acquisition_budget':d['budget'],'experimental_budget':'one of the same16 queries'}
