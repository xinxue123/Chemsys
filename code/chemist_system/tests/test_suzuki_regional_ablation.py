from copy import deepcopy
import numpy as np
import pytest
import torch

from chem.benchmarks import regional_ablation_suzuki_engine as e
from chem.benchmarks import regional_ablation_suzuki_study as s

torch.set_num_threads(1)


@pytest.fixture
def pool():
    return {'ids':[f'p{i:05}' for i in range(8)],
        'z':[[.1+.1*i,.25+.04*i,.8-.08*i,(i+.5)/8] for i in range(8)],
        'points':[{'catalyst':cat} for cat in e.CATALYSTS]}


@pytest.fixture
def knowledge():
    return {'confidence':.7,'clauses':[{'id':'H1','coordinate':0,'bounds':[.2,.8],'categories':[]},
                                      {'id':'H2','coordinate':3,'bounds':[],'categories':[0,3]}]}


def test_legacy_conversion_preserves_exact_coordinate_membership(pool):
    advice={'confidence':.58,'clauses':[{'id':'H1','coordinate':0,'interval':[.2,.8]},
                                      {'id':'H2','coordinate':3,'interval':[.25,.5]}]}
    k=e.from_advice(advice)
    assert np.array_equal(e.membership(pool,k),e.common.regional.clause_membership(pool['z'],advice))
    assert k['confidence']==.58 and k['clauses'][1]['categories']==[2,3]


def test_noncontiguous_categories_are_a_set_and_label_invariant():
    first=e.SetRegionalTilt([0,3]);second=e.SetRegionalTilt([0,3],[19,-3,101,7,0,12,-9,4])
    for tilt in (first,second):tilt.set_box_region([0.]*4,[1.]*4)
    assert torch.equal(first._a,second._a)
    assert torch.all(first._a[~torch.isin(first.canonical_category,torch.tensor([0,3]))]==0)


def test_region_bank_uses_geometry_not_reason_or_clause_id(knowledge):
    other=deepcopy(knowledge);other['clauses'][0]['id']='H4'
    assert e.region_key(knowledge,['H1','H2'])==e.region_key(other,['H4','H2'])
    assert e.region_key(knowledge,['H1','H2'])!=e.region_key(knowledge,['H1'])


def test_filter_initial_amplitude_matches_baseline_and_advances_elapsed_noise():
    bank=e.new_bank(3);prior=e.forecast(bank,3)
    assert prior['amplitude']==1.
    bank['last_round']=3;later=e.forecast(bank,7)
    assert np.allclose(np.diag(later['covariance']),[.0025+4*.000025,.25+4*.0025])


def test_zero_innovation_preserves_amplitude_and_joseph_covariance_is_psd():
    bank=e.new_bank(1);pr=e.forecast(bank,1)
    after,m=e.update_bank(bank,pr,1,.1,.2,.02,.3)
    assert np.allclose(after['mean'],[0.,1.]) and abs(m['innovation'])<1e-14
    assert np.linalg.eigvalsh(after['covariance']).min()>0


def test_failed_regional_residual_shrinks_amplitude():
    bank=e.new_bank(1);pr=e.forecast(bank,1)
    after,m=e.update_bank(bank,pr,1,.1,.2,.01,.2)
    assert 0<after['mean'][1]<1 and m['innovation']<0
    assert m['measurement_variance']==pytest.approx(.01**2+.03**2)


def test_high_GP_uncertainty_downweights_measurement():
    bank=e.new_bank(1);pr=e.forecast(bank,1)
    low,_=e.update_bank(bank,pr,1,.1,.2,.01,.2)
    high,_=e.update_bank(bank,pr,1,.1,.2,1.,.2)
    assert abs(high['mean'][1]-1)<abs(low['mean'][1]-1)


def test_deployment_clips_mean_without_initial_uncertainty_discount():
    bank=e.new_bank(1);bank['mean'][1]=-2;assert e.forecast(bank,1)['amplitude']==0
    bank['mean'][1]=3;assert e.forecast(bank,1)['amplitude']==1


def test_scaling_changes_only_same_native_mean_field(pool,tmp_path,knowledge):
    history=[{'candidate_id':pool['ids'][i],'y':y} for i,y in enumerate((.1,.3,.15,.5))]
    cache=e.RegionalAblationCache(pool,tmp_path/'cache',1)
    spec=e.region(knowledge,['H1','H2']);tilt,value,_=cache.make_tilt(history,spec)
    x=cache.z.unsqueeze(1)
    with torch.no_grad():
        base=value['model'].posterior(x);native=tilt.posterior(value['model'],x)
        unity=e.ScaledRegionModel(value['model'],tilt,1).posterior(x)
        half=e.ScaledRegionModel(value['model'],tilt,.5).posterior(x)
    assert torch.equal(native.mean,unity.mean) and torch.equal(native.variance,unity.variance)
    assert torch.allclose(half.mean-base.mean,.5*(native.mean-base.mean),rtol=0,atol=1e-14)
    assert torch.equal(half.variance,native.variance)


def test_empty_knowledge_and_zero_amplitude_recover_base(pool,tmp_path,knowledge):
    history=[{'candidate_id':pool['ids'][i],'y':y} for i,y in enumerate((.1,.3,.15,.5))]
    cache=e.RegionalAblationCache(pool,tmp_path/'cache',1)
    base,_=cache.score(history)
    zero,delta,_=cache.calibrated_score(history,knowledge,['H1','H2'],0.)
    empty,_,_=cache.calibrated_score(history,knowledge,[],1.)
    assert np.array_equal(base,zero) and np.array_equal(base,empty) and np.all(delta==0)


def test_refresh_retain_preserves_exact_geometry_and_confidence(pool,knowledge):
    plan={'actions':[{'id':h['id'],'operation':'retain','bounds':[],'categories':[],
                      'reason':'Keep own supplied preference','support':[1]} for h in knowledge['clauses']], 'summary':'Keep'}
    valid=e.validate_refresh(plan,knowledge,{'H1':0,'H2':3},4,pool)
    assert valid['knowledge']==knowledge


def test_refresh_can_revoke_and_restore_only_its_locked_axis(pool,knowledge):
    plan={'actions':[{'id':'H1','operation':'revoke','bounds':[],'categories':[],'reason':'Revoke','support':[1]},
                      {'id':'H2','operation':'revise','bounds':[],'categories':[2,6],'reason':'Revise','support':[2]}], 'summary':'Update'}
    valid=e.validate_refresh(plan,knowledge,{'H1':0,'H2':3},4,pool)
    assert valid['knowledge']['clauses']==[{'id':'H2','coordinate':3,'bounds':[],'categories':[2,6]}]
    assert valid['knowledge']['confidence']==knowledge['confidence']


def test_refresh_rejects_future_support_and_confidence_changes(pool,knowledge):
    plan={'actions':[{'id':h['id'],'operation':'retain','bounds':[],'categories':[],
                      'reason':'Keep','support':[5]} for h in knowledge['clauses']], 'summary':'Keep'}
    with pytest.raises(ValueError):e.validate_refresh(plan,knowledge,{'H1':0,'H2':3},4,pool)
    plan['confidence']=.99
    with pytest.raises(ValueError):e.validate_refresh(plan,knowledge,{'H1':0,'H2':3},4,pool)


def test_loaded_initial_knowledge_does_not_trigger_round1_refresh():
    class Diagnosis:conflict_kind='quality_exclusion'
    state={'history':[{'y':.2}]*4,'decisions':[],'knowledge':{'clauses':[{}]},'calls':[]}
    assert e.event_reasons(state,{'diagnosis':Diagnosis()},1)==[]


def test_refresh_schedule_preserves_cooldown_and_three_call_limit():
    class Diagnosis:conflict_kind='none'
    state={'history':[{'y':.2}]*7,'decisions':[{'measurement':{'innovation_z':0}}]*3,
           'knowledge':{'clauses':[{}]},'calls':[]}
    assert e.event_reasons(state,{'diagnosis':Diagnosis()},4)==['stagnation']
    state['calls']=[{'iteration':4}];assert e.event_reasons(state,{'diagnosis':Diagnosis()},5)==[]
    state['calls']=[{'iteration':4},{'iteration':6},{'iteration':8}]
    assert e.event_reasons(state,{'diagnosis':Diagnosis()},10)==[]


def test_ladder_and_inference_settings_are_frozen():
    assert s.METHODS==('G0_gate','G1_gate_KF','G2_gate_KF_B4','G3_gate_KF_B4_event')
    assert s.PARAMETERS==s.prior.PARAMETERS
    assert e.PARAMETERS['deployment_clip']==[0.,1.]
