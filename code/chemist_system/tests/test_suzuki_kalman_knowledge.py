import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
import torch

from chem.benchmarks import kalman_knowledge_suzuki_engine as e
from chem.benchmarks import kalman_knowledge_suzuki_study as s

Z=np.array([[.1,.1,.2,.0625],[.7,.8,.3,.1875],[.3,.4,.8,.3125],[.8,.9,.7,.4375]])


def hypothesis(key='H1',variable='temperature',bounds=None,categories=None,effect=10,op='add'):
    return {'id':key,'operation':op,'variable':variable,'bounds':bounds if bounds is not None else [.5,1.],
            'categories':categories if categories is not None else [],'effect_pp':effect,'reason':'Tentative contrast from observations','support':[1]}


def plan(hs):return {'hypotheses':hs,'revocations':[],'summary':'Testable residual contrasts'}


def calibrated():
    p=e.validate_plan(plan([hypothesis()]),[],4,Z);return e.apply_plan(e.initial_filter(),[],p,Z)


def test_feature_pool_centered_in_physical_units():
    state,hs=calibrated();x=e.features(Z,hs)
    assert x.shape==(4,5) and np.allclose(x[:,1].mean(),0)
    assert np.ptp(x[:,1])==pytest.approx(.1) and hs[0]['pool_fraction']==.5
    assert np.allclose(x[:,0],1)


def test_catalysts_have_set_membership_without_ordering():
    h=hypothesis(variable='catalyst',bounds=[],categories=[0,2]);p=e.validate_plan(plan([h]),[],4,Z)
    assert e.membership(Z,p['hypotheses']).tolist()==[[True,False,True,False]]


@pytest.mark.parametrize('change',[{'effect_pp':float('nan')},{'bounds':[0,1]},{'bounds':[.50001,1.]},{'categories':[1]},{'support':[5]},{'operation':'retain'}])
def test_invalid_hypotheses_rejected(change):
    h=hypothesis();h.update(change)
    with pytest.raises(ValueError):e.validate_plan(plan([h]),[],4,Z)


def test_lifecycle_accounts_for_old_ids_and_preserves_retained_state():
    state,hs=calibrated();state['theta'][1]=.2;state['covariance'][0][1]=.0001;state['covariance'][1][0]=.0001
    retain=hypothesis(op='retain');p=e.validate_plan(plan([retain]),hs,5,Z);new,nh=e.apply_plan(state,hs,p,Z)
    assert new==state and nh[0]['version']==1
    with pytest.raises(ValueError):e.validate_plan(plan([]),hs,5,Z)
    retain['effect_pp']=5
    with pytest.raises(ValueError):e.validate_plan(plan([retain]),hs,5,Z)


def test_revision_resets_only_own_coefficient_and_cross_covariance():
    state,hs=calibrated();state['theta'][0]=.1;state['theta'][1]=-.5;state['covariance'][0][1]=.001;state['covariance'][1][0]=.001
    p=e.validate_plan(plan([hypothesis(op='revise',effect=5)]),hs,5,Z);new,nh=e.apply_plan(state,hs,p,Z)
    assert new['theta'][:2]==[.1,1.] and new['covariance'][0][1]==0 and new['covariance'][1][1]==.25
    assert new['covariance'][0][0]==state['covariance'][0][0] and nh[0]['version']==2


def test_revocation_removes_state_and_process_noise():
    state,hs=calibrated();p={'hypotheses':[],'revocations':[{'id':'H1','reason':'unsupported','support':[1]}],'summary':'Remove'}
    new,nh=e.apply_plan(state,hs,e.validate_plan(p,hs,5,Z),Z);f=e.forecast(new,nh)
    assert not nh and f['theta'][1]==f['covariance'][1][1]==0 and not f['weights']


def test_joseph_update_matches_direct_formula_and_is_positive_semidefinite():
    state,hs=calibrated();prior=e.forecast(state,hs);feature=e.features(Z,hs)[1]
    new,m=e.update_filter(state,prior,feature,.4,.02,.3)
    p=np.array(prior['covariance']);r=.02**2+.03**2;gain=p@feature/(feature@p@feature+r)
    expected=np.array(prior['theta'])+gain*(.3-.4-feature@np.array(prior['theta']))
    assert np.allclose(new['theta'],expected) and m['residual']==pytest.approx(-.1)
    assert np.linalg.eigvalsh(new['covariance']).min()>-1e-14
    assert m['prequery_mu_y']==.4 and m['measurement_variance']==pytest.approx(r)


def test_disconfirming_prequery_residual_reduces_trust():
    state,hs=calibrated();prior=e.forecast(state,hs);feature=e.features(Z,hs)[1]
    new,_=e.update_filter(state,prior,feature,.5,.01,.1)
    assert e.forecast(new,hs)['weights']['H1']<prior['weights']['H1']


def test_no_feature_exposure_does_not_create_knowledge_credit():
    state,hs=calibrated();prior=e.forecast(state,hs)
    new,m=e.update_filter(state,prior,np.array([1.,0,0,0,0]),.4,.03,.8)
    assert new['theta'][1]==1. and new['covariance'][1][1]==prior['covariance'][1][1]
    assert m['predictions']['GP_joint_KF']==m['predictions']['GP_offset_null']==.4
    assert new['theta'][0]==pytest.approx(new['null_mean'])


def test_negative_coefficients_disable_instead_of_reversing_knowledge():
    state,hs=calibrated();state['theta'][1]=-.5
    assert e.forecast(state,hs)['weights']['H1']==0


def test_mean_cap_and_no_offset_leak():
    state,hs=calibrated();state['theta'][0]=.9;prior=e.forecast(state,hs)
    change=e.correction(Z,hs,{'H1':1.},np.full(4,.02),['H1'])
    assert np.allclose(change,[-.01,.01,-.01,.01])
    assert np.allclose(e.correction(Z,hs,prior['weights'],np.ones(4),[]),0)


def test_b4_joint_conflict_is_detected_and_lexical_tie_is_frozen():
    hs=[hypothesis('H1',bounds=[0,.5]),hypothesis('H2',bounds=[.6,1.])]
    d=e.diagnose(np.array([1.,.9,.8,.7]),Z,hs,{'H1':.8,'H2':.8},1.,np.zeros(4,dtype=bool),list('abcd'))
    assert d.conflict_kind=='empty_intersection' and d.retained_ids==('H1',)
    assert ('H1','H2') in d.minimal_conflict_sets


def test_budget_guard_and_failed_knowledge_fallback():
    class Cache:
        def score(self,h):return np.array([0.,-.5,-5.,-1.]),{'key':'base','fit_key':'fit'}
        def knowledge_score(self,*args):return np.array([-2.,2.,100.,0.]),{'key':'guide'}
        def posterior_stats(self,h):return {'key':'post','mu_y':np.zeros(4),'sd_y':np.ones(4),'sd_predictive_y':np.ones(4)}
    state,hs=calibrated();prior=e.forecast(state,hs)
    selected,event=e.choose(Cache(),[],Z,list('abcd'),hs,prior,1)
    assert selected==1 and event['base_logei_loss']==.5
    selected,event=e.choose(Cache(),[],Z,list('abcd'),hs,prior,16)
    assert selected==0 and event['base_logei_loss']==0
    selected,event=e.choose(Cache(),[],Z,list('abcd'),hs,prior,1,False)
    assert selected==0 and not event['diagnosis']['retained_ids']


def test_stagnation_and_predictive_error_event_caps():
    class D:conflict_kind='quality_exclusion'
    history=[{'y':.5}]*8;events=[{'measurement':{'innovation_z':3.}}]*4
    reasons=e.event_reasons(5,history,events,[{}],D(),[1])
    assert reasons==['stagnation','consecutive_predictive_errors','B4_quality_exclusion']
    assert e.event_reasons(2,history,events,[],D(),[1])==[]
    assert e.event_reasons(15,history,events,[],D(),[1])==[]
    assert e.event_reasons(9,history,events,[],D(),[1,3,5,7])==[]
    assert e.event_reasons(1,[],[],[],D(),[])==['initial']


def test_gp_wrapper_preserves_covariance_and_caps_physical_mean():
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal
    class Base(torch.nn.Module):
        num_outputs=1;batch_shape=torch.Size([])
        def posterior(self,x,observation_noise=False,**kwargs):
            mean=torch.zeros(x.shape[:-1],dtype=x.dtype);variance=torch.full_like(mean,.04 if not observation_noise else .09)
            return GPyTorchPosterior(MultivariateNormal(mean,torch.diag_embed(variance)))
    state,hs=calibrated();model=e.ResidualMeanModel(Base(),hs,{'H1':1.},['H1'],.2)
    x=torch.tensor(Z,dtype=torch.double).unsqueeze(1);p=model.posterior(x);pn=model.posterior(x,observation_noise=True)
    assert torch.allclose(p.mean.reshape(-1),torch.tensor([-.1,.1,-.1,.1],dtype=torch.double))
    assert torch.allclose(p.variance,torch.full_like(p.variance,.040001))
    assert torch.allclose(pn.variance,torch.full_like(pn.variance,.090001))
    assert torch.allclose(p.mean,pn.mean)


def test_api_retry_records_first_valid_plan_and_omits_raw_reasoning(tmp_path,monkeypatch):
    class Fake:config=s.api.APIConfig('https://api.deepseek.com','fake')
    calls=[]
    def generate(*args):
        calls.append(1)
        if len(calls)==1:httpx.Response(429,request=httpx.Request('POST','https://api.deepseek.com/chat/completions')).raise_for_status()
        content=plan([hypothesis()]) if len(calls)>2 else {'bad':True}
        return {'id':str(len(calls)),'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]}
    monkeypatch.setattr(s.previous,'complete',generate);monkeypatch.setattr(s.time,'sleep',lambda _:None)
    c={'public':{'task':'test'},'history':[{}]*4,'context_sha256':'test'}
    valid=s.get_plan(Fake(),c,[],Z,tmp_path)
    assert valid['accepted_attempt']==3 and valid['received_responses']==2 and len(calls)==3
    body=s.read(tmp_path/'attempt03/request.json');assert body['model']=='deepseek-flash' and body['max_tokens']==16384
    assert 'fake' not in json.dumps(body) and len(body['messages'])==4


def test_empty_plan_is_valid_and_schema_failure_never_invents_knowledge(tmp_path,monkeypatch):
    assert e.validate_plan(plan([]),[],4,Z)['hypotheses']==[]
    class Fake:config=s.api.APIConfig('https://api.deepseek.com','fake')
    calls=[]
    def generate(*args):
        calls.append(1);return {'id':'invalid','model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    monkeypatch.setattr(s.previous,'complete',generate)
    with pytest.raises(RuntimeError):s.get_plan(Fake(),{'public':{},'history':[{}]*4,'context_sha256':'x'},[],Z,tmp_path)
    assert len(calls)==3 and not (tmp_path/'validated.json').exists()


def test_context_contains_only_own_observations_and_saved_prequery_errors():
    pool={'ids':['a','b','c','d'],'z':Z.tolist(),'points':[{'catalyst':s.CATALYSTS[i]} for i in range(4)]}
    state={'task':'K01','history':[{'candidate_id':'a','y':.4}], 'decisions':[],'hypotheses':[],'filter':e.initial_filter()}
    class D:
        def to_dict(self):return {'conflict_kind':'none'}
    stats={'mu_y':np.array([.6]*4),'sd_y':np.array([.1]*4)}
    public=s.public_request(state,pool,stats,np.arange(4.),D(),['initial'])
    assert public['observations']==[{'index':1,'cat':0,'z':[.1,.1,.2],'yield_percent':40.}]
    assert all('yield_percent' not in row for row in public['gp_top3_per_catalyst'])
    assert 'prequery_residual_pp' not in public['observations'][0]
