from copy import deepcopy
import numpy as np
import pytest
import torch
from gpytorch.kernels import RBFKernel

from chem.benchmarks import mixed_kalman_suzuki_engine as e
from chem.benchmarks import mixed_kalman_suzuki_study as s

torch.set_num_threads(1)


@pytest.fixture
def toy():
    z=[[.1+.1*k,.2+.08*k,.8-.07*k,(k+.5)/8] for k in range(8)]
    return {'ids':[f'p{k:05}' for k in range(8)],'z':z,'points':[{'catalyst':c} for c in e.CATALYSTS]}


def history(pool):return [{'candidate_id':pool['ids'][i],'y':v} for i,v in enumerate((.1,.25,.08,.4))]


def test_only_continuous_inputs_are_normalized_and_category7_survives(toy):
    x=e.encode_mixed(toy);assert np.array_equal(x[:,3],np.arange(8))
    assert np.allclose(x[:,:3],np.array(toy['z'])[:,:3])


def test_legacy_ordinal_bin_changes_do_not_enter_gp_features(toy):
    copy=deepcopy(toy)
    for i,z in enumerate(copy['z']):z[3]=99-i
    assert np.array_equal(e.encode_mixed(copy),e.encode_mixed(toy))


def test_categories_require_unique_nominal_labels(toy):
    with pytest.raises(ValueError):e.encode_mixed(toy,[0]*8)


def test_category_kernel_has_equal_similarity_for_any_distinct_labels(toy):
    x=torch.tensor(e.encode_mixed(toy),dtype=torch.double);x[:,:3]=.5
    model=e.MixedSingleTaskGP(x,torch.arange(8,dtype=torch.double).unsqueeze(1),cat_dims=[3])
    with torch.no_grad():k=model.covar_module(x).to_dense().numpy()
    mask=~np.eye(8,dtype=bool);assert np.ptp(k[mask])<1e-12 and np.ptp(np.diag(k))<1e-12
    for module in model.covar_module.modules():
        if isinstance(module,RBFKernel):assert module.active_dims.tolist()==[0,1,2]


@pytest.fixture
def caches(toy,tmp_path):
    first=e.MixedKnowledgeCache(toy,tmp_path/'one',1)
    second=e.MixedKnowledgeCache(toy,tmp_path/'two',1,category_map=[19,-3,101,7,0,12,-9,4])
    return first,second


def test_fitted_posterior_invariant_to_category_bijection(toy,caches):
    first,second=caches;h=history(toy);a=first.posterior_stats(h);b=second.posterior_stats(h)
    assert np.allclose(a['mu_y'],b['mu_y'],rtol=0,atol=1e-12)
    assert np.allclose(a['sd_y'],b['sd_y'],rtol=0,atol=1e-12)
    assert np.all(a['sd_predictive_y']>=a['sd_y'])


def test_fitted_base_qlogei_invariant_and_repeat_does_not_refit(toy,caches):
    first,second=caches;h=history(toy);a,_=first.score(h);b,_=second.score(h)
    assert np.allclose(a,b,rtol=0,atol=1e-10)
    fits=first.counters['fits'];first.score(h);first.posterior_stats(h);assert fits==first.counters['fits']==1


def test_guided_scores_and_actual_choice_invariant_under_category_relabeling(toy,caches):
    first,second=caches;h=history(toy);p=e.policy
    plan=p.validate_plan({'hypotheses':[{'id':'H1','operation':'add','variable':'catalyst','bounds':[],'categories':[0,4,6],
        'effect_pp':5.,'reason':'Synthetic test','support':[1]}],'revocations':[],'summary':'Test'},[],4,toy['z'])
    state,hs=p.apply_plan(p.initial_filter(),[],plan,toy['z']);prior=p.forecast(state,hs)
    a,am=first.knowledge_score(h,hs,prior['weights'],['H1']);b,bm=second.knowledge_score(h,hs,prior['weights'],['H1'])
    assert np.allclose(a,b,rtol=0,atol=1e-10)
    ia,ra=p.choose(first,h,toy['z'],toy['ids'],hs,prior,1);ib,rb=p.choose(second,h,toy['z'],toy['ids'],hs,prior,1)
    assert ia==ib and ra['base_logei_loss']==pytest.approx(rb['base_logei_loss'],abs=1e-10)


def test_zero_mean_wrapper_adds_only_common_jitter(toy,caches):
    cache,_=caches;_,value=cache.model(history(toy));x=cache.z.unsqueeze(1)
    with torch.no_grad():
        raw=value['model'].posterior(x);wrapped=e.MixedResidualMeanModel(value['model'],[],{},[],value['std']).posterior(x)
    assert torch.equal(raw.mean,wrapped.mean)
    assert torch.allclose(raw.variance+1e-6,wrapped.variance,rtol=0,atol=1e-12)


def test_nominal_knowledge_shift_and_cap_match_shared_policy(toy,caches):
    cache,_=caches;_,value=cache.model(history(toy));p=e.policy
    hs=[{'id':'H1','variable':'catalyst','categories':[0,6],'bounds':[],'effect_pp':20.,'pool_fraction':.25}];weights={'H1':.8}
    with torch.no_grad():
        raw=value['model'].posterior(cache.z.unsqueeze(1));guided=e.MixedResidualMeanModel(value['model'],hs,weights,['H1'],value['std']).posterior(cache.z.unsqueeze(1))
    sd=raw.variance.reshape(-1).sqrt().numpy()*float(value['std']);expected=p.correction(toy['z'],hs,weights,sd,['H1'])
    actual=(guided.mean-raw.mean).reshape(-1).numpy()*float(value['std'])
    assert np.allclose(actual,expected,rtol=0,atol=1e-12)


def test_model_cache_records_actual_mixed_inputs_not_legacy_bins(toy,caches):
    cache,_=caches;key,value=cache.model(history(toy));saved=torch.load(cache.directory/(key+'.pt'),weights_only=False)
    assert saved['gp_class']=='MixedSingleTaskGP' and saved['cat_dims']==[3]
    assert [row[3] for row in saved['train_X']]==[0.,1.,2.,3.]


def test_policy_and_api_settings_are_unchanged():
    assert s.PARAMETERS==s.old.PARAMETERS and s.SYSTEM_PROMPT==s.old.SYSTEM_PROMPT
    assert e.policy.PARAMETERS['max_calls']==4 and e.policy.PARAMETERS['budget_ratio']==4
    assert s.WORKERS==2 and s.METHODS[2]=='Mixed_qLogEI_B0'
