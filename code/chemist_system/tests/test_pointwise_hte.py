from copy import deepcopy
import json
import numpy as np
import torch
import pytest
from botorch.models.gp_regression_mixed import MixedSingleTaskGP

from chem.benchmarks import pointwise_hte_engine as e
from chem.benchmarks import pointwise_hte_llm as l
from chem.benchmarks import pointwise_hte_study as s


def model_and_tilt():
    torch.set_num_threads(1)
    x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double)
    y=torch.tensor([[.1],[.3],[.8],[.4]],dtype=torch.double);y=(y-y.mean())/y.std(unbiased=False)
    model=MixedSingleTaskGP(x,y,cat_dims=[0,1]);model.eval()
    grid=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[2.,0.],[2.,1.]],dtype=torch.double)
    tilt=e.CorrectedNominalTilt(grid,torch.full((6,),1/6,dtype=torch.double))
    tilt.fit_lambda_by_delta(model,1.2);return model,tilt


def test_cached_and_uncached_same_posterior_with_output_scaling():
    model,tilt=model_and_tilt();xx=torch.tensor([[2.,0.],[1.,1.]],dtype=torch.double)
    assert float(model.outcome_transform.stdvs.square())==pytest.approx(4/3)
    with torch.no_grad():
        uncached=tilt.posterior(model,xx);tilt.prepare_cache(model);cached=tilt.posterior(model,xx)
        torch.testing.assert_close(cached.mean,uncached.mean,rtol=0,atol=1e-11)
        torch.testing.assert_close(cached.mvn.covariance_matrix,uncached.mvn.covariance_matrix,rtol=0,atol=1e-12)


def test_corrected_field_matches_direct_joint_posterior_covariance():
    model,tilt=model_and_tilt();xx=torch.tensor([[2.,0.],[1.,1.]],dtype=torch.double);tilt.prepare_cache(model)
    with torch.no_grad():
        joint=model.posterior(torch.cat([xx,tilt.Xg])).mvn.covariance_matrix
        expected=tilt._lam*(joint[:2,2:]@tilt._a)
        actual=(tilt.posterior(model,xx).mean-model.posterior(xx).mean).reshape(-1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-11)
        base=e.nominal.JitterModel(model).posterior(xx)
        torch.testing.assert_close(tilt.posterior(model,xx).mvn.covariance_matrix,base.mvn.covariance_matrix)


def test_region_can_select_outside_bo_top5():
    base=np.array([1.,.99,.98,.97,.96,.95,.94]);ids=[f'p{i}' for i in range(7)]
    knowledge={'clauses':[{'id':'H1'}]};membership=np.ones((1,7),dtype=bool)
    guided=lambda retained:np.array([1.,2.,3.,4.,5.,6.,100.])
    for method in ('LGBO_region','LGBO_gate','B4_LGBO'):
        selected,_=e.policies.choose_policy(method,base,knowledge,membership,1.4,np.zeros(7,dtype=bool),ids,guided)
        assert selected==6


def test_rbo_confidence_one_not_three_and_top5_guard():
    pool={'task_id':'test','axes':['x'],'levels':{'x':[str(i) for i in range(7)]},
        'points':[{'x':str(i)} for i in range(7)],'ids':[f'p{i}' for i in range(7)]}
    raw={'comment':'test','keywords':'x','hypotheses':[
        {'strategy':str(i),'rationale':'test','confidence':c,'parameter_sets':[pool['points'][i]]}
        for i,c in [(0,'medium'),(2,'high'),(1,'high')]]}
    assert l.validate_rbo(raw,pool,[],pool['ids'][:5])['selected_id']=='p2'
    raw['hypotheses'][0]['parameter_sets']=[pool['points'][6]]
    with pytest.raises(ValueError):l.validate_rbo(raw,pool,[],pool['ids'][:5])


def test_counter_separates_observations_and_query_round():
    template=l.optimization_template()
    text=template.format(iteration=4,query_round=1,target='max',retrieved_context=None,insight_history='x',
        trial_data='4 real rows',bo_recommendations='five',constraint='known',parameter_definitions='nominal')
    assert 'at query round 1.' in text and 'data from 4 completed experiments' in text


def test_null_innovation_does_not_break_events():
    state={'method':'New_R2','history':[{'candidate_id':str(i),'y':.5} for i in range(7)],
        'decisions':[{'measurement':None}]*3,'calls':[],
        'knowledge':{'clauses':[]}}
    class Diagnosis:conflict_kind='none'
    assert 'stagnation' in e.event_reasons(state,{'diagnosis':Diagnosis()},4)


def test_sobol_initial_is_features_only_and_shared():
    pool={'axes':['x','y'],'levels':{'x':['a','b','c'],'y':['m','n','o']},
        'points':[{'x':a,'y':b} for a in ['a','b','c'] for b in ['m','n','o']],
        'ids':[f'p{i}' for i in range(9)]}
    a=s.sobol_initial(pool);b=s.sobol_initial(deepcopy(pool));assert a==b and len(set(a))==4


def test_nominal_gp_invariant_to_arbitrary_codes(tmp_path):
    pool={'task_id':'toy','axes':['x','y'],'levels':{'x':['a','b','c'],'y':['m','n']},
        'points':[{'x':a,'y':b} for a in ['a','b','c'] for b in ['m','n']],
        'ids':[f'p{i}' for i in range(6)]}
    history=[{'candidate_id':key,'y':y} for key,y in zip(['p0','p1','p3','p4'],[.1,.3,.8,.4])]
    torch.set_num_threads(1)
    a=e.Cache(pool,tmp_path/'a');b=e.Cache(pool,tmp_path/'b',permutation={'x':[9,2,100],'y':[7,3]})
    np.testing.assert_allclose(a.stats(history)['mu_y'],b.stats(history)['mu_y'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(a.stats(history)['sd_y'],b.stats(history)['sd_y'],rtol=0,atol=1e-10)


def test_native_justify_runs16_single_queries_without_feedback_leak():
    from chem.benchmarks import pointwise_hte_justify as j
    from types import SimpleNamespace
    points=[{'x':a,'y':b} for a in range(4) for b in range(6)]
    pool={'ids':[f'p{i:02d}' for i in range(24)],'axes':['x','y'],'points':points,'reaction_class':'synthetic test'}
    ids=pool['ids'];X=torch.tensor([[p['x'],p['y']] for p in points],dtype=torch.double)
    initial=[{'candidate_id':ids[i],'y':float(i)/100} for i in range(4)];history=deepcopy(initial)
    def score(prefix):return np.arange(24,dtype=float),np.zeros(24),{'synthetic_only':True}
    cache=SimpleNamespace(X=X,task_seed=0,score=score)
    def propose(prefix,prompt,t):
        assert len(prefix)==t+3
        key=next(key for key in reversed(ids) if key not in {h['candidate_id'] for h in prefix})
        return {'candidate_id':key},{'context_sha256':'SYNTHETIC_ONLY'}
    def objective(key,event,prefix):
        assert {h['candidate_id']:h['y'] for h in prefix}=={h['candidate_id']:h['y'] for h in history}
        assert key not in {h['candidate_id'] for h in history}
        y=float(len(history))/100;history.append({'candidate_id':key,'y':y});return y
    result=j.execute(pool,{'history':initial},cache,propose,objective)
    assert len(history)==20 and result['native_gate_calls']==16


def test_native_rbo_loop_has_correct4_observation_count(tmp_path,monkeypatch):
    pool={'task_id':'synthetic','axes':['x'],'levels':{'x':[str(i) for i in range(9)]},
        'points':[{'x':str(i)} for i in range(9)],'ids':[f'p{i}' for i in range(9)]}
    from chem.benchmarks import reasoning_bo_data as d
    monkeypatch.setattr(d,'ROOT',tmp_path/'inputs');monkeypatch.setattr(l,'ROOT',tmp_path/'requests')
    d.write(d.ROOT/'synthetic/compass.json',{'target':'maximize','constraint':'known options',
        'parameter_definitions':[{'display_name':'x','bounds':list(pool['levels']['x'])}]})
    raw={'comment':'synthetic','keywords':'x','hypotheses':[
        {'strategy':str(i),'rationale':'synthetic','confidence':c,'parameter_sets':[pool['points'][i]]}
        for i,c in [(4,'medium'),(5,'high'),(6,'high')]]}
    def fake(client,context,folder,system,validator):
        assert context['public']['observation_count']==4
        prompt=context['public']['official_prompt_with_pointwise_count_fix']
        assert 'query round 1.' in prompt and 'data from 4 completed experiments' in prompt
        return {'raw':raw,'plan':validator(raw)}
    monkeypatch.setattr(l.nominal,'get',fake)
    state={'task':'synthetic','method':'ReasoningBO_point','history':[{'candidate_id':f'p{i}','y':i*.1} for i in range(4)],
        'decisions':[],'initial_knowledge':{'confidence':.7,'clauses':[]},'protocol_sha256':'SYNTHETIC_ONLY'}
    reasoner=l.make_reasoner(None,pool,tmp_path/'native','SYNTHETIC_ONLY')
    key,_,_=l.rbo_call(reasoner,pool,state,[4,5,6,7,8]);assert key=='p5'
