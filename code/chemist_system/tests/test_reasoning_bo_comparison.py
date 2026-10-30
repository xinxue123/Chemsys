"""Scientific contracts for the batch extension, not duplicated implementation."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_engine as engine
from chem.benchmarks import reasoning_bo_llm as llm
from chem.benchmarks import reasoning_bo_study as study


@pytest.fixture
def pool():
    axes=['ligand','base']
    points=[{'ligand':a,'base':b} for a in ('A','B','C') for b in ('X','Y')]
    return {'task_id':'contract_synthetic','axes':axes,'levels':{'ligand':['A','B','C'],'base':['X','Y']},
        'points':points,'ids':[f'p{i}' for i in range(6)]}


def insight(pool):
    return {'comment':'observed-only contract test','keywords':'ligand',
        'hypotheses':[{'strategy':str(i),'rationale':'synthetic input','confidence':confidence,
            'parameter_sets':[pool['points'][j]]} for i,(confidence,j) in enumerate(
                [('medium',0),('high',1),('high',2),('low',3),('medium',4)])]}


def test_native_confidence_sort_stable_not_reimplemented(pool):
    raw=insight(pool)
    assert llm.extract(raw)==[pool['points'][1],pool['points'][2],pool['points'][0]]
    assert all(len(r['sha256'])==64 for r in llm.SOURCE_RECEIPT.values())


def test_paper_subset_and_three_distinct_enforced(pool):
    raw=insight(pool)
    checked=llm.validate_insight(raw,pool,[],pool['ids'][:5])
    assert checked['selected_ids']==['p1','p2','p0']
    with pytest.raises(ValueError,match='five BO'):
        llm.validate_insight(raw,pool,[],['p0','p1','p3','p4','p5'])
    raw['hypotheses'][2]['parameter_sets']=[pool['points'][1]]
    with pytest.raises(ValueError,match='distinct'):
        llm.validate_insight(raw,pool,[],pool['ids'][:5])


def test_no_already_observed_point(pool):
    with pytest.raises(ValueError,match='observed'):
        llm.validate_insight(insight(pool),pool,[{'candidate_id':'p1','y':.9}],pool['ids'][:5])


def test_b4_requires_one_common_witness_not_different_points():
    pool={'axes':['x','y'],'ids':['p0','p1','p2'],
        'points':[{'x':'a','y':'z'},{'x':'b','y':'w'},{'x':'b','y':'z'}]}
    knowledge={'clauses':[{'id':'H1','axis':'x','allowed':['a']},{'id':'H2','axis':'y','allowed':['w']}]}
    d=engine.batch_diagnosis(np.array([0.]),[[0,1,2]],pool,knowledge,1.)
    assert d.retained_ids==('H1',)
    assert d.revoked_ids==('H2',)
    assert d.conflict_kind=='empty_intersection'


def test_batch_kf_reduces_exactly_to_frozen_scalar():
    bank=engine.nominal.new_bank(1);prior=engine.nominal.forecast(bank,1)
    old,m=engine.nominal.update_bank(bank,prior,1,.02,.6,.04,.7)
    new,n=engine.joint_kf(bank,prior,1,{'raw_region_shift_y':[.02],
        'prequery_mu_y':[.6],'prequery_observation_covariance':[[.04**2]]},[.7])
    np.testing.assert_allclose(new['mean'],old['mean'],atol=1e-15,rtol=0)
    np.testing.assert_allclose(new['covariance'],old['covariance'],atol=1e-15,rtol=0)
    assert n['innovation_z']==pytest.approx(abs(m['innovation_z']))


def test_joint_kf_permutation_invariant_and_positive_covariance():
    bank=engine.nominal.new_bank(1);prior=engine.nominal.forecast(bank,1)
    m={'raw_region_shift_y':[.02,.06,-.01],'prequery_mu_y':[.2,.3,.4],
        'prequery_observation_covariance':[[.001,.0002,.0001],[.0002,.003,.0003],[.0001,.0003,.002]]}
    y=np.array([.24,.5,.3]);a,receipt=engine.joint_kf(bank,prior,1,m,y)
    ix=[2,0,1]
    perm={'raw_region_shift_y':np.array(m['raw_region_shift_y'])[ix],
        'prequery_mu_y':np.array(m['prequery_mu_y'])[ix],
        'prequery_observation_covariance':np.array(m['prequery_observation_covariance'])[ix][:,ix]}
    b,_=engine.joint_kf(bank,prior,1,perm,y[ix])
    np.testing.assert_allclose(a['mean'],b['mean'],atol=1e-13,rtol=0)
    np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-13,rtol=0)
    assert np.linalg.eigvalsh(a['covariance']).min()>0


def test_all_category_gp_invariant_to_arbitrary_bijections(pool,tmp_path):
    torch.set_num_threads(1)
    history=[{'candidate_id':k,'y':y} for k,y in zip(['p0','p3','p4'],[.1,.4,.8])]
    a=engine.BatchCache(pool,tmp_path/'a')
    b=engine.BatchCache(pool,tmp_path/'b',permutation={'ligand':[9,2,100],'base':[13,4]})
    ma=a.stats(history);mb=b.stats(history)
    np.testing.assert_allclose(ma['mu_y'],mb['mu_y'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(ma['sd_y'],mb['sd_y'],atol=1e-10,rtol=0)


def test_commit_persisted_before_any_batch_outcome(tmp_path,monkeypatch):
    receipt={'task':'synthetic','iteration':1,'selected_ids':['a','b','c'],
        'simultaneous_commit':True,'history_count':3}
    def spy(task,ids):
        saved=json.loads((tmp_path/'batches/01_committed.json').read_text(encoding='utf8'))
        assert saved==receipt and ids==['a','b','c']
        return [{'candidate_id':key,'y':.5} for key in ids]
    monkeypatch.setattr(data,'reveal',spy)
    assert len(study.commit_and_reveal(tmp_path,receipt))==3
    receipt['selected_ids']=['a','a','c']
    with pytest.raises(AssertionError):study.commit_and_reveal(tmp_path,receipt)


def test_event_cooldown_and_cap():
    state={'method':'New_R2','calls':[],'decisions':[{'incumbent_gain':0.,'measurement':{'innovation_z':3.}}]*3}
    assert not engine.event_reasons(state,{'fallback':True},2)
    assert set(engine.event_reasons(state,{'fallback':True},3))=={'B4_conflict','stagnation_3_batches','bad_joint_innovation_2_batches'}
    state['calls']=[{'iteration':3},{'iteration':7},{'iteration':11}]
    assert not engine.event_reasons(state,{'fallback':True},15)


def test_full_data_no_entry_or_target_feature():
    for task,count in [('Perera_Suzuki',3696),('Doyle_Buchwald_Hartwig',3955)]:
        p=data.load_pool(task)
        assert len(p['ids'])==count and 'entry' not in p['axes'] and 'yield' not in p['axes']
        assert len(p['axes'])==(5 if task=='Perera_Suzuki' else 4)
        assert all(set(point)==set(p['axes']) for point in p['points'])


def test_official_loop_uses_exact_current_prefix_and_native_sort(tmp_path,monkeypatch):
    points=[{'x':str(i)} for i in range(9)]
    pool={'task_id':'synthetic_native','axes':['x'],'levels':{'x':[str(i) for i in range(9)]},
        'points':points,'ids':[f'p{i}' for i in range(9)]}
    compass={'application_context':'synthetic test','description':'test','target':'maximize',
        'constraint':'known options only','parameter_definitions':[{'display_name':'x','bounds':list(pool['levels']['x'])}]}
    monkeypatch.setattr(data,'ROOT',tmp_path/'inputs')
    monkeypatch.setattr(llm,'ROOT',tmp_path/'requests_only_synthetic')
    data.write(data.ROOT/pool['task_id']/'compass.json',compass)
    raw={'comment':'synthetic only','keywords':'x','hypotheses':[
        {'strategy':str(i),'rationale':'synthetic test','confidence':c,'parameter_sets':[points[i]]}
        for i,c in [(3,'medium'),(4,'high'),(5,'high')]]}
    def fake(client,context,folder,system,validator):
        prompt=context['public']['official_prompt']
        assert '12.3' in prompt and '45.6' in prompt and '78.9' in prompt
        assert context['observed_prefix']==history
        return {'raw':raw,'plan':validator(raw),'response_id':'SYNTHETIC_TEST_ONLY','context_sha256':context['context_sha256']}
    monkeypatch.setattr(llm.nominal,'get',fake)
    r=llm.make_reasoner(None,pool,0,tmp_path/'native','synthetic_protocol')
    history=[{'candidate_id':f'p{i}','y':y} for i,y in enumerate([.123,.456,.789])]
    state={'history':history,'initial_insight':raw,'decisions':[]}
    selected,returned,_=llm.iteration_reasoning(r,state,{'ids':[f'p{i}' for i in range(3,8)]},1)
    assert selected==['p4','p5','p3'] and returned==raw
    assert len((tmp_path/'native/trial_data/Yield.csv').read_text(encoding='utf8').splitlines())==4
