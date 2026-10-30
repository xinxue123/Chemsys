from pathlib import Path
from copy import deepcopy
import numpy as np
from chem.benchmarks import stable_r2_fit as fit
from chem.benchmarks import stable_r2_engine as engine
from chem.benchmarks import stable_r2_policy as policy
from chem.benchmarks import pointwise_hte_study as common
from chem.benchmarks import reasoning_bo_data as data


def known_prefix(task):
    root=Path('data/runs/hte_r2_tuning_20261028/focused_countercheck_v3_shuffle_s0')
    state=data.read(root/'runs'/(task+'_s0_focused_countercheck_v3_shuffle_s0')/'state.json')
    return data.load_pool(task),state['history'][:6],root/'cache'/task


def test_recovers_failed_Perera_fit_only_from_its_observed_prefix(tmp_path):
    common.warm();pool,hist,source=known_prefix('Perera_Suzuki')
    cache=fit.Cache(pool,tmp_path/'p');cache.sources=[source]
    key,value=cache.model(hist);record=deepcopy(cache.last_health)
    assert record['health']['trigger'] and record['used_robust']
    assert record['total_MLL_gain']>2.9
    assert key!=record['base_key']
    assert len(record['attempts'])==2
    assert cache.model(hist)[0]==key and cache.last_health==record
    cache.release()
    assert cache.model(hist)[0]==key and cache.last_health==record


def test_healthy_Doyle_retains_native_fit_and_scores(tmp_path):
    common.warm();pool,hist,source=known_prefix('Doyle_Buchwald_Hartwig')
    native=fit.base.Cache(pool,tmp_path/'native');native.sources=[source]
    robust=fit.Cache(pool,tmp_path/'robust');robust.sources=[source]
    oldkey,_=native.model(hist);newkey,_=robust.model(hist)
    assert oldkey==newkey and not robust.last_health['health']['trigger']
    assert not robust.last_health['used_robust'] and not robust.last_health['attempts']
    np.testing.assert_array_equal(native.score(hist)[0],robust.score(hist)[0])
    np.testing.assert_allclose(native.stats(hist)['mu_y'],robust.stats(hist)['mu_y'],atol=1e-14,rtol=0)


def test_priorities_vary_only_independent_tie_seed_and_round():
    s0=engine.priority(100,3,0);s1=engine.priority(100,3,1)
    np.testing.assert_array_equal(s0,engine.priority(100,3,0))
    assert not np.array_equal(s0,s1)
    scores=np.arange(100,dtype=float)
    np.testing.assert_array_equal(engine.ordered(scores,np.ones(100,dtype=bool),3,1),np.arange(99,-1,-1))


def test_ID_rename_does_not_choose_shortlist(monkeypatch):
    monkeypatch.setattr(policy,'settings',lambda s:{'high_yield_threshold_percent':95.,'counterexample_shortlist_size':5})
    pool={'axes':['x','y'],'levels':{'x':['a','b'],'y':[str(i) for i in range(10)]},
        'points':[{'x':a,'y':str(i)} for a in ['a','b'] for i in range(10)],'ids':[f'p{i:02d}' for i in range(20)]}
    s={'history':[{'candidate_id':'p00','y':.6},{'candidate_id':'p11','y':.1}],
        'knowledge':{'confidence':.7,'clauses':[{'id':'H1','axis':'x','allowed':['a']},{'id':'H2','axis':'y','allowed':['0']}]},
        'validations':[{'query':2}],'decisions':[{}, {}, {}],'tie_seed':1}
    d={'base':np.zeros(20),'reference':0,'eligible':np.ones(20,dtype=bool),
        'stats':{'mu_y':np.ones(20)*.3,'sd_y':np.ones(20)*.2}}
    before=policy.build_candidates(pool,s,d)
    new=deepcopy(pool);new['ids']=[f'z{100-i}' for i in range(20)]
    renamed=deepcopy(s);renamed['history'][0]['candidate_id']=new['ids'][0];renamed['history'][1]['candidate_id']=new['ids'][11]
    after=policy.build_candidates(new,renamed,d)
    assert [c['point'] for c in before['candidates']]==[c['point'] for c in after['candidates']]
    assert len({c['base_rank'] for c in before['candidates']})==1
