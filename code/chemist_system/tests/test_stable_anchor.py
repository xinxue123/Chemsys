from copy import deepcopy
import numpy as np
from chem.benchmarks import stable_r2_engine as e
from chem.decision.domain_guard import select_candidate


def test_anchor_ties_are_feature_based_and_never_cross_distinct_scores(monkeypatch):
    pool={'ids':['observed','z1','a2','b3'],'axes':['x','y'],
        'points':[{'x':'a','y':'m'},{'x':'b','y':'m'},{'x':'b','y':'n'},{'x':'a','y':'n'}]}
    scores=np.array([0.,1.,1.,1.]);available=np.array([False,True,True,True])
    h=np.ones((1,4),dtype=bool)
    dg=select_candidate(scores,h,0.,candidate_ids=pool['ids'],hypothesis_ids=['H1'],queried_mask=~available)
    state={'history':[{'candidate_id':'observed','y':.9}],
        'knowledge':{'clauses':[{'id':'H1','axis':'x','allowed':['a','b']}]},'tie_rule':'anchor_on_recovery'}
    d={'base':scores,'score':scores,'available':available,'eligible':available,'selected':dg.candidate_index,
       'reference':dg.candidate_index,'diagnosis':dg,'fallback':False,'stats':{'sd_y':np.array([.1,.2,.5,.3])}}
    monkeypatch.setattr(e.fixed,'setup',lambda *args:deepcopy(d))
    monkeypatch.setattr(e.original.nominal,'membership',lambda *args:h)
    cache=type('Cache',(),{'pool':pool,'last_health':{'used_robust':True}})()
    for seed in [0,1,2]:
        current=deepcopy(state);current['tie_seed']=seed
        assert e.setup(cache,current,3)['selected']==3
    d['score']=np.array([0.,1.,1.01,1.])
    d['selected']=2
    assert e.setup(cache,deepcopy(state),3)['selected']==2
