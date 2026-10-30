import numpy as np
from copy import deepcopy
from chem.benchmarks import stable_agent_bridge as bridge, stable_agent_policy as policy


def example():
    pool={'ids':['p0','p1','p2','p3','p4'],'axes':['x','y'],
        'levels':{'x':['a','b','c'],'y':['m','n']},
        'points':[{'x':'a','y':'m'},{'x':'a','y':'n'},{'x':'b','y':'m'},{'x':'b','y':'n'},{'x':'c','y':'m'}]}
    state={'history':[{'candidate_id':'p0','y':.96},{'candidate_id':'p3','y':.93}],
        'knowledge':{'confidence':.7,'clauses':[]},'fit_recovery_seen':True,'recombination_calls':[],
        'tie_seed':0,'decisions':[{},{}]}
    d={'available':np.array([False,True,True,False,True]),'eligible':np.array([False,False,True,False,True]),
        'base':np.ones(5),'reference':2,'stats':{'mu_y':np.ones(5)*.9,'sd_y':np.ones(5)*.05}}
    return pool,state,d


def test_recombination_uses_only_observed_supported_levels_and_original_gate():
    pool,s,d=example();p=bridge.candidates(pool,s,d,3)
    assert [c['candidate_id'] for c in p['candidates']]==['p2']
    assert p['candidates'][0]['controlled_anchors']==[{'observation_index':1,'changed_axis':'x'},{'observation_index':2,'changed_axis':'y'}]
    assert s['recombination_calls']==[]


def test_recombination_does_not_invent_evidence_or_spend_extra_slots():
    pool,s,d=example();p=bridge.candidates(pool,s,d,3)
    raw={'candidate_id':'p2','prediction':'Test the one-factor comparison.','reason':'Supported values.','support':[1]}
    assert bridge.validate(raw,p,s['history'])==raw
    for change in [{'fit_recovery_seen':False},{'recombination_calls':[{},{}]}]:
        q={**s,**change};assert not bridge.candidates(pool,q,d,3)['candidates']
    assert not bridge.candidates(pool,s,d,7)['candidates']
    q=deepcopy(s);q['history'][0]['y']=.99
    assert not bridge.candidates(pool,q,d,3)['candidates']


def test_shortlist_keeps_complete_tied_class(monkeypatch):
    pool,s,d=example();s['validations']=[{}]
    rows=[{'candidate_id':k,'base_qLogEI':1.} for k in pool['ids']]
    monkeypatch.setattr(policy.base,'focused_candidates',lambda *args:{'candidates':deepcopy(rows)})
    monkeypatch.setattr(policy.base,'settings',lambda s:{'counterexample_shortlist_size':2})
    got=policy.build_candidates(pool,s,d)
    assert len(got['candidates'])==5
    assert len({c['base_rank'] for c in got['candidates']})==1
