import numpy as np
from chem.benchmarks import r2_tuning_policy_v3 as p


def fixture(monkeypatch):
    monkeypatch.setattr(p.base,'settings',lambda s:{'high_yield_threshold_percent':95.,'counterexample_shortlist_size':5})
    pool={'axes':['x','y'],'levels':{'x':['a','b'],'y':[str(i) for i in range(10)]},
        'points':[{'x':a,'y':str(i)} for a in ['a','b'] for i in range(10)],'ids':[f'p{i:02d}' for i in range(20)]}
    s={'history':[{'candidate_id':'p00','y':.6},{'candidate_id':'p11','y':.1}],
        'knowledge':{'confidence':.7,'clauses':[{'id':'H1','axis':'x','allowed':['a']},{'id':'H2','axis':'y','allowed':['0']}]},
        'validations':[]}
    d={'base':np.zeros(20),'reference':0,'eligible':np.ones(20,dtype=bool),
        'stats':{'mu_y':np.ones(20)*.3,'sd_y':np.ones(20)*.2}}
    return pool,s,d


def test_first_validation_unchanged(monkeypatch):
    pool,s,d=fixture(monkeypatch)
    assert p.build_candidates(pool,s,d)==p.base.build_candidates(pool,s,d)


def test_later_pool_is_BO_top5_with_explicit_tie_order(monkeypatch):
    pool,s,d=fixture(monkeypatch);s['validations']=[{'changed_axis':'x'}]
    got=p.build_candidates(pool,s,d)
    assert got['counterexample_pool_before_shortlist']==8
    assert [c['candidate_id'] for c in got['candidates']]==['p02','p03','p04','p05','p06']


def test_surrogate_score_can_change_shortlist_without_outcomes(monkeypatch):
    pool,s,d=fixture(monkeypatch);s['validations']=[{'changed_axis':'x'}];d['base'][9]=10.
    assert p.build_candidates(pool,s,d)['candidates'][0]['candidate_id']=='p09'


def test_incumbent_gate_still_skips(monkeypatch):
    pool,s,d=fixture(monkeypatch);s['history'][0]['y']=.96;s['validations']=[{'changed_axis':'x'}]
    assert p.build_candidates(pool,s,d)['candidates']==[]
