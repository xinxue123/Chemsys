from copy import deepcopy
import numpy as np
from chem.benchmarks import r2_tuning_policy_v2 as p


def example(monkeypatch):
    monkeypatch.setattr(p,'settings',lambda s:{'high_yield_threshold_percent':95.})
    pool={'axes':['x','y'],'levels':{'x':['a','b'],'y':['m','n','o','p']},
        'points':[{'x':a,'y':b} for a in ['a','b'] for b in ['m','n','o','p']],
        'ids':[f'p{i}' for i in range(8)]}
    state={'history':[{'candidate_id':'p0','y':.6},{'candidate_id':'p5','y':.1}],
        'knowledge':{'confidence':.7,'clauses':[{'id':'H1','axis':'x','allowed':['a']},{'id':'H2','axis':'y','allowed':['m']}]},
        'validations':[]}
    d={'base':np.arange(8,dtype=float),'reference':7,'eligible':np.zeros(8,dtype=bool),
        'stats':{'mu_y':np.ones(8)*.3,'sd_y':np.ones(8)*.2}}
    return pool,state,d


def test_high_observed_incumbent_skips_forced_validation(monkeypatch):
    pool,s,d=example(monkeypatch);s['history'][0]['y']=.9557
    assert p.build_candidates(pool,s,d)['candidates']==[]


def test_first_test_public_pool_unchanged(monkeypatch):
    pool,s,d=example(monkeypatch)
    assert p.build_candidates(pool,s,d)==p.base.build_candidates(pool,s,d)


def test_later_test_focuses_unseen_values_on_other_dimension(monkeypatch):
    pool,s,d=example(monkeypatch);s['validations']=[{'changed_axis':'x'}]
    got=p.build_candidates(pool,s,d)
    assert {c['candidate_id'] for c in got['candidates']}=={'p2','p3'}
    assert got['coverage_focus']['axis']=='y'
    assert all(c['changed_axis']=='y' for c in got['candidates'])


def test_focus_is_nominal_not_an_integer_distance(monkeypatch):
    pool,s,d=example(monkeypatch);s['validations']=[{'changed_axis':'x'}]
    before=deepcopy(s);a=p.build_candidates(pool,s,d)
    pool['levels']['y']=['p','o','n','m'];b=p.build_candidates(pool,s,d)
    assert [c['candidate_id'] for c in a['candidates']]==[c['candidate_id'] for c in b['candidates']]
    assert before==s
