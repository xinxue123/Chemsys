from copy import deepcopy
import numpy as np
import pytest
from chem.benchmarks import r2_tuning_policy as p


def example():
    pool={'axes':['x','y'],'levels':{'x':['a','b','c'],'y':['m','n']},
        'points':[{'x':a,'y':b} for a in ['a','b','c'] for b in ['m','n']],
        'ids':[f'p{i}' for i in range(6)]}
    state={'history':[{'candidate_id':'p0','y':.6},{'candidate_id':'p3','y':.1}],
        'knowledge':{'confidence':.7,'clauses':[{'id':'H1','axis':'x','allowed':['a']}]}}
    d={'base':np.arange(6,dtype=float),'reference':5,'eligible':np.zeros(6,dtype=bool),
        'stats':{'mu_y':np.ones(6)*.3,'sd_y':np.ones(6)*.2}}
    return pool,state,d


def raw():return {'candidate_id':'p2','clause_id':'H1','prediction':'counterexample','reason':'controlled test','support':[1]}


def test_pool_is_one_factor_unqueried_counterexamples():
    pool,s,d=example();x=p.build_candidates(pool,s,d)
    assert x['anchor_candidate_id']=='p0'
    assert {c['candidate_id'] for c in x['candidates']}=={'p2','p4'}
    assert all(c['changed_axis']=='x' and c['challenged_clause_ids']==['H1'] for c in x['candidates'])


def test_no_future_yields_needed_or_exported():
    pool,s,d=example();x=p.build_candidates(pool,s,d)
    assert all('y' not in c and 'yield_percent' not in c for c in x['candidates'])
    assert p.validate(raw(),x,s['history'])['changed_axis']=='x'


def test_validation_can_be_outside_normal_gate():
    pool,s,d=example();x=p.build_candidates(pool,s,d)
    assert all(not c['normal_gate_eligible'] for c in x['candidates'])
    assert p.validate(raw(),x,s['history'])['candidate_id']=='p2'


@pytest.mark.parametrize('changes',[{'candidate_id':'p5'},{'clause_id':'H2'},{'support':[3]},
    {'support':[1,1]},{'prediction':''},{'support':[{}]},{'candidate_id':{}}])
def test_reject_invalid_or_unobserved_choices(changes):
    pool,s,d=example();x=p.build_candidates(pool,s,d);r=raw();r.update(changes)
    with pytest.raises(ValueError):p.validate(r,x,s['history'])


def test_no_knowledge_means_no_validation_pool():
    pool,s,d=example();s['knowledge']['clauses']=[]
    assert p.build_candidates(pool,s,d)['candidates']==[]


def test_candidate_builder_does_not_mutate_input():
    pool,s,d=example();before=deepcopy(s);p.build_candidates(pool,s,d);assert s==before
