from copy import deepcopy
import numpy as np
from chem.benchmarks import stable_counter_bridge as b,stable_counter_engine as e


def example():
    pts=[{'x':x,'c':c,'z':'m'} for x in ['a','b'] for c in ['old','new']]
    pool={'ids':['p0','p1','p2','p3'],'axes':['x','c','z'],'levels':{'x':['a','b'],'c':['old','new'],'z':['m']},'points':pts}
    s={'history':[{'candidate_id':'p0','y':.96},{'candidate_id':'p2','y':.2}],
       'knowledge':{'clauses':[{'id':'H1','axis':'x','allowed':['a']}]},'recombination_calls':[],'tie_seed':0}
    d={'base':np.array([0.,0.,-5.,-5.]),'reference':1,'eligible':np.array([False,True,False,False]),
       'available':np.array([False,True,False,True]),'stats':{'mu_y':np.ones(4)*.3,'sd_y':np.ones(4)*.1},
       'model_identifiability':{'extreme_unsupported':True,'critical_axis':'x','isolated_pairs':{'x':0,'c':0,'z':0},
         'unseen_levels':{'x':[],'c':['new'],'z':[]},'lengthscales':{'x':1e-6,'c':1e4,'z':1e4}}}
    return pool,s,d


def test_joint_challenge_can_leave_gate_only_in_declared_pool():
    pool,s,d=example();p=b.candidates(pool,s,d,1)
    assert [c['candidate_id'] for c in p['candidates']]==['p3']
    assert not p['candidates'][0]['normal_gate_eligible']
    assert p['critical_axis']=='x' and p['context_axis']=='c'
    raw={'candidate_id':'p3','prediction':'A joint test.','reason':'Different context.','support':[1,2]}
    q=b.validate(raw,p,s['history'])
    assert q['violated_preferences']==['H1']
    assert q['critical_levels']==['b']


def test_no_unsupported_extreme_no_test_or_extra_query():
    pool,s,d=example();d['model_identifiability']['extreme_unsupported']=False
    assert not b.candidates(pool,s,d,1)['candidates']
    d['model_identifiability']['extreme_unsupported']=True
    assert not b.candidates(pool,s,d,3)['candidates']
    s['recombination_calls']=[{},{}];assert not b.candidates(pool,s,d,1)['candidates']


def test_successful_joint_counterexample_refresh_stays_within_budget(monkeypatch):
    monkeypatch.setattr(e,'_original_event_reasons',lambda *args:[])
    s={'recombination_calls':[{'iteration':1,'observed_percent':99.,'incumbent_before_percent':95.,
        'validated':{'plan':{'violated_preferences':['H1']}}}],'calls':[]}
    assert e.event_reasons(s,{},2)==['successful_joint_counterexample']
    s['calls']=[{'iteration':2},{'iteration':4},{'iteration':6}]
    assert e.event_reasons(s,{},7)==[]
