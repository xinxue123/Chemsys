import numpy as np
from chem.benchmarks import stable_tie_bridge as b


def fixture():
    pool={'ids':[f'p{i}' for i in range(8)],'axes':['x'],
        'levels':{'x':[str(i) for i in range(8)]},'points':[{'x':str(i)} for i in range(8)]}
    s={'history':[{'candidate_id':'p0','y':.95}],'recombination_calls':[],'tie_seed':0}
    d={'base':np.array([-1.,1.,1.,1.,1.,1.,1.,0.]),'available':np.array([False]+[True]*7),
       'stats':{'mu_y':np.ones(8)*.8,'sd_y':np.ones(8)*.2}}
    return pool,s,d


def test_all_maximum_members_are_retained_without_ID_cutoff():
    pool,s,d=fixture();p=b.candidates(pool,s,d,1)
    assert set(p['candidate_ids'])==set(pool['ids'][1:7])
    assert len(p['candidates'])==6
    assert set(r[0] for r in p['candidates'])==set(p['candidate_ids'])
    assert p['category_structures']['x']['7']=='7'


def test_only_early_maximum_ties_and_valid_observed_evidence():
    pool,s,d=fixture();p=b.candidates(pool,s,d,1)
    raw={'candidate_id':'p2','prediction':'Resolve a hypothesis.','reason':'Use observed chemistry.','support':[1]}
    assert b.validate(raw,p,s['history'])==raw
    assert not b.candidates(pool,s,d,3)['candidate_ids']
    s['recombination_calls']=[{},{}];assert not b.candidates(pool,s,d,1)['candidate_ids']
    s['recombination_calls']=[];d['base'][1]=1.01
    assert not b.candidates(pool,s,d,1)['candidate_ids']
