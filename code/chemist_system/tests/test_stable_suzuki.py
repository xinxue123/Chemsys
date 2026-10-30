from copy import deepcopy
import numpy as np
from chem.benchmarks import stable_suzuki_engine as e,stable_suzuki_policy as p


def test_region_risk_does_not_reset_with_confidence_or_clause_name():
    a={'confidence':.8,'clauses':[{'id':'old','coordinate':3,'bounds':[],'categories':[2,1]}]}
    b={'confidence':.9,'clauses':[{'id':'new','coordinate':3,'bounds':[],'categories':[1,2]}]}
    assert e.trust.geometry_key(e.canonical_region(a,['old']))==e.trust.geometry_key(e.canonical_region(b,['new']))


def test_no_nearest_neighbor_substitution():
    pool={'ids':['z','a'],'points':[{'catalyst':'P1-L1','t_res':100.,'temperature':40.,'catalyst_loading':1.},
        {'catalyst':'P1-L2','t_res':100.1,'temperature':40.,'catalyst_loading':1.}],
        'z':[[.1,.1,.1,.0625],[.1001,.1,.1,.3125]]}
    state={'history':[{'candidate_id':'z','y':.5}],'validations':[],'knowledge':{'confidence':.8,
        'clauses':[{'id':'H4','coordinate':3,'categories':[0],'bounds':[]}]}}
    d={'base':np.array([0.,1.]),'available':np.array([False,True]),'eligible':np.array([False,True]),
       'reference':1,'stats':{'mu_y':np.array([.5,.5]),'sd_y':np.array([.1,.1])}}
    prepared=p.build_candidates(pool,state,d)
    assert prepared['candidates']==[] and 'no exact' in prepared['skip_reason']
    pool['z'][1][0]=.1;pool['points'][1]['t_res']=100.
    prepared=p.build_candidates(pool,state,d);assert len(prepared['candidates'])==1
    assert prepared['candidates'][0]['candidate_id']=='a'


def test_same_region_requires_later_positive_feedback_and_clear_B4():
    d={'region':{'allowed':{'catalyst':[1,2]}},'fallback':True,'amplitude':0.}
    state={'history':[{}]*4,'region_trust':{}}
    first=e.trust.permission(state,d,1);assert first['trust_blocked']
    d.update(fallback=False,amplitude=.8)
    assert e.trust.permission(state,d,2)['trust_blocked']
    key=first['geometry_key'];entry=state['region_trust'][key]
    entry['pending_evidence']={'after_observation_count':5,'positive_evidence':True}
    assert e.trust.permission(state,d,2)['trust_blocked']
    state['history'].append({})
    assert e.trust.permission(state,d,2)['recovered']


def test_information_proxy_invariant_to_nominal_relabeling():
    X=np.array([[.1,.2,.3,0],[.1,.2,.3,1],[.1,.8,.3,0],[.7,.8,.3,1]])
    renamed=X.copy();renamed[:,3]=np.where(X[:,3]==0,101,-30)
    a=e.information_proxy(X,[0,3]);b=e.information_proxy(renamed,[0,3])
    assert all(np.array_equal(x,y) for x,y in zip(a,b))


def test_random_priority_not_candidate_ids_or_global_rng():
    a=e.priority(100,3,0);np.random.seed(42);np.random.normal(size=1000)
    assert np.array_equal(a,e.priority(100,3,0))
    assert not np.array_equal(a,e.priority(100,4,0))
