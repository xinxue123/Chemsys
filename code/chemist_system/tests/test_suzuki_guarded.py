"""Meaningful new-policy semantics; original experiments stay immutable."""
import numpy as np
import pytest
from chem.benchmarks import guarded_suzuki_engine as e


def fixture():
    base=np.array([0.,-.10,-.25,-1.2])
    member=np.array([[True,False,False,True],[False,False,True,True]])
    advice={'confidence':.7,'clauses':[{'id':'H1','coordinate':0,'interval':[0,.5]},
                                      {'id':'H2','coordinate':1,'interval':[0,.5]}]}
    return base,member,advice,np.zeros(4,dtype=bool),['a','b','c','d']


def test_budget_is_log_ratio_and_reaches_zero():
    assert e.budget_at(4,1)==np.log(4)
    assert e.budget_at(16,16)==0


def test_joint_revocation_and_soft_prior_semantics_are_explicit():
    base,member,advice,q,ids=fixture()
    guided=lambda retained:np.array([0.,.2,.5,2.])
    hard,receipt=e.choose_policy('B4_base',base,advice,member,.3,q,ids,guided)
    soft,event=e.choose_policy('B4_LGBO',base,advice,member,.3,q,ids,guided)
    assert hard==0 and receipt['retained_ids']==['H1']
    assert soft==2 and event['retained_ids']==['H1']
    assert not event['selected_satisfies_retained_box']
    assert event['retained_box_witness_count']==1 and not event['budget_violation']


def test_single_candidate_and_constrained_maximization_are_distinct():
    base,member,advice,q,ids=fixture();guided=lambda retained:np.array([0.,.2,.5,2.])
    single,_=e.choose_policy('B3_gate',base,advice,member,.3,q,ids,guided)
    gate,_=e.choose_policy('LGBO_gate',base,advice,member,.3,q,ids,guided)
    assert single==0 and gate==2


def test_revoke_all_does_not_silently_become_minimum_revocation():
    base,member,advice,q,ids=fixture()
    selected,event=e.choose_policy('A2_LGBO',base,advice,member,.3,q,ids,lambda r:base)
    assert selected==0 and event['retained_ids']==[]


def test_no_advice_reduces_all_policies_to_same_base_choice():
    base,_,_,q,ids=fixture();a={'confidence':.7,'clauses':[]};h=np.empty((0,4),dtype=bool)
    assert all(e.choose_policy(m,base,a,h,.3,q,ids,lambda r:base)[0]==0 for m in e.METHODS)
