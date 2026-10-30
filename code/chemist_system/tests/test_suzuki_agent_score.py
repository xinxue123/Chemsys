import numpy as np
import pytest
from chem.benchmarks import agent_score_suzuki_engine as e


def card():return {'bias':[0]*8,'target':[.5]*3,'curvature':[4]*3,'novelty':2,'anchors':[],'reason':'observed initial point','support':[1]}


def test_full_score_formula_with_same_category_novelty():
    z=np.array([[.5,.5,.5],[1,.5,.5],[.5,.5,.5]])
    got=e.score_pool(z,[0,0,1],z[:1],[0],card())
    assert np.allclose(got,[0,-1+1/np.sqrt(3),2])


def test_signed_memory_anchor_only_applies_same_catalyst():
    a=card();a['curvature']=[0]*3;a['novelty']=0;a['anchors']=[[1,-3,.2]]
    z=np.array([[.5,.5,.5],[.5,.5,.5]])
    assert np.array_equal(e.score_pool(z,[0,1],z[:1],[0],a),[-3,0])


def test_no_repeats_and_deterministic_lexical_tie():
    assert e.select(np.array([2.,2.,3.]),['b','a','c'],{'c'})==1


def test_card_cannot_reference_future_observation():
    a=card();a['anchors']=[[2,1,.3]]
    with pytest.raises(ValueError):e.validate_card(a,[{'y':.5}])


def test_score_api_has_no_outcome_vector():
    import inspect
    assert set(inspect.signature(e.score_pool).parameters)=={'z','categories','history_z','history_categories','card'}


def test_frozen_score_ignores_future_experience():
    a=card();z=np.array([[.1,.2,.3],[.8,.8,.8]]);c=[0,0]
    frozen=e.score_pool(z,c,z[:1],[0],a)
    adaptive=e.score_pool(z,c,z,[0,0],a)
    assert frozen[1]>adaptive[1]
