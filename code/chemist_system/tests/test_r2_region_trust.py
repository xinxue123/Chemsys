from copy import deepcopy
import pytest
from chem.benchmarks import r2_region_trust_engine as e


def fixture():
    spec={'allowed':{'x':['a','b'],'y':['m']},'confidence':.7}
    state={'history':[{'candidate_id':str(i),'y':.3} for i in range(4)]}
    d={'region':spec,'fallback':False,'amplitude':.8,'selected':0,
        'stats':{'mu_y':[.3]},'prior':{'amplitude':.8},'_measurement':{'raw_region_shift_y':.1}}
    return state,d


def lock(s,d):
    d['fallback']=True;d['trust']=e.permission(s,d,1);d['fallback']=False


def test_no_previous_conflict_does_not_change_permission():
    s,d=fixture();p=e.permission(s,d,1)
    assert p['amplitude_after_trust']==.8 and not p['trust_blocked'] and s['region_trust']=={}


def test_revoke_without_feedback_cannot_unlock_same_region():
    s,d=fixture();lock(s,d);p=e.permission(s,d,1)
    assert p['trust_blocked'] and p['amplitude_after_trust']==0 and p['b4_fallback'] is False


def test_geometry_risk_survives_reorder_and_confidence_edit():
    s,d=fixture();lock(s,d);d['region']={'allowed':{'y':['m'],'x':['b','a']},'confidence':.2}
    assert e.permission(s,d,1)['trust_blocked']


def test_different_geometry_keeps_original_permission():
    s,d=fixture();lock(s,d);d['region']['allowed']['y']=['n']
    assert e.permission(s,d,1)['amplitude_after_trust']==.8


@pytest.mark.parametrize('outcome',[.2,.3])
def test_unsupported_real_feedback_does_not_unlock(outcome):
    s,d=fixture();lock(s,d);d['trust']=e.permission(s,d,1)
    r=e.feedback(s,d,1,outcome,{'mean':[0.,.9]});s['history'].append({'candidate_id':'new','y':outcome})
    assert not r['positive_evidence'] and e.permission(s,d,2)['trust_blocked']


def test_positive_feedback_releases_only_on_later_clear_B4():
    s,d=fixture();lock(s,d);d['trust']=e.permission(s,d,1)
    r=e.feedback(s,d,1,.4,{'mean':[0.,.7]})
    assert r['positive_evidence'] and s['region_trust'][e.geometry_key(d['region'])]['blocked']
    assert e.permission(s,d,1)['trust_blocked']  # same prefix cannot use future feedback
    s['history'].append({'candidate_id':'new','y':.4});d['amplitude']=.7
    p=e.permission(s,d,2);assert p['recovered'] and p['amplitude_after_trust']==.7


def test_new_B4_conflict_invalidates_old_supporting_feedback():
    s,d=fixture();lock(s,d);d['trust']=e.permission(s,d,1)
    e.feedback(s,d,1,.4,{'mean':[0.,.7]});s['history'].append({'candidate_id':'new','y':.4})
    d['fallback']=True;e.permission(s,d,2);d['fallback']=False
    assert e.permission(s,d,2)['trust_blocked']


def test_zero_field_does_not_create_positive_evidence():
    s,d=fixture();lock(s,d);d['trust']=e.permission(s,d,1);d['_measurement']['raw_region_shift_y']=0.
    assert not e.feedback(s,d,1,.4,{'mean':[0.,.7]})['positive_evidence']


def test_resume_preserves_the_gate():
    s,d=fixture();lock(s,d);restored=deepcopy(s)
    assert e.permission(s,d,1)==e.permission(restored,d,1)


def test_blocked_deployment_really_uses_base_choice(monkeypatch):
    from types import SimpleNamespace
    s,d=fixture();lock(s,d)
    d.update(selected=1,reference=0,score=[0.,2.],shift_std=[.1,.1],guided_meta={})
    monkeypatch.setattr(e.original,'setup',lambda cache,state,t:deepcopy(d))
    cache=SimpleNamespace(score=lambda history:([2.,0.],[0.,0.],{'base':True}))
    actual=e.setup(cache,s,1)
    assert actual['selected']==0 and actual['amplitude']==0 and actual['shift_std']==[0.,0.]


def test_reply_reuse_only_tolerates_float_roundoff():
    from chem.benchmarks.r2_region_trust_study import equivalent
    a={'mu':.3,'point':'A','observed':[{'index':1,'y':.2}]}
    b=deepcopy(a);b['mu']+=1e-15;assert equivalent(a,b)
    b['mu']+=1e-4;assert not equivalent(a,b)
    b=deepcopy(a);b['point']='B';assert not equivalent(a,b)
    b=deepcopy(a);b['observed'][0]['index']=2;assert not equivalent(a,b)


def test_native_B4_tuples_equal_their_saved_JSON_arrays():
    import json
    from chem.benchmarks.r2_region_trust_study import equivalent
    native={'retained_ids':('H1','H2'),'subset_checks':({'bits':1,'retained_ids':('H1',)},)}
    assert equivalent(native,json.loads(json.dumps(native)))
