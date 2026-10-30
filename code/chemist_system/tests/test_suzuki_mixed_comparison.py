from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from chem.benchmarks import mixed_comparison_suzuki_engine as e
from chem.benchmarks import mixed_comparison_suzuki_study as s

torch.set_num_threads(1)


@pytest.fixture
def pool():
    return {'ids':[f'p{i:05}' for i in range(8)],
            'z':[[.1+.1*i,.25+.04*i,.8-.08*i,(i+.5)/8] for i in range(8)],
            'points':[{'catalyst':cat,'t_res':60+540*(.1+.1*i),
                       'temperature':30+80*(.25+.04*i),'catalyst_loading':.5+2*(.8-.08*i)}
                      for i,cat in enumerate(e.CATALYSTS)]}


def test_quadrature_is_balanced_over_real_categories_and_numeric_points():
    tilt=e.NominalRegionalTilt()
    assert tilt.Xg.shape==(512,4)
    assert torch.equal(tilt.Xg[:8,:3],tilt.Xg[:1,:3].expand(8,3))
    assert torch.equal(torch.bincount(tilt.Xg[:,3].long()),torch.full((8,),64))


def test_categorical_preference_is_a_set_not_smoothed_distance():
    tilt=e.NominalRegionalTilt();tilt.set_box_region([0,0,0,.25],[1,1,1,.375])
    assert tilt.allowed_categories==(2,)
    for k in range(8):
        weights=tilt._a[tilt.canonical_category==k]
        assert bool((weights>0).all()) if k==2 else bool((weights==0).all())
    assert float(tilt._a.sum())==pytest.approx(1.)


def test_category_renaming_preserves_weights_and_numeric_grid():
    first=e.NominalRegionalTilt();second=e.NominalRegionalTilt([19,-3,101,7,0,12,-9,4])
    for tilt in (first,second):tilt.set_box_region([.2,.1,.3,.25],[.8,.9,.7,.5])
    assert torch.equal(first._a,second._a) and torch.equal(first.Xg[:,:3],second.Xg[:,:3])
    assert first.allowed_categories==second.allowed_categories==(2,3)


def test_vendor_calibration_and_mean_tilt_are_inherited_unchanged():
    for name in ('fit_lambda_by_delta','prepare_cache','posterior'):
        assert getattr(e.NominalRegionalTilt,name) is getattr(e.native.LinearExponentialRegionalMeanTiltPlugAndPlay,name)


def test_actual_fitted_guidance_is_invariant_to_category_names(pool,tmp_path):
    h=[{'candidate_id':pool['ids'][i],'y':y} for i,y in enumerate((.1,.3,.15,.5))]
    a=e.MixedRegionalCache(pool,tmp_path/'a',1)
    b=e.MixedRegionalCache(pool,tmp_path/'b',1,category_map=[19,-3,101,7,0,12,-9,4])
    box=([.2,0,0,.25],[1,1,1,.5])
    ba,_=a.score(h);bb,_=b.score(h);ga,meta=a.score(h,box);gb,_=b.score(h,box)
    assert np.allclose(ba,bb,rtol=0,atol=1e-10)
    assert np.allclose(ga,gb,rtol=0,atol=1e-10)
    assert meta['category_names']==['P1-L2','P1-L3']
    assert a.counters['fits']==a.counters['regional_acquisitions']==1
    a.score(h,box);assert a.counters['fits']==1


def test_empty_category_set_is_rejected():
    with pytest.raises(ValueError):e.NominalRegionalTilt().set_box_region([0,0,0,.01],[1,1,1,.02])


def test_point_response_uses_observed_evidence_and_physical_domain(pool):
    h=[{'candidate_id':'p00000','y':.1}]
    payload={'point':pool['points'][1],'reason':'Explore an unobserved category','support':[1]}
    answer={'model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload)}}]}
    assert e.parse_point(answer,h)==payload
    payload['support']=[2];answer['choices'][0]['message']['content']=json.dumps(payload)
    with pytest.raises(ValueError):e.parse_point(answer,h)


def test_point_request_excludes_scores_truth_labels_and_unobserved_results(pool):
    prior={'confidence':.7,'clauses':[{'id':'H1','coordinate':3,'interval':[.25,.375]}]}
    h=[{'candidate_id':'p00000','y':.1}]
    public=e.public_point_request(h,'original prompt',1,pool,prior,'X01')
    assert public['observations']==[{'index':1,'point':pool['points'][0],'yield_fraction':.1}]
    assert public['uncertain_preferences']['clauses'][0]['preferred']==['P1-L2']
    assert not {'case','regime','GP','scores','oracle','candidate_id'}&set(public)


def test_all_four_conditions_and_five_requested_policies_are_preserved():
    assert s.REGIMES==('correct','wrong_category','mixed','assistant_initial')
    assert set(s.METHODS)=={'LGBO_region','LLINBO_matched','LGBO_gate','B4_base','B4_LGBO'}
    assert s.PARAMETERS==s.previous.PARAMETERS
