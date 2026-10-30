from copy import deepcopy
from pathlib import Path
import numpy as np
from chem.benchmarks import stable_information_engine as e,stable_complete_policy as policy
from chem.benchmarks import stable_agent_policy as legacy,reasoning_bo_data as data,pointwise_hte_study as common
from chem.benchmarks import stable_complete_study,stable_complete_report


def test_information_is_nominal_and_catalog_order_invariant():
    x=np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[2,0,0],[0,2,0],[1,1,0]],dtype=float);observed=[0,1]
    result,pairs,unseen=e.information_proxy(x,observed)
    recoded=x.copy();maps=[{0:31,1:-2,2:173},{0:-51,1:43,2:9},{0:100,1:-4}]
    for j,m in enumerate(maps):recoded[:,j]=[m[v] for v in x[:,j]]
    r,a,b=e.information_proxy(recoded,observed);np.testing.assert_array_equal(result,r);np.testing.assert_array_equal(a,pairs);np.testing.assert_array_equal(b,unseen)
    permutation=np.array([4,1,6,0,3,2,5]);inverse=np.argsort(permutation)
    r,_,_=e.information_proxy(x[permutation],inverse[observed]);np.testing.assert_array_equal(r,result[permutation])
    assert result[2]>result[0] and result[3]>result[0]


def test_real_tie_uses_isolated_contrast_proxy_and_keeps_top_acquisition(tmp_path):
    common.warm();task='Doyle_Buchwald_Hartwig';pool=data.load_pool(task)
    root=Path('data/runs/stable_r2_20261029/followup_t0');p=next((root/'runs').glob('Doyle*/state.json'));saved=data.read(p)
    ready=data.read(root/'initialization'/task/'ready.json')
    state={**saved,'history':saved['history'][:6],'decisions':saved['decisions'][:2],
        'knowledge':ready['knowledge'],'initial_knowledge':ready['knowledge'],'banks':{},'region_trust':{},
        'calls':[],'fit_recovery_seen':False,'validations':saved['validations'][:1]}
    for event in saved['decisions'][:2]:
        commit=data.read(p.parent/'rounds'/f"{event['iteration']:02d}_committed.json")
        key=commit['decision']['bank_key']
        if key:state['banks'][key]=deepcopy(event['calibration_after'])
        state['region_trust']=deepcopy(event['region_trust_after_feedback'])
    cache=e.Cache(pool,tmp_path);cache.sources=[root/'cache'/task]
    decision=e.setup(cache,state,3);mask=decision['eligible']&(decision['score']==decision['score'][decision['eligible']].max())
    assert mask.sum()==3 and decision['tie_order']['information_active']
    j=decision['selected'];assert mask[j]
    assert decision['stats']['information_proxy'][j]==decision['stats']['information_proxy'][mask].max()
    best=saved['history'][1]['candidate_id'];ix={k:i for i,k in enumerate(pool['ids'])}
    # Base differs from the current best too; the selected extra contrast is on Additive,
    # whose global domain has more unobserved levels than the other tied axes.
    initial_anchor=pool['points'][ix[max(ready['history'],key=lambda h:h['y'])['candidate_id']]]
    assert [a for a in pool['axes'] if pool['points'][j][a]!=initial_anchor[a]]==['Additive']
    assert policy.SYSTEM==legacy.SYSTEM and not np.isnan(decision['stats']['information_proxy']).any()
