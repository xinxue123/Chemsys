from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from chem.benchmarks import stable_productive_bridge as b,stable_productive_engine as e
from chem.benchmarks import reasoning_bo_data as data,pointwise_hte_study as common


def test_candidates_only_change_two_axes_at_own_best_background():
    pool=data.load_pool('Perera_Suzuki');root=Path('data/runs/stable_r2_20261029/structure_t0')
    s=data.read(next((root/'runs').glob('Perera*/state.json')))
    state={**s,'history':s['history'][:4],'decisions':[],'recombination_calls':[],'knowledge':s['initial_knowledge']}
    ix={k:i for i,k in enumerate(pool['ids'])};axes=pool['axes'];critical=axes[1]
    seen={a:{pool['points'][ix[h['candidate_id']]][a] for h in state['history']} for a in axes}
    d={'available':~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]),'model_identifiability':{
        'extreme_unsupported':True,'critical_axis':critical,'isolated_pairs':{a:0 for a in axes},
        'unseen_levels':{a:list(set(pool['levels'][a])-seen[a]) for a in axes}}}
    p=b.candidates(pool,state,d,1);assert len(p['candidates'])==14
    anchor=pool['points'][ix[p['anchor_candidate_id']]]
    assert p['anchor_candidate_id']==max(state['history'],key=lambda h:h['y'])['candidate_id']
    for row in p['candidates']:
        point=pool['points'][ix[row['candidate_id']]]
        assert set(a for a in axes if point[a]!=anchor[a])=={critical,p['context_axis']}
        assert not any('GP_mean' in k for k in row)
    raw={'ratings':[{'candidate_id':r['candidate_id'],'expected_yield_percent':77.,'reason':'test'} for r in p['candidates']],
        'support':[2],'summary':'all tied'}
    assert b.validate(raw,p,state['history'])['candidate_id']==p['candidates'][0]['candidate_id']
    raw['ratings'][-1]['expected_yield_percent']=94.
    assert b.validate(raw,p,state['history'])['candidate_id']==raw['ratings'][-1]['candidate_id']
    bad=deepcopy(raw);bad['ratings'][0]=bad['ratings'][1]
    with pytest.raises(ValueError):b.validate(bad,p,state['history'])
    bad=deepcopy(raw);bad['ratings'][0]['expected_yield_percent']=float('nan')
    with pytest.raises(ValueError):b.validate(bad,p,state['history'])


def test_healthy_initial_path_does_not_enable_productive_call(tmp_path):
    common.warm();pool=data.load_pool('Doyle_Buchwald_Hartwig')
    ready=data.read(Path('data/runs/stable_r2_20261029/structure_t0/initialization')/pool['task_id']/'ready.json')
    state={**ready,'method':'New_R2','banks':{},'region_trust':{},'calls':[],'validations':[],
        'decisions':[],'recombination_calls':[],'tie_seed':0,'tie_rule':'random','fit_recovery_seen':False}
    c=e.Cache(pool,tmp_path);c.sources=[Path('data/runs/stable_r2_20261029/structure_t0/cache')/pool['task_id']]
    d=e.setup(c,state,1);assert not d['model_identifiability']['extreme_unsupported']
    assert not b.candidates(pool,state,d,1)['candidates']
