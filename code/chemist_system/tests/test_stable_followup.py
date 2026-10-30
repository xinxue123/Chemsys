from copy import deepcopy
from pathlib import Path
import numpy as np
from chem.benchmarks import stable_followup_bridge as b,stable_diverse_bridge as previous
from chem.benchmarks import reasoning_bo_data as data


def context():
    pool=data.load_pool('Perera_Suzuki');s=data.read(next(Path('data/runs/stable_r2_20261029/diverse_t0/runs').glob('Perera*/state.json')))
    state={**s,'history':s['history'][:6],'decisions':s['decisions'][:2],
        'recombination_calls':s['recombination_calls'][:2],'knowledge':s['initial_knowledge']}
    ix={k:i for i,k in enumerate(pool['ids'])};axes=pool['axes'];seen={a:{pool['points'][ix[h['candidate_id']]][a] for h in state['history']} for a in axes}
    d={'available':~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]),'model_identifiability':{
        'extreme_unsupported':True,'critical_axis':axes[1],'isolated_pairs':{a:0 for a in axes},
        'unseen_levels':{a:[v for v in pool['levels'][a] if v not in seen[a]] for a in axes}}}
    return pool,state,d


def test_followup_keeps_two_categories_and_changes_only_one_other_axis():
    pool,state,d=context();p=b.candidates(pool,state,d,3);assert p['kind']=='local_followup' and len(p['candidates'])==12
    ix={k:i for i,k in enumerate(pool['ids'])};anchor=pool['points'][ix[p['anchor_candidate_id']]]
    assert p['anchor_candidate_id']==state['recombination_calls'][-1]['selected_id']
    for row in p['candidates']:
        point=pool['points'][ix[row['candidate_id']]]
        changed={a for a in pool['axes'] if anchor[a]!=point[a]}
        assert len(changed)==1 and not changed & {p['critical_axis'],p['context_axis']}
    raw={'ratings':[{'candidate_id':r['candidate_id'],'expected_yield_percent':70.,'reason':'test'} for r in p['candidates']],
        'support':[6],'summary':'neighborhood comparison'}
    plan=b.validate(raw,p,state['history']);assert plan['kind']=='local_followup' and plan['candidate_id']==p['candidates'][0]['candidate_id']
    assert 'LOCAL FOLLOW-UP' in b.FOLLOWUP_SYSTEM and 'These are JOINT tests' not in b.FOLLOWUP_SYSTEM
    assert b.SYSTEM==previous.SYSTEM


def test_no_extra_followup_without_productive_latest_result_or_at_other_round():
    pool,state,d=context();bad=deepcopy(state);bad['recombination_calls'][-1]['observed_percent']=74.99
    assert not b.candidates(pool,bad,d,3)['candidates']
    assert not b.candidates(pool,state,d,4)['candidates']
