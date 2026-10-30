from pathlib import Path
import numpy as np
from chem.benchmarks import stable_diverse_bridge as b,stable_productive_bridge as previous
from chem.benchmarks import reasoning_bo_data as data


def test_first_pool_unchanged_but_second_does_not_globalize_confounded_counts():
    pool=data.load_pool('Perera_Suzuki');root=Path('data/runs/stable_r2_20261029/productive_t0/runs')
    saved=data.read(next(root.glob('Perera*/state.json')))
    ix={k:i for i,k in enumerate(pool['ids'])};axes=pool['axes'];critical=axes[1]
    def state_and_decision(n):
        state={**saved,'history':saved['history'][:n+4],'decisions':saved['decisions'][:n],
            'recombination_calls':saved['recombination_calls'][:n],'knowledge':saved['initial_knowledge']}
        seen={a:{pool['points'][ix[h['candidate_id']]][a] for h in state['history']} for a in axes}
        d={'available':~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]),'model_identifiability':{
            'extreme_unsupported':True,'critical_axis':critical,'isolated_pairs':{a:0 for a in axes},
            'unseen_levels':{a:[v for v in pool['levels'][a] if v not in seen[a]] for a in axes}}}
        return state,d
    state,d=state_and_decision(0)
    assert b.SYSTEM==previous.SYSTEM
    assert b.candidates(pool,state,d,1)==previous.candidates(pool,state,d,1)
    state,d=state_and_decision(1);new=b.candidates(pool,state,d,2);old=previous.candidates(pool,state,d,2)
    tested=pool['points'][ix[state['recombination_calls'][0]['selected_id']]];value=tested[critical]
    assert not any(pool['points'][ix[r['candidate_id']]][critical]==value for r in old['candidates'])
    assert any(pool['points'][ix[r['candidate_id']]][critical]==value for r in new['candidates'])
    context=new['context_axis'];levels=pool['levels'][context];matrix=b.catalog(pool['task_id'])['axes'][context]['Tanimoto']
    for row in new['candidates']:
        point=pool['points'][ix[row['candidate_id']]]
        assert matrix[levels.index(point[context])][levels.index(tested[context])]<.5
    assert len(new['candidates'])==10
