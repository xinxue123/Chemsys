"""Check relocated scientific code against the original first rollout decision."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
sys.path[:0]=[str(PORTABLE/'code/chemist_system'),str(PORTABLE/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from fast_hit_bo.model import SharedGP
from qlogei_rollout.policy import decision


def main():
    paths=[Path('code/experiment_tools/fast_hit_bo')/name for name in ('__init__.py','model.py','policy.py')]
    paths+=[Path('code/experiment_tools/qlogei_rollout')/name for name in
            ('__init__.py','acquisition.py','policy.py','validation.py','study.py')]
    for path in paths:
        assert (ROOT/path).read_bytes()==(PORTABLE/path).read_bytes(),str(path)
    pool=knowledge_gp.load_task(PORTABLE,'Reizman_Suzuki_Case_1',seed=0)
    sf=ROOT/'experiments/runs/fast_hit_qlogei_rollout_20261104/runs/s4_middle_rollout_qlogei/state.json'
    recorded=json.loads(sf.read_text(encoding='utf-8'))
    first=knowledge_gp.read_observation(PORTABLE,pool,recorded['history'][0]['candidate_id'])
    assert first==recorded['history'][0]
    gp=SharedGP(pool,PORTABLE/'experiments/runs/fast_hit_bo_20261104/cache')
    context=gp.context([first],2)
    expected=recorded['decisions'][1]['decision']
    actual=decision(gp,context,2)
    for key in ('selected_id','BO_reference_id','outer_ids','future_ids','fantasy_seed','future_seed'):
        assert actual[key]==expected[key],key
    errors={key:float(np.max(np.abs(np.array(actual[key])-np.array(expected[key]))))
            for key in ('first_logei','future_logei','outer_values')}
    assert max(errors.values())<1e-10
    assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
    result=dict(status='PASS',scientific_source_files_identical=len(paths),
        initial_observation_matches=True,selected_id=actual['selected_id'],
        outer_candidates=len(actual['outer_ids']),all_future_choices_match=True,
        max_absolute_errors=errors,operator_compiles_in_loop=0,new_experiments=0)
    result_path=PORTABLE/'results/fast_hit_qlogei_rollout_20261104/portable_verification.json'
    result_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
