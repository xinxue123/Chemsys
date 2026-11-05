"""Relocated first proposal, normal qLogEI and prospective KF feedback proof."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
sys.path[:0]=[str(PORTABLE/'code/chemist_system'),str(PORTABLE/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from fast_hit_bo.model import SharedGP
from llm_event_start import policy
from llm_screen_qlogei import knowledge


def main():
    paths=[Path('code/experiment_tools/llm_event_start')/name for name in ('__init__.py','policy.py','prompts.py','validation.py')]
    paths += [Path('code/experiment_tools/llm_screen_qlogei')/name for name in ('__init__.py','knowledge.py','policy.py','validation.py')]
    paths += [Path('code/experiment_tools/fast_hit_bo')/name for name in ('__init__.py','model.py','policy.py')]
    paths += [Path('code/experiment_tools/reasoning_memory_v13/calibration.py')]
    for path in paths:
        assert (ROOT/path).read_bytes()==(PORTABLE/path).read_bytes(),str(path)
    directory=PORTABLE/'results/llm_event_start_qlogei_20261104'
    state=json.loads((directory/'event_reference.json').read_text(encoding='utf-8'))
    pool=knowledge_gp.load_task(PORTABLE,'Reizman_Suzuki_Case_1',seed=0)
    c=state['decisions'][0]
    plan=policy.validate(c['LLM']['raw'],pool,[],1)
    first=policy.decide(pool,None,plan['notes'],plan['proposal'],1,0,'event')
    assert first==c['decision']
    gp=SharedGP(pool,PORTABLE/'experiments/runs/llm_event_start_qlogei_20261104/cache')
    counts=[]
    for q in (2,3):
        c=state['decisions'][q-1]
        context=gp.context(state['history'][:q-1],q)
        assert context['key']==c['fit_key']
        packet=policy.decide(pool,context,c['notes_at_commit'],None,q,0,'event')
        for key,value in packet.items():
            assert value==c['decision'][key],key
        after,events=knowledge.feedback(c['notes_at_commit'],packet['forecasts'],state['history'][q-1],q)
        assert after==state['decisions'][q]['notes_at_commit']
        counts.append(len(events))
    assert counts[-1]>0 and gp.counters['fits']==0
    result=dict(status='PASS',scientific_files_identical=len(paths),first_proposal_exact=True,
        normal_qLogEI_choices_exact=True,preoutcome_forecasts_exact=True,
        actual_KF_feedback_recomputed=True,KF_updates_checked=counts,new_GP_fits=0,new_experiments=0)
    (directory/'portable_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
