"""Post hoc condition differences: LLM first proposal versus every pool optimum.

These oracle-derived tables are analysis artifacts, never decision-agent inputs.
No new LLM call, GP fit, optimization trajectory or experimental query is made.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp as native
from grounded_first_start.proposal import background

BASE=ROOT/'experiments/runs/grounded_first_start_20261105'
REPORT=ROOT/'experiments/reports/grounded_first_start_20261105'
SHORT=dict(zip(native.TASKS,('c1','c2','c3','c4','perera','doyle','snar','hplc')))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LABELS={
    'BrC1=CC=C(N=CC=C2)C2=C1':'heteroaryl bromide',
    'IC1=CC=C(N=CC=C2)C2=C1':'same-scaffold heteroaryl iodide',
    'CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O':'organoboronic acid',
    'CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4':'same-scaffold pinacol boronate ester (Bpin)',
    'P(C1CCCCC1)(C2CCCCC2)C3CCCCC3':'tricyclohexylphosphine (PCy3)',
    'CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1':'di-tert-butyl(4-dimethylaminophenyl)phosphine',
    'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C':'tri(o-tolyl)phosphine',
    'COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2':'dicyclohexyl dimethoxybiaryl phosphine',
    '[K+].[OH-]':'potassium hydroxide (KOH)',
    'CC([O-])C.[Li+]':'lithium isopropoxide (LiOiPr)',
    'CO':'methanol',
    'CN1CCCN2C1=NCCC2':'MTBD',
    'CC1=NOC=C1':'methylisoxazole',
    'C1(C2=CC=CC=C2)=CON=C1':'phenylisoxazole',
    'IC1=CC=C(C(F)(F)F)C=C1':'para-trifluoromethyl-substituted iodobenzene',
    'IC1=NC=CC=C1':'2-iodopyridine',
    'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC':'di-tert-butyl dimethoxybiaryl phosphine',
    'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2':'di-tert-butyl biaryl phosphine without methoxy groups',
}


def label(card,axis,value):
    variable=next(v for v in card['inputs'] if v['name']==axis)
    if variable['kind']=='continuous':return str(value)
    if axis=='catalyst':
        identity=variable['chemical_identities'][value]
        return value+' / '+identity['ligand']['name']
    return LABELS.get(value,variable['chemical_identities'].get(value,{}).get('name',value))


def main():
    protocol=read(BASE/'protocol.json');details=[];summaries=[];plans=[];inputs={}
    for task in native.TASKS:
        state=read(BASE/'runs'/SHORT[task]/'state.json')
        assert state['status']=='COMPLETE' and len(state['history'])==20
        first=state['decisions'][0]['decision'];pool=native.load_task(ROOT,task,seed=0);card=background(pool)
        assert first['conditions']==pool.points[pool.ix[first['selected_id']]]
        oracle=pool.source_path.with_name('oracle.npz')
        for path in (pool.source_path,oracle):
            key=path.relative_to(ROOT).as_posix();inputs[key]=sha(path)
            assert protocol['inputs_sha256'][key]==inputs[key]
        with np.load(oracle,allow_pickle=False) as data:
            values=data['raw_values'].copy() if 'raw_values' in data else data['values'].copy()*pool.scale
        assert values.shape==(len(pool.ids),) and np.isfinite(values).all()
        llm_value=float(values[pool.ix[first['selected_id']]])
        assert abs(llm_value-state['history'][0]['raw_value'])<1e-10
        maximum=float(values.max());indices=np.flatnonzero(values==maximum)
        assert maximum>=max(v['raw_value'] for v in state['history'])
        units={v['name']:v.get('unit') for v in card['inputs']}
        for variant,j in enumerate(indices,1):
            optimum=pool.points[int(j)];same=[];changed=[]
            for axis,kind in zip(pool.axes,pool.kinds):
                before=first['conditions'][axis];after=optimum[axis]
                equal=before==after if kind=='categorical' else math.isclose(before,after,rel_tol=0.,abs_tol=1e-10)
                (same if equal else changed).append(axis)
                details.append(dict(task=SHORT[task],optimum_variant=variant,optimum_count=len(indices),
                    LLM_candidate_id=first['selected_id'],optimum_candidate_id=pool.ids[int(j)],
                    axis=axis,kind=kind,input_unit=units[axis],LLM_value=before,optimum_value=after,
                    LLM_label=label(card,axis,before),optimum_label=label(card,axis,after),
                    identical=bool(equal),optimum_minus_LLM=float(after-before) if kind=='continuous' else None))
            summaries.append(dict(task=SHORT[task],optimum_variant=variant,optimum_count=len(indices),
                LLM_candidate_id=first['selected_id'],optimum_candidate_id=pool.ids[int(j)],
                LLM_first_feedback=llm_value,pool_global_best=maximum,objective_unit=pool.unit,
                objective_gap=maximum-llm_value,same_axes=';'.join(same),different_axes=';'.join(changed),
                identical_input_count=len(same),total_input_count=len(pool.axes)))
            plans.append(dict(task=SHORT[task],optimum_variant=variant,LLM_candidate_id=first['selected_id'],
                optimum_candidate_id=pool.ids[int(j)],LLM_conditions=first['conditions'],optimum_conditions=optimum,
                LLM_first_feedback=llm_value,pool_global_best=maximum,objective_unit=pool.unit,
                same_axes=same,different_axes=changed))
    pd.DataFrame(details).to_csv(REPORT/'first_vs_optimum_by_axis.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(REPORT/'first_vs_optimum_summary.csv',index=False,encoding='utf-8-sig')
    record=dict(scope='Post hoc comparison with exact maximum of the current finite input pool, not an unrestricted continuous-space optimum.',
        generated_UTC=datetime.now(timezone.utc).isoformat(),all_tied_maxima_retained=True,
        experimental_queries_added=0,LLM_calls_added=0,GP_fits_added=0,
        source_sha256=sha(__file__),inputs_sha256=inputs,plans=plans)
    (REPORT/'first_vs_optimum_conditions.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(summaries)[['task','optimum_variant','optimum_count','LLM_first_feedback','pool_global_best','same_axes','different_axes']].to_string(index=False))
    print(json.dumps(dict(tasks=8,optimum_plans=len(plans),axis_comparisons=len(details),new_queries=0)))


if __name__=='__main__':main()
