"""Stratified four-prefix resume spot check of the secondary HEBO repair."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import contextlib
import io
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_limits
from chem.benchmarks.suzuki_domain import same_point
from chem.benchmarks.suzuki_policies import hebo_select
from chem.decision.replay import write_json_atomic


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    rows=[]
    with threadpool_limits(1):
        for case in range(1,5):
            path=Path(f'data/runs/suzuki_baselines/hebo_default64/c{case}_s0_HEBO/state.json')
            state=json.loads(path.read_text(encoding='utf8'))
            h=state['history'][:19]
            d=state['decisions'][-1]
            output=io.StringIO()
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
                p,_=hebo_select([r['point'] for r in h],[r['yield_percent']/100 for r in h],d['seed'])
            equal=same_point(p,d['point'],1e-10)
            assert equal
            rows.append({'case':case,'seed':0,'iteration':16,'same_proposal':equal,
                         'abandoned_training_epochs':output.getvalue().count('give up fitting GP'),
                         'random_prediction_fallbacks':output.getvalue().count('output random predictions')})
    receipt={'status':'PASS','scope':'4 stratified final-prefix resume spot checks; not a full 320-decision repaired replay',
             'oracle_queries':0,'rows':rows}
    write_json_atomic('research/suzuki_baselines/hebo_precision_resume_check.json',receipt)
    print(json.dumps(receipt))
