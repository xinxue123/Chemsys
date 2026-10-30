"""A separately frozen numerical repair of HEBO, preserving the native study.

Only torch's default parameter dtype changes to float64. Upstream HEBO sources,
hyperparameters, MACE, evolution, seeds, initial observations and budget remain
fixed. Explicit upstream FloatTensor inputs remain float32; actual GP prediction
dtypes are measured. This is not claimed to be an all-float64 implementation.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from concurrent.futures import ProcessPoolExecutor
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from chem.benchmarks.suzuki_policies import hebo_select
from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle, validate_point, same_point, digest
from chem.benchmarks.suzuki_runner import read_json, check_protocol, installed_sources
from chem.decision.replay import write_json_atomic

ROOT = Path('data/runs/suzuki_baselines')
DEST = ROOT / 'hebo_default64'
FREEZE = Path('research/suzuki_baselines/hebo_precision_protocol.json')


def worker(spec):
    output = DEST / spec['run_id'] / 'state.json'
    state = read_json(output) if output.exists() else {
        **spec, 'status':'CREATED','history':[], 'decisions':[], 'optimization_seconds':0., 'oracle_seconds':0.}
    if state['status'] in ('COMPLETED','FAILED'):
        return {'run_id':spec['run_id'],'status':state['status']}
    with threadpool_limits(1):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        _ = torch.linalg.cholesky(torch.eye(4))
        oracle = FrozenSuzukiOracle(ROOT / 'oracles',spec['case'])
        if not state['history']:
            y = oracle.predict(spec['initial_points'])
            state['history'] = [{'query':i+1,'point':p,'yield_percent':float(v*100),'initial':True}
                                for i,(p,v) in enumerate(zip(spec['initial_points'],y))]
            write_json_atomic(output,state)
        try:
            from hebo.models.gp.gp import GP
            while len(state['history']) < 20:
                t = len(state['history']) - 3
                seed = 20261026 + spec['case']*10000 + spec['seed']*100 + t
                stream = io.StringIO()
                dtypes = set()
                original = GP.predict
                def measured_predict(self,*args,**kwargs):
                    result = original(self,*args,**kwargs)
                    dtypes.add((str(result[0].dtype),str(result[1].dtype)))
                    return result
                started = time.perf_counter()
                with patch.object(GP,'predict',measured_predict), contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                    point,detail = hebo_select([h['point'] for h in state['history']],
                                               [h['yield_percent']/100 for h in state['history']],seed)
                state['optimization_seconds'] += time.perf_counter()-started
                text = stream.getvalue()
                detail.update(iteration=t, query=len(state['history'])+1,
                              fit_epochs_abandoned=text.count('give up fitting GP'),
                              random_prediction_fallback_calls=text.count('output random predictions'),
                              measured_gp_prediction_dtypes=[list(d) for d in sorted(dtypes)],
                              default_parameter_dtype=str(torch.get_default_dtype()))
                if text:
                    log = output.parent / f't{t:02d}.log'
                    log.write_text(text,encoding='utf8')
                if detail['fit_epochs_abandoned'] or detail['random_prediction_fallback_calls']:
                    state['numerical_failure'] = detail
                    raise ValueError('upstream numerical fallback remains after default-float64 repair')
                if dtypes != {('torch.float64','torch.float64')}:
                    raise ValueError('GP output precision did not actually change to float64')
                point = validate_point(point)
                if any(same_point(point,h['point']) for h in state['history']):
                    raise ValueError('repeat query after precision repair')
                started = time.perf_counter()
                y = float(oracle.predict([point])[0])
                state['oracle_seconds'] += time.perf_counter()-started
                detail['point'] = point
                state['decisions'].append(detail)
                state['history'].append({'query':len(state['history'])+1,'point':point,'yield_percent':y*100,'initial':False})
                state['status'] = 'RUNNING'
                write_json_atomic(output,state)
            state['status'] = 'COMPLETED'
        except Exception as exc:
            state.update(status='FAILED',error=repr(exc))
        state['precision_protocol_sha256'] = read_json(FREEZE)['precision_protocol_sha256']
        write_json_atomic(output,state)
    return {'run_id':spec['run_id'],'status':state['status']}


if __name__ == '__main__':
    parent = check_protocol(ROOT)
    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    if FREEZE.exists():
        frozen = read_json(FREEZE)
        assert frozen['script_sha256'] == source_hash and frozen['parent_protocol_sha256'] == parent['protocol_sha256']
    else:
        frozen = {'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'parent_protocol_sha256':parent['protocol_sha256'], 'script_sha256':source_hash,
                  'selection_reason':'Numerical NaN failures in 5/320 original native decisions; all five failing prefixes became finite with default float64. No optimizer result ranking used to choose repair.',
                  'change':'torch default parameter dtype float64; measured GP means/variances must both be float64; upstream explicit FloatTensor inputs remain float32',
                  'unchanged':'HEBO source,100 GP epochs,MACE,100x100 evolution,seeds,4 paired initial observations,16 further queries,Cases1-4',
                  'role':'secondary numerical repair; original 60 primary trajectories remain intact and are reported',
                  'failure_policy':'stop repaired run on any upstream abandoned training epoch or random prediction fallback; retain its planned denominator',
                  'resources':{'workers':2,'threads_each':1,'operator_compiles_in_loop':0},
                  'hebo_sources':installed_sources()}
        frozen['precision_protocol_sha256'] = digest(frozen)
        write_json_atomic(FREEZE,frozen)
    DEST.mkdir(parents=True,exist_ok=True)
    manifest = [s for s in read_json(ROOT / 'manifest.json') if s['method'] == 'HEBO']
    with ProcessPoolExecutor(max_workers=2) as pool:
        states = list(pool.map(worker,manifest))
    result = {'precision_protocol_sha256':frozen['precision_protocol_sha256'],'planned_runs':20,
              'completed':sum(s['status']=='COMPLETED' for s in states), 'failed':sum(s['status']=='FAILED' for s in states),'runs':states}
    write_json_atomic(DEST / 'status.json',result)
    print(json.dumps(result))
