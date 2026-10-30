"""Reproduce all native HEBO suggestions; capture upstream numerical fallbacks.

No simulator queries and no changed optimizer settings. Original outputs remain
untouched; this is an execution audit, not another optimization study.
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
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits
from chem.benchmarks.suzuki_policies import hebo_select
from chem.benchmarks.suzuki_domain import same_point
from chem.decision.replay import write_json_atomic

ROOT = Path('data/runs/suzuki_baselines')


def audit(path):
    state = json.loads(Path(path).read_text(encoding='utf8'))
    results = []
    with threadpool_limits(1):
        for d in state['decisions']:
            prefix = state['history'][:d['query'] - 1]
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                point, detail = hebo_select([r['point'] for r in prefix],
                                            [r['yield_percent']/100 for r in prefix],d['seed'])
            text = output.getvalue()
            results.append({'iteration':d['iteration'], 'same_proposal':same_point(point,d['point'],1e-10),
                            'fit_epochs_abandoned':text.count('give up fitting GP'),
                            'random_prediction_fallback_calls':text.count('output random predictions'),
                            'log_sha256':sha256(text.encode('utf8')).hexdigest(),
                            'replay_details':detail})
            if text:
                log = ROOT / 'hebo_audit_logs' / state['run_id'] / f"t{d['iteration']:02d}.txt"
                log.parent.mkdir(parents=True,exist_ok=True)
                log.write_text(text,encoding='utf8')
    result = {'run_id':state['run_id'],'iterations':results}
    write_json_atomic(ROOT / 'hebo_replays' / (state['run_id'] + '.json'),result)
    return result


if __name__ == '__main__':
    started = time.perf_counter()
    paths = sorted((ROOT/'runs').glob('*_HEBO/state.json'))
    with ProcessPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(audit, map(str,paths)))
    decisions = [d for r in runs for d in r['iterations']]
    report = {'scope':'all native HEBO decisions, no oracle queries, no algorithm changes',
              'decisions':len(decisions),'same_proposals':sum(d['same_proposal'] for d in decisions),
              'steps_with_abandoned_fit_epochs':sum(d['fit_epochs_abandoned'] > 0 for d in decisions),
              'abandoned_fit_epochs':sum(d['fit_epochs_abandoned'] for d in decisions),
              'steps_with_random_prediction_fallback':sum(d['random_prediction_fallback_calls'] > 0 for d in decisions),
              'random_prediction_fallback_calls':sum(d['random_prediction_fallback_calls'] for d in decisions),
              'seconds':time.perf_counter()-started,'runs':runs}
    write_json_atomic(ROOT / 'hebo_numerical_audit.json',report)
    write_json_atomic('research/suzuki_baselines/hebo_numerical_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}))
