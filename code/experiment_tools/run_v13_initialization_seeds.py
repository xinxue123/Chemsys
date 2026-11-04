"""Frozen V13 multi-seed launcher and fresh-agent queue metadata."""
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from v13_initialization_seeds import study
from state_split_v13_experiment import transport


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'check', 'preflight', 'batch', 'next', 'finish', 'status'))
    p.add_argument('names', nargs='*', help='Exact returned canonical task_name from collaboration.spawn_agent')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if a.action in ('prepare', 'check'):
        result = study.prepare(ROOT) if a.action == 'prepare' else study.check(ROOT, True)
        result = {k: result[k] for k in ('variant', 'protocol_sha256', 'jobs')}
    elif a.action == 'preflight':
        result = study.preflight(ROOT, a.workers)
        result = dict(status=result['status'], pairs=len(result['rows']), BO_queries_audited=sum(r['baseline_queries_audited'] for r in result['rows']), new_queries=0)
    elif a.action == 'batch':
        result = study.batch(ROOT, a.workers)
    else:
        if a.action == 'finish':
            rows = {'/root/' + r['invocation']['task_name']: r for r in study.pending(ROOT)}
            for name in a.names:
                row = rows[name]
                with patch.object(transport, 'base', return_value=study.base(ROOT, row['task'], row['seed'])):
                    transport.record_spawn(ROOT, row['request'], {'task_name': name})
                    transport.seal(ROOT, row['request'])
        result = study.status(ROOT)
        if a.action != 'status':
            result.pop('rows')
            result['pending'] = [dict(task=r['invocation']['task_name'],
                input=(ROOT / transport.read(Path(r['request']).parent / 'export.json')['input']).as_posix(),
                assigned=r['assigned'], output_exists=r['output_exists']) for r in study.pending(ROOT)]
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
