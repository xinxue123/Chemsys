"""Runnable local experiment and metadata CLI; no credentials used."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from state_split_v13_experiment import study, transport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'batch', 'status', 'record', 'seal'))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--receipts', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        p = study.prepare(ROOT)
        result = dict(protocol_sha256=p['protocol_sha256'], tasks=p['tasks'])
    elif args.action == 'batch':
        result = study.batch(ROOT, args.workers)
    elif args.action == 'status':
        result = study.status(ROOT)
    elif args.action == 'record':
        result = [transport.record_spawn(ROOT, row['request'], row['spawn_result']) for row in transport.read(args.receipts)]
    else:
        result = [transport.seal(ROOT, row['request']) for row in transport.pending(ROOT) if row['assigned'] and row['output_exists']]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
