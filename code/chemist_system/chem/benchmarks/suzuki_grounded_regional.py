"""python -m chem.benchmarks.suzuki_grounded_regional ACTION"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'qualify', 'prepare', 'run', 'status', 'report', 'verify'))
    parser.add_argument('--config', default='../config.txt')
    args = parser.parse_args()
    from . import grounded_regional_suzuki_study as study
    if args.action in ('preflight', 'prepare', 'run'):
        result = getattr(study, args.action)(args.config)
    elif args.action in ('qualify', 'status'):
        result = getattr(study, args.action)()
    else:
        from . import grounded_regional_suzuki_report as report
        result = getattr(report, args.action)()
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
