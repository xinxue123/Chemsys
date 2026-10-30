"""R7-D0 reachability diagnostics and R7-S passive-covariance mechanism study."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_r7_study as study
    from chem.benchmarks import continuous_r7_engine as engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qualify', 'prepare', 'check', 'diagnose',
                                           'run', 'status', 'audit', 'report'))
    parser.add_argument('--task', choices=study.TASKS, default='SnAr')
    parser.add_argument('--arm', choices=engine.ARMS, default='B0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default='../config.txt')
    parser.add_argument('--r6-root', type=str,
                        help='Original R6 data/runs/continuous_r6_20261030 directory for diagnose')
    parser.add_argument('--output', type=str,
                        help='Optional R7 diagnostic result JSON path')
    args = parser.parse_args()
    if args.action == 'run':
        result = study.run_one(args.task, args.arm, args.seed, args.config)
    elif args.action == 'diagnose':
        from chem.benchmarks import continuous_r7_diagnostics as diagnostics
        result = diagnostics.run(seed=args.seed, r6_root=args.r6_root, output=args.output)
    else:
        result = getattr(study, args.action)()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
