"""R6 separate-arm, same-four-start qualitative-knowledge BO study."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_r6_study as study
    from chem.benchmarks import continuous_r6_engine as engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qualify', 'prepare', 'run', 'status', 'audit', 'report'))
    parser.add_argument('--task', choices=study.TASKS, default='SnAr')
    parser.add_argument('--arm', choices=engine.ARMS, default='A0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default='../config.txt')
    args = parser.parse_args()
    if args.action == 'run':
        result = study.run_one(args.task, args.arm, args.seed, args.config)
    else:
        result = getattr(study, args.action)()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
