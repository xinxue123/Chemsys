"""Run strictly separated R7-S on OpenAI gpt-6.1-terra, medium reasoning."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_r7_openai_study as study

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qualify', 'prepare', 'check',
                                           'run', 'status', 'audit', 'report'))
    parser.add_argument('--task', choices=study.TASKS, default='SnAr')
    parser.add_argument('--arm', choices=study.ARMS, default='B0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key-file', type=str,
                        help='Absolute path to the raw OpenAI API key (never copied or logged)')
    a = parser.parse_args()
    if a.action == 'run':
        result = study.run_one(a.task, a.arm, a.seed, a.key_file or '')
    else:
        result = getattr(study, a.action)()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
