"""Root-independent entry point; does not synchronize historical source links."""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'code/chemist_system'))
    os.chdir(root/'experiments/chemist_system')
    from chem.benchmarks import knowledge_study as study
    from chem.benchmarks.knowledge_gp import TASKS
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run', 'status', 'check'])
    ap.add_argument('--variant', default='v3')
    ap.add_argument('--seed', type=int, choices=[0, 1, 2], default=None,
                    help='prepare: registered trial seed (default0); run: verify frozen seed or infer it')
    ap.add_argument('--task', action='append', choices=TASKS)
    ap.add_argument('--mode', choices=['bo','shadow','deployment','full','verification'], default='full')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--condition', choices=['agent_initial_terra','correct','wrong_category','mixed'], default='agent_initial_terra')
    args = ap.parse_args()
    if args.seed is not None and args.action not in ('prepare', 'run'):
        ap.error('--seed applies only to prepare or run')
    if args.action == 'run':
        tasks = args.task or (TASKS if args.condition == 'agent_initial_terra' else TASKS[:4])
        study.batch(root, args.variant, tasks, args.mode, args.workers, args.condition, args.seed)
    else:
        value = (study.prepare(root, args.variant, seed=0 if args.seed is None else args.seed)
                 if args.action == 'prepare' else getattr(study, args.action)(root, args.variant))
        if args.action == 'prepare':
            value = {k:v for k,v in value.items() if k not in ('source_sha256','inputs')}
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
