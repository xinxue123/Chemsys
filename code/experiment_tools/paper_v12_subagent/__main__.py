"""Use this package directory directly or run python -m paper_v12_subagent."""
from pathlib import Path
import argparse
import json
import os
import sys


def main(argv=None):
    root_default = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=root_default)
    ap.add_argument('--job', nargs=3, metavar=('TASK', 'SEED', 'ARM'))
    ap.add_argument('--wait-seconds', type=float, default=-1)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--pending', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--copy-cache', action='store_true')
    group.add_argument('--export-request', action='store_true')
    group.add_argument('--assign-agent')
    group.add_argument('--submit', type=Path)
    ap.add_argument('--request', type=Path)
    ap.add_argument('--spawn-evidence', type=Path)
    ap.add_argument('--agent-id')
    args = ap.parse_args(argv)
    root = args.root.resolve()
    sys.path[:0] = [str(root / 'code/chemist_system'), str(root / 'code/experiment_tools')]
    # Set before importing Torch/NumPy; CPU numerics remain the original recipe.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from paper_v12_subagent import study, transport
    job = (args.job[0], int(args.job[1]), args.job[2]) if args.job else None
    if args.request and not args.request.resolve().is_relative_to(study.directory(root)):
        ap.error('Request must be inside this registered subagent namespace')
    if args.export_request:
        if not args.request:
            ap.error('--export-request requires --request')
        out = transport.export_request(root, args.request)
    elif args.assign_agent:
        if not args.request or not args.spawn_evidence:
            ap.error('--assign-agent requires --request and --spawn-evidence')
        out = transport.assign(args.request, args.assign_agent, transport.read(args.spawn_evidence))
    elif args.submit:
        if not args.request or not args.agent_id:
            ap.error('--submit requires --request and --agent-id')
        out = transport.submit(args.request, args.submit, args.agent_id)
    elif args.pending:
        out = study.pending(root, job)
    elif args.status:
        out = study.status(root)
    elif args.check:
        p = study.check(root, True)
        out = dict(status='PASS', protocol_sha256=p['protocol_sha256'], jobs=len(p['jobs']))
    elif args.copy_cache:
        study.check(root)
        out = dict(GP_cache_files_copied=study.copy_cache(root))
    elif job:
        os.chdir(root / 'experiments/chemist_system')
        try:
            out = study.run_one(root, *job, wait_seconds=args.wait_seconds)
        except transport.PendingRequest as exc:
            print(json.dumps(dict(status='PENDING', request=str(exc.path))))
            return 75
    else:
        ap.error('Choose --job, --pending, --status, --check, --copy-cache, --export-request, --assign-agent or --submit')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
