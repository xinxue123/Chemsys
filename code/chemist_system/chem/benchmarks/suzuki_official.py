"""Official-source Suzuki study: python -m chem.benchmarks.suzuki_official."""
import argparse
import json
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'


def main():
    p=argparse.ArgumentParser(description='Unchanged upstream LLINBO/LGBO/HEBO, Suzuki Cases1-4 only')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare')
    s=sub.add_parser('serve');s.add_argument('--workers',type=int,default=4)
    sub.add_parser('status')
    q=sub.add_parser('pending');q.add_argument('--compact',action='store_true')
    a=sub.add_parser('submit');a.add_argument('--answers',type=Path,required=True)
    r=sub.add_parser('report');r.add_argument('--replay-official',action='store_true')
    sub.add_parser('verify')
    args=p.parse_args()
    if args.action in ('report','verify'):
        from . import official_suzuki_report as report
        value=report.report(args.replay_official) if args.action=='report' else report.verify()
    else:
        from . import official_suzuki_runner as runner
        if args.action=='prepare':
            protocol=runner.prepare();value={'protocol_sha256':protocol['protocol_sha256'],'planned_runs':60}
        elif args.action=='serve':value=runner.serve(args.workers)
        elif args.action=='status':value=runner.status()
        elif args.action=='pending':value=runner.pending(args.compact)
        else:value=runner.submit(args.answers)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')))


if __name__=='__main__':main()
