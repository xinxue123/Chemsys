"""python -m chem.benchmarks.suzuki_uncertainty prepare|develop|freeze|test|report|verify|audit."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','develop','freeze','test','report','verify','audit'])
    a=p.parse_args()
    from . import uncertain_suzuki_study as s
    if a.action=='prepare':v=s.prepare()
    elif a.action in ('develop','test'):v=s.run_stage('development' if a.action=='develop' else 'test')
    elif a.action=='freeze':v=s.select_parameters()
    else:
        from . import uncertain_suzuki_report as r
        v={'report':r.report,'verify':r.verify,'audit':r.audit_llinbo}[a.action]()
    print(json.dumps(v,ensure_ascii=False,separators=(',',':')))

if __name__=='__main__':main()
