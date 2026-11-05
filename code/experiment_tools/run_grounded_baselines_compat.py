"""Resume frozen prefixes with exact native posterior acquisition transport."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from grounded_baselines import study as s,compat
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import traceback

def run_job(job):
    try:return s.run_one(*job)
    except Exception as exc:
        s.write(s.path_for(*job)/'execution_error.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),replace=True)
        print(json.dumps(dict(job=job,error=type(exc).__name__,message=str(exc))),flush=True)
        raise

if __name__=='__main__':
    if '--qualify' in sys.argv:print(json.dumps(compat.qualified_receipt()))
    else:
        p=s.check(True);receipt=s.read(s.BASE/'execution_compatibility.json')
        for path,h in receipt['new_source_sha256'].items():assert s.sha(s.ROOT/path)==h,path
        with ProcessPoolExecutor(max_workers=8,initializer=compat.initialize) as executor:
            for future in as_completed([executor.submit(run_job,j) for j in s.JOBS]):
                print(json.dumps(future.result()),flush=True)
