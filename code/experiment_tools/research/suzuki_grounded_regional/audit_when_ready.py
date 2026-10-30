"""Overlap independent CPU audits with the two API workers, once cases finish."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem.benchmarks import grounded_regional_suzuki_report as report
from chem.benchmarks import grounded_regional_suzuki_study as study


def main():
    study.check_protocol()
    manifest = study.read(study.ROOT / 'manifest.json')
    futures = {}
    done = {}
    errors = {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, initializer=study.worker_init) as executor:
        while len(done) + len(errors) < 4:
            if time.monotonic() - started > 7200:
                raise RuntimeError('audit dispatcher exceeded two hours')
            for case in (1, 2, 3, 4):
                if case in futures or case in done or case in errors:
                    continue
                paths = [study.ROOT / 'runs' / v['run_id'] / 'state.json' for v in manifest if v['case'] == case]
                if all(path.exists() and study.read(path)['status'] == 'COMPLETE' for path in paths):
                    futures[case] = executor.submit(report.audit_checkpoint, case)
            for case, future in list(futures.items()):
                if not future.done():
                    continue
                try:
                    done[case] = future.result()
                    print(json.dumps({'audit_case': case, 'status': 'PASS', 'choices': done[case]['choices']}), flush=True)
                except Exception as error:
                    errors[case] = {'kind': type(error).__name__, 'message': str(error)[:250]}
                    print(json.dumps({'audit_case': case, 'status': 'FAILED', 'error': errors[case]}), flush=True)
                del futures[case]
            study.write(study.OUT / 'audit_dispatch_status.json',
                {'completed': sorted(done), 'running': sorted(futures), 'errors': errors})
            if len(done) + len(errors) < 4:
                time.sleep(10)
    if errors:
        raise RuntimeError('independent case audits failed; inspect audit_dispatch_status.json')


if __name__ == '__main__':
    main()
