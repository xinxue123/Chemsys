"""Wait for the sealed study, then build audited artifacts for visual review.

Never edits scientific policy, writes a decision response or launches an
additional experiment. Publication packaging remains a separate final step.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'experiments/runs/v13_late_validation_20261103'
LOG=ROOT/'experiments/logs/paper_v13_late'
PY=ROOT/'runtime/environments/.venv-official/Scripts/python.exe'
DOC=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(x):
    temp=LOG/'pipeline_status.tmp';temp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');temp.replace(LOG/'pipeline_status.json')


def main(from_step='native_replay'):
    LOG.mkdir(parents=True,exist_ok=True);p=read(B/'protocol.json');last=-1
    while True:
        complete=queries=0
        for j in p['jobs']:
            sf=B/j['id']/'runs'/j['id']/'state.json'
            if sf.exists():
                s=read(sf);complete+=s['status']=='COMPLETE';queries+=max(0,len(s['history'])-j['prefix_queries'])
        write(dict(status='WAITING_FOR_REGISTERED_STUDY',complete=complete,total=75,new_queries=queries,total_new_queries=840))
        if queries//30!=last:
            last=queries//30;print(json.dumps(dict(complete=complete,new_queries=queries)),flush=True)
        if complete==75 and (B/'batch_receipt.json').exists() and (B/'RBO_compat_batch_receipt.json').exists():break
        time.sleep(10)
    commands=[('native_replay',PY,['code/analysis/audit_v13_late_validation.py']),
        ('assets',PY,['code/publication_v13_late/assets.py','--fresh']),
        ('forecast_arithmetic',PY,['code/publication_v13_late/traces.py','--fresh']),
        ('candidate_coverage',PY,['code/publication_v13_late/coverage.py','--fresh']),
        ('model_and_request_binding',PY,['code/publication_v13_late/bind.py']),
        ('manuscript_prose',PY,['code/publication_v13_late/render.py']),
        ('portable_arithmetic',DOC,['paper/v13_late/verify_results.py']),
        ('typeset',PY,['code/publication_v13_late/typeset.py']),
        ('render_every_page',DOC,['code/publication_v13_late/pdf_qa.py'])]
    names=[x[0] for x in commands]
    if from_step not in names:raise ValueError('Unknown build stage: '+from_step)
    for name,exe,args in commands[names.index(from_step):]:
        write(dict(status='BUILDING',step=name,complete=75,new_queries=840))
        with (LOG/(name+'.log')).open('w',encoding='utf-8') as out:
            result=subprocess.run([str(exe),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            write(dict(status='BUILD_FAILURE',step=name,returncode=result.returncode,log=str(LOG/(name+'.log'))))
            raise RuntimeError('Publication stage failed: '+name)
        print(name+' PASS',flush=True)
    write(dict(status='READY_FOR_VISUAL_REVIEW',complete=75,new_queries=840,
        publication_packaged=False,original_release_unchanged=True,completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-step',default='native_replay')
    main(parser.parse_args().from_step)
