"""Resume only interrupted M2 trajectories with the logged null-data adapter."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import reasoning_bo_study as study
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_compat as compat


def main():
    p=study.check_protocol();specs=[]
    for spec in data.read(study.ROOT/'manifest.json'):
        path=study.ROOT/'runs'/spec['run_id']/'state.json'
        if path.exists() and data.read(path)['status']=='INTERRUPTED':specs.append(spec)
    if not specs:
        print('No interrupted trajectories.');return
    receipt={'protocol_sha256':p['protocol_sha256'],'scientific_source_files_unchanged':True,
        'compat_sha256':data.filehash(compat.__file__),'resume_script_sha256':data.filehash(__file__),
        'API_workers':2,'supported_entrypoint_sha256':data.filehash('reasoning_bo.py'),
        'bug':'measurement=None has no innovation; frozen event check called None.get instead of defaulting missing evidence to0.',
        'fix':'Read-only state view normalizes None measurement to{} before unchanged event_reasons. No observation, bank, score, confidence, prompt or decision changes.',
        'preserved':'Every completed batch, initialization, first-valid LLM response, GP/cache and query budget. Continue at the next unfinished batch.',
        'affected_runs':[s['run_id'] for s in specs],
        'before_state_sha256':{s['run_id']:data.filehash(study.ROOT/'runs'/s['run_id']/'state.json') for s in specs},
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    target=study.OUT/'execution_compatibility.json'
    if target.exists():
        old=data.read(target);assert old['compat_sha256']==receipt['compat_sha256']
        data.write(study.OUT/('execution_compatibility_resume_'+str(time.time_ns())+'.json'),receipt)
    else:data.write(target,receipt)
    with ProcessPoolExecutor(max_workers=2,initializer=study._setup_worker) as ex:
        for f in as_completed([ex.submit(compat.run_one,s) for s in specs]):
            print(json.dumps(f.result()),flush=True)
    print(json.dumps({k:v for k,v in study.progress().items() if k!='unfinished'}),flush=True)


if __name__=='__main__':main()
