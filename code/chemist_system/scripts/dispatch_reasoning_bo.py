"""Operational parallelism only; imports the frozen scientific executor.

Run from chemist_system. An already running bootstrap uses2 API slots; this
dispatcher uses6 until all20 initializations exist, then8. No method, seed,
query, prompt, model fit or outcome-dependent scheduling rule changes.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import reasoning_bo_study as study
from chem.benchmarks import reasoning_bo_data as data


def main():
    p=study.check_protocol()
    amendments=study.OUT/'operational_concurrency.json'
    record={'protocol_sha256':p['protocol_sha256'],'scientific_protocol_changed':False,
        'dispatcher_sha256':data.filehash(__file__),
        'previous_API_workers':2,'new_total_API_workers':8,'GP_workers':6,
        'API_slots_while_bootstrap_running':{'bootstrap':2,'dispatcher':6},
        'CPU_threads_per_worker':1,'operator_compiles_in_loop':0,
        'reason':'Observed ready task queue, local20 logical CPUs and31.8GB RAM (10.9GB available before expansion); initial requests had no429. Expand independent trajectories, never batch or scientific settings.',
        'provider_capacity_reference':'https://api-docs.deepseek.com/quick_start/pricing/',
        'provider_published_Flash_concurrency':2500,
        'failure_policy':'Built-in429 retry/backoff remains. Do not resample responses or repeat completed batches. Stop new submissions on observed402; already in-flight calls finish.',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    if amendments.exists():
        saved=data.read(amendments)
        assert saved['dispatcher_sha256']==record['dispatcher_sha256'] and saved['protocol_sha256']==record['protocol_sha256']
    else:data.write(amendments,record)
    manifest=data.read(study.ROOT/'manifest.json');attempted=set();futures={};resource_pause=False
    with ProcessPoolExecutor(max_workers=8,initializer=study._setup_worker) as api_ex, \
            ProcessPoolExecutor(max_workers=6,initializer=study._setup_worker) as cpu_ex:
        while True:
            ready={(t,s) for t in data.SPECS for s in study.SEEDS if
                (study.ROOT/'initialization'/f'{t}_s{s:02d}'/'ready.json').exists()}
            api_limit=8 if len(ready)==20 else 6
            counts={kind:sum(kind==value['kind'] for value in futures.values()) for kind in ('api','cpu')}
            for spec in manifest:
                key=spec['run_id']
                if key in attempted or (spec['task'],spec['seed']) not in ready:continue
                path=study.ROOT/'runs'/key/'state.json'
                if path.exists() and data.read(path)['status']=='COMPLETE':attempted.add(key);continue
                kind='api' if spec['method'] in ('ReasoningBO','New_R2') else 'cpu'
                if counts[kind]>=(api_limit if kind=='api' else 6):continue
                if kind=='api' and resource_pause:continue
                ex=api_ex if kind=='api' else cpu_ex
                futures[ex.submit(study.run_one,spec)]={'run_id':key,'kind':kind}
                counts[kind]+=1;attempted.add(key)
            if futures:
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:
                    spec=futures.pop(f)
                    try:print(json.dumps(f.result()),flush=True)
                    except Exception as error:print(json.dumps({'run_id':spec['run_id'],'worker_error':type(error).__name__}),flush=True)
                if done:
                    errors=list((study.ROOT/'requests').glob('*/attempt*/transport_error.json'))
                    if any(data.read(x).get('error',{}).get('status')==402 for x in errors):
                        resource_pause=True
                        data.write(study.ROOT/'resource_pause.json',{'kind':'HTTP402','new_API_submissions_stopped':True})
            else:
                if len(ready)==20 or resource_pause:break
                time.sleep(5)
            if len(attempted)==len(manifest) and not futures:break
    print(json.dumps({k:v for k,v in study.progress().items() if k!='unfinished'}),flush=True)


if __name__=='__main__':main()
