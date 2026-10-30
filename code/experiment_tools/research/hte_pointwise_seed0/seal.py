"""Seal this new single-seed evidence without copying credentials or old caches."""
from hashlib import sha256
from pathlib import Path
from collections import Counter
import json
import sys
import time

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
ROOT=BASE/'data/runs/hte_pointwise_seed0'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def hashed(p):return sha256(p.read_bytes()).hexdigest()
def entry(p,base=BASE):return {'path':p.relative_to(base).as_posix(),'bytes':p.stat().st_size,'sha256':hashed(p)}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    protocol=read(OUT/'protocol.json');validation=read(OUT/'validation_receipt.json');results=read(OUT/'results.json')
    assert validation['status']=='PASS_COMPLETE' and validation['choices_audited']==256
    assert results['status']=='COMPLETE' and results['completed']==16
    assert read(OUT/'preservation_receipt.json')['status']=='PASS'
    assert read(OUT/'trace_audit.json')['status']=='PASS'
    for path,h in {**protocol['source_sha256'],**protocol['inputs'],**protocol['protected']}.items():assert hashed(BASE/path)==h
    paths=[BASE/p for p in protocol['source_sha256']]
    for folder in (ROOT,OUT):
        paths += [p for p in folder.rglob('*') if p.is_file() and 'cache' not in p.relative_to(folder).parts
            and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp')
            and p.name not in ('release_inventory.json','release_receipt.json')]
    sys.path.insert(0,str(BASE))
    from chem.benchmarks.deepseek_suzuki_client import load_config
    secret=load_config(BASE.parent/'config.txt').api_key.encode('utf8')
    assert secret and all(secret not in p.read_bytes() for p in set(paths)),'credential found in release'
    cache=ROOT/'cache';rows=[entry(p,cache) for p in sorted(cache.rglob('*'))
        if p.is_file() and p.suffix not in ('.lock','.tmp')]
    tree={'files':len(rows),'bytes':sum(x['bytes'] for x in rows),
        'sha256':sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()}
    save(OUT/'release_inventory.json',{'files':[entry(p) for p in sorted(set(paths))],'cache_tree':tree})
    responses=[read(p) for p in (ROOT/'requests').glob('*/attempt*/response.json')]
    receipt={'status':'PASS_COMPLETE','trajectories':16,'queries_per_trajectory':{'initial':4,'added':16},
        'logical_observations':320,'unique_committed_decisions':256,'seed':0,'population_significance_claim':False,
        'protocol_sha256':protocol['protocol_sha256'],'inventory_sha256':hashed(OUT/'release_inventory.json'),
        'results_sha256':hashed(OUT/'results.json'),'validation_sha256':hashed(OUT/'validation_receipt.json'),
        'cache_tree':tree,'API':results['API'],'old_M1_M2_preserved':True,'credentials_included':False,
        'response_models':dict(Counter(x['model'] for x in responses)),
        'system_fingerprints':dict(Counter(x['system_fingerprint'] for x in responses)),
        'validation_repairs':len(list((ROOT/'requests').glob('*/attempt*/validation_error.json'))),
        'HTTP_errors':len(list((ROOT/'requests').glob('*/attempt*/http_error.json'))),
        'credential_scan_files':len(set(paths)),
        'operator_compiles_in_loop':0,'visual_QA':'comparison.png manually inspected',
        'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(OUT/'release_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('API','cache_tree')},indent=2))


if __name__=='__main__':main()
