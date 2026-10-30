"""Seal complete M2 scientific evidence; exclude environment and credentials."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_study as study


def main():
    base=Path('.').resolve();p=study.check_protocol();progress=study.progress()
    assert progress['completed']==140 and progress['committed_batches_completed']==4200
    validation=data.read(study.OUT/'validation_receipt.json');assert validation['status']=='PASS_COMPLETE'
    assert data.read(study.OUT/'numeric_audit.json')['status']=='PASS_COMPLETE_FIXED_PREFIXES'
    assert data.read(study.OUT/'input_audit.json')['status']=='PASS_COMPLETE_STUDY_INPUTS'
    assert data.read(study.OUT/'M1_preservation_receipt.json')['status']=='PASS'
    assert data.read(study.OUT/'qualification_receipt.json')['tests']==13
    assert data.read(study.OUT/'results.json')['status']=='COMPLETE'
    inventory={}
    def add(path):
        path=Path(path).resolve();assert path.is_relative_to(base)
        name=path.relative_to(base).as_posix()
        assert 'config.txt' not in name and '.venv' not in name and '__pycache__' not in name
        inventory[name]={'path':name,'bytes':path.stat().st_size,'sha256':data.filehash(path)}
    for name in p['scientific_sha256']:add(name)
    for path in Path('chem/benchmarks').glob('reasoning_bo_*.py'):add(path)
    for path in Path('scripts').glob('*reasoning_bo*.py'):add(path)
    for name in ('reasoning_bo.py','scripts/reconcile_perera_paper_data.py','docs/reasoning_bo_comparison.md'):add(name)
    for row in data.read(data.UPSTREAM/'SOURCE_PROVENANCE.json')['files']:
        path=data.UPSTREAM/row['path'];assert data.filehash(path)==row['local_sha256'];add(path)
    add(data.UPSTREAM/'SOURCE_PROVENANCE.json')
    add(study.ROOT/'protocol.json');add(study.ROOT/'manifest.json')
    for path in Path('data/benchmarks/reaction_transfer/raw').glob('*'):
        if path.is_file():add(path)
    for folder in (data.ROOT,study.OUT,study.ROOT/'initialization',study.ROOT/'requests',study.ROOT/'runs'):
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix in ('.tmp','.lock'):continue
            if path.name in ('release_inventory.json','release_receipt.json'):continue
            add(path)
    cache=study.ROOT/'cache'
    rows=[{'path':path.relative_to(cache).as_posix(),'bytes':path.stat().st_size,'sha256':data.filehash(path)}
        for path in sorted(cache.rglob('*')) if path.is_file() and path.suffix not in ('.lock','.tmp')]
    cache_tree={'files':len(rows),'bytes':sum(r['bytes'] for r in rows),
        'sha256':sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()}
    inv={'protocol_sha256':p['protocol_sha256'],'files':[inventory[k] for k in sorted(inventory)],'cache_tree':cache_tree,
        'credentials_included':False,'runtime_environment_included':False}
    data.write(study.OUT/'release_inventory.json',inv)
    receipt={'status':'PASS_COMPLETE','completed':140,'planned':140,'batches':4200,
        'logical_observations':13020,'protocol_sha256':p['protocol_sha256'],
        'inventory_sha256':data.filehash(study.OUT/'release_inventory.json'),
        'results_sha256':data.filehash(study.OUT/'results.json'),'files':len(inventory),'cache_tree':cache_tree,
        'API':progress['api'],'operator_compiles_in_loop':0,'credentials_included':False,
        'M1_preserved':True,'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    data.write(study.OUT/'release_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
