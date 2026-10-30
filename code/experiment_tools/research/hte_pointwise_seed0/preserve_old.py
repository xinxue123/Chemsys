"""Read-only verification of protected M1/M2 files, including their GP caches."""
from hashlib import sha256
from pathlib import Path
import json

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]


def read(p):return json.loads(p.read_text(encoding='utf8'))
def hashed(p):return sha256(p.read_bytes()).hexdigest()


def tree(folder,skip_temporary=False):
    rows=[{'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':hashed(p)}
        for p in sorted(folder.rglob('*')) if p.is_file() and
        (not skip_temporary or p.suffix not in ('.lock','.tmp'))]
    return {'files':len(rows),'bytes':sum(r['bytes'] for r in rows),
        'sha256':sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()}


def main():
    m1=read(BASE/'research/milestones/M1_Reizman_seed0_20261028/milestone.json')
    m2=read(BASE/'research/reasoning_bo_comparison/release_inventory.json')
    old_receipt=read(BASE/'research/reasoning_bo_comparison/release_receipt.json')
    assert hashed(BASE/'research/reasoning_bo_comparison/release_inventory.json')==old_receipt['inventory_sha256']
    for row in m1['files']+m2['files']:assert hashed(BASE/row['path'])==row['sha256'],row['path']
    assert hashed(BASE/'research/milestones/M1_Reizman_seed0_20261028'/m1['archive']['name'])==m1['archive']['sha256']
    t1={k:tree(BASE/'data/runs/suzuki_grounded_regional'/k) for k in m1['cache_trees_verified']}
    assert t1==m1['cache_trees_verified']
    t2=tree(BASE/'data/runs/reasoning_bo_comparison/cache',True);assert t2==old_receipt['cache_tree']
    receipt={'status':'PASS','M1_sealed_files':len(m1['files']),'M2_sealed_files':len(m2['files']),
        'M1_archive_sha256':m1['archive']['sha256'],'M1_cache_trees':t1,'M2_cache_tree':t2,
        'old_sources_results_and_caches_modified':False,
        'scope':'Byte preservation only; historical numerical claims are not re-certified.'}
    (OUT/'preservation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps({'status':'PASS','M1_sealed_files':len(m1['files']),'M2_sealed_files':len(m2['files']),
        'old_cache_files':sum(x['files'] for x in t1.values())+t2['files']}))


if __name__=='__main__':main()
