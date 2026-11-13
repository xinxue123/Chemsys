"""Materialize native input-path aliases from the frozen pools, without new queries."""
import csv
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    with (ROOT/'data/dataset_registry.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        source=ROOT/'data/pools'/row['task']
        target=(ROOT/row['pool_source']).parent
        assert target.resolve().is_relative_to(ROOT/'experiments')
        target.mkdir(parents=True,exist_ok=True)
        for name in ('pool.json','oracle.npz'):
            path=target/name
            if path.exists():assert path.read_bytes()==(source/name).read_bytes(),str(path)
            else:shutil.copy2(source/name,path)
        print(row['task'],target.relative_to(ROOT).as_posix())
    (ROOT/'experiments/chemist_system').mkdir(parents=True,exist_ok=True)
    print('PASS: public input-path aliases only; no model call, GP fit or query.')

if __name__=='__main__':main()
