"""Verify the GitHub edition from published data and derived tables only."""
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_hypothesis_scope_method import load_npz
LOCAL={'output','__pycache__','.git','archives','experiments','evidence','runtime'}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset_check():
    catalogue=rows(ROOT/'data/tasks.csv');report={}
    assert {r['task'] for r in catalogue}=={'c1','c2','c3','c4','perera','doyle','snar','hplc','aryl','cpa','dfl','bor','chx'}
    for row in catalogue:
        pool_path,oracle_path=ROOT/row['pool'],ROOT/row['oracle']
        pool=read(pool_path);arrays=load_npz(oracle_path.read_bytes());n=int(row['candidates'])
        assert sha(pool_path)==row['pool_sha256'] and sha(oracle_path)==row['oracle_sha256']
        assert len(pool['ids'])==len(set(pool['ids']))==len(pool['points'])==len(arrays['values'])==n
        raw=arrays['raw_values'] if 'raw_values' in arrays else [v*float(pool.get('objective_scale',100.)) for v in arrays['values']]
        assert all(math.isfinite(float(x)) for x in raw)
        assert abs(max(raw)-float(row['maximum_raw']))<1e-8
        if (pool_path.parent/'candidates.csv').exists():
            data=rows(pool_path.parent/'candidates.csv')
            assert [r['candidate_id'] for r in data]==pool['ids']
            assert all(abs(float(r['objective_raw'])-v)<1e-7 for r,v in zip(data,raw))
        report[row['task']]={'candidates':n,'pool_oracle_hashes_and_ID_order':'PASS'}
    return report

def source_check():
    changes=read(ROOT/'provenance/portability_adaptations.json')+read(ROOT/'provenance/publication_adaptations.json')
    by=defaultdict(list)
    for c in changes:by[c['file']].append(c)
    sources=read(ROOT/'provenance/source_index.json')
    for item in sources:
        raw=(ROOT/item['path']).read_bytes()
        for c in reversed(by[item['path']]):
            a,b=c['before'].encode(),c['after'].encode()
            assert raw.count(b)==1,(item['path'],'adaptation missing')
            raw=raw.replace(b,a,1)
        assert hashlib.sha256(raw).hexdigest()==item['source_sha256'],item['path']
    for p in list((ROOT/'code').rglob('*.py'))+list((ROOT/'tools').glob('*.py'))+[ROOT/'research.py']:
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    return dict(published_source_records=len(sources),filesystem_adaptations_documented=True,scientific_algorithm_changes=0,Python_syntax='PASS')

def result_check():
    catalogue={r['task']:r for r in rows(ROOT/'data/tasks.csv')}
    # Development curves contain derived running-best values, not raw decisions.
    dev=rows(ROOT/'results/grounded_baselines_20261105/running_best.csv')
    curves=defaultdict(list)
    for r in dev:curves[r['task'],int(r['seed']),r['method']].append(r)
    assert len(curves)==160
    per_seed={}
    for key,values in curves.items():
        values=sorted(values,key=lambda r:int(r['query']))
        assert [int(r['query']) for r in values]==list(range(1,21))
        best=[float(r['cumulative_best']) for r in values]
        assert all(a<=b+1e-9 for a,b in zip(best,best[1:]))
        scale=100/float(catalogue[key[0]]['maximum_raw'])
        per_seed[key]=(st.mean(best)*scale,best[-1]*scale)
    method_rows=rows(ROOT/'results/hypothesis_scope_method_evidence_20261105/method_by_task.csv')
    for r in method_rows:
        values=[per_seed[r['task'],s,r['method']] for s in range(4)]
        assert abs(st.mean(v[0] for v in values)-float(r['auc_pct_of_max']))<1e-8
        assert abs(st.mean(v[1] for v in values)-float(r['final_pct_of_max']))<1e-8
    # Paired primary/DeepSeek values are recomputed from the derived curves.
    pairs=rows(ROOT/'results/deepseek_table4_20261110/paired_results.csv')
    table4=rows(ROOT/'results/deepseek_table4_20261110/table4_model_extension.csv')
    by_key=defaultdict(list)
    for r in rows(ROOT/'results/deepseek_table4_20261110/running_best.csv'):
        by_key[r['task'],r['model'],r['method'],int(r['pair'])].append(r)
    assert len(pairs)==len(by_key)==288
    scores={};groups=defaultdict(list)
    for row in pairs:
        key=(row['task'],row['model'],row['method'],int(row['pair']))
        curve=sorted(by_key[key],key=lambda r:int(r['query']))
        assert [int(r['query']) for r in curve]==list(range(1,21))
        best=[float(r['running_best_pct_of_max']) for r in curve]
        assert all(a<=b+1e-9 for a,b in zip(best,best[1:]))
        auc,final=st.mean(best),best[-1]
        assert abs(auc-float(row['SearchAUC']))<1e-8 and abs(final-float(row['final_best']))<1e-8
        assert abs(float(row['SearchAUC'])-float(row['BO_SearchAUC'])-float(row['paired_difference']))<1e-8
        scores[key]=(auc,final);groups[key[:3]].append(row)
    for key,data in groups.items():assert len(data)==24 and len({r['pair'] for r in data})==24,key
    for r in table4:
        data=groups[r['task'],r['model'],r['method']]
        assert abs(st.mean(float(v['SearchAUC']) for v in data)-float(r['SearchAUC']))<1e-8
        assert abs(st.mean(float(v['final_best']) for v in data)-float(r['final_best']))<1e-8
    claims=rows(ROOT/'results/hypothesis_scope_v2_mechanism_20261106/claims.csv')
    contrasts=rows(ROOT/'results/hypothesis_scope_v2_mechanism_20261106/contrasts.csv')
    assert len({r['key'] for r in claims})==99 and len(contrasts)==68
    coding=ROOT/'results/hypothesis_scope_v2_mechanism_20261106/claim_coding.json'
    assert sha(coding)==coding.with_suffix('.sha256').read_text().split()[0]
    tex=(ROOT/'paper/hypothesis_scope_jcim/main.tex').read_text(encoding='utf-8')
    table=tex[tex.index(r'\label{tab:map}'):];table=table[:table.index(r'\end{table}')];displayed=0
    for line in table.splitlines():
        cells=line.strip().removesuffix(r'\\').strip().split(' & ')
        if len(cells)!=7 or not cells[3].isdigit():continue
        n=int(cells[3]);effect=float(cells[4].strip('$'));win,loss=map(int,cells[5].split('/'))
        assert any(int(r['contrasts'])==n and int(r['wins'])==win and int(r['losses'])==loss and r['verdict']==cells[6].lower() and abs(float(r['mean_effect'])-effect)<=.005000001 for r in contrasts)
        displayed+=1
    assert displayed==11
    return dict(development_trajectories=160,development_method_task_cells=len(method_rows),heldout_model_pairs=288,Table_4_rows=len(table4),knowledge_hypotheses=99,comparative_contrasts=68,Table_6_rows=11,scope='Arithmetic from published derived curves and tables; raw decision chains, GP fitting and LLM execution are not replayed.')

def credentials_check():
    token=re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    endings={'.py','.json','.csv','.md','.txt','.tex','.toml','.yaml','.yml','.env'}
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or rel.parts[0] in LOCAL:continue
        assert p.name not in ('config.txt','openai-key.txt','openai-apikey.txt','.env') and p.suffix not in ('.pem','.key')
        if p.suffix in endings:assert not token.search(p.read_bytes()),rel.as_posix()
        elif p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if Path(info.filename).suffix in endings:assert not token.search(z.read(info)),rel.as_posix()
    return 'PASS'

def manifest_check():
    path=ROOT/'MANIFEST.json'
    if not path.exists():return 'NOT_SEALED_YET'
    manifest=read(path)
    def check(row):
        p=ROOT/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    with ThreadPoolExecutor(max_workers=min(16,os.cpu_count() or 1)) as workers:list(workers.map(check,manifest['files']))
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] not in LOCAL and '__pycache__' not in p.parts}
    assert actual=={r['path'] for r in manifest['files']}|{'MANIFEST.json','VALIDATION.json'}
    return 'PASS'

def main():
    report=dict(status='PASS',edition='GitHub paper companion',datasets=dataset_check(),sources=source_check(),results=result_check(),credentials=credentials_check(),manifest=manifest_check(),raw_native_replay=False,new_model_calls=0,new_GP_fits=0,new_experimental_queries=0)
    print(json.dumps(report,ensure_ascii=False,indent=2));return report

if __name__=='__main__':main()
