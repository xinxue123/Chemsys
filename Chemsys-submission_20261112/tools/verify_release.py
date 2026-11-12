"""Read-only integrity, data and saved-result checks. No model calls or GP fits."""
import ast
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

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset_check():
    catalogue=rows(ROOT/'data/tasks.csv')
    assert {r['task'] for r in catalogue}=={'c1','c2','c3','c4','perera','doyle','snar','hplc','aryl','cpa','dfl','bor','chx'}
    report={}
    for row in catalogue:
        p=ROOT/row['pool'];o=ROOT/row['oracle'];pool=read(p);arrays=load_npz(o.read_bytes())
        assert sha(p)==row['pool_sha256'] and sha(o)==row['oracle_sha256'],row['task']
        ids=pool['ids'];values=arrays['values'];n=int(row['candidates'])
        assert len(ids)==len(pool['points'])==len(values)==len(set(ids))==n
        if 'raw_values' in arrays:raw=arrays['raw_values']
        else:
            scale=float(pool.get('objective_scale',100.))
            raw=[float(x)*scale for x in values]
        assert all(math.isfinite(float(x)) for x in raw)
        assert abs(max(raw)-float(row['maximum_raw']))<1e-8
        if (p.parent/'candidates.csv').exists():
            csv_rows=rows(p.parent/'candidates.csv')
            assert [r['candidate_id'] for r in csv_rows]==ids
            assert all(abs(float(r['objective_raw'])-v)<1e-7 for r,v in zip(csv_rows,raw))
        report[row['task']]={'candidates':n,'pool_and_oracle_hashes':'PASS','ID_order':'PASS','finite_objectives':'PASS'}
    return report

def source_check():
    changes=read(ROOT/'provenance/portability_adaptations.json')
    groups={}
    for change in changes:groups.setdefault(change['file'],[]).append(change)
    index=read(ROOT/'provenance/source_index.json')
    for item in index:
        path=ROOT/item['path'];b=path.read_bytes()
        for change in reversed(groups.get(item['path'],[])):
            a,new=change['before'].encode(),change['after'].encode()
            assert b.count(new)==1,(item['path'],'adaptation missing')
            b=b.replace(new,a,1)
        assert hashlib.sha256(b).hexdigest()==item['source_sha256'],item['path']
    for p in (ROOT/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    for p in (ROOT/'tools').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    return dict(indexed_sources=len(index),path_only_adaptations=len(changes),native_algorithms='UNCHANGED',Python_syntax='PASS')

def result_check():
    by=rows(ROOT/'paper/hypothesis_scope_jcim/figure_data/fig2_development/panel_a_by_task.csv')
    native=rows(ROOT/'results/hypothesis_scope_method_evidence_20261105/method_by_task.csv')
    ni={(r['task'],r['method']):r for r in native}
    # Exported figure labels use presentation names rather than the internal arm IDs.
    mapping={'Scope controller':'hyp','LLM-start BO':'first','Random-start BO':'bo','LGBO':'lgbo','Reasoning BO':'rbo'}
    tasks={'Reizman 1':'c1','Reizman 2':'c2','Reizman 3':'c3','Reizman 4':'c4','Perera':'perera','Doyle':'doyle','SnAr':'snar','HPLC':'hplc'}
    for r in by:
        key=(tasks.get(r['task'],r['task']),mapping.get(r['method'],r['method']))
        a=ni[key]
        # Frozen figure exports are rounded to four decimal places.
        assert abs(float(r['searchauc_pct_of_max'])-float(a['auc_pct_of_max']))<=5.0001e-5
        assert abs(float(r['final_best_pct_of_max'])-float(a['final_pct_of_max']))<=5.0001e-5
    primary=read(ROOT/'results/hypothesis_scope_v2_shared_start_20261107/shared_start.json')
    deep=rows(ROOT/'results/deepseek_table4_20261110/paired_results.csv')
    assert len(deep)==288,len(deep)
    groups={}
    for r in deep:groups.setdefault((r['model'],r['task'],r['method']),[]).append(r)
    for key,data in groups.items():
        assert len(data)==24 and len({int(r['pair']) for r in data})==24,key
        assert all(abs(float(r['SearchAUC'])-float(r['BO_SearchAUC'])-float(r['paired_difference']))<1e-8 for r in data)
    starts={(r['task'],int(r['pair'])):r for r in read(ROOT/'evidence/deepseek/frozen/shared_starts.json')}
    catalogue={r['task']:r for r in rows(ROOT/'data/tasks.csv')}
    for row in deep:
        task,model,arm,pair=row['task'],row['model'],row['method'],int(row['pair'])
        start=starts[task,pair]
        if model=='original':
            member=start['baseline_state'] if arm=='bo' else start['controller_state'] if arm=='r1' else f'shared_start/runs/{arm}/s{pair}/aryl/state.json'
            path=ROOT/'evidence/deepseek/frozen'/member
        elif arm=='bo':path=ROOT/f'evidence/deepseek/local_bo/{task}/s{pair}/state.json'
        else:path=ROOT/'evidence/deepseek/new'/('shared_start' if task=='aryl' else 'cpa')/'runs'/arm/f's{pair}'/task/'state.json'
        state=read(path);history=state['history']
        assert state['status']=='COMPLETE' and len(history)==20 and len({h['candidate_id'] for h in history})==20
        assert history[0]['candidate_id']==row['start']
        top=float(catalogue[task]['maximum_raw']);values=[100*float(h['raw_value'])/top for h in history]
        best=[max(values[:i+1]) for i in range(20)]
        assert abs(st.mean(best)-float(row['SearchAUC']))<1e-8
        assert abs(best[-1]-float(row['final_best']))<1e-8
    audit=read(ROOT/'results/hypothesis_scope_v2_mechanism_20261106/mechanism.json')
    claims=rows(ROOT/'results/hypothesis_scope_v2_mechanism_20261106/claims.csv')
    contrasts=rows(ROOT/'results/hypothesis_scope_v2_mechanism_20261106/contrasts.csv')
    assert len({r['key'] for r in claims})==99 and len(contrasts)==68
    table = rows(ROOT / "results/hypothesis_scope_v2_mechanism_20261106/table6_displayed.csv")
    for item in table:
        identity = (item["task"], item["axis"], item["higher"], item["lower"])
        expected = (item["pairs"], item["wins"], item["losses"], item["verdict"])
        matching = [row for row in contrasts
                    if (row["task"], row["axis"], row["high"], row["low"]) == identity
                    and (row["contrasts"], row["wins"], row["losses"], row["verdict"]) == expected]
        assert any(abs(float(row["mean_effect"]) - float(item["effect"])) <= .005000001
                   for row in matching), item
    displayed = len(table)
    assert displayed == 11
    lineage = read(ROOT / "provenance/claim_coding_lineage.json")
    for relative in lineage["analysis_files"]:
        assert sha(ROOT / relative) == lineage["analysis_sha256"], relative
    hypothesis_file = ROOT / "results/hypothesis_scope_v2_mechanism_20261106/claim_coding.json"
    assert hypothesis_file.with_suffix(".sha256").read_text().split()[0] == lineage["analysis_sha256"]
    coding = read(hypothesis_file)
    assert set(coding) == set(lineage["retained_fields"])
    hypotheses = coding["hypotheses"]
    assert len(hypotheses) == lineage["hypotheses"] == 99
    assert sum(bool(h["contrasts"]) for h in hypotheses) == lineage["comparative_hypotheses"] == 62
    assert sum(len(h["contrasts"]) for h in hypotheses) == lineage["contrasts"] == 68
    return dict(development_figure_cells=len(by),heldout_model_pair_rows_recomputed_from_states=len(deep),knowledge_hypotheses=len({r['key'] for r in claims}),hypothesis_stream_rows=len(claims),comparative_contrasts=len(contrasts),Table_6_rows_checked=displayed,saved_result_checks='PASS')

def credentials_check():
    bad=[]
    token=re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    text_ext={'.py','.json','.csv','.md','.txt','.tex','.toml','.yaml','.yml','.env'}
    for p in ROOT.rglob('*'):
        if not p.is_file() or 'output' in p.relative_to(ROOT).parts:continue
        if p.name in ('config.txt','openai-key.txt','openai-apikey.txt','.env') or p.suffix in ('.pem','.key'):
            bad.append(p.relative_to(ROOT).as_posix())
        elif p.suffix.lower() in text_ext and token.search(p.read_bytes()):bad.append(p.relative_to(ROOT).as_posix())
        elif p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if Path(info.filename).suffix.lower() in text_ext and token.search(z.read(info)):
                        bad.append(p.relative_to(ROOT).as_posix()+'::'+info.filename)
    assert not bad,{'credential_paths_only':bad}
    return 'PASS'

def manifest_check():
    p=ROOT/'MANIFEST.json'
    if not p.exists():return 'NOT_SEALED_YET'
    manifest=read(p)
    def one(row):
        p=ROOT/row['path']
        return p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    with ThreadPoolExecutor(max_workers=min(16,os.cpu_count() or 1)) as pool:
        assert all(pool.map(one,manifest['files'])),'Manifest hash mismatch'
    expected={r['path'] for r in manifest['files']}|{'MANIFEST.json','VALIDATION.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not {'output','__pycache__','.git'}&set(p.relative_to(ROOT).parts)}
    assert actual==expected,{'untracked':sorted(actual-expected),'missing':sorted(expected-actual)}
    return 'PASS'

def main():
    report={'status':'PASS','datasets':dataset_check(),'sources':source_check(),'results':result_check(),'credentials':credentials_check(),'manifest':manifest_check(),'new_model_calls':0,'new_GP_fits':0,'new_experimental_queries':0}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':main()
