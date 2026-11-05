"""All registered V13-era paths, with byte-level reference deduplication."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import io
import json

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/v13_all_paths_20261104'
READS={}
FULL_TO_SHORT={**{f'Reizman_Suzuki_Case_{i}':f'c{i}' for i in range(1,5)},
    'Perera_Suzuki':'perera','Doyle_Buchwald_Hartwig':'doyle','SnAr':'snar','HPLC':'hplc',
    'Direct_Arylation':'aryl'}


def tracked(path):
    p=Path(path).resolve();b=p.read_bytes()
    READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
    return b


def read(path):
    return json.loads(tracked(path).decode('utf-8-sig'))


def rows(path):
    return list(csv.DictReader(io.StringIO(tracked(path).decode('utf-8-sig'))))


def save(name,records):
    if not records:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                     for k,v in r.items()} for r in records)


def close(a,b):
    assert np.isclose(float(a),float(b),atol=1e-8,rtol=1e-10),(a,b)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tracked(__file__);tracked(ROOT/'docs/v13_all_paths_20261104.txt')
    registry=[]
    def add(family,task,seed,method,backend,path,expected=None,initial=None,queries=None,
            stored_auc=None,stored_final=None,replay=False,primary=False,replicate=''):
        registry.append(dict(registry_row=len(registry),family=family,task=FULL_TO_SHORT.get(task,task),
            seed=int(seed or 0),method=method,backend=backend,source_path=path,
            expected_sha256=expected,initial=initial,queries=queries,stored_auc=stored_auc,
            stored_final=stored_final,explicit_replay=replay,primary=primary,replicate=replicate))
    main_rows=rows(ROOT/'paper/v13_late/data/metrics.csv')
    for r in main_rows:
        add('main_comparison',r['task'],r['seed'],r['method'],r['backend'],r['state_path'],
            r['state_sha256'],int(r['initial']),int(r['queries']),r['SearchAUC'],r['final_best'],primary=True)
    for r in rows(ROOT/'paper/v12_subagent/data/metrics.csv'):
        special=r['arm'] in ('no_kf','no_refresh')
        add('v12_ablation' if special else 'v12_registry_reference',r['task'],r['initialization_seed'],
            'v12_'+r['arm'] if special else r['method'],
            'none' if r['arm']=='bo' else 'fresh_subagent',r['source'],r['source_sha256'],4,16,
            r['SearchAUC'],r['final_best'])
    for r in rows(ROOT/'experiments/reports/v13_targeted_ablation_20261103/metrics.csv'):
        add('v13_targeted_replay' if r['origin']=='frozen_response_replay' else
            'v13_targeted' if r['origin']=='conditional_control' else 'targeted_reference',
            r['task'],r['seed'],'v13_'+r['arm'],'none' if r['arm']=='bo' else 'fresh_subagent',
            r['state_path'],None,0,20,r['SearchAUC'],r['final_best'],replay=r['origin']=='frozen_response_replay')
    for r in rows(ROOT/'paper/v13_late/data/final_metrics.csv'):
        add('late_'+r['kind'],r['task'],r['prefix_seed'],r['arm'],
            'none' if r['arm']=='bo' else 'fresh_subagent',r['state_path'],r['state_sha256'],
            int(r['prefix_queries']),int(r['new_queries']),r['SearchAUC20'],r['final_best'],replicate=r['repetition'])
    for r in rows(ROOT/'paper/v12/data/metrics.csv'):
        add('predecessor_API',r['task'],r['initialization_seed'],r['method'].lower().replace(' ','_'),
            'none' if r['method']=='BO' else 'historical_API',r['source'],r['source_sha256'],4,16,
            r['SearchAUC'],r['final_best'],replicate=r['replicate'])
    # Include all six selected V14 extensions; reused comparison rows remain
    # references rather than adding another copy of the main-comparison state.
    for r in rows(ROOT/'experiments/reports/reason_cards_v14_20261103/metrics.csv'):
        task=FULL_TO_SHORT.get(r['task'],r['task'])
        if r['method']=='v14':
            path=f"experiments/runs/reason_cards_v14_20261103/runs/v14_{task}_s{r['seed']}/state.json"
        else:
            match=next(x for x in main_rows if x['regime']=='cold' and x['task']==task and
                       x['seed']==r['seed'] and x['method']==r['method'])
            path=match['state_path']
        add('selected_v14' if r['method']=='v14' else 'v14_reference',task,r['seed'],r['method'],
            'none' if r['method']=='bo' else 'fresh_subagent',path,None,0,20,r['SearchAUC'],r['final_best'])
    for r in rows(ROOT/'experiments/reports/program_kf_weight_20261103/metrics.csv'):
        add('programmatic_KF',r['task'],r['seed'],r['method'],'programmatic_ranker',
            r['state_path'],None,0,20,r['SearchAUC'],r['final_best'])
    confirm=read(ROOT/'experiments/runs/program_kf_confirmation_20261104/protocol.json')
    for r in rows(ROOT/'experiments/reports/program_kf_confirmation_20261104/metrics.csv'):
        if r['arm'] in ('fixed','dynamic'):
            job=next(j for j in confirm['jobs'] if j['cell']==r['cell'] and j['arm']==r['arm'])
            path=f"experiments/runs/program_kf_confirmation_20261104/runs/{job['id']}/state.json"
            task,seed=job['task'],job['seed']
        else:
            path=confirm['references'][r['cell']+'_'+r['arm']]
            s=read(ROOT/path);task=s['task'];seed=next(j['seed'] for j in confirm['jobs'] if j['cell']==r['cell'])
        add('programmatic_KF_confirmation',task,seed,r['arm'],'programmatic_ranker',
            path,None,0,20,r['SearchAUC'],r['final_best'])

    unique={};catalogue=[];all_rounds=[]
    for r in registry:
        path=(ROOT/r['source_path']).resolve();b=tracked(path);h=hashlib.sha256(b).hexdigest()
        if r['expected_sha256']:assert h==r['expected_sha256'],r['source_path']
        r['actual_sha256']=h
        r['canonical_path']=path.relative_to(ROOT).as_posix()
        r['duplicate_of']=unique.get(h,'')
        if r['duplicate_of']:continue
        ident=f'path{len(catalogue):04d}';unique[h]=ident
        s=json.loads(b.decode('utf-8-sig'))
        assert s['status']=='COMPLETE' and len(s['history'])==20,r['source_path']
        values=np.array([o['raw_value'] for o in s['history']],dtype=float)
        assert np.isfinite(values).all()
        assert len({o['candidate_id'] for o in s['history']})==20
        best=np.maximum.accumulate(values)
        ini=int(r['initial']);queries=int(r['queries'])
        continuation=r['family'] in ('late_prefix','late_holdout') and ini>0
        assert ini+queries==20
        actual_auc=float(best.mean() if continuation else best[ini:].mean())
        close(actual_auc,r['stored_auc']);close(best[-1],r['stored_final'])
        if not continuation:assert len(s['decisions'])==queries,r['source_path']
        record=dict(path_id=ident,family=r['family'],primary=r['primary'],task=r['task'],seed=r['seed'],
            replicate=r['replicate'],method=r['method'],backend=r['backend'],
            regime='continuation10+10' if continuation else 'cold' if ini==0 else 'warm',
            initial=ini,queries=queries,observations=20,explicit_replay=r['explicit_replay'],
            source_path=r['canonical_path'],source_sha256=h,SearchAUC=actual_auc,
            continuation_SearchAUC=float(best[ini:].mean()) if continuation else None,
            final_best=float(best[-1]),initial_best=float(best[ini-1]) if ini else None,
            first_final_total=int(np.flatnonzero(values==best[-1])[0])+1,
            physical_sequence_sha256=hashlib.sha256(json.dumps(dict(task=r['task'],history=
                [(x['candidate_id'],x['raw_value']) for x in s['history']]),sort_keys=True).encode()).hexdigest(),
            primary_registry_row=r['registry_row'])
        catalogue.append(record)
        for q,obs in enumerate(s['history'],1):
            all_rounds.append(dict(path_id=ident,family=r['family'],task=r['task'],method=r['method'],
                total_observations=q,candidate_id=obs['candidate_id'],objective=obs['raw_value'],
                cumulative_best=float(best[q-1]),prefix_observation=q<=ini))
    save('registry_references.csv',registry);save('path_catalogue.csv',catalogue)
    save('all_observed_sequences.csv',all_rounds)
    summary=[]
    for family in sorted({r['family'] for r in catalogue}):
        subset=[r for r in catalogue if r['family']==family]
        summary.append(dict(family=family,unique_states=len(subset),
            distinct_observed_paths=len({r['physical_sequence_sha256'] for r in subset}),
            complete_from_start=sum(r['regime']!='continuation10+10' and not r['explicit_replay'] for r in subset),
            conditional_continuations=sum(r['regime']=='continuation10+10' for r in subset),
            explicit_replays=sum(r['explicit_replay'] for r in subset)))
    save('census_summary.csv',summary)
    for name,h in READS.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    facts=dict(status='PASS_COMPLETE_CENSUS',registry_records=len(registry),unique_state_files=len(catalogue),
        duplicate_registry_references=sum(bool(r['duplicate_of']) for r in registry),
        primary_main_paths=sum(r['primary'] for r in catalogue),
        complete_from_start_nonreplay=sum(r['regime']!='continuation10+10' and not r['explicit_replay'] for r in catalogue),
        continuations=sum(r['regime']=='continuation10+10' for r in catalogue),
        explicit_frozen_replays=sum(r['explicit_replay'] for r in catalogue),
        observed_history_records=len(all_rounds),families=summary,new_GP_fits=0,new_LLM_calls=0,new_experiments=0)
    (OUT/'census_audit.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    (OUT/'census_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
