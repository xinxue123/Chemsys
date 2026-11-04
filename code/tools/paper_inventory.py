"""Read-only historical experiment catalogue; no oracle or response files read."""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import csv,json,os,re,math
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/paper_v12'

def normalize_task(x,path):
    text=str(x or path)
    m=re.search(r'(?:Case_|case_|case|/c)([1-4])',text)
    if m:return 'Reizman_Suzuki_Case_'+m[1]
    for t in ('Perera_Suzuki','Doyle_Buchwald_Hartwig','SnAr','HPLC'):
        if t.lower() in text.lower():return t
    return str(x or 'unresolved')

def collect():
    paths=[]
    for tree in ('experiments/runs','experiments/pr_validation'):
        for d,dirs,files in os.walk(ROOT/tree):
            dirs[:]=[x for x in dirs if x not in ('cache','requests','__pycache__','snapshots','preflight','paper_v12')]
            if 'state.json' in files:paths.append(Path(d)/'state.json')
    rows=[];seen={};errors=[];protocols={}
    for path in sorted(paths):
        rel=path.relative_to(ROOT).as_posix();raw=path.read_bytes();h=sha256(raw).hexdigest()
        try:
            s=json.loads(raw.decode('utf-8-sig'));header=s.get('header') or {};cfg=header.get('settings',{})
            task=normalize_task(s.get('task',header.get('task_id',s.get('case'))),rel)
            if task in ('1','2','3','4'):task='Reizman_Suzuki_Case_'+task
            hist=s.get('history',s.get('oracle_ledger',[]));values=[];key=None
            for k in ('raw_value','yield_percent','y','value'):
                if hist and all(isinstance(r,dict) and type(r.get(k)) in (int,float) and math.isfinite(r[k]) for r in hist):
                    values=[float(r[k]) for r in hist];key=k;break
            protocol_path=None;pd={}
            for ancestor in path.parents:
                if ancestor==ROOT:break
                candidate=ancestor/'protocol.json'
                if candidate.exists():
                    protocol_path=candidate
                    if str(candidate) not in protocols:protocols[str(candidate)]=json.loads(candidate.read_text(encoding='utf-8-sig'))
                    pd=protocols[str(candidate)];break
            budget=pd.get('budget',{});budget=budget if isinstance(budget,dict) else {}
            init=cfg.get('initial_queries',pd.get('initial_queries',pd.get('initial',budget.get('initial'))))
            init_source='header_or_protocol'
            if not isinstance(init,int):
                flags=[r.get('initial') for r in hist] if hist else []
                if flags and all(type(x) is bool for x in flags):init=sum(flags);init_source='explicit_observation_flags'
                else:init=4;init_source='assumed_four_not_verified'
            declared=cfg.get('additional_queries',pd.get('additional_queries',pd.get('queries')))
            if declared is None and type(budget.get('queries_per_trajectory')) is int:declared=budget['queries_per_trajectory']-init
            unit='raw objective (see task protocol)' if key=='raw_value' else ('yield percent' if key=='yield_percent' else 'stored objective; scale requires protocol')
            if key=='raw_value' and task.startswith('Reizman'):unit='simulated yield percent'
            if key=='raw_value' and task=='HPLC':unit='source HPLC peak area'
            curve=[];best=-math.inf
            for v in values:best=max(best,v);curve.append(best)
            decisions=s.get('decisions',[]);nq=len(values)-init if values else len(decisions)
            cohort=rel.split('/')[2] if rel.startswith('experiments/runs/') else 'pr_validation'
            protocol=s.get('protocol_sha256',s.get('precision_protocol_sha256',header.get('task_sha256','')))
            initial_ids=[r.get('candidate_id',r.get('point')) for r in hist[:init]]
            dup=seen.get(h,'');seen.setdefault(h,rel)
            method=s.get('method',s.get('arm',s.get('mode',header.get('method','unknown'))))
            seed=s.get('seed',header.get('seed'))
            initialization_seed=seed
            if 'v12_seed123/' in rel or 'paper_v12/' in rel:
                m=re.search(r'/(?:v12_seed123|paper_v12)/s([0-9]+)/',rel)
                if m:initialization_seed=int(m[1])
            comparison=dict(task=task,protocol=protocol,initial_ids=initial_ids,initial=init,queries=nq,unit=unit,metric_field=key)
            group=sha256(json.dumps(comparison,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            row=dict(source=rel,source_sha256=h,byte_identical_to=dup,cohort=cohort,task=task,method=method,
                native_seed=seed,initialization_seed=initialization_seed,knowledge_condition=s.get('condition',s.get('regime','not recorded at state top level')),
                status=s.get('status','UNKNOWN'),measured_points=len(values),initial_queries=init,initial_count_source=init_source,post_initial_queries=nq,declared_post_initial_queries=declared,
                metric_field=key,objective_unit=unit,SearchAUC_stored_scale=sum(curve[init:])/len(curve[init:]) if len(curve)>init and init_source!='assumed_four_not_verified' else None,
                final_best_stored_scale=curve[-1] if curve else None,first_best_query=max(0,values.index(curve[-1])+1-init) if curve else None,
                protocol_sha256=protocol,protocol_path=protocol_path.relative_to(ROOT).as_posix() if protocol_path else '',
                protocol_file_sha256=sha256(protocol_path.read_bytes()).hexdigest() if protocol_path else '',
                model=pd.get('model',pd.get('api',{}).get('model','not recorded') if isinstance(pd.get('api',{}),dict) else 'not recorded'),
                initial_ids_sha256=sha256(json.dumps(initial_ids,sort_keys=True).encode()).hexdigest(),comparability_key=group,
                evidence_class='historical_development_not_independent_confirmation',metric_reconstruction=('AUC_WITHHELD_UNVERIFIED_INITIAL_BUDGET' if init_source=='assumed_four_not_verified' else 'PASS_FROM_MEASURED_PREFIX') if values else 'UNSUPPORTED_STATE_SCHEMA')
            rows.append(row)
        except Exception as e:errors.append(dict(source=rel,kind=type(e).__name__,message=str(e)))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'historical_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'historical_runs.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    summary=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        rs=[r for r in rows if r['cohort']==cohort]
        summary.append(dict(cohort=cohort,records=len(rs),byte_distinct=len({r['source_sha256'] for r in rs}),statuses=dict(Counter(r['status'] for r in rs)),
            tasks=sorted({r['task'] for r in rs}),metric_reconstructed=sum(r['metric_reconstruction']=='PASS_FROM_MEASURED_PREFIX' for r in rs)))
    result=dict(scope='Canonical experiments/runs and experiments/pr_validation only; compatibility junctions are not separately enumerated',
        state_records=len(rows),byte_distinct=len(seen),byte_identical_aliases=sum(bool(r['byte_identical_to']) for r in rows),errors=errors,cohorts=summary,
        caveat='Byte equality is content deduplication, not independence. COMPLETE is the stored execution status, not scientific qualification. Stored-scale metrics are not pooled across protocols or tasks. All original losses and partial runs are retained.')
    (OUT/'historical_inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cohorts'},indent=2))
if __name__=='__main__':collect()
