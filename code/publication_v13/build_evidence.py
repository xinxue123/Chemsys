"""Read-only V13 publication export and an additive, hash-verified milestone.

No GP fitting, acquisition evaluation, oracle access or LLM requests occur here.
Run with the existing .venv-official Python from the repository root.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/v13'
MILESTONE = ROOT / 'experiments/milestones/v13_publication_20261103'
VARIANTS = ('v13_state_split_20261102', 'v13_seed123_20261102',
            'cold0_20261102', 'cold_s123_20261102')
SHORT = dict(zip(tuple(f'Reizman_Suzuki_Case_{i}' for i in range(1, 5)) +
                 ('Perera_Suzuki', 'Doyle_Buchwald_Hartwig', 'SnAr', 'HPLC'),
                 ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')))
REPORTS = ('v13_state_split_experiment_20261102', 'v13_seed123_20261102',
           'cold0_20261102', 'cold_s123_20261102', 'v13_loss_diagnosis_20261103')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Empty export: ' + str(path))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def finite_close(a, b):
    return math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-7)


def registrations(protocols):
    rows = []
    warm0, warm3, cold0, cold3 = protocols
    for task, short in SHORT.items():
        rows.append(('warm', task, 0, 'v13', ROOT / 'experiments/runs' / VARIANTS[0] / 'runs' / short / 'state.json', 'fresh_subagent'))
        for method, value in warm0['controls'][task].items():
            rows.append(('warm', task, 0, method, ROOT / value['path'], 'none' if method == 'bo' else ('historical_API' if method in ('reasoning_bo', 'v12_api') else 'fresh_subagent')))
        for seed in (1, 2, 3):
            rows.append(('warm', task, seed, 'v13', ROOT / 'experiments/runs' / VARIANTS[1] / f's{seed}' / short / 'runs' / f'm{seed}_{short}' / 'state.json', 'fresh_subagent'))
            control = next(x for x in warm3['controls'] if x['task'] == task and x['initialization_seed'] == seed)
            for method, value in control['arms'].items():
                rows.append(('warm', task, seed, method, ROOT / value['path'], 'none' if method == 'bo' else 'fresh_subagent'))
        for seed in (0, 1, 2, 3):
            for method, arm in (('v13', 'v13'), ('bo', 'bo'), ('reasoning_bo', 'rbo')):
                if seed == 0:
                    path = ROOT / 'experiments/runs' / VARIANTS[2] / 'runs' / f'{short}_{arm}' / 'state.json'
                else:
                    path = ROOT / 'experiments/runs' / VARIANTS[3] / f's{seed}' / 'runs' / f'ms{seed}_{short}_{arm}' / 'state.json'
                rows.append(('cold', task, seed, method, path, 'none' if method == 'bo' else 'fresh_subagent'))
    return rows


def export(protocols, registered):
    metrics, rounds, forecasts, inventory, model_links = [], [], [], [], []
    states = {}
    for regime, task, seed, method, path, backend in registered:
        state = read(path)
        n0, budget = (4, 16) if regime == 'warm' else (0, 20)
        h, ds = state['history'], state['decisions']
        if state['status'] != 'COMPLETE' or len(h) != 20 or len(ds) != budget:
            raise ValueError('Incomplete registered evidence: ' + str(path))
        if len({x['candidate_id'] for x in h}) != len(h):
            raise ValueError('Repeated paid point: ' + str(path))
        raw = [float(x['raw_value']) for x in h]
        bests = [max(raw[:n0 + q]) for q in range(1, budget + 1)]
        auc, final = sum(bests) / budget, max(raw)
        first = 0 if n0 and max(raw[:n0]) == final else next(q for q, b in enumerate(bests, 1) if b == final)
        key = (regime, SHORT[task], seed, method)
        states[key] = state
        metrics.append(dict(regime=regime, task=SHORT[task], seed=seed, method=method,
            backend=backend, initial=n0, queries=budget, SearchAUC=auc, final_best=final,
            first_final_best_query=first, initial_best=max(raw[:n0]) if n0 else '',
            state_path=rel(path), state_sha256=sha(path)))
        inventory.append(dict(regime=regime, task=SHORT[task], seed=seed, method=method,
            path=rel(path), sha256=sha(path), decisions=len(ds), observations=len(h)))
        for q, (d, b) in enumerate(zip(ds, bests), 1):
            obs = h[n0 + q - 1]
            if d['selected_id'] != obs['candidate_id'] or not finite_close(d['raw_value'], obs['raw_value']):
                raise ValueError('Decision/outcome mismatch')
            before = max(raw[:n0 + q - 1]) if n0 + q - 1 else None
            if before is not None and before > b + 1e-8:
                raise ValueError('Nonmonotonic incumbent')
            events = d.get('feedback', [])
            rounds.append(dict(regime=regime, task=SHORT[task], seed=seed, method=method,
                query=q, total_observations=n0 + q, candidate_id=obs['candidate_id'],
                objective=obs['raw_value'], incumbent_before='' if before is None else before,
                cumulative_best=b, BO_reference_id=d.get('BO_reference_id', ''),
                changed_from_own_GP=int(d.get('BO_reference_id', obs['candidate_id']) != obs['candidate_id']),
                forecast_count=len(events), refresh=int(q in state.get('edit_calls', [])),
                review=int(q in state.get('review_queries', [])), state_path=rel(path)))
            if method == 'v13':
                fit = path.parent / 'rounds' / f'{q:02}_fit.json'
                if fit.is_file():
                    value = read(fit)
                    model_links.append(dict(regime=regime, task=SHORT[task], seed=seed, query=q,
                        fit_record=rel(fit), fit_sha256=sha(fit), fit_key=value.get('fit_key', ''),
                        no_data=(regime == 'cold' and q == 1)))
                for e in events:
                    c = e['calibration']
                    forecasts.append(dict(regime=regime, task=SHORT[task], seed=seed, query=q,
                        candidate_id=obs['candidate_id'], use=e['use'], improved=int(e['improved']),
                        accurate_failure=int(e.get('accurate_failure_forecast', False)),
                        Brier_gain=c['Brier_gain'], reliability_before=c['before']['reliability'],
                        reliability_after=c['after']['reliability'], kalman_gain=c['kalman_gain'],
                        permission_before=e['recommendation_before'], permission_after=e['recommendation_after'],
                        event_id=e['event_id']))
    # Verify shared warm initial observations, not just seed labels.
    for task in SHORT.values():
        for seed in range(4):
            ref = states['warm', task, seed, 'v13']['history'][:4]
            for method in ('bo', 'reasoning_bo', 'v12_subagent'):
                other = states['warm', task, seed, method]['history'][:4]
                if [(x['candidate_id'],x['raw_value']) for x in ref] != [(x['candidate_id'],x['raw_value']) for x in other]:
                    raise ValueError('Warm initialization mismatch')
    # Cross-check independent arithmetic against all four native metric exports.
    native = [('warm',0,ROOT / 'experiments/reports/v13_state_split_experiment_20261102/metrics.csv'),
              ('warm',None,ROOT / 'experiments/reports/v13_seed123_20261102/metrics.csv'),
              ('cold',None,ROOT / 'experiments/reports/cold_s123_20261102/metrics_all.csv')]
    indexed = {(r['regime'],r['task'],r['seed'],r['method']):r for r in metrics}
    checks = 0
    for regime, fixed_seed, path in native:
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                method = row['method'].lower()
                k = (regime, row['task'], int(row.get('seed', fixed_seed)), method)
                actual = indexed[k]
                for field in ('SearchAUC','final_best','first_final_best_query'):
                    if not finite_close(row[field],actual[field]):
                        raise ValueError('Native metric mismatch: ' + str(k) + field)
                checks += 1
    paired = []
    for r in metrics:
        if r['method'] != 'v13':
            continue
        for comparator in ('bo','reasoning_bo','v12_subagent'):
            k=(r['regime'],r['task'],r['seed'],comparator)
            if k not in indexed:
                continue
            b=indexed[k]
            delta=r['SearchAUC']-b['SearchAUC']
            paired.append(dict(regime=r['regime'],task=r['task'],seed=r['seed'], comparator=comparator,
                V13_SearchAUC=r['SearchAUC'],control_SearchAUC=b['SearchAUC'],delta=delta,
                paired_relative_percent=100*delta/b['SearchAUC'],
                final_best_delta=r['final_best']-b['final_best'],
                result='tie' if abs(delta)<1e-8 else 'win' if delta>0 else 'loss',
                backend_match=(r['backend']==b['backend'] or comparator=='bo')))
    summary=[]
    for cohort,seeds in (('warm_development',(0,)),('warm_multistart',(1,2,3)),('cold_multistart',(0,1,2,3))):
        regime=cohort.split('_')[0]
        for task in SHORT.values():
            for method in ('bo','reasoning_bo','v13','v12_subagent'):
                rows=[x for x in metrics if x['regime']==regime and x['task']==task and x['method']==method and x['seed'] in seeds]
                if not rows:
                    continue
                summary.append(dict(cohort=cohort,task=task,method=method,n=len(rows),
                    mean_SearchAUC=statistics.mean(x['SearchAUC'] for x in rows),
                    sd_SearchAUC=statistics.stdev(x['SearchAUC'] for x in rows) if len(rows)>1 else 0.,
                    mean_final_best=statistics.mean(x['final_best'] for x in rows),
                    sd_final_best=statistics.stdev(x['final_best'] for x in rows) if len(rows)>1 else 0.,
                    mean_first_final_best_query=statistics.mean(x['first_final_best_query'] for x in rows)))
    for name,rows in [('metrics',metrics),('rounds',rounds),('paired',paired),('summary',summary),('forecasts',forecasts),('state_inventory',inventory),('fit_inventory',model_links)]:
        write_csv(PAPER/'data'/f'{name}.csv',rows)
    counts=Counter((r['regime'],r['method']) for r in metrics)
    mechanism={}
    for cohort,seeds in (('warm_development',(0,)),('warm_multistart',(1,2,3)),('cold_multistart',(0,1,2,3))):
        regime=cohort.split('_')[0]
        f=[x for x in forecasts if x['regime']==regime and x['seed'] in seeds]
        rr=[x for x in rounds if x['regime']==regime and x['seed'] in seeds and x['method']=='v13']
        mechanism[cohort]=dict(trajectories=len(rr)//(16 if regime=='warm' else 20), decisions=len(rr),
            note_forecasts=len(f), KF_increases=sum(x['reliability_after']>x['reliability_before']+1e-12 for x in f),
            KF_decreases=sum(x['reliability_after']<x['reliability_before']-1e-12 for x in f),
            KF_unchanged=sum(abs(x['reliability_after']-x['reliability_before'])<=1e-12 for x in f),
            accurate_failure_events=sum(x['accurate_failure'] for x in f),
            new_suppressions=sum(x['permission_after']=='suppressed' and x['permission_before']!='suppressed' for x in f),
            new_allowed=sum(x['permission_after']=='allowed' and x['permission_before']!='allowed' for x in f),
            changed_from_own_GP=sum(x['changed_from_own_GP'] for x in rr),
            refresh_calls=sum(x['refresh'] for x in rr),review_queries=sum(x['review'] for x in rr),
            forecast_use_counts=dict(Counter(x['use'] for x in f)))
    write_json(PAPER/'data/mechanisms.json',mechanism)
    audit=dict(status='PASS',native_metric_rows_crosschecked=checks,total_registered_trajectories=len(metrics),
        V13_trajectories=sum(x['method']=='v13' for x in metrics),
        V13_paid_queries=sum(x['queries'] for x in metrics if x['method']=='v13'),
        counts={f'{a}/{b}':n for (a,b),n in counts.items()},
        all_warm_initializations_matched=True,all_trajectories_complete=True,
        new_experiments=0,new_GP_fits=0,new_LLM_calls=0,new_oracle_calls=0,
        metric='mean of B post-query cumulative incumbents; B=16 warm, B=20 cold; no missing values dropped')
    write_json(PAPER/'audit/arithmetic.json',audit)
    return audit


def seal(protocols, registered):
    MILESTONE.mkdir(parents=True,exist_ok=True)
    if (MILESTONE/'receipt.json').exists():
        receipt=read(MILESTONE/'receipt.json')
        for name,h in receipt['artifact_sha256'].items():
            if sha(MILESTONE/name)!=h:
                raise ValueError('Sealed publication milestone changed')
        return receipt
    selected=set()
    cache_counts={}
    def add(path):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Source outside workspace')
        if not path.is_file():
            raise ValueError('Missing archival source: '+str(path))
        if path.name.lower() in ('config.txt','openai-key.txt') or path.suffix.lower() in ('.pem','.key'):
            raise ValueError('Credential excluded')
        selected.add(path)
    def tree(folder, cache=False):
        for path in Path(folder).rglob('*'):
            if not path.is_file():
                continue
            parts=path.relative_to(folder).parts
            if '__pycache__' in parts or path.suffix in ('.lock','.pyc'):
                continue
            if not cache and 'cache' in parts:
                continue
            add(path)
    for variant,p in zip(VARIANTS,protocols):
        base=ROOT/'experiments/runs'/variant
        tree(base)
        files=[f for f in (base/'cache').rglob('*') if f.is_file() and f.suffix in ('.pt','.npz','.json')]
        cache_counts[variant]=dict(files=len(files),pt_files=sum(f.suffix=='.pt' for f in files),bytes=sum(f.stat().st_size for f in files))
        for f in files:
            add(f)
        for name,h in p['sources_sha256'].items():
            path=ROOT/name
            if sha(path)!=h:
                raise ValueError('Frozen scientific source changed: '+name)
            add(path)
        for field in ('protected_sha256','protected_seed0_sha256','inputs_sha256'):
            for name,h in p.get(field,{}).items():
                if sha(ROOT/name)!=h:
                    raise ValueError('Frozen protected input changed: '+name)
                add(ROOT/name)
    for row in protocols[0]['inputs']:
        for name,h in row['files_sha256'].items():
            if sha(ROOT/name)!=h:
                raise ValueError('Frozen dataset changed')
            add(ROOT/name)
    # Native controls and their committed numerical/decision records are retained.
    for _,_,_,_,path,_ in registered:
        tree(path.parent)
    for report in REPORTS:
        tree(ROOT/'experiments/reports'/report)
    # The V14 selected-task extension is kept as negative development evidence.
    tree(ROOT/'experiments/reports/reason_cards_v14_20261103')
    for p in (ROOT/'code/publication_v13').glob('*.py'):
        add(p)
    old=ROOT/'experiments/milestones/v13_warmstart_20261102'
    old_receipt=read(old/'receipt.json')
    for name,h in old_receipt['artifact_sha256'].items():
        if sha(old/name)!=h:
            raise ValueError('Previous 4+16 milestone changed')
    add(old/'manifest.json');add(old/'receipt.json')
    ordered=sorted(selected,key=rel)
    print('Hashing',len(ordered),'publication evidence and GP cache files',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes=list(pool.map(sha,ordered))
    names={rel(p):p for p in ordered}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='V13: all 32 warm 4+16 and all 32 cold 0+20 trajectories, matched controls, datasets, source, prompts and native GP caches',
        source_files=len(names),byte_count=sum(p.stat().st_size for p in ordered),
        protocols={v:p['protocol_sha256'] for v,p in zip(VARIANTS,protocols)},
        model_assets=cache_counts,
        LLM_assets='Hosted model weights are not locally available. Registered model/effort, public prompts, returned response bytes and transport receipts are retained; no checkpoint or exact server reproducibility is claimed.',
        exclusions=['credentials','private model reasoning','environment binaries','transient locks and bytecode'],
        earlier_warm_milestone=dict(path=rel(old),artifact_sha256=old_receipt['artifact_sha256']),
        files_sha256=dict(zip((rel(p) for p in ordered),hashes)))
    write_json(MILESTONE/'manifest.json',manifest)
    model_index=[dict(path=rel(p),sha256=h,bytes=p.stat().st_size) for p,h in zip(ordered,hashes) if p.suffix=='.pt']
    write_csv(MILESTONE/'GP_checkpoints.csv',model_index)
    archive=MILESTONE/'evidence_and_GP_models.zip'
    temporary=MILESTONE/'evidence_and_GP_models.zip.partial'
    if temporary.exists():
        raise ValueError('Unexpected partial archive; inspect before continuing')
    print('Archiving',round(manifest['byte_count']/1024**2,1),'MiB',flush=True)
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,path in names.items():
            if sha(path)!=manifest['files_sha256'][name]:
                raise ValueError('Source changed during archive')
            z.write(path,name)
    temporary.replace(archive)
    print('Verifying every archived member',flush=True)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(names):
            raise ValueError('Incomplete archive')
        for name in z.namelist():
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(2**20),b''):
                    h.update(chunk)
            if h.hexdigest()!=manifest['files_sha256'][name]:
                raise ValueError('Archive bytes changed: '+name)
    receipt=dict(status='SEALED_AND_VERIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
        file_count=len(names),GP_checkpoint_files=len(model_index),archive_bytes=archive.stat().st_size,
        V13_warm_trajectories=32,V13_cold_trajectories=32,original_files_modified=False,
        all_registered_scientific_sources_and_protected_inputs_verified=True,
        artifact_sha256={n:sha(MILESTONE/n) for n in ('manifest.json','GP_checkpoints.csv','evidence_and_GP_models.zip')})
    write_json(MILESTONE/'receipt.json',receipt)
    return receipt


def main():
    protocols=[read(ROOT/'experiments/runs'/v/'protocol.json') for v in VARIANTS]
    for p in protocols:
        if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
            raise ValueError('Protocol digest mismatch')
    registered=registrations(protocols)
    audit=export(protocols,registered)
    print(json.dumps(audit),flush=True)
    receipt=seal(protocols,registered)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
