"""Append both registered publication cohorts to the frozen historical index."""
from pathlib import Path
from hashlib import sha256
import csv,json,sys
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent';OUT=ROOT/'experiments/reports/paper_v12_subagent'
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12 import study as api
from paper_v12_subagent import study as agent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return sha256(p.read_bytes()).hexdigest()
def main():
    with (ROOT/'paper/v12/data/historical_runs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    legacy=len(rows);fields=list(rows[0])+['execution_backend','full_registered_budget_complete','safe_execution_error','derived_record_note']
    unverified_initialization_rows=0
    for r in rows:
        r.update(execution_backend='historical_see_protocol',full_registered_budget_complete='',safe_execution_error='',derived_record_note='')
        if r['initial_count_source']=='assumed_four_not_verified':
            # Keep the frozen original index unchanged, but do not propagate its
            # fallback of four into new query-index or matching claims.
            for key in ('initial_queries','post_initial_queries','first_best_query','initial_ids_sha256','comparability_key','SearchAUC_stored_scale'):
                r[key]=''
            r['metric_reconstruction']='INITIALIZATION_DEPENDENT_FIELDS_WITHHELD'
            r['derived_record_note']='Legacy inventory did not verify the initialization budget; derived query indices and pairing keys are withheld here. Original inventory and native state remain unchanged.'
            unverified_initialization_rows+=1
    seen={r['source'] for r in rows};unit={d['task']:d['unit'] for d in read(P/'data/evidence.json')['datasets']}
    for module,backend in ((api,'configured_API'),(agent,'codex_subagent')):
        protocol_path=module.directory(ROOT)/'protocol.json';protocol=read(protocol_path)
        for task,seed,arm in module.JOBS:
            path=module.run_path(ROOT,task,seed,arm)/'state.json';source=path.relative_to(ROOT).as_posix()
            if source in seen:raise ValueError('Duplicate state path')
            s=read(path) if path.exists() else {};hist=s.get('history',[]);ys=[r['raw_value'] for r in hist]
            init=[r['candidate_id'] for r in hist[:4]];nq=len(s.get('decisions',[]));complete=s.get('status')=='COMPLETE' and len(ys)==20 and nq==16
            curve=[max(ys[:i]) for i in range(1,len(ys)+1)];error_path=path.parent/'execution_error.json';err=read(error_path) if error_path.exists() else {}
            comparison=dict(task=task,protocol=protocol['protocol_sha256'],initial_ids=init,initial=4,queries=nq,unit=unit[task],metric_field='raw_value')
            r={k:'' for k in fields};r.update(source=source,source_sha256=sha(path) if path.exists() else '',cohort=protocol['variant'],task=task,method=arm,
                native_seed=0,initialization_seed=seed,knowledge_condition='own_observations_and_public_semantics' if arm!='bo' else 'no_LLM',
                status=s.get('status','NOT_STARTED'),measured_points=len(ys),initial_queries=4,initial_count_source='registered_protocol',post_initial_queries=nq,declared_post_initial_queries=16,
                metric_field='raw_value',objective_unit=unit[task],SearchAUC_stored_scale=sum(curve[4:])/16 if complete else '',
                final_best_stored_scale=max(ys) if ys else '',first_best_query=max(0,ys.index(max(ys))+1-4) if ys else '',
                protocol_sha256=protocol['protocol_sha256'],protocol_path=protocol_path.relative_to(ROOT).as_posix(),protocol_file_sha256=sha(protocol_path),
                model=protocol['model'] if arm!='bo' else 'none',initial_ids_sha256=api.digest(init),comparability_key=api.digest(comparison),
                evidence_class='registered_publication_cohort_not_external_test',metric_reconstruction='PASS_FULL_MEASURED_BUDGET' if complete else 'AUC_WITHHELD_INCOMPLETE_REGISTERED_BUDGET',
                execution_backend=backend if arm!='bo' else 'none',full_registered_budget_complete=complete,
                safe_execution_error=err.get('safe_message',err.get('kind','')))
            rows.append(r);seen.add(source)
    receipt=dict(legacy_state_records=legacy,registered_API_cells=71,registered_subagent_cells=68,total_catalogue_rows=len(rows),unverified_initialization_rows=unverified_initialization_rows,
        missing_state_files=sum(not r['source_sha256'] for r in rows),scope='Frozen legacy catalogue plus every original API and subagent publication cell, including not-started/partial cells. Units and cohorts must not be pooled.')
    for base in (OUT,P/'data'):
        base.mkdir(parents=True,exist_ok=True)
        with (base/'all_experiment_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        (base/'catalogue_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
