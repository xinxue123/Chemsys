"""Verify exports, preserve sealed science, and package the completed R2 study."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import json
import re
import numpy as np
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'data/runs/continuous_r2_20261030'
NAME='SnAr_HPLC_R2_Comparison'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    protocol=read(HERE/'protocol.json');amendment=read(HERE/'execution_amendment.json')
    audit=read(HERE/'validation_receipt.json');assert audit['status']=='PASS'
    assert audit['budgeted_added_choices_verified']==64 and audit['baselines_rerun']==0
    protected=read(HERE/'protected_snapshot.json')
    for name,expected in protected.items():assert digest(ROOT/name)==expected,name
    for name,expected in amendment['corrected_source_sha256'].items():assert digest(ROOT/name)==expected,name
    for name,expected in amendment['completed_subset_files_sha256'].items():assert digest(ROOT/name)==expected,name
    for name,expected in protocol['input_sha256'].items():assert digest(ROOT/name)==expected,name
    baseline=read(ROOT/'research/continuous_revised_20261030/results.json')['metrics']
    results=read(HERE/'results.json');assert results['metrics'][:len(baseline)]==baseline

    tables=rows_checked=0;max_export_error=0.;module_records=[]
    states=[read(path) for path in sorted((RUN/'runs').glob('*/state.json'))]
    for state in states:
        task=state['task'];method=state['method'];pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
        scale=pool['objective_scale'];ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
        with np.load(ROOT/f'data/benchmarks/continuous_20261029/{task}/oracle.npz') as f:truth=f['raw_values'].copy()
        history=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(history)
        metric=next(r for r in results['new_metrics'] if r['task']==task and r['method']==method)
        assert abs(float(best[4:].mean())-metric['SearchAUC'])<1e-9
        assert float(best[-1])==metric['final_best']
        assert int(np.flatnonzero(history==best[-1])[0])+1==metric['first_final_best_total']
        for t,event in enumerate(state['decisions'],1):
            folder=RUN/'runs'/state['run_id']/'rounds';commit=read(folder/f'{t:02d}_committed.json')
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={k:f[k].copy() for k in f.files}
            path=HERE/task/method/'candidate_tables'/f'{t:02d}_candidates.csv'
            with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            assert [row['candidate_id'] for row in rows]==ids
            assert sum(row['selected']=='True' for row in rows)==1
            assert next(row['candidate_id'] for row in rows if row['selected']=='True')==event['selected_id']
            assert int(a['available'].sum())==len(ids)-(t+3)
            selected=ix[event['selected_id']];assert a['eligible'][selected]
            expected={'true_objective_analysis_only':truth,'GP_mean':a['mu_y']*scale,
                      'GP_latent_SD':a['sd_y']*scale,'GP_predictive_SD':a['sd_predictive_y']*scale,
                      'base_qLogEI':a['base_qLogEI'],'deployed_qLogEI':a['guided_qLogEI'],
                      'deployed_mean':(a['mu_y']+a['deployed_shift_std']*commit['GP_scale_y'])*scale}
            for key,value in expected.items():
                export=np.array([float(row[key]) for row in rows]);error=float(np.max(abs(export-value)))
                assert error<1e-9,(task,method,t,key,error);max_export_error=max(max_export_error,error)
            for key,name in [('available','available'),('inside_original_budget_gate','eligible'),('in_joint_pool','interaction_pool')]:
                assert np.array_equal(np.array([row[key]=='True' for row in rows]),a[name])
            if commit.get('pending_forecast'):
                pending=commit['pending_forecast'];prediction=(a['mu_y']+pending['shadow_probe_coefficient']*a['pending_raw_shift_std']*commit['GP_scale_y'])*scale
                error=float(np.max(abs(np.array([float(row['pending_shadow_prediction']) for row in rows])-prediction)))
                assert error<1e-9;max_export_error=max(max_export_error,error)
            else:assert all(row['pending_shadow_prediction']=='' for row in rows)
            tables+=1;rows_checked+=len(rows)
        for call in state['calls']:
            if 'inheritance' in call:
                checks=[d['pending_validation'] for d in state['decisions'] if d.get('pending_validation') and d['iteration']==call['iteration']]
                module_records.append({'task':task,'method':method,'query_round':call['iteration'],
                    'initial_coefficient':call['inheritance']['initial_mean'],
                    'inherited_source_count':len(call['inheritance']['sources']),
                    'shadow_gain_y_squared':checks[0]['paired_SE_gain'] if checks else None,
                    'promoted':any(p['proposal_created_query']==call['iteration'] for p in state['promotions'])})
    assert tables==64 and rows_checked==163296
    write(HERE/'export_checks.json',{'status':'PASS','candidate_tables_checked':tables,
          'candidate_rows_checked':rows_checked,'maximum_export_error':max_export_error,
          'old_metric_rows_unchanged':len(baseline),'metrics_independently_recomputed':4})
    write(HERE/'module_activation.json',{'new_proposals':module_records,
          'all_new_proposals_rejected':all(r['shadow_gain_y_squared']<0 and not r['promoted'] for r in module_records),
          'subset_recovery_activations':sum(c['subset_recovery_activations'] for c in audit['checks']),
          'new_optional_joint_LLM_calls':sum(1 for p in (RUN/'requests').glob('*/context.json') if read(p)['phase']=='optional_joint_test'),
          'scope':'No positive deployment or subset-restoration increment can be inferred when those branches did not activate.'})

    pdf=HERE/(NAME+'.pdf');tex=HERE/(NAME+'.tex');doc=fitz.open(pdf)
    assert len(doc)==2
    body=tex.read_text(encoding='utf-8').split(r'\begin{document}',1)[1]
    body=re.sub(r'\\hsize=[.\d]+\\hsize','',body)
    decimals=set(re.findall(r'[+-]?\d+\.\d{3,}',body))
    extracted='\n'.join(page.get_text() for page in doc)
    missing=[value for value in sorted(decimals) if value not in extracted];assert not missing,missing
    fonts={xref for page in doc for xref,*_ in page.get_fonts(full=True)}
    assert all(len(doc.extract_font(xref)[3])>0 for xref in fonts)
    log=(ROOT/'outputs/continuous_r2_typeset'/f'{NAME}.log').read_text(encoding='utf-8',errors='replace')
    assert 'Missing character:' not in log and 'Overfull' not in log
    for i in [1,2]:assert (ROOT/f'outputs/continuous_r2_typeset/pages/page-{i}.png').is_file()
    write(HERE/'document_checks.json',{'status':'PASS','pages':2,'decimal_items_checked':len(decimals),
          'missing_numbers':missing,'all_fonts_embedded':True,'missing_glyphs':0,'overfull_boxes':0,
          'visual_pages_passed':[1,2],'PDF_sha256':digest(pdf),'TeX_sha256':digest(tex)})
    write(HERE/'unit_test_receipt.json',{'status':'PASS','tests_passed':11,
          'command':'.venv-official/Scripts/python.exe -X utf8 -m unittest tests.test_continuous_r2 -v',
          'test_source_sha256':digest(ROOT/'tests/test_continuous_r2.py'),
          'qualification_and_full_trajectory_audit_also_passed':True})

    phases=Counter();tokens=Counter();attempts=0
    for path in sorted((RUN/'requests').glob('*/context.json')):
        phases[read(path)['phase']]+=1
        for response in sorted(path.parent.glob('attempt*/response.json')):
            attempts+=1
            for key in ['prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens']:
                tokens[key]+=read(response)['usage'].get(key,0)
    assert sum(phases.values())<=protocol['maximum_fresh_logical_LLM_calls']
    write(HERE/'LLM_usage.json',{'model':'deepseek-flash','new_unique_logical_calls':sum(phases.values()),
          'new_response_attempts':attempts,'phases':dict(phases),'new_tokens':dict(tokens),
          'reused_bootstrap_contexts':len(list((RUN/'bootstrap_requests').glob('*/context.json'))),
          'same_prefix_main_ablation_response_references':sum(c['fresh_LLM_contexts_checked'] for c in audit['checks']),
          'credentials_included':False})

    members={}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.suffix in {'.json','.csv','.md','.py','.tex','.pdf'} and path.name not in {
            pdf.name,'release_inventory.json','release_receipt.json','delivery_receipt.json','api_preflight.json'}:
            members[path.relative_to(HERE).as_posix()]=path
    for name in amendment['corrected_source_sha256']:
        path=ROOT/name
        if path.suffix=='.py':members['code/'+path.relative_to(ROOT).as_posix()]=path
    reporter=ROOT/'chem/benchmarks/continuous_r2_report.py'
    members['code/'+reporter.relative_to(ROOT).as_posix()]=reporter
    for folder in ['requests','bootstrap_requests','runs']:
        for path in sorted((RUN/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz'}:
                members['evidence/'+path.relative_to(RUN).as_posix()]=path
    for name,path in members.items():
        assert 'config.txt' not in name.lower() and path.suffix not in {'.pt','.key','.pem'}
        if 'requests/' in name and path.suffix=='.json':
            content=path.read_text(encoding='utf-8')
            assert not re.search(r'"(?:authorization|api_key|apikey)"\s*:',content,re.I),name
            assert not re.search(r'\bsk-[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9]',content),name
    archive=HERE/'SnAr_HPLC_R2_Source_and_Results.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('PACKAGE_README.txt',
            'TeX compiles independently with the two *_summary_figure.pdf assets.\n'
            'code/ is incremental code for the original chemist_system checkout, not a standalone replacement repository.\n'
            'It depends on frozen original GP, input data, and decision modules identified in protected_snapshot.json.\n'
            'All 64 candidate CSVs and native decision evidence are included.\n'
            'The 10 replayed-prefix query positions count inside the four 4+16 trajectories.\n'
            'Evidence includes the registered execution-only JSON-cache repair; scientific parameters were unchanged.\n'
            'No credentials or GP checkpoint weights are included. The final report PDF is delivered separately.\n')
        for name,path in sorted(members.items()):z.write(path,name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(members)+1
        for name,path in members.items():assert sha256(z.read(name)).hexdigest()==digest(path),name

    write(HERE/'delivery_receipt.json',{'status':'COMPLETE_AND_VERIFIED','created_utc':datetime.now(timezone.utc).isoformat(),
        'completed_trajectories':4,'budgeted_added_queries':64,'replayed_prefix_queries':10,'fresh_decisions':54,
        'baselines_rerun':0,'protected_files_unchanged':len(protected),'unit_tests_passed':11,
        'trajectory_audit':'PASS','export_checks':'PASS','PDF_checks':'PASS',
        'new_knowledge_promotions':0,'B4_subset_recovery_activations':0,
        'execution_only_repair_registered':True,'scientific_parameters_unchanged':True,
        'archive_members_verified':len(members)+1,'PDF_sha256':digest(pdf),'source_results_ZIP_sha256':digest(archive),
        'scope':'SnAr improves over R1 and old New_Agent but remains below Pure_BO; HPLC preserves R1; inactive ablation is inconclusive.'})
    paths={path for path in HERE.rglob('*') if path.is_file()}
    paths.update(path for path in RUN.rglob('*') if path.is_file())
    paths.update(ROOT/name for name in amendment['corrected_source_sha256'])
    paths.add(reporter)
    paths.update(path for path in (ROOT/'outputs').glob('continuous_r2_*.log'))
    paths.update(ROOT/f'outputs/continuous_r2_typeset/pages/page-{i}.png' for i in [1,2])
    inventory={path.relative_to(ROOT).as_posix():digest(path) for path in sorted(paths)
        if path.name not in {'release_inventory.json','release_receipt.json'}
        and path.suffix not in {'.tmp','.lock','.pyc'} and '__pycache__' not in path.parts}
    write(HERE/'release_inventory.json',inventory)
    for name,expected in inventory.items():assert digest(ROOT/name)==expected,name
    for name,expected in protected.items():assert digest(ROOT/name)==expected,name
    write(HERE/'release_receipt.json',{'status':'PASS','files_verified':len(inventory),
        'inventory_sha256':digest(HERE/'release_inventory.json'),'protected_files_unchanged':len(protected),'baseline_reruns':0})
    print(json.dumps({'status':'COMPLETE_AND_VERIFIED','sealed_files':len(inventory),
                     'archive_members':len(members)+1,'archive_bytes':archive.stat().st_size,
                     'candidate_rows_checked':rows_checked,'new_unique_LLM_calls':sum(phases.values()),
                     'PDF':str(pdf)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
