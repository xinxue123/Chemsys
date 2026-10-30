"""Verify R3 exports and package the frozen incremental code and native evidence."""
from collections import Counter
from datetime import datetime,timezone
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
RUN=ROOT/'data/runs/continuous_r3_20261030'
NAME='SnAr_HPLC_R3_Comparison'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,v):Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    protocol=read(HERE/'protocol.json');audit=read(HERE/'validation_receipt.json')
    assert audit['status']=='PASS' and audit['budgeted_added_choices_verified']==32 and audit['baselines_rerun']==0
    protected=read(HERE/'protected_snapshot.json')
    for pinset in [protected,protocol['source_sha256'],protocol['input_sha256'],protocol['bootstrap_source_sha256']]:
        for name,expected in pinset.items():assert sha(ROOT/name)==expected,name
    results=read(HERE/'results.json');old=read(ROOT/'research/continuous_r2_20261030/results.json')['metrics']
    assert results['metrics'][:len(old)]==old
    tables=rows_checked=0;max_error=0.;actual_count=0
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        state=read(RUN/'runs'/(task+'_New_Agent_R3')/'state.json');pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
        scale=pool['objective_scale'];ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
        with np.load(ROOT/f'data/benchmarks/continuous_20261029/{task}/oracle.npz') as f:truth=f['raw_values'].copy()
        y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y)
        metric=next(row for row in results['new_metrics'] if row['task']==task)
        assert abs(metric['SearchAUC']-float(best[4:].mean()))<1e-9 and metric['final_best']==float(best[-1])
        assert metric['first_final_best_total']==int(np.flatnonzero(y==best[-1])[0]+1)
        ceiling=float((best[4:4+prefix].sum()+(16-prefix)*truth.max())/16)
        assert metric['protected_prefix_SearchAUC_upper_bound']==ceiling
        for t,d in enumerate(state['decisions'],1):
            folder=RUN/'runs'/state['run_id']/'rounds';c=read(folder/f'{t:02d}_committed.json')
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={key:f[key].copy() for key in f.files}
            with (HERE/task/'candidate_tables'/f'{t:02d}_candidates.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
            assert [row['candidate_id'] for row in rows]==ids
            assert sum(row['selected']=='True' for row in rows)==1
            assert rows[ix[d['selected_id']]]['selected']=='True' and a['eligible'][ix[d['selected_id']]]
            assert int(a['available'].sum())==len(ids)-(t+3)
            shift=a['deployed_shift_std']*c['GP_scale_y'] if t<=prefix else a['local_shift_y']
            expected={'true_objective_analysis_only':truth,'GP_mean':a['mu_y']*scale,
                'GP_latent_SD':a['sd_y']*scale,'GP_predictive_SD':a['sd_predictive_y']*scale,
                'base_qLogEI':a['base_qLogEI'],'guided_qLogEI':a['guided_qLogEI'],
                'deployed_mean':(a['mu_y']+shift)*scale,'local_shift':shift*scale,
                'base_loss':a['base_qLogEI'][ix[d['reference_id']]]-a['base_qLogEI']}
            for key,value in expected.items():
                export=np.array([float(row[key]) for row in rows]);error=float(np.max(abs(export-value)))
                assert error<1e-9,(task,t,key,error);max_error=max(max_error,error)
            for key,source in [('available','available'),('inside_original_budget','eligible')]:
                assert np.array_equal([row[key]=='True' for row in rows],a[source])
            if t>prefix:
                assert np.array_equal([float(row['local_coefficient']) for row in rows],a['local_coefficient'])
                assert {row['candidate_id'] for row in rows if row['in_LLM_card']=='True'}=={row['candidate_id'] for row in c['candidate_card']}
                actual_count+=d['local_feedback']['actual_local_update'] is not None
            tables+=1;rows_checked+=len(rows)
        with (HERE/task/'trajectory.csv').open(encoding='utf-8-sig') as stream:trajectory=list(csv.DictReader(stream))
        assert [row['candidate_id'] for row in trajectory]==[d['selected_id'] for d in state['decisions']]
        for row,d in zip(trajectory,state['decisions']):assert float(row['objective'])==d['raw_value']
    assert tables==32 and rows_checked==81648 and actual_count==4
    write(HERE/'export_checks.json',{'status':'PASS','candidate_tables_checked':tables,'candidate_rows_checked':rows_checked,
        'maximum_export_error':max_error,'old_metric_rows_unchanged':len(old),'metrics_independently_recomputed':2})

    pdf=HERE/(NAME+'.pdf');tex=HERE/(NAME+'.tex');doc=fitz.open(pdf);assert len(doc)==2
    text='\n'.join(page.get_text() for page in doc);body=tex.read_text(encoding='utf-8').split(r'\begin{document}',1)[1]
    body=re.sub(r'\\hsize=[.\d]+\\hsize','',body);values=set(re.findall(r'[+-]?\d+\.\d{3,}',body))
    assert all(value in text for value in values)
    fonts={xref for page in doc for xref,*_ in page.get_fonts(full=True)}
    assert all(len(doc.extract_font(xref)[3])>0 for xref in fonts)
    log=(ROOT/'outputs/continuous_r3_typeset'/f'{NAME}.log').read_text(encoding='utf-8',errors='replace')
    assert 'Missing character:' not in log and 'Overfull' not in log
    for i,page in enumerate(doc,1):
        picture=ROOT/f'outputs/continuous_r3_typeset/pages/page-{i}.png';assert picture.is_file()
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4];assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
    visual=read(HERE/'visual_review.json');assert visual['PDF_sha256']==sha(pdf) and visual['pages_passed']==[1,2]
    write(HERE/'document_checks.json',{'status':'PASS','pages':2,'decimal_items_checked':len(values),
        'all_fonts_embedded':True,'missing_glyphs':0,'overfull_boxes':0,'visual_pages_passed':[1,2],
        'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex)})
    write(HERE/'unit_test_receipt.json',{'status':'PASS','tests_passed_before_scientific_freeze':12,
        'command':'.venv-official/Scripts/python.exe -X utf8 -m unittest tests.test_continuous_r3 -v',
        'test_source_sha256':sha(ROOT/'tests/test_continuous_r3.py'),
        'recovery_path_behavior_test_passed':True,'observed_live_recoveries':0})

    phases=Counter();tokens=Counter();attempts=0;contexts={}
    for path in sorted((RUN/'requests').glob('*/context.json')):
        context=read(path);phases[context['phase']]+=1;contexts[path.parent.name]=context
        for response in sorted(path.parent.glob('attempt*/response.json')):
            attempts+=1
            for key in ['prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens']:
                tokens[key]+=read(response)['usage'].get(key,0)
    assert len(contexts)==25 and attempts>=25
    repairs=[]
    for key,context in contexts.items():
        folder=RUN/'requests'/key;accepted=read(folder/'validated.json')
        assert accepted['received_responses']==len(list(folder.glob('attempt*/response.json')))
        for failure in sorted(folder.glob('attempt*/validation_error.json')):
            repairs.append({'context_sha256':key,'task':context['task'],'phase':context['phase'],
                'attempt':failure.parent.name,'validation_error':read(failure)['error']})
    assert attempts==25+len(repairs)
    write(HERE/'LLM_usage.json',{'model':'deepseek-flash','new_unique_logical_calls':25,'response_attempts':attempts,
        'phases':dict(phases),'new_tokens':dict(tokens),'format_or_schema_repairs':repairs,
        'repair_policy':'Existing bounded transport repair, before selected outcome lookup; all failed replies retained.',
        'protected_prefix_budget_positions_replayed':7,
        'LLM_call_budget_matched_to_baselines':False,'credentials_included':False})

    members={}
    excluded={pdf.name,'release_inventory.json','release_receipt.json','delivery_receipt.json','api_preflight.json'}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.suffix in {'.json','.csv','.md','.py','.tex','.pdf'} and path.name not in excluded:
            members[path.relative_to(HERE).as_posix()]=path
    for name in protocol['source_sha256']:
        path=ROOT/name
        if path.suffix=='.py':members['code/'+path.relative_to(ROOT).as_posix()]=path
    reporter=ROOT/'chem/benchmarks/continuous_r3_report.py';members['code/'+reporter.relative_to(ROOT).as_posix()]=reporter
    for folder in ['requests','runs']:
        for path in sorted((RUN/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz'}:members['evidence/'+path.relative_to(RUN).as_posix()]=path
    for name,path in members.items():
        assert 'config.txt' not in name.lower() and path.suffix not in {'.pt','.pem','.key'}
        if path.suffix in {'.json','.py','.md','.tex'}:
            content=path.read_text(encoding='utf-8')
            assert not re.search(r'"(?:authorization|api_key|apikey)"\s*:',content,re.I),name
            assert not re.search(r'\bsk-[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9]',content),name
    archive=HERE/'SnAr_HPLC_R3_Source_and_Results.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('PACKAGE_README.txt',
            'Incremental R3 code for the original chemist_system checkout. Original modules and input data remain required.\n'
            'Includes 32 complete candidate CSVs and native decision/LLM evidence.\n'
            'Each trajectory has initial4 +16 queries. Seven of 32 query positions replay the R2 protected prefix;25 are new.\n'
            'TeX compiles independently with the two *_summary_figure.pdf assets. Final report PDF is delivered separately.\n'
            'No API credentials, config, or GP checkpoint weights are included.\n'
            'R3 ties R2 on both SearchAUC and final best; technical validation is not scientific superiority.\n')
        for name,path in sorted(members.items()):z.write(path,name)
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(members)+1
        for name,path in members.items():assert sha256(z.read(name)).hexdigest()==sha(path),name
    write(HERE/'delivery_receipt.json',{'status':'COMPLETE_AND_VERIFIED','created_utc':datetime.now(timezone.utc).isoformat(),
        'completed_trajectories':2,'budgeted_added_positions':32,'protected_prefix_positions':7,'fresh_decisions':25,
        'baselines_rerun':0,'protected_files_unchanged':len(protected),'pre_run_unit_tests_passed':12,
        'full_trajectory_audit':'PASS','export_checks':'PASS','PDF_checks':'PASS',
        'scientific_source_unchanged_after_freeze':True,'actual_local_updates':4,'local_guidance_selection_changes':0,
        'actual_recoveries':0,'archive_members_verified':len(members)+1,'PDF_sha256':sha(pdf),'ZIP_sha256':sha(archive),
        'scope':'Single-seed protected-prefix continuation. R3 equals R2 on both primary metrics; no search increment shown.'})
    paths={p for p in HERE.rglob('*') if p.is_file()}
    paths.update(p for p in RUN.rglob('*') if p.is_file());paths.update(ROOT/name for name in protocol['source_sha256'])
    paths.add(reporter);paths.update((ROOT/'outputs').glob('continuous_r3_*.log'))
    paths.update(ROOT/f'outputs/continuous_r3_typeset/pages/page-{i}.png' for i in [1,2])
    inventory={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)
        if p.name not in {'release_inventory.json','release_receipt.json'}
        and p.suffix not in {'.pyc','.tmp','.lock'} and '__pycache__' not in p.parts}
    write(HERE/'release_inventory.json',inventory)
    for name,expected in inventory.items():assert sha(ROOT/name)==expected,name
    for name,expected in protected.items():assert sha(ROOT/name)==expected,name
    write(HERE/'release_receipt.json',{'status':'PASS','files_verified':len(inventory),
        'inventory_sha256':sha(HERE/'release_inventory.json'),'protected_files_unchanged':len(protected),'baseline_reruns':0})
    print(json.dumps({'status':'COMPLETE_AND_VERIFIED','sealed_files':len(inventory),'archive_members':len(members)+1,
        'archive_bytes':archive.stat().st_size,'candidate_rows_checked':rows_checked,'new_LLM_calls':len(contexts),
        'PDF':str(pdf)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
