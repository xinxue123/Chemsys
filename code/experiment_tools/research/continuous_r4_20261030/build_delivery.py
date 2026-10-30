"""Verify R4 exports and package the immutable incremental study."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import csv
import json
import re
import numpy as np
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'data/runs/continuous_r4_20261030'
NAME='SnAr_HPLC_R4_Comparison'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=read(HERE/'protocol.json');audit=read(HERE/'validation_receipt.json');assert audit['status']=='PASS'
    assert audit['budgeted_added_choices_verified']==32 and audit['baselines_rerun']==0
    protected=read(HERE/'protected_snapshot.json')
    for pins in [protected,p['source_sha256'],p['runtime_sha256'],p['input_sha256'],p['bootstrap_source_sha256']]:
        for name,value in pins.items():assert sha(ROOT/name)==value,name
    results=read(HERE/'results.json');baseline=read(ROOT/'research/continuous_r3_20261030/results.json')['metrics']
    assert results['metrics'][:len(baseline)]==baseline
    tables=rows_checked=0;error_max=0.;preGP_receipts=0
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        state=read(RUN/'runs'/(task+'_New_Agent_R4')/'state.json');pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
        ids=pool['ids'];ix={key:j for j,key in enumerate(ids)};scale=pool['objective_scale']
        with np.load(ROOT/f'data/benchmarks/continuous_20261029/{task}/oracle.npz') as f:truth=f['raw_values'].copy()
        y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y)
        metric=next(v for v in results['new_metrics'] if v['task']==task)
        assert abs(float(best[4:].mean())-metric['SearchAUC'])<1e-9 and float(best[-1])==metric['final_best']
        assert int(np.flatnonzero(y==best[-1])[0]+1)==metric['first_final_best_total']
        for t,event in enumerate(state['decisions'],1):
            folder=RUN/'runs'/state['run_id']/'rounds';commit=read(folder/f'{t:02d}_committed.json')
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={k:f[k].copy() for k in f.files}
            with (HERE/task/'candidate_tables'/f'{t:02d}_candidates.csv').open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
            assert [r['candidate_id'] for r in rows]==ids and sum(r['selected']=='True' for r in rows)==1
            assert rows[ix[event['selected_id']]]['selected']=='True' and a['eligible'][ix[event['selected_id']]]
            assert int(a['available'].sum())==len(ids)-(t+3)
            shift=a['deployed_shift_std']*commit['GP_scale_y'] if t<=prefix else a['local_shift_y']
            expected={'true_objective_analysis_only':truth,'GP_mean':a['mu_y']*scale,'GP_latent_SD':a['sd_y']*scale,
                'GP_predictive_SD':a['sd_predictive_y']*scale,'base_qLogEI':a['base_qLogEI'],'guided_qLogEI':a['guided_qLogEI'],
                'deployed_mean':(a['mu_y']+shift)*scale,'local_shift':shift*scale,
                'base_loss':a['base_qLogEI'][ix[event['reference_id']]]-a['base_qLogEI']}
            for key,value in expected.items():
                export=np.array([float(r[key]) for r in rows]);err=float(np.max(abs(export-value)))
                assert err<1e-9,(task,t,key,err);error_max=max(error_max,err)
            for key,source in [('available','available'),('inside_original_budget','eligible')]:assert np.array_equal([r[key]=='True' for r in rows],a[source])
            if t>prefix:
                assert np.array_equal([r['validation_candidate']=='True' for r in rows],a['validation_candidate'])
                proposal=folder/f'{t:02d}_blind_proposal.json';assert proposal.stat().st_mtime_ns<=(folder/f'{t:02d}_committed.json').stat().st_mtime_ns
                assert read(proposal)['GP_current_round_computed'] is False;preGP_receipts+=1
            tables+=1;rows_checked+=len(rows)
        with (HERE/task/'trajectory.csv').open(encoding='utf-8-sig') as stream:trajectory=list(csv.DictReader(stream))
        assert [r['candidate_id'] for r in trajectory]==[d['selected_id'] for d in state['decisions']]
        assert [float(r['objective']) for r in trajectory]==[d['raw_value'] for d in state['decisions']]
    assert tables==32 and rows_checked==81648 and preGP_receipts==25
    write(HERE/'export_checks.json',{'status':'PASS','candidate_tables_checked':tables,'candidate_rows_checked':rows_checked,
        'maximum_export_error':error_max,'old_metric_rows_unchanged':len(baseline),'new_metrics_independently_recomputed':2,'pre_GP_stage_files_checked':preGP_receipts})

    pdf=HERE/(NAME+'.pdf');tex=HERE/(NAME+'.tex');doc=fitz.open(pdf);assert len(doc)==2
    body=tex.read_text(encoding='utf-8').split(r'\begin{document}',1)[1];body=re.sub(r'\\hsize=[.\d]+\\hsize','',body)
    values=set(re.findall(r'[+-]?\d+\.\d{3,}',body));extracted='\n'.join(page.get_text() for page in doc)
    assert all(v in extracted for v in values)
    fonts={xref for page in doc for xref,*_ in page.get_fonts(full=True)};assert all(len(doc.extract_font(x)[3])>0 for x in fonts)
    log=(ROOT/'outputs/continuous_r4_typeset'/f'{NAME}.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and 'Missing character:' not in log
    for i,page in enumerate(doc,1):
        assert (ROOT/f'outputs/continuous_r4_typeset/pages/page-{i}.png').is_file()
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4];assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
    visual=read(HERE/'visual_review.json');assert visual['PDF_sha256']==sha(pdf) and visual['pages_passed']==[1,2]
    write(HERE/'document_checks.json',{'status':'PASS','pages':2,'decimal_items_checked':len(values),'all_fonts_embedded':True,
        'missing_glyphs':0,'overfull_boxes':0,'visual_pages_passed':[1,2],'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex)})
    unit_log=ROOT/'outputs/continuous_r4_unit.log';unit=unit_log.read_text(encoding='utf-8-sig')
    assert 'Ran 17 tests' in unit and '\nOK' in unit and 'FAILED' not in unit
    write(HERE/'unit_test_receipt.json',{'status':'PASS','tests_passed_before_freeze':17,
        'source_sha256':sha(ROOT/'tests/test_continuous_r4.py'),'log_sha256':sha(unit_log),
        'scope':'behavior tests include blind order, two-point timing, scheduled validation, pause recovery and old/new inheritance; not benchmark superiority'})

    tokens=Counter();attempts=0;contexts=[];repairs=[]
    for path in sorted((RUN/'requests').glob('*/context.json')):
        contexts.append(read(path));responses=list(path.parent.glob('attempt*/response.json'));accepted=read(path.parent/'validated.json')
        assert accepted['received_responses']==len(responses)
        for response in responses:
            attempts+=1
            for key in ['prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens']:tokens[key]+=read(response)['usage'].get(key,0)
        for failure in path.parent.glob('attempt*/validation_error.json'):repairs.append({'context':path.parent.name,'error':read(failure)['error']})
    assert len(contexts)==8 and attempts==len(contexts)+len(repairs)
    write(HERE/'LLM_usage.json',{'model':'deepseek-flash','new_unique_logical_calls':len(contexts),'response_attempts':attempts,
        'new_tokens':dict(tokens),'schema_or_format_repairs':repairs,'LLM_call_budget_matched_to_baselines':False,
        'public_GP_information':False,'credentials_included':False})

    members={};excluded={pdf.name,'release_inventory.json','release_receipt.json','delivery_receipt.json','api_preflight.json'}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.suffix in {'.json','.csv','.md','.py','.tex','.pdf'} and path.name not in excluded:members[path.relative_to(HERE).as_posix()]=path
    for name in p['source_sha256']:
        path=ROOT/name
        if path.suffix=='.py':members['code/'+path.relative_to(ROOT).as_posix()]=path
    reporter=ROOT/'chem/benchmarks/continuous_r4_report.py';members['code/'+reporter.relative_to(ROOT).as_posix()]=reporter
    for folder in ['requests','runs']:
        for path in sorted((RUN/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json','.npz'}:members['evidence/'+path.relative_to(RUN).as_posix()]=path
    for name,path in members.items():
        assert 'config.txt' not in name.lower() and path.suffix not in {'.pt','.pem','.key'}
        if path.suffix in {'.json','.py','.md','.tex'}:
            content=path.read_text(encoding='utf-8')
            assert not re.search(r'"(?:authorization|api_key|apikey)"\s*:',content,re.I),name
            assert not re.search(r'\bsk-[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9]',content),name
    archive=HERE/'SnAr_HPLC_R4_Source_and_Results.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('PACKAGE_README.txt','Incremental R4 code for original chemist_system; original modules, input data and environment required.\n'
            'Includes 32 complete candidate CSVs and native pre-GP proposal, pre-outcome choice, array and LLM evidence.\n'
            'Two initial4+16 trajectories, seven reused prefix positions and 25 fresh decisions. No comparator reruns.\n'
            'TeX compiles with two *_summary_figure.pdf assets; final report PDF delivered separately.\n'
            'No credentials, config or GP checkpoint weights.\n'
            'SnAr improves over R3/R2 on this seed, HPLC ties. No active local validation/mean deployment/inheritance increment demonstrated.\n')
        for name,path in sorted(members.items()):z.write(path,name)
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(members)+1
        for name,path in members.items():assert sha256(z.read(name)).hexdigest()==sha(path),name
    write(HERE/'delivery_receipt.json',{'status':'COMPLETE_AND_VERIFIED','created_utc':datetime.now(timezone.utc).isoformat(),
        'completed_trajectories':2,'budgeted_added_positions':32,'protected_prefix_positions':7,'fresh_decisions':25,
        'baselines_rerun':0,'protected_files_unchanged':len(protected),'pre_run_unit_tests_passed':17,
        'trajectory_audit':'PASS','exports':'PASS','PDF_checks':'PASS','frozen_scientific_source_unchanged':True,
        'scheduled_local_validation_activations':0,'local_mean_deployments':0,'inheritance_activations':0,
        'archive_members_verified':len(members)+1,'PDF_sha256':sha(pdf),'ZIP_sha256':sha(archive),
        'scope':'Single-seed protected continuation; SnAr improves over R3/R2, HPLC ties; inactive modules lack performance attribution.'})
    paths={x for x in HERE.rglob('*') if x.is_file()};paths.update(x for x in RUN.rglob('*') if x.is_file())
    paths.update(ROOT/name for name in p['source_sha256']);paths.add(reporter)
    paths.update((ROOT/'outputs').glob('continuous_r4_*.log'));paths.update(ROOT/f'outputs/continuous_r4_typeset/pages/page-{i}.png' for i in [1,2])
    inventory={x.relative_to(ROOT).as_posix():sha(x) for x in sorted(paths) if x.name not in {'release_inventory.json','release_receipt.json'}
        and x.suffix not in {'.pyc','.tmp','.lock'} and '__pycache__' not in x.parts}
    write(HERE/'release_inventory.json',inventory)
    for name,value in inventory.items():assert sha(ROOT/name)==value,name
    for name,value in protected.items():assert sha(ROOT/name)==value,name
    write(HERE/'release_receipt.json',{'status':'PASS','files_verified':len(inventory),'inventory_sha256':sha(HERE/'release_inventory.json'),
        'protected_files_unchanged':len(protected),'baseline_reruns':0})
    print(json.dumps({'status':'COMPLETE_AND_VERIFIED','sealed_files':len(inventory),'archive_members':len(members)+1,
        'archive_bytes':archive.stat().st_size,'candidate_rows_checked':rows_checked,'new_LLM_calls':len(contexts),'PDF':str(pdf)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
