"""Final study custody check. No optimization or new simulator queries."""
from __future__ import annotations
from hashlib import sha256
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from chem.benchmarks.suzuki_runner import check_protocol, read_json, validate_response, engine_files
from chem.benchmarks.suzuki_domain import digest
from chem.decision.replay import write_json_atomic

ROOT=Path('data/runs/suzuki_baselines')
RESEARCH=Path('research/suzuki_baselines')


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=check_protocol(ROOT)
    result=read_json(RESEARCH/'results.json')
    manifest=read_json(ROOT/'manifest.json')
    assert len(manifest)==60 and {r['case'] for r in manifest}=={1,2,3,4}
    assert result['counts']['completed_runs']==60 and result['counts']['queries']==1200
    assert result['counts']['llm_responses']==640 and result['counts']['hard_domain_violations']==result['counts']['duplicate_queries']==0
    assert result['hebo_precision_check']['completed_runs']==20
    assert result['hebo_precision_check']['measured_float64_gp_decisions']==320
    assert result['hebo_precision_check']['oracle_queries']==400
    audit=read_json(ROOT/'independent_selection_audit.json')
    assert audit['decisions']==640 and audit['exact_point_matches']+audit['machine_precision_ties']==640
    assert audit['protocol_sha256']==protocol['protocol_sha256']
    for run_id,expected in audit['input_state_sha256'].items():
        assert file_hash(ROOT/'runs'/run_id/'state.json')==expected
    for row in audit['rows']:
        assert row['status']=='PASS'
        if row['machine_precision_tie']:
            d=row['tie_diagnostic']
            assert d['absolute_score_gap']<=d['score_equivalence_bound_64_ulp']
    independent=read_json(RESEARCH/'independent_metrics.json')
    primary={r['run_id']:r for r in result['runs']}
    repaired={r['run_id']:r for r in read_json(RESEARCH/'hebo_precision_results.json')['runs']}
    for row in independent['runs']:
        source=(ROOT/'runs' if row['role']=='primary' else ROOT/'hebo_default64')/row['run_id']/'state.json'
        assert file_hash(source)==row['state_sha256']
        actual=(primary if row['role']=='primary' else repaired)[row['run_id']]
        assert abs(row['search_auc_percent']/100-actual['search_auc'])<1e-12
        assert abs(row['final_best_percent']/100-actual['final_best_yield'])<1e-12
    answer_count=0
    for batch in range(1,17):
        raw_path=RESEARCH/f'assistant_round{batch:02d}.json'
        expanded_path=raw_path.with_suffix('.expanded.json')
        raw,expanded=read_json(raw_path),read_json(expanded_path)
        receipt=read_json(expanded_path.with_suffix('.receipt.json'))
        assert receipt['answers_file_sha256']==file_hash(expanded_path)
        assert len(raw['answers'])==len(expanded['responses'])==40
        for a in expanded['responses']:
            expected=dict(raw['answers'][a['run_id']])
            expected.update({k:a[k] for k in ('run_id','iteration','context_sha256')})
            assert expected==a and a['iteration']==batch
            req=read_json(ROOT/'requests'/a['context_sha256']/'request.json')
            assert validate_response(a,req)==read_json(ROOT/'requests'/a['context_sha256']/'response.json')
            answer_count+=1
    assert answer_count==640
    hebo=read_json(RESEARCH/'hebo_numerical_audit.json')
    assert hebo['decisions']==hebo['same_proposals']==320
    assert hebo['steps_with_random_prediction_fallback']==5
    tests={}
    for name in ('unit_tests','regression_tests'):
        suites=ET.parse(ROOT/(name+'.xml')).getroot()
        count={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
        assert count['failures']==count['errors']==count['skipped']==0
        tests[name]=count
    assert tests['unit_tests']['tests']==8 and tests['regression_tests']['tests']==70
    pip=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True,check=True)
    historical=subprocess.run([str(Path('.venv-reproduce/Scripts/python.exe').resolve()),'-c',
        'from chem.decision.evaluation import engine_fingerprint; print(engine_fingerprint())'],text=True,capture_output=True,check=True).stdout.strip()
    assert historical==read_json('research/chapter5/validation_receipt.json')['engine_sha256']
    sources=[*Path('chem/benchmarks').glob('*.py'),*Path('scripts').glob('*suzuki*.py'),Path('tests/test_suzuki_baselines.py')]
    for p in sources: ast.parse(p.read_text(encoding='utf8'))
    for p in (RESEARCH/'results_tables.md',RESEARCH/'hebo_precision_comparison.md',Path('docs/suzuki_three_baselines.md')):
        assert '\ufffd' not in p.read_text(encoding='utf8')
    figure_files=list((RESEARCH/'figures').glob('suzuki_learning_curves.*'))
    assert {p.suffix for p in figure_files}=={'.png','.pdf','.svg'}
    files={*sources,Path('docs/suzuki_three_baselines.md'),Path('README.md')}
    for folder in (ROOT,RESEARCH):
        files.update(p for p in folder.rglob('*') if p.is_file() and p.name not in ('delivery_artifacts.json','validation_receipt.json') and '__pycache__' not in p.parts)
    inventory={'files':[{'path':p.as_posix(),'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(files)],
               'excludes':'this inventory and the final self-referencing receipt'}
    write_json_atomic(ROOT/'delivery_artifacts.json',inventory)
    receipt={'status':'COMPLETE_WITH_DISCLOSED_LIMITATIONS','scope':'only Suzuki Cases1-4; implementation, execution and evidence checks, not universal scientific superiority',
             'primary_runs':60,'precision_check_runs':20,'total_oracle_queries':1600,'actual_assistant_logical_responses':answer_count,
             'assistant_batches':16,'independent_api_calls':False,'shared_llm_context':True,
             'independently_recomputed_metric_rows':len(independent['runs']),
             'independent_llm_decisions':audit['decisions'],'exact_point_matches':audit['exact_point_matches'],
             'machine_precision_ties':audit['machine_precision_ties'],
             'native_hebo_exact_replayed_decisions':320,'native_hebo_steps_with_numerical_fallback':5,
             'repaired_hebo_measured_float64_decisions':320,'repaired_hebo_fallback_steps':0,
             'repaired_hebo_resume_spot_checks':read_json(RESEARCH/'hebo_precision_resume_check.json'),
             'tests':tests,'pip_check':pip.stdout.strip(),'historical_engine_sha256_in_original_environment':historical,
             'current_engine_files_sha256':engine_files(),'utf8_ast_checked_files':len(sources),
             'exported_figure_formats':['png','pdf','svg'],'visual_review':'PNG inspected: all four panels, axes, legend and seed-SD bands readable; native fallback and shared-chat caveats carried into title',
             'jit_basis':'No JIT/torch.compile calls in study or HEBO source; prebuilt pymoo extensions confirmed; worker warm-up precedes optimization. No global compiler instrumentation is claimed.',
             'inventory_files':len(inventory['files']),'inventory_bytes':sum(r['bytes'] for r in inventory['files']),
             'artifact_inventory_sha256':file_hash(ROOT/'delivery_artifacts.json'),'protocol_sha256':protocol['protocol_sha256'],
             'limitations':result['limitations']}
    write_json_atomic(RESEARCH/'validation_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('limitations','repaired_hebo_resume_spot_checks','current_engine_files_sha256')},ensure_ascii=False))


if __name__=='__main__': main()
