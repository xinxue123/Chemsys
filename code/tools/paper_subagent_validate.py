"""Validate the new evidence, preserving the explicit full-goal completion gate."""
from pathlib import Path
from hashlib import sha256
import argparse,csv,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent';OUT=ROOT/'experiments/reports/paper_v12_subagent'
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12_subagent import study
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');a=ap.parse_args()
    protocol=study.check(ROOT,True);e=read(P/'data/evidence.json');issues=[];sources={}
    expected=set(study.JOBS);actual=set()
    for r in e['rows']:
        path=ROOT/r['source'];sources[r['source']]=sha(path)
        if sources[r['source']]!=r['source_sha256']:issues.append('State hash drift: '+r['source'])
        s=read(path);values=[x['raw_value'] for x in s['history']]
        if len(values)!=20 or len(s['decisions'])!=16 or s['status']!='COMPLETE':issues.append('Incomplete trajectory included')
        if values!=r['values'] or len(set(r['initial_ids']+r['selected_ids']))!=20:issues.append('Observation/choice custody')
        if not np.isclose(np.maximum.accumulate(values)[4:].mean(),r['SearchAUC'],rtol=0,atol=1e-10):issues.append('SearchAUC mismatch')
        if max(values)!=r['final_best'] or max(0,values.index(max(values))+1-4)!=r['first_best_query']:issues.append('Final outcome metric mismatch')
        if r['cohort']=='codex_subagent':actual.add((r['task'],r['initialization_seed'],r['arm']))
    if not actual<=expected or len(actual)!=e['completed_trajectories']:issues.append('Registered matrix mismatch')
    if e['reused_BO_controls']!=32 or sum(r['arm']=='bo' for r in e['rows'])!=32:issues.append('BO control denominator')
    source_rows={r['source']:r for r in e['rows']}
    for pair in e['paired']:
        x=source_rows[pair['source']];y=source_rows[pair['control_source']]
        if x['initial_ids']!=y['initial_ids'] or x['values'][:4]!=y['values'][:4]:issues.append('Unequal paired initialization')
        if not np.isclose(pair['AUC_delta'],x['SearchAUC']-y['SearchAUC'],rtol=0,atol=1e-10):issues.append('Paired metric mismatch')
        if pair['comparison'] in ('V12_vs_RBO','KF_removal','refresh_removal') and {x['cohort'],y['cohort']}!={'codex_subagent'}:issues.append('Cross-backend algorithm comparison')
    text=(P/'main.tex').read_text(encoding='utf-8');methods=(P/'methods.tex').read_text(encoding='utf-8');bib=(P/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+)',bib))
    for group in re.findall(r'\\cite\w*\{([^}]+)\}',text+methods):
        if not set(group.split(','))<=keys:issues.append('Missing bibliography key')
    abstract=(P/'generated/abstract.tex').read_text(encoding='utf-8')
    if len(abstract.split())>150:issues.append('Abstract exceeds150words')
    main=text.split(r'\section*{Methods}')[0]
    main=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}','',main,flags=re.S)
    main=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',main,flags=re.S)
    def include(match):
        path=P/match[1]
        if not path.suffix:path=path.with_suffix('.tex')
        return '' if 'datasets' in match[1] else path.read_text(encoding='utf-8')
    main=re.sub(r'\\input\{([^}]+)\}',include,main)
    words=len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",main))
    if words>3500:issues.append('Main text exceeds3500words')
    displays=len(re.findall(r'\\begin\{figure\}',text))+1
    if displays>6 or r'\tableofcontents' in text:issues.append('Main display/contents constraint')
    audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else {}
    if audit.get('auditor_sha256')!=sha(ROOT/'code/tools/paper_subagent_audit.py'):issues.append('Audit uses a different auditor source version')
    audited_inputs=audit.get('completed_input_sha256',{})
    if audit.get('completed_trajectories') and not audited_inputs:issues.append('Audit does not bind its completed input bytes')
    for f,expected_sha in audited_inputs.items():
        path=ROOT/f
        if not path.is_file() or sha(path)!=expected_sha:issues.append('Input changed since completed audit: '+f)
    if e.get('transport_continuations',[])!=audit.get('transport_continuations',[]):issues.append('Interrupted-request audit missing from exported evidence')
    complete=len(actual)==68 and not e['pending']
    for name in ('text_generation.json','figure_generation.json','mechanism_generation.json'):
        generated=read(OUT/name)
        generator={'text_generation.json':'paper_subagent_text.py',
                   'figure_generation.json':'paper_subagent_figures.py',
                   'mechanism_generation.json':'paper_subagent_mechanisms.py'}[name]
        if generated.get('generator_sha256')!=sha(ROOT/'code/tools'/generator):issues.append('Generation receipt uses a different generator source version: '+name)
        if generated['evidence_sha256']!=sha(P/'data/evidence.json'):issues.append('Manuscript numbers/plots use stale evidence: '+name)
        for f,expected_sha in {**generated.get('input_sha256',{}),**generated['files']}.items():
            if sha(P/f)!=expected_sha:issues.append('Generated numerical content/input changed: '+f)
        if name=='mechanism_generation.json':
            if generated['subagent_rounds']!=16*len(actual) or generated['reused_BO_rounds']!=512:issues.append('Per-round export denominator mismatch')
            for f,expected_sha in generated['source_sha256'].items():
                if sha(ROOT/f)!=expected_sha:issues.append('Round/forecast source changed: '+f)
    if a.require_complete:
        if not complete or actual!=expected:issues.append('All68 trajectories not complete')
        if audit.get('status')!='PASS' or audit.get('completed_trajectories')!=68 or audit.get('audited_queries')!=1088:issues.append('Full1088query audit missing')
        for r in e['rows']:
            if r['cohort']=='codex_subagent' and audited_inputs.get(r['source'])!=r['source_sha256']:issues.append('Exported trajectory not covered by the exact completed audit: '+r['source'])
        if sum(r['comparison']=='V12_vs_RBO' and r['seed'] in (1,2,3) for r in e['paired'])!=24:issues.append('Matched primary matrix incomplete')
        if sum(r['comparison'] in ('KF_removal','refresh_removal') for r in e['paired'])!=12:issues.append('Removal matrix incomplete')
        with (P/'data/all_experiment_records.csv').open(encoding='utf-8-sig',newline='') as f:catalogue=list(csv.DictReader(f))
        if len(catalogue)!=6822:issues.append('Unified historical/registered catalogue incomplete')
        for r in catalogue:
            path=ROOT/r['source']
            if not r['source_sha256'] or not path.exists() or sha(path)!=r['source_sha256']:issues.append('Catalogue source missing or changed: '+r['source'])
            if r['initial_count_source']=='assumed_four_not_verified' and any(r[k] for k in ('initial_queries','post_initial_queries','first_best_query','initial_ids_sha256','comparability_key','SearchAUC_stored_scale')):
                issues.append('Unverified historical initialization used for a derived query metric: '+r['source'])
        if sum(r['cohort']=='paper_v12_subagent' and r['status']=='COMPLETE' for r in catalogue)!=68:issues.append('Unified ledger missing completed subagent cells')
        from paper_subagent_compile import source_hashes
        build=read(OUT/'compile_receipt.json')
        if source_hashes()!=build['source_sha256'] or len(build['documents'])!=2 or any(r['returncode'] for r in build['documents']):issues.append('Current manuscript compilation unverified')
        qa=read(OUT/'pdf_qa.json') if (OUT/'pdf_qa.json').exists() else {}
        if len(qa.get('documents',[]))!=2:issues.append('Both PDF QA receipts required')
        for d in qa.get('documents',[]):
            if d['pdf_sha256']!=sha(P/(d['name']+'.pdf')) or d['automated_issues'] or d['visual_review']!='PASS_ASSISTANT_VISUAL_INSPECTION':issues.append('Latest PDF not visually approved')
    receipt=dict(status='FAIL' if issues else ('PASS_COMPLETE_EVIDENCE' if a.require_complete else 'PASS_LISTED_DRAFT_CHECKS'),
        issues=issues,require_complete=a.require_complete,completed_trajectories=len(actual),registered_trajectories=68,reused_BO_controls=32,
        abstract_words=len(abstract.split()),approximate_main_words=words,main_displays=displays,
        historical_files_checked=len(protocol['protected_controls_sha256']),audit_status=audit.get('status','NOT_AUDITED'),
        completion_gate_satisfied=bool(a.require_complete and not issues),claim_boundary='Execution/record validation is separate from optimizer superiority; cohorts remain separate.')
    for name in ('protocol.json',):sources['experiments/runs/paper_v12_subagent/'+name]=sha(study.directory(ROOT)/name)
    for name in ('audit.json','pdf_qa.json'):
        if (OUT/name).exists():sources[(OUT/name).relative_to(ROOT).as_posix()]=sha(OUT/name)
    (OUT/'manuscript_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (P/'evidence_manifest.json').write_text(json.dumps(dict(sources=sources,validation=receipt),indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
