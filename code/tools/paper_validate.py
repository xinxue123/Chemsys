"""Check the manuscript evidence contract and produce a file-level receipt."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import csv,json,re,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12';OUT=ROOT/'experiments/reports/paper_v12'

def sha(path):return sha256(path.read_bytes()).hexdigest()
def main():
    e=json.loads((P/'data/evidence.json').read_text(encoding='utf-8'));rows=e['rows'];issues=[]
    protocol=json.loads((ROOT/'experiments/runs/paper_v12/protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/name)!=expected:issues.append('frozen implementation drift: '+name)
    if (e['completed_supplemental'],e['pending_supplemental'])!=(15,56):issues.append('manuscript execution-status prose requires update')
    for r in rows:
        p=ROOT/r['source']
        if sha(p)!=r['source_sha256']:issues.append('source drift: '+r['source'])
        if len(r['values'])!=20 or len(r['selected_ids'])!=16:issues.append('budget: '+r['source'])
        if not np.isclose(np.maximum.accumulate(r['values'])[4:].mean(),r['SearchAUC'],rtol=0,atol=1e-10):issues.append('AUC: '+r['source'])
    for r in e['paired']:
        a=next(x for x in rows if x['source']==r['source']);b=next(x for x in rows if x['source']==r['BO_source'])
        if a['initial_ids']!=b['initial_ids']:issues.append('unmatched initialization')
    with (P/'data/historical_runs.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
    if len(history)!=6683:issues.append('historical catalogue denominator')
    if any(r['initial_count_source']=='assumed_four_not_verified' and r['SearchAUC_stored_scale'] for r in history):issues.append('imputed AUC under unknown budget')
    init=[r for r in rows if r['phase']=='initialization' and r['method']=='V12']
    expected={'forecasts':81,'weight_decreases':38,'pauses':6,'retractions':3,'resumes':1,'edits':36}
    if {k:sum(r[k] for r in init) for k in expected}!=expected:issues.append('manuscript mechanism counts')
    for task,disagreement in [('Reizman_Suzuki_Case_2',1/6),('Reizman_Suzuki_Case_4',23/48),('HPLC',35/48)]:
        rr=[r for r in rows if r['phase']=='LLM_repeat' and r['task']==task]
        actual=np.mean([np.mean(np.array(rr[i]['selected_ids'])!=np.array(rr[j]['selected_ids'])) for i in range(3) for j in range(i+1,3)])
        if not np.isclose(actual,disagreement):issues.append('choice disagreement '+task)
    maintex=(P/'main.tex').read_text(encoding='utf-8');bib=(P/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+)',bib))
    for text in [maintex,(P/'methods.tex').read_text(encoding='utf-8')]:
        for group in re.findall(r'\\cite\w*\{([^}]+)\}',text):
            for key in group.split(','):
                if key not in keys:issues.append('missing citation '+key)
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',maintex,re.S)[1]
    if len(abstract.split())>150:issues.append('abstract word limit')
    if r'\tableofcontents' in maintex:issues.append('unrequested contents')
    # Article text, excluding abstract, figures and captions, frontmatter and Methods.
    text=maintex.split(r'\section*{Methods}')[0]
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}','',text,flags=re.S)
    text=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',text,flags=re.S)
    words=len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",text))
    if words>3500:issues.append('main text limit')
    sources={r['source']:r['source_sha256'] for r in rows}
    for r in e['datasets']:sources[r['input_source']]=r['input_sha256']
    for name in ('docs/paper_goal_20261101.txt','docs/paper_v12_supplement_protocol_20261101.txt','experiments/runs/paper_v12/protocol.json','experiments/reports/paper_v12/audit.json','experiments/reports/paper_v12/transport_diagnostic.json'):
        sources[name]=sha(ROOT/name)
    receipt=dict(status='PASS_LISTED_EVIDENCE_CHECKS' if not issues else 'FAIL',issues=issues,selected_complete_trajectories=len(rows),historical_state_records=len(history),unverified_initial_budgets_with_AUC_withheld=sum(r['metric_reconstruction']=='AUC_WITHHELD_UNVERIFIED_INITIAL_BUDGET' for r in history),frozen_implementation_files_checked=len(protocol['source_sha256']),
        abstract_words=len(abstract.split()),approximate_main_words=words,main_display_items=6,frozen_source_edited=False,new_supplemental_complete=15,new_supplemental_pending=56,
        claim_boundary='Working manuscript; no causal module attribution, population significance, whole-eight-task V12 validation or publication claim.')
    (OUT/'manuscript_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (P/'evidence_manifest.json').write_text(json.dumps(dict(sources=sources,validation=receipt),indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
