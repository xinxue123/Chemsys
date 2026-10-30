"""Verify the consultation document and package only its public artifacts."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import json
import re
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='Chem_Expert_Consultation'


def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    pdf=HERE/(NAME+'.pdf');tex=HERE/(NAME+'.tex');doc=fitz.open(pdf);assert len(doc)==9
    text='\n'.join(p.get_text() for p in doc);body=tex.read_text(encoding='utf-8').split(r'\begin{document}',1)[1]
    assert not re.search(r'(?<!\\)%',body),'unescaped percent could hide prose'
    numbers=set(re.findall(r'(?<![A-Za-z\d])[+-]?\d+\.\d{3,}',body));assert all(v in text for v in numbers)
    for term in ['0.444','81.700%','98.287%','10227.031','10373.540','232']:
        assert term in text,term
    fonts={xref for page in doc for xref,*_ in page.get_fonts(full=True)}
    assert all(doc.extract_font(x)[3] for x in fonts)
    for n,page in enumerate(doc,1):
        assert len(page.get_text())>500 and (ROOT/f'outputs/expert_consultation_20261030/pages/page-{n}.png').is_file()
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(n,block[:4])
    log=(ROOT/'outputs/expert_consultation_20261030'/f'{NAME}.log').read_text(encoding='utf-8',errors='replace')
    assert 'Missing character:' not in log and 'Overfull' not in log
    visual=read(HERE/'visual_review.json')
    assert visual['PDF_sha256']==sha(pdf) and visual['pages_passed']==list(range(1,10))
    snapshot=read(HERE/'reviewed_snapshot.json');assert snapshot['verification']['metric_rows_recomputed']==232
    with (HERE/'experiment_results_complete.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==232
    for row,original in zip(rows,snapshot['rows']):
        assert row['state_source']==original['state_source'] and row['method']==original['method']
        for key in ['SearchAUC','final_best','initial_best']:assert float(row[key])==original[key]
        assert int(row['first_final_best_total'])==original['first_final_best_total']
    calls=snapshot['R5_calls'];impossible=[c for c in calls if c['menu_inside_gate']==0]
    assert len(impossible)==1 and impossible[0]['task']=='SnAr' and impossible[0]['query']==14
    assert sum(c['GP_accepted'] for c in calls)==2 and all(c['semantic_ok'] for c in calls)
    sources=read(HERE/'sources.json')
    for name,value in sources.items():assert sha(ROOT/name)==value,name
    # Guard both the last full release and all older comparator evidence.
    old=read(ROOT/'research/continuous_r5_20261030/release_inventory.json')
    protected=read(ROOT/'research/continuous_r5_20261030/protected_snapshot.json')
    for pins in [old,protected]:
        for name,value in pins.items():assert sha(ROOT/name)==value,name
    receipt={'status':'PASS','PDF_pages':9,'visual_pages_passed':list(range(1,10)),'numeric_strings_checked':len(numbers),
        'all_fonts_embedded':True,'missing_glyphs':0,'overfull_boxes':0,'CSV_metric_rows_verified':232,
        'original_trajectory_metrics_recomputed':snapshot['verification'],'source_files_unchanged':len(sources),
        'R5_release_files_unchanged':len(old),'older_protected_files_unchanged':len(protected),
        'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex),'new_experiments':0,'new_LLM_calls':0}
    write(HERE/'validation_receipt.json',receipt)
    members=[x for x in HERE.iterdir() if x.is_file() and x.suffix in {'.pdf','.tex','.py','.csv','.json'} and x.name!='delivery_receipt.json']
    for path in members:
        assert 'config' not in path.name.lower()
        if path.suffix in {'.json','.py','.tex'}:
            content=path.read_text(encoding='utf-8')
            assert not re.search(r'"(?:api_key|apikey|authorization)"\s*:',content,re.I),path.name
            assert not re.search(r'\bsk-[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9]',content),path.name
    archive=HERE/'Chem_Expert_Consultation_Source_and_Data.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('READ_ME.txt',
            'Expert consultation materials as of 2026-10-30. No new experiments or LLM calls.\n'
            'Open Chem_Expert_Consultation.pdf for the nine-page briefing.\n'
            'The TeX source compiles with continuous_best.pdf in the same directory.\n'
            'experiment_results_complete.csv includes 232 archived comparison rows; these are not independent repetitions.\n'
            'Rows retain task, knowledge regime, method, units and original trajectory source.\n'
            'R5_call_feasibility.csv gives eight actual calls, menu/gate intersection and GP review.\n'
            'sources.json provides original source hashes. Native trajectories remain in the original project.\n'
            'To run the builder/verifier, keep these files under research/expert_consultation_20261030 in the original project.\n'
            'No API keys, model weights, private reasoning or new benchmark claims are included.\n')
        for path in sorted(members):z.write(path,path.name)
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(members)+1
        for path in members:assert sha256(z.read(path.name)).hexdigest()==sha(path)
    write(HERE/'delivery_receipt.json',{'status':'COMPLETE_AND_VERIFIED','PDF_sha256':sha(pdf),'ZIP_sha256':sha(archive),
        'archive_members_verified':len(members)+1,'archive_bytes':archive.stat().st_size,'PDF_pages':9,
        'source_files_unchanged':len(sources),'CSV_rows':232,'new_experiments':0,'new_LLM_calls':0})
    print(json.dumps({'status':'COMPLETE_AND_VERIFIED','PDF_pages':9,'CSV_rows':232,'source_files_unchanged':len(sources),
        'R5_release_files_unchanged':len(old),'older_protected_files_unchanged':len(protected),
        'archive_bytes':archive.stat().st_size,'archive_members':len(members)+1},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
