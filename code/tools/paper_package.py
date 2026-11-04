"""Package only reviewed manuscript inputs; no raw requests or credentials."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,re,zipfile

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v12';OUT=ROOT/'experiments/reports/paper_v12'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return sha256(path.read_bytes()).hexdigest()

def main():
    qa=read(OUT/'pdf_qa.json');validation=read(OUT/'manuscript_validation.json')
    if validation['status']!='PASS_LISTED_EVIDENCE_CHECKS':raise ValueError('Evidence validation required')
    for d in qa['documents']:
        if d['visual_review']!='PASS_ASSISTANT_VISUAL_INSPECTION' or d['automated_issues']:
            raise ValueError('Latest PDF visual inspection required')
        if sha(P/f"{d['name']}.pdf")!=d['pdf_sha256']:raise ValueError('PDF changed after review')
    allowed={'main.tex','methods.tex','supplement.tex','references.bib','README.txt','evidence_manifest.json'}
    files=[P/name for name in sorted(allowed)]
    for folder,extensions in [('figures',{'.pdf','.png'}),('generated',{'.tex'}),('data',{'.csv','.json'})]:
        files.extend(p for p in (P/folder).iterdir() if p.is_file() and p.suffix in extensions)
    for source in (P/'main.tex',P/'methods.tex',P/'supplement.tex'):
        tex=source.read_text(encoding='utf-8')
        for name in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',tex):
            path=P/name
            if not path.suffix:path=path.with_suffix('.tex')
            if path not in files:raise ValueError('Missing portable TeX input: '+name)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p.relative_to(P).as_posix():sha(p) for p in sorted(files)},
                  pdfs={d['name']+'.pdf':d['pdf_sha256'] for d in qa['documents']},
                  status='REVIEWED_WORKING_DRAFT_WITH_PENDING_EXPERIMENTS',
                  evidence_validation='PASS_LISTED_EVIDENCE_CHECKS',visual_review='PASS_ASSISTANT_VISUAL_INSPECTION')
    mf=P/'artifact_manifest.json';mf.write_text(json.dumps(manifest,indent=2),encoding='utf-8');files.append(mf)
    archive=P.parent/'v12_source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,Path('v12')/p.relative_to(P))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
        for name,expected in manifest['files'].items():
            if sha256(z.read('v12/'+name)).hexdigest()!=expected:raise ValueError('Archive content drift')
    receipt=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),
                 portable_tex_inputs_checked=True,archive_crc_checked=True,raw_API_files_included=False,credentials_included=False)
    (OUT/'package.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
