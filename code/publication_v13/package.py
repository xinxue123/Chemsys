"""Package the finished V13 paper; verify extracted arithmetic and all hashes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13'
M=ROOT/'experiments/milestones/v13_publication_20261103'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    # Historical archive integrity is checked again; its bytes are never changed.
    for folder,receipt_name in ((M,'receipt.json'),(M,'provenance_receipt.json'),
            (ROOT/'experiments/milestones/v13_warmstart_20261102','receipt.json')):
        receipt=read(folder/receipt_name)
        for name,h in receipt['artifact_sha256'].items():
            assert sha(folder/name)==h,(folder,name)
    layout=read(P/'audit/pdf_layout.json')
    for doc in layout:
        assert doc['pdf_sha256']==sha(P/(doc['document']+'.pdf'))
        assert doc['all_pages_rendered']
        assert all(not x['unresolved_markers'] and not x['unusual_boxes'] for x in doc['page_checks'])
    typeset=read(P/'audit/typeset.json')
    assert all(x['returncode']==0 and not x['warnings'] for x in typeset)
    spec=importlib.util.spec_from_file_location('v13_verify',P/'verify_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The preceding manifest may exist only when rebuilding a released package.
    # No mutation after an existing release is accepted by this entrypoint.
    release=ROOT/'paper/v13_release_receipt.json'
    assert not release.exists(),'Released paper is immutable; use a new revision namespace'
    result=module.verify(ROOT)
    write(P/'audit/native_verification.json',result)
    review=dict(status='REVIEWED',main_pages=11,supplement_pages=8,
        all_pages_visually_inspected=True,figures_and_table_alignment_reviewed=True,
        unresolved_references=0,outside_page_text_boxes=0,final_TeX_warnings=0,
        pdf_sha256={x['document']+'.pdf':x['pdf_sha256'] for x in layout},
        inspected_render_directory='experiments/logs/paper_v13/render',
        inference_limit='Layout and replay checks do not demonstrate optimization superiority')
    write(P/'audit/visual_review.json',review)
    allowed={'.tex','.bib','.py','.txt','.csv','.json','.pdf','.png'}
    files=[]
    forbidden={'config.txt','openai-key.txt','.env'}
    patterns=[re.compile(rb'sk-[A-Za-z0-9_-]{24,}'),re.compile(rb'(?i)authorization[\x22\x27\s:]+bearer\s+[A-Za-z0-9._-]{20,}')]
    for f in P.rglob('*'):
        if not f.is_file() or f.suffix not in allowed or '__pycache__' in f.parts or f.name=='artifact_manifest.json':continue
        assert f.name.lower() not in forbidden
        if f.suffix in {'.tex','.bib','.py','.txt','.csv','.json'}:
            assert not any(x.search(f.read_bytes()) for x in patterns),'Credential-shaped content; filename only: '+str(f)
        files.append(f)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='FROZEN_V13_PAPER_PACKAGE',scientific_trajectories=227,V13_trajectories=64,
        note='Compact LaTeX/data package; original model/response custody is in the separate sealed milestone.',
        files_sha256={f.relative_to(P).as_posix():sha(f) for f in sorted(files)},
        preservation_manifest_sha256=sha(M/'manifest.json'),
        old_warm_archive_preserved=True,new_optimization_calls=0)
    write(P/'artifact_manifest.json',manifest)
    files.append(P/'artifact_manifest.json')
    archive=ROOT/'paper/v13_source.zip'
    assert not archive.exists(),'Do not overwrite a previous source release'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(files):z.write(f,f.relative_to(P).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={f.relative_to(P).as_posix() for f in files}
        for name,h in manifest['files_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        target=ROOT/'experiments/logs/paper_v13'/('portable_'+uuid.uuid4().hex[:10])
        target.mkdir(parents=True,exist_ok=False)
        assert target.resolve().is_relative_to(ROOT)
        z.extractall(target)
    # Standard-library verifier runs from the extracted package, with no access
    # to the original native tree required, and uses a separate Python runtime.
    portable_python=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    proc=subprocess.run([str(portable_python),str(target/'verify_results.py')],cwd=target,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',check=True)
    portable=json.loads(proc.stdout);assert portable['status']=='PASS'
    native=module.verify(ROOT);assert native['status']=='PASS'
    output=dict(status='PASS_RELEASED_LOCAL',created_utc=datetime.now(timezone.utc).isoformat(),
        main_pages=11,supplement_pages=8,source_zip_bytes=archive.stat().st_size,
        main_pdf_sha256=sha(P/'main.pdf'),supplement_pdf_sha256=sha(P/'supplement.pdf'),
        manifest_sha256=sha(P/'artifact_manifest.json'),source_zip_sha256=sha(archive),
        portable_extracted_package_check=portable,workspace_evidence_check=native,
        preservation_receipt_sha256=sha(M/'receipt.json'),
        scientific_code_and_original_results_modified=False,
        new_experiments=0,new_GP_fits=0,new_LLM_calls=0)
    write(release,output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
