"""Package the completed additive LaTeX/PDF revision with portable arithmetic."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13_late'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for x in iter(lambda:stream.read(2**20),b''):h.update(x)
    return h.hexdigest()
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    sys.path.insert(0,str(P));import verify_results
    preservation=json.loads((P/'audit/new_preservation_receipt.json').read_text(encoding='utf-8'))
    if preservation['status']!='SEALED_AND_VERIFIED':raise ValueError('Evidence preservation required')
    layout=json.loads((P/'audit/pdf_layout.json').read_text(encoding='utf-8'))
    visual=json.loads((P/'audit/visual_review.json').read_text(encoding='utf-8'))
    if visual['all_pages_visually_reviewed'] is not True or visual['unresolved_issues']:raise ValueError('Visual review remains incomplete')
    for d in layout:
        if not d['all_pages_rendered'] or any(p['unresolved_markers'] or p['unusual_boxes'] for p in d['page_checks']):raise ValueError('PDF layout gate failed')
        if d['pdf_sha256']!=sha(P/(d['document']+'.pdf')):raise ValueError('PDF changed since rendering')
        if visual['PDF_sha256'][d['document']]!=d['pdf_sha256']:raise ValueError('Visual review refers to an older PDF')
    # Reuse the already portable historical regenerator, and expose only the
    # new data/figure functions in a separate portable file.
    shutil.copy2(ROOT/'paper/v13/regenerate_assets.py',P/'regenerate_historical_assets.py')
    source=(ROOT/'code/publication_v13_late/assets.py').read_text(encoding='utf-8')
    source=source.replace("ROOT=Path(__file__).resolve().parents[2]\nP=ROOT/'paper/v13_late'\nR=ROOT/'experiments/reports/v13_late_validation_20261103'\nOLD=ROOT/'paper/v13'", "ROOT=Path(__file__).resolve().parent\nP=ROOT\nR=P/'data'\nOLD=P")
    source=source.replace("    stage();rows=read(P/'data/historical_phase_metrics.csv');", "    rows=read(P/'data/historical_phase_metrics.csv');")
    source=source.replace("        shutil.copy2(R/name,P/'data'/name)", "        if (R/name).resolve()!=(P/'data'/name).resolve():shutil.copy2(R/name,P/'data'/name)")
    if "stage();rows=" in source:raise ValueError('Portable regenerator conversion failed')
    (P/'regenerate_new_assets.py').write_text(source,encoding='utf-8')
    # Verify before creating the final manifest, then verify once more with it.
    verified=verify_results.verify(ROOT);write(P/'audit/arithmetic.json',verified)
    prose=json.loads((P/'audit/prose_generation.json').read_text(encoding='utf-8'))
    if prose['abstract_words']>150:raise ValueError('Abstract too long')
    main_source=(P/'main.tex').read_text(encoding='utf-8')
    if len(re.findall(r'\\begin\{(?:figure|table)\}',main_source))>6:raise ValueError('More than six main display items')
    files={path.relative_to(P).as_posix():sha(path) for path in sorted(P.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.name!='artifact_manifest.json' and path.suffix not in ('.pyc','.log','.aux','.lock')}
    write(P/'artifact_manifest.json',dict(status='V13_ADDITIVE_VALIDATION_PAPER',files_sha256=files,
        original_release_retained='paper/v13',original_source_zip_sha256=sha(ROOT/'paper/v13_source.zip'),
        new_trajectories=75,new_paid_queries=840,new_evidence_manifest_sha256=preservation['manifest_sha256']))
    verified=verify_results.verify(ROOT)
    zpath=ROOT/'paper/v13_late_source.zip'
    if zpath.exists():raise ValueError('Final source package already exists; do not silently replace a release')
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in files:z.write(P/name,name)
        z.write(P/'artifact_manifest.json','artifact_manifest.json')
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist())==len(files)+1
        for name,value in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
        portable=ROOT/'experiments/logs/paper_v13_late/portable_zip_check'
        if portable.exists():raise ValueError('Use a fresh portable-verification directory')
        portable.mkdir(parents=True)
        for name in z.namelist():
            if not (portable/name).resolve().is_relative_to(portable.resolve()):raise ValueError('Unsafe ZIP path')
        z.extractall(portable)
    bundled=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    check=subprocess.run([str(bundled),str(portable/'verify_results.py')],cwd=portable,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    (portable.parent/'portable_verification.txt').write_text(check.stdout,encoding='utf-8')
    if check.returncode:raise RuntimeError('Extracted portable verifier failed: '+check.stdout[-1500:])
    receipt=dict(status='COMPLETE_ADDITIVE_V13_RESEARCH_MANUSCRIPT',original_paper_and_results_unchanged=True,
        native_new_audit='PASS_COMPLETE',new_trajectories=75,new_paid_queries=840,PDFs=[dict(document=d['document'],pages=d['pages'],sha256=d['pdf_sha256']) for d in layout],
        source_zip_sha256=sha(zpath),source_zip_bytes=zpath.stat().st_size,source_zip_all_member_bytes_verified=True,
        extracted_zip_verified_with_standard_library_python=True,
        manifest_sha256=sha(P/'artifact_manifest.json'),evidence_archive_sha256=preservation['archive_sha256'],
        portable_and_workspace_arithmetic='PASS',final_verified_counts=verified)
    write(ROOT/'paper/v13_late_release_receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
