"""Compile the new manuscript without modifying the delivered API manuscript."""
from pathlib import Path
from hashlib import sha256
import json,os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent';OUT=ROOT/'experiments/reports/paper_v12_subagent'
def source_hashes():
    paths=list(P.rglob('*.tex'))+list(P.glob('*.bib'))+list((P/'figures').glob('*.pdf'))
    return {p.relative_to(P).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def main():
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'runtime/cache/tex'));receipts=[];OUT.mkdir(parents=True,exist_ok=True);sources=source_hashes()
    for name in ('main','supplement'):
        folder=ROOT/'experiments/logs/paper_v12_subagent/typeset'/name;folder.mkdir(parents=True,exist_ok=True)
        result=subprocess.run([str(ROOT/'runtime/tools/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--keep-intermediates','--outdir',str(folder),str(P/f'{name}.tex')],cwd=P,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (OUT/f'{name}_compile.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        receipts.append(dict(document=name,returncode=result.returncode))
        (OUT/'compile_receipt.json').write_text(json.dumps(dict(documents=receipts,source_sha256=sources),indent=2),encoding='utf-8')
        if result.returncode:print((result.stdout+'\n'+result.stderr)[-2500:]);raise SystemExit(result.returncode)
        shutil.copy2(folder/f'{name}.pdf',P/f'{name}.pdf')
        receipts[-1]['pdf_sha256']=sha256((P/f'{name}.pdf').read_bytes()).hexdigest()
    if source_hashes()!=sources:raise ValueError('Typesetting inputs changed during build')
    (OUT/'compile_receipt.json').write_text(json.dumps(dict(documents=receipts,source_sha256=sources),indent=2),encoding='utf-8');print(json.dumps(receipts))
if __name__=='__main__':main()
