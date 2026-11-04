"""Compile the two manuscripts with the existing Tectonic runtime."""
from pathlib import Path
import json,os,subprocess,shutil,sys
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper/v12';OUT=ROOT/'experiments/reports/paper_v12'
def main():
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'runtime/cache/tex'))
    receipts=[]
    for name in ('main','supplement'):
        dest=ROOT/'experiments/logs/paper_v12/typeset'/name;dest.mkdir(parents=True,exist_ok=True)
        p=subprocess.run([str(ROOT/'runtime/tools/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--keep-intermediates','--outdir',str(dest),str(PAPER/(name+'.tex'))],cwd=PAPER,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (OUT/(name+'_compile.log')).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
        receipts.append(dict(name=name,returncode=p.returncode))
        if p.returncode:
            print((p.stdout+'\n'+p.stderr)[-2200:]);raise SystemExit(p.returncode)
        shutil.copy2(dest/(name+'.pdf'),PAPER/(name+'.pdf'))
        print(json.dumps(dict(name=name,pdf_bytes=(PAPER/(name+'.pdf')).stat().st_size)))
    (OUT/'compile_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
if __name__=='__main__':main()
