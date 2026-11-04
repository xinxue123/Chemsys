"""Compile both new V13 documents without touching older manuscript namespaces."""
from pathlib import Path
import json
import os
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13'
TEX=ROOT/'runtime/tools/tex_tools/tectonic/tectonic.exe'


def main():
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'runtime/cache/tex')
    result=[]
    for name in ('main','supplement'):
        out=ROOT/'experiments/logs/paper_v13/typeset'/name
        out.mkdir(parents=True,exist_ok=True)
        proc=subprocess.run([str(TEX),'--keep-logs','--keep-intermediates','--outdir',str(out),str(P/(name+'.tex'))],cwd=P,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
        (out/'compiler_output.txt').write_text(proc.stdout,encoding='utf-8')
        if proc.returncode:
            print('\n'.join(proc.stdout.splitlines()[-55:]));raise RuntimeError('TeX compilation failed: '+name)
        shutil.copy2(out/(name+'.pdf'),P/(name+'.pdf'))
        warnings=[s for s in proc.stdout.splitlines() if any(w in s.lower() for w in ('warning','overfull','undefined','underfull'))]
        result.append(dict(document=name,returncode=proc.returncode,warnings=warnings))
        print(name,'compiled',flush=True)
    (P/'audit').mkdir(exist_ok=True)
    (P/'audit/typeset.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
