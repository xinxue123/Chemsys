"""Render every manuscript page with Poppler and collect layout/text checks."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

import fitz
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13'
POPPLER=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')


def main():
    out=ROOT/'experiments/logs/paper_v13/render'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ('main','supplement'):
        pdf=P/(name+'.pdf');folder=out/name;folder.mkdir(exist_ok=True)
        subprocess.run([str(POPPLER),'-r','90','-png',str(pdf),str(folder/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        doc=fitz.open(pdf);pages=[]
        for q,page in enumerate(doc,1):
            words=page.get_text('words');txt=page.get_text()
            excursions=[dict(word=w[4],box=list(w[:4])) for w in words if w[0]<48 or w[2]>page.rect.width-40 or w[1]<25 or w[3]>page.rect.height-24]
            pages.append(dict(page=q,words=len(words),unresolved_markers=('??' in txt),
                unusual_boxes=excursions,first_line=txt.splitlines()[0] if txt else '',last_lines=txt.splitlines()[-3:]))
        pngs=sorted(folder.glob('page-*.png'))
        for sheet in range(math.ceil(len(pngs)/6)):
            canvas=Image.new('RGB',(1500,1470),'#CCD4DA');draw=ImageDraw.Draw(canvas)
            for slot,path in enumerate(pngs[sheet*6:(sheet+1)*6]):
                im=Image.open(path).convert('RGB');im.thumbnail((475,675))
                x=15+(slot%3)*500;y=28+(slot//3)*735
                canvas.paste(im,(x,y));draw.text((x,y-18),f'{name}: page {sheet*6+slot+1}',fill='black')
            canvas.save(folder/f'contact_{sheet+1}.png')
        records.append(dict(document=name,pages=len(doc),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            all_pages_rendered=len(pngs)==len(doc),page_checks=pages))
        print(name,len(doc),'pages, rendered',len(pngs),flush=True)
    (P/'audit/pdf_layout.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
