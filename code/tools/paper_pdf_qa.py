"""Render each delivered PDF with Poppler; record text and layout checks."""
from pathlib import Path
from hashlib import sha256
import json,re,subprocess
import fitz
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper/v12';OUT=ROOT/'experiments/reports/paper_v12'
POP=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')

def main():
    rows=[]
    for name in ('main','supplement'):
        path=PAPER/(name+'.pdf');pdf_hash=sha256(path.read_bytes()).hexdigest()
        folder=ROOT/'experiments/logs/paper_v12/render'/name/pdf_hash[:12];folder.mkdir(parents=True,exist_ok=True)
        r=subprocess.run([str(POP),'-r','75','-png',str(path),str(folder/'page')],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Poppler rendering failed: '+name)
        doc=fitz.open(path);issues=[];texts=[]
        for i,page in enumerate(doc):
            txt=page.get_text();texts.append(txt)
            if '??' in txt:issues.append(dict(page=i+1,kind='unresolved_reference'))
            for block in page.get_text('dict')['blocks']:
                if 'lines' not in block:continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if x0<-1 or y0<-1 or x1>page.rect.width+1 or y1>page.rect.height+1:issues.append(dict(page=i+1,kind='text_outside_page',text=span['text'][:80]))
        (folder/'extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
        images=sorted(folder.glob('page-*.png'))
        if len(images)!=len(doc):raise ValueError('Rendered page count mismatch')
        for start in range(0,len(images),6):
            ims=[]
            for idx,p in enumerate(images[start:start+6],start+1):
                im=Image.open(p).convert('RGB');im.thumbnail((375,540));tile=Image.new('RGB',(395,575),'#e8edf0');tile.paste(im,((395-im.width)//2,25));ImageDraw.Draw(tile).text((12,6),f'{name} p{idx}',fill='black');ims.append(tile)
            contact=Image.new('RGB',(395*3,575*2),'#e8edf0')
            for j,im in enumerate(ims):contact.paste(im,((j%3)*395,(j//3)*575))
            contact.save(folder/f'contact_{start//6+1}.png')
        rows.append(dict(name=name,pages=len(doc),pdf_sha256=pdf_hash,render_directory=folder.relative_to(ROOT).as_posix(),rendered_pages=len(images),automated_issues=issues,visual_review='PENDING_HUMAN_ASSISTANT_INSPECTION'))
    tex=(PAPER/'main.tex').read_text(encoding='utf-8');abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S)[1]
    report=dict(documents=rows,abstract_words=len(abstract.split()),main_display_items=len(re.findall(r'\\begin\{figure\}',tex))+1,table_of_contents=False)
    (OUT/'pdf_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
