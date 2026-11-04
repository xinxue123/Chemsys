"""Render and mechanically check each new PDF; visual sign-off is separate."""
from pathlib import Path
from hashlib import sha256
import json,re,subprocess,unicodedata
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent';OUT=ROOT/'experiments/reports/paper_v12_subagent'
POP=Path('C:/Users/lihao/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
def expanded_source(path):
    text=path.read_text(encoding='utf-8')
    def include(match):
        child=P/match[1]
        if not child.suffix:child=child.with_suffix('.tex')
        return expanded_source(child)
    return re.sub(r'\\input\{([^}]+)\}',include,text)

def normalized(text):
    return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',text).lower())

def completeness_issues(source,text,prefix):
    issues=[]
    for environment,label in (('figure','Figure'),('(table|longtable)','Table')):
        expected=len(re.findall(r'\\begin\{'+environment+r'\}',source))
        if label=='Figure':expected+=len(re.findall(r'\\captionof\{figure\}',source))
        actual=set(re.findall(label+r'\s+('+prefix+r'\d+):',text))
        missing={prefix+str(i) for i in range(1,expected+1)}-actual
        if missing:issues.append(dict(kind='missing_display_caption',label=label,missing=sorted(missing)))
    folded=normalized(text)
    for title in re.findall(r'\\(?:section|subsection)\*?\{([^}]+)\}',source):
        if normalized(title) not in folded:issues.append(dict(kind='missing_section_heading',title=title))
    return issues

def main():
    from paper_subagent_compile import source_hashes
    build=json.loads((OUT/'compile_receipt.json').read_text(encoding='utf-8'))
    if len(build['documents'])!=2 or any(r['returncode'] for r in build['documents']) or source_hashes()!=build['source_sha256']:
        raise ValueError('Both current sources must compile before PDF inspection')
    rows=[]
    for name in ('main','supplement'):
        path=P/f'{name}.pdf';sha=sha256(path.read_bytes()).hexdigest();folder=ROOT/'experiments/logs/paper_v12_subagent/render'/name/sha[:12];folder.mkdir(parents=True,exist_ok=True)
        if sha!=next(r['pdf_sha256'] for r in build['documents'] if r['document']==name):raise ValueError('PDF differs from compiled receipt')
        result=subprocess.run([str(POP),'-r','75','-png',str(path),str(folder/'page')],capture_output=True)
        if result.returncode:raise RuntimeError('Render failed: '+name)
        doc=fitz.open(path);issues=[];text=[]
        for i,page in enumerate(doc):
            s=page.get_text();text.append(s)
            if '??' in s:issues.append(dict(page=i+1,kind='unresolved_reference'))
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if x0<-1 or y0<-1 or x1>page.rect.width+1 or y1>page.rect.height+1:
                            issues.append(dict(page=i+1,kind='text_outside_page',text=span['text'][:60]))
        full_text='\n\n'.join(text)
        issues.extend(completeness_issues(expanded_source(P/f'{name}.tex'),full_text,'S' if name=='supplement' else ''))
        (folder/'extracted.txt').write_text(full_text,encoding='utf-8')
        images=sorted(folder.glob('page-*.png'))
        if len(images)!=len(doc):raise ValueError('Render count mismatch')
        for start in range(0,len(images),6):
            contact=Image.new('RGB',(1185,1150),'#e8edf0')
            for j,path in enumerate(images[start:start+6]):
                im=Image.open(path).convert('RGB');im.thumbnail((375,540));tile=Image.new('RGB',(395,575),'#e8edf0');tile.paste(im,((395-im.width)//2,25));ImageDraw.Draw(tile).text((12,6),f'{name} p{start+j+1}',fill='black');contact.paste(tile,((j%3)*395,(j//3)*575))
            contact.save(folder/f'contact_{start//6+1}.png')
        rows.append(dict(name=name,pages=len(doc),pdf_sha256=sha,render_directory=folder.relative_to(ROOT).as_posix(),rendered_pages=len(images),automated_issues=issues,visual_review='PENDING_ASSISTANT_INSPECTION'))
    (OUT/'pdf_qa.json').write_text(json.dumps(dict(documents=rows),indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
