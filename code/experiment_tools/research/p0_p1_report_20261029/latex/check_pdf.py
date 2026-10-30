"""Numerical preservation, page bounds, font warnings and render QA helpers."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse
import json
import math
import re
import subprocess
import sys
import fitz
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TMP = ROOT / 'outputs/p0_p1_typeset'


def h(path):return sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def table_texts(node, inside=False):
    if isinstance(node, list):
        for v in node:yield from table_texts(v, inside)
    elif isinstance(node, dict):
        inside = inside or node.get('t') == 'Table'
        if inside and node.get('t') == 'Str':
            yield node['c']
        else:
            for v in node.values():yield from table_texts(v, inside)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    pdf=HERE/'P0_P1_Report.pdf';doc=fitz.open(pdf)
    text='\n'.join(p.get_text() for p in doc)
    (TMP/'extracted_pdf.txt').write_text(text,encoding='utf-8')
    # All decimal cells from the reviewed Markdown tables must survive conversion.
    raw=subprocess.check_output(['pandoc',str(HERE.parent/'REPORT.md'),'-t','json'],encoding='utf-8')
    source_ast=json.loads(raw)
    pattern=r'(?<![\d.])[+-]?\d+\.\d+(?![\d.])'
    expected=Counter(re.findall(pattern,' '.join(table_texts(source_ast))))
    actual=Counter(re.findall(pattern,text.replace('\u2212','-')))
    missing={k:v-actual[k] for k,v in expected.items() if actual[k]<v}
    # Validate the new chart geometry uses exactly the old source rows.
    records=read(TMP/'figure_source_rows.json');metric_data=read(HERE.parent/'data.json')
    expected_metrics={str(ROOT/r['state_source']):r for key in ('P0','P1','P1_development') for r in metric_data[key]}
    for record in records:
        y=record['yield_percent'];best=record['cumulative_best_percent']
        assert len(y)==20 and best==[max(y[:i]) for i in range(1,21)]
        m=expected_metrics[record['source']]
        assert abs(math.fsum(best[4:])/16-m['SearchAUC'])<1e-10
        assert max(y)==m['final_best_percent']
    log=(TMP/'P0_P1_Report.log').read_text(encoding='utf-8',errors='replace')
    tex=(HERE/'P0_P1_Report.tex').read_text(encoding='utf-8')
    cases=Counter(re.findall(r'\\caption\{P0 Case ([1-4]) ',tex))
    assert cases==Counter({'1':4,'2':4,'3':4,'4':4}),cases
    warnings={k:len(re.findall(p,log)) for k,p in {
        'missing_characters':r'Missing character:',
        'overfull_hbox':r'Overfull \\hbox',
        'overfull_vbox':r'Overfull \\vbox',
        'underfull_hbox':r'Underfull \\hbox',
        'undefined_references':r'There were undefined references',
        'rerun_required':r'Rerun to get|Rerun to get the outlines'}.items()}
    pages=[];outside=[]
    for i,page in enumerate(doc,1):
        blocks=page.get_text('dict')['blocks']
        for b in blocks:
            x0,y0,x1,y1=b['bbox']
            if x0 < -1 or y0 < -1 or x1 > page.rect.width+1 or y1 > page.rect.height+1:
                outside.append({'page':i,'bbox':b['bbox']})
        pages.append({'page':i,'text_characters':len(page.get_text()),
                      'width':page.rect.width,'height':page.rect.height,
                      'images':len(page.get_images())})
    assert h(HERE.parent/'REPORT.md')==read(HERE/'build_provenance.json')['source_markdown_sha256']
    result={'status':'NUMERICAL_PASS_VISUAL_REVIEW_PENDING' if not missing and not outside and not warnings['missing_characters'] and not warnings['overfull_hbox'] and not warnings['overfull_vbox'] else 'REPAIR_REQUIRED',
        'pdf_sha256':h(pdf),'tex_sha256':h(HERE/'P0_P1_Report.tex'),'page_count':len(doc),
        'decimal_cells_checked':sum(expected.values()),'decimal_cells_missing':missing,
        'chart_series_checked':len(records),'chart_values_checked':sum(len(r['yield_percent']) for r in records),
        'table_count':read(HERE/'build_provenance.json')['tables_preserved'],'warnings':warnings,
        'P0_case_captions_verified':dict(cases),
        'page_bounds_failures':outside,'pages':pages}
    if args.render:
        folder=TMP/('pages_'+h(pdf)[:12]);folder.mkdir(exist_ok=True)
        poppler=Path(r'C:\Users\lihao\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
        subprocess.run([str(poppler),'-r','110','-png',str(pdf),str(folder/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        pngs=sorted(folder.glob('page-*.png'))
        assert len(pngs)==len(doc)
        contacts=[]
        for start in range(0,len(pngs),9):
            selection=pngs[start:start+9];thumbs=[]
            for k,path in enumerate(selection,start+1):
                im=Image.open(path).convert('RGB');im.thumbnail((300,424))
                panel=Image.new('RGB',(320,455),'#dce1e6')
                panel.paste(im,((320-im.width)//2,24))
                ImageDraw.Draw(panel).text((9,6),f'Page {k}',fill='black')
                thumbs.append(panel)
            sheet=Image.new('RGB',(960,455*math.ceil(len(thumbs)/3)),'white')
            for k,im in enumerate(thumbs):sheet.paste(im,((k%3)*320,(k//3)*455))
            target=TMP/f'contact_{h(pdf)[:8]}_{start+1:02d}_{start+len(selection):02d}.png'
            sheet.save(target);contacts.append(str(target))
        result['rendered_pages']=len(pngs)
        result['rendered_page_folder']=str(folder)
        result['contact_sheets']=contacts
    (TMP/'pdf_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pages'},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
