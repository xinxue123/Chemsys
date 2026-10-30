"""Display exact pending observations; transcribe actual assistant decisions.

No oracle, surrogate, ranking or proposal generation. Unit conversion and fixed
category-bin transport are the only changes to manually supplied conditions.
"""
import argparse
import json
from pathlib import Path

ROOT=Path('data/runs/suzuki_official')
CATS=['P1-L1','P2-L1','P1-L2','P1-L3','P1-L4','P1-L5','P1-L6','P1-L7']


def read(p):return json.loads(p.read_text(encoding='utf8'))


def waiting():
    values={}
    for f in sorted((ROOT/'runs').glob('*/state.json')):
        s=read(f)
        if s['status']=='WAITING_LLM':
            q=read(ROOT/'requests'/s['pending_context']/'request.json')
            if not (ROOT/'requests'/s['pending_context']/'response.json').exists(): values[s['run_id']]=q
    return values


def encode(point):
    c,t,T,l=point
    return [(t-60)/540,(T-30)/80,(l-.5)/2,(CATS.index(c)+.5)/8]


def text_for(a,method,previous):
    if method=='LLINBO': return json.dumps(encode(a['point']),separators=(',',':'))
    if a.get('retain_previous'):
        # Retain only the exact preference payload; the assistant explicitly
        # supplies the new rationale and prefix citations for every context.
        final=previous['upstream_reply'].split('Final Answer:',1)[1].strip()
    else:
        if a.get('point') is not None:
            c,t,T,l=a['point'];payload=[t,T,l,(CATS.index(c)+.5)/8]
            final='[point, '+json.dumps(payload)+', '+str(a['confidence'])+']'
        else:
            lb,ub=a['lb'],a['ub']
            final='[region, '+json.dumps([lb,ub])+', '+str(a['confidence'])+']'
    return 'Thinking:\n'+a['reasoning']+'\nFinal Answer:\n'+final


def main():
    p=argparse.ArgumentParser();p.add_argument('--show',choices=['full','latest']);p.add_argument('--answers',type=Path)
    args=p.parse_args();requests=waiting()
    if args.show:
        for key,q in requests.items():
            h=q['history'];selected=h if args.show=='full' else h[-1:]
            row={'id':key,'t':q['iteration'],'H':[[r['query'],r['point']['catalyst'],
                *[round(r['point'][n],3) for n in ('t_res','temperature','catalyst_loading')],round(r['y']*100,3)] for r in selected]}
            if args.show=='latest':
                best=max(h,key=lambda r:r['y']);row['best']=[best['query'],best['point'],round(best['y']*100,3)]
            print(json.dumps(row,separators=(',',':')))
        return
    raw=read(args.answers);rows=[]
    for key,a in raw['answers'].items():
        q=requests[key]
        if q['iteration']!=raw['round']: raise ValueError('answers target a different round')
        reply=text_for(a,q['method'],q['previous_response'])
        rows.append({**{k:q[k] for k in ('run_id','iteration','context_sha256')},
                     'upstream_reply':reply,'reasoning':a['reasoning'],'supporting_queries':a['supporting_queries'],
                     'origin':'current_assistant','transport':'exact deterministic normalization/formatting only'})
    target=args.answers.with_suffix('.expanded.json')
    target.write_text(json.dumps({'responses':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'expanded_file':str(target),'responses':len(rows)}))


if __name__=='__main__':main()
