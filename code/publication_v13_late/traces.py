"""Export traceable, outcome-complete mechanism records, without new selection."""
from pathlib import Path
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13_late'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):raise ValueError((a,b))


def scalar(packet,event,arm):
    c=event['calibration'];b=c['before'];a=c['after'];s=float(event['improved'])
    p=packet['p_LLM'];g=packet['p_GP'];z=(s-g)**2-(s-p)**2
    R=max(.0025,4*(p-g)**2*g*(1-g));pred=b['variance']+.0025;K=pred/(pred+R)
    mean=b['mean']+K*(z-b['mean']);var=(1-K)**2*pred+K*K*R
    prob=.5*math.erfc(-mean/math.sqrt(2*var));reliability=.8*prob
    if z<=1e-12:reliability=min(reliability,b['reliability'])
    for name,value in dict(Brier_gain=z,measurement_variance=R,predicted_variance=pred,kalman_gain=K).items():close(c[name],value)
    actual=c['proposed_after'] if arm=='kf_fixed' else a
    if arm=='coupled':
        actual=event['split_state_proposal']['calibration']['after']
        lc=event['legacy_filter'];la=lc['after'];lb=lc['before']
        # Legacy retirement can set deployed weight to zero. Its scalar m/P
        # update otherwise uses the identical predictive-skill equations.
        close(la['mean'],mean);close(la['variance'],var);close(la['probability_positive'],prob)
        for key in ('mean','variance','probability_positive'):close(a[key],la[key])
        close(a['reliability'],la['weight'])
        if la['status']!='retracted':close(la['weight'],reliability)
    for key,value in dict(mean=mean,variance=var,probability_positive=prob,reliability=reliability).items():close(actual[key],value)
    if arm=='kf_fixed':
        for key in ('mean','variance','probability_positive','reliability'):close(a[key],b[key])
        if c.get('applied') is not False:raise ValueError('Numerical freeze missing')
    return dict(current_incumbent=packet['incumbent'],p_LLM=p,p_GP=g,Brier_gain=z,
        mean_before=b['mean'],variance_before=b['variance'],reliability_before=b['reliability'],
        measurement_variance=R,predicted_variance=pred,kalman_gain=K,
        proposed_mean=mean,proposed_variance=var,proposed_reliability=reliability,
        actual_mean=a['mean'],actual_variance=a['variance'],actual_reliability=a['reliability'])


def main(fresh=False):
    traces=[];forecasts=[]
    historical=[r for r in rows(P/'data/metrics.csv') if r['method']=='v13']
    jobs=[dict(id=f"historical_{r['regime']}_{r['task']}{r['seed']}",kind='historical',task=r['task'],
        prefix_seed=r['seed'],repetition=0,arm='full',prefix_queries=int(r['initial']),state_path=r['state_path']) for r in historical]
    if fresh:
        jobs.extend(dict(r,prefix_queries=(10 if r['kind']=='prefix' else 0)) for r in rows(P/'data/final_metrics.csv'))
    for job in jobs:
        sf=ROOT/job['state_path'];state=read(sf)
        for d in state['decisions']:
            q=d['query']
            if job['kind']=='prefix' and q<=10:continue
            c=read(sf.parent/'rounds'/f'{q:02}_commit.json')
            reasoning=c['reasoning'];raw=reasoning['raw'] if reasoning else None
            if raw and 'insight' in raw:insight=raw['insight']
            else:insight=raw
            h=state['history'][int(job['prefix_queries'])+q-1] if job['kind']=='historical' and int(job['prefix_queries'])==4 else state['history'][q-1]
            # Historical warm query indices start after the four observations.
            before=state['history'][:int(job['prefix_queries'])+q-1] if job['kind']=='historical' and int(job['prefix_queries'])==4 else state['history'][:q-1]
            common=dict(job=job['id'] if 'id' in job else job['job'],kind=job['kind'],task=job['task'],prefix_seed=job['prefix_seed'],
                repetition=job['repetition'],arm=job['arm'],query=q,selected_id=c['selected_id'],
                BO_reference_id=c['BO_reference_id'],selected_is_own_GP_top=int(c['selected_id']==c['BO_reference_id']),
                measured=h['raw_value'],incumbent_before=max((x['raw_value'] for x in before),default=None),
                editor_this_query=int(c['memory'] is not None),applicable_forecasts=len(d['feedback']),
                declared_knowledge_uses=d.get('knowledge_uses',[]),refresh_trigger=c.get('refresh'),
                public_comment=insight['comment'] if insight else None,commit_path=(sf.parent/'rounds'/f'{q:02}_commit.json').resolve().relative_to(ROOT).as_posix())
            traces.append(common)
            if not d['feedback']:continue
            packets={f['commit_sha256']:f for f in c['decision']['forecasts']}
            notes={(n['id'],n['version']):n for n in c['decision']['notes_at_commit']}
            for e in d['feedback']:
                packet=packets[e['commit_sha256']];n=notes[packet['id'],packet['version']]
                value=scalar(packet,e,job['arm'])
                if bool(e['accurate_failure_forecast']) != (not e['improved'] and packet['p_LLM']<.5 and value['Brier_gain']>1e-12):raise ValueError('Wrong failure flag')
                forecasts.append(dict(job=common['job'],kind=job['kind'],task=job['task'],prefix_seed=job['prefix_seed'],
                    repetition=job['repetition'],arm=job['arm'],query=q,selected_id=c['selected_id'],note_id=packet['id'],version=packet['version'],
                    claim=n['claim'],focus=n['focus'],scope=n['scope'],effect=n['effect'],use=packet['use'],
                    improved=int(e['improved']),accurate_failure=int(e['accurate_failure_forecast']),
                    permission_before=e['recommendation_before'],permission_after=e['recommendation_after'],
                    **value,forecast_sha256=packet['commit_sha256'],event_id=e['event_id']))
    (P/'data/mechanism_traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (P/'data/late_forecast_calculations.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(forecasts[0]));w.writeheader()
        for x in forecasts:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items()})
    print(json.dumps(dict(status='PASS',trace_queries=len(traces),forecast_events=len(forecasts),fresh_included=fresh,
        GP_refits=0,LLM_calls=0,new_experiment_queries=0)))


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--fresh',action='store_true');args=a.parse_args();main(args.fresh)
