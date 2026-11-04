"""Recompute manuscript values from original measured trajectories."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import csv,json,math,sys,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12 import study
PAPER=ROOT/'paper/v12'; RUN=ROOT/'experiments/runs/knowledge_loop_20261101'; OUT=ROOT/'experiments/reports/paper_v12'
LABEL={t:(f'Suzuki {t[-1]}' if 'Case_' in t else {'Perera_Suzuki':'Perera','Doyle_Buchwald_Hartwig':'Doyle','SnAr':'SnAr','HPLC':'HPLC'}[t]) for t in study.TASKS}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def csvfile(p,rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def record(path,task,method,phase,seed=0,rep=None):
    if not path.exists():return None
    st=read(path)
    if st['status']!='COMPLETE':return None
    ys=[r['raw_value'] for r in st['history']]
    if len(ys)!=20 or len(st['decisions'])!=16:raise ValueError('Not a 4+16 complete trajectory: '+str(path))
    best=[];v=-math.inf
    for y in ys:v=max(v,y);best.append(v)
    auc=sum(best[4:])/16
    # Independent definition: integrate the post-query step sequence explicitly.
    check=sum(max(ys[:4+t]) for t in range(1,17))/16
    assert abs(auc-check)<1e-10
    ev=[e for d in st['decisions'] for e in d.get('KF_feedback',[])]
    r=dict(task=task,label=LABEL[task],method=method,phase=phase,initialization_seed=seed,replicate=rep,
        SearchAUC=auc,final_best=max(ys),first_best_query=max(0,ys.index(max(ys))+1-4),initial_best=max(ys[:4]),
        source=path.relative_to(ROOT).as_posix(),source_sha256=sha256(path.read_bytes()).hexdigest(),
        initial_ids=[x['candidate_id'] for x in st['history'][:4]],selected_ids=[x['candidate_id'] for x in st['history'][4:]],
        values=ys,best_curve=best[3:],forecasts=len(ev),Brier_gain_sum=sum(e['Brier_gain'] for e in ev),
        weight_decreases=sum(e['after']['weight']<e['before']['weight'] for e in ev),
        pauses=sum(e['reason']=='automatic_pause' for e in ev),retractions=sum(e['reason']=='automatic_retract' for e in ev),
        resumes=sum(e['reason']=='automatic_resume' for e in ev),edits=len(st.get('edit_calls',[])),
        different_from_same_history_BO=sum(d.get('selected_id')!=d.get('BO_reference_id') for d in st['decisions'] if d.get('BO_reference_id')))
    return r

def main():
    rows=[]
    def add(*args,**kw):
        r=record(*args,**kw)
        if r:rows.append(r)
    for task in study.TASKS:
        for arm,name in [('bo','BO'),('reasoning_bo','Reasoning BO'),('kf_rank','V9')]:add(RUN/'v9_kf_rank/runs'/f'{task}_{arm}/state.json',task,name,'seed0')
        add(RUN/'v8/runs'/f'{task}_full_agent_initial_terra/state.json',task,'V8','seed0')
        add(RUN/'v10_information/runs'/f'{task}_information/state.json',task,'V10','seed0')
        add(RUN/'v11_reasoning_memory/runs'/f'{task}_rbo_memory/state.json',task,'V11','seed0')
        if task in study.FOCUS:add(RUN/'v12_reasoning_memory/runs'/f'{task}_memory/state.json',task,'V12','seed0')
        else:add(study.run_path(ROOT,task,0,'memory')/'state.json',task,'V12','seed0')
        for seed in (1,2,3):
            for arm,name in [('bo','BO'),('memory','V12'),('reasoning_bo','Reasoning BO')]:
                if task in study.FOCUS and arm!='reasoning_bo':path=study.seeds.run_path(ROOT,task,seed,arm)/'state.json'
                else:path=study.run_path(ROOT,task,seed,arm)/'state.json'
                add(path,task,name,'initialization',seed)
    for task in study.FOCUS:
        for rep in ('r01','r02','r03'):
            add(RUN/'v12_llm_repeats/replicates'/task/rep/'runs'/f'{task}_memory/state.json',task,'V12','LLM_repeat',0,rep)
        if task!='HPLC':add(RUN/'v12_reasoning_memory/runs'/f'{task}_category/state.json',task,'V12 category','category',0)
        for seed in (0,3):
            for arm in ('no_kf','no_refresh'):add(study.run_path(ROOT,task,seed,arm)/'state.json',task,arm,'ablation',seed)
    datasets=[]
    for task in study.TASKS:
        p=study.NATIVE_LOAD(ROOT,task,seed=0)
        datasets.append(dict(task=task,label=LABEL[task],group=p.group,candidate_count=len(p.ids),axes=list(p.axes),kinds=list(p.kinds),unit=p.unit,scale=p.scale,
            input_source=p.source_path.relative_to(ROOT).as_posix(),input_sha256=sha256(p.source_path.read_bytes()).hexdigest()))
    pairs=[]
    for phase in ('seed0','initialization','LLM_repeat','ablation'):
        for r in [x for x in rows if x['phase']==phase and x['method']!='BO']:
            bp='seed0' if phase=='LLM_repeat' or phase=='ablation' and r['initialization_seed']==0 else ('initialization' if phase=='ablation' else phase)
            b=next(x for x in rows if x['task']==r['task'] and x['phase']==bp and x['method']=='BO' and x['initialization_seed']==r['initialization_seed'])
            assert r['initial_ids']==b['initial_ids']
            pairs.append(dict(task=r['task'],label=r['label'],phase=phase,method=r['method'],seed=r['initialization_seed'],replicate=r['replicate'],
                BO_AUC=b['SearchAUC'],method_AUC=r['SearchAUC'],AUC_delta=r['SearchAUC']-b['SearchAUC'],AUC_percent=100*(r['SearchAUC']/b['SearchAUC']-1),
                BO_best=b['final_best'],method_best=r['final_best'],best_delta=r['final_best']-b['final_best'],best_percent=100*(r['final_best']/b['final_best']-1),
                source=r['source'],BO_source=b['source']))
    summary=[]
    for task in study.FOCUS:
        for phase in ('initialization','LLM_repeat'):
            ps=[x for x in pairs if x['task']==task and x['phase']==phase and x['method']=='V12'];x=[r['method_AUC'] for r in ps];b=[r['BO_AUC'] for r in ps]
            summary.append(dict(task=task,label=LABEL[task],phase=phase,n=len(ps),AUC_mean=float(np.mean(x)),AUC_sample_sd=float(np.std(x,ddof=1)),
                BO_mean=float(np.mean(b)),ratio_of_means_percent=100*(np.mean(x)/np.mean(b)-1),wins=sum(r['AUC_delta']>1e-9 for r in ps),
                ties=sum(abs(r['AUC_delta'])<=1e-9 for r in ps),losses=sum(r['AUC_delta']<-1e-9 for r in ps)))
    scalar=[{k:v for k,v in r.items() if k not in ('initial_ids','selected_ids','values','best_curve')} for r in rows]
    result=dict(rows=rows,datasets=datasets,paired=pairs,summary=summary,completed_supplemental=sum(x['status']=='COMPLETE' for x in study.status(ROOT)),pending_supplemental=sum(x['status']!='COMPLETE' for x in study.status(ROOT)))
    dump(OUT/'evidence.json',result);dump(PAPER/'data/evidence.json',result)
    for name,data in [('metrics',scalar),('paired_comparisons',pairs),('summary',summary),('datasets',datasets)]:
        csvfile(OUT/(name+'.csv'),data);shutil.copy2(OUT/(name+'.csv'),PAPER/'data'/(name+'.csv'))
    if (OUT/'historical_runs.csv').exists():shutil.copy2(OUT/'historical_runs.csv',PAPER/'data/historical_runs.csv')
    print(json.dumps(dict(selected_complete_trajectories=len(rows),datasets=datasets,summary=summary),indent=2))
if __name__=='__main__':main()
