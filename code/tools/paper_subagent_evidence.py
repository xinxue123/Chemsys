"""Keep subagent results separate while reusing exactly matched BO controls."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12_subagent import study
from paper_evidence import record,csvfile,dump,read
from paper_audit import control_run
OUT=ROOT/'experiments/reports/paper_v12_subagent';PAPER=ROOT/'paper/v12_subagent'
NAMES={'memory':'V12','reasoning_bo':'Reasoning BO','no_kf':'No KF','no_refresh':'No refresh'}

def main():
    p=study.check(ROOT);rows=[];pending=[]
    for task,seed,arm in study.JOBS:
        path=study.run_path(ROOT,task,seed,arm)/'state.json'
        r=record(path,task,NAMES[arm],'subagent',seed)
        if r:rows.append(dict(r,cohort='codex_subagent',arm=arm))
        else:pending.append(dict(task=task,seed=seed,arm=arm))
    for task in study.original.TASKS:
        for seed in (0,1,2,3):
            r=record(control_run(ROOT,task,seed)/'state.json',task,'BO','subagent_control',seed)
            if r is None:raise ValueError('Missing frozen BO control')
            rows.append(dict(r,cohort='reused_deterministic_BO',arm='bo'))
    by={(r['task'],r['initialization_seed'],r['arm']):r for r in rows};pairs=[]
    for task in study.original.TASKS:
        for seed in (0,1,2,3):
            for arm,control,kind in [('memory','bo','V12_vs_BO'),('reasoning_bo','bo','RBO_vs_BO'),
                ('memory','reasoning_bo','V12_vs_RBO'),('memory','no_kf','KF_removal'),('memory','no_refresh','refresh_removal')]:
                a=by.get((task,seed,arm));b=by.get((task,seed,control))
                if a is None or b is None:continue
                if a['initial_ids']!=b['initial_ids']:raise ValueError('Initial IDs do not match')
                # Outcome equality matters as well as ID equality.
                if a['values'][:4]!=b['values'][:4]:raise ValueError('Initial outcomes do not match')
                pairs.append(dict(task=task,label=a['label'],seed=seed,comparison=kind,
                    method=a['method'],control=b['method'],SearchAUC=a['SearchAUC'],control_SearchAUC=b['SearchAUC'],
                    AUC_delta=a['SearchAUC']-b['SearchAUC'],AUC_percent=100*(a['SearchAUC']/b['SearchAUC']-1),
                    final_best=a['final_best'],control_final_best=b['final_best'],best_delta=a['final_best']-b['final_best'],
                    source=a['source'],control_source=b['source']))
    summary=[]
    for task in study.original.TASKS:
        for kind in ('V12_vs_BO','RBO_vs_BO','V12_vs_RBO'):
            rr=[r for r in pairs if r['task']==task and r['comparison']==kind and r['seed'] in (1,2,3)]
            if not rr:continue
            a=np.array([r['SearchAUC'] for r in rr]);b=np.array([r['control_SearchAUC'] for r in rr]);d=a-b
            summary.append(dict(task=task,label=rr[0]['label'],comparison=kind,n=len(rr),complete_three_seed_panel=len(rr)==3,
                mean_SearchAUC=float(a.mean()),sd_SearchAUC=float(a.std(ddof=1)) if len(rr)>1 else None,
                control_mean_SearchAUC=float(b.mean()),mean_paired_delta=float(d.mean()),
                paired_delta_sd=float(d.std(ddof=1)) if len(rr)>1 else None,ratio_of_means_percent=100*(a.mean()/b.mean()-1),
                wins=int((d>1e-9).sum()),ties=int((np.abs(d)<=1e-9).sum()),losses=int((d<-1e-9).sum())))
    auditpath=OUT/'audit.json';audit=read(auditpath) if auditpath.exists() else {}
    datasets=read(ROOT/'paper/v12/data/evidence.json')['datasets']
    evidence=dict(cohort='codex_subagent',protocol_sha256=p['protocol_sha256'],context_policy=p['context_policy'],
        requested_model=p['model'],requested_reasoning_effort=p['reasoning_effort'],
        registered_trajectories=68,completed_trajectories=sum(r['cohort']=='codex_subagent' for r in rows),
        pending=pending,reused_BO_controls=32,rows=rows,paired=pairs,summary=summary,datasets=datasets,
        audit_status=audit.get('status','NOT_YET_AUDITED'),audit_completed=audit.get('completed_trajectories',0),
        transport_continuations=audit.get('transport_continuations',[]),
        historical_API_results_are_separate=True,claim_boundary='Do not pool with API repetitions. A removal run also draws fresh LLM responses; single-cell differences do not isolate module causality.')
    scalar=[{k:v for k,v in r.items() if k not in ('initial_ids','selected_ids','values','best_curve')} for r in rows]
    for base in (OUT,PAPER/'data'):
        dump(base/'evidence.json',evidence)
        csvfile(base/'metrics.csv',scalar);csvfile(base/'paired_comparisons.csv',pairs);csvfile(base/'summary.csv',summary)
    print(json.dumps(dict(complete=evidence['completed_trajectories'],pending=len(pending),BO_controls=32,
                         audited=evidence['audit_completed'],audit_status=evidence['audit_status'],comparisons=len(pairs)),indent=2))

if __name__=='__main__':main()
