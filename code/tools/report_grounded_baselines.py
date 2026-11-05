"""Report all preregistered trajectories, retaining incomplete/failure status."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,math,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
OUT=ROOT/'experiments/reports/grounded_baselines_20261105'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
NAMES=dict(bo='Pure BO',first='Same first + BO',hyp='Hypothesis/KF/events',lgbo='LGBO',rbo='Reasoning BO')
TASK_LABELS=dict(c1='Reizman Case 1',c2='Reizman Case 2',c3='Reizman Case 3',c4='Reizman Case 4',
    perera='Perera Suzuki',doyle='Doyle Buchwald-Hartwig',snar='SnAr',hplc='HPLC')
AXIS_LABELS=dict(c1='Best simulated yield (%)',c2='Best simulated yield (%)',c3='Best simulated yield (%)',c4='Best simulated yield (%)',
    perera='Best UV product area (%)',doyle='Best reported yield (%)',snar='Best STY (kg m$^{-3}$ h$^{-1}$)',hplc='Best peak area (a.u.)')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    if (BASE/'completion_receipt.json').exists():
        print(json.dumps(dict(status='FROZEN_REPORT_RETAINED',report=str(OUT/'report.txt'))));return
    p=read(BASE/'protocol.json');OUT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'code/chemist_system'))
    from chem.benchmarks.knowledge_gp import load_task
    pools={short:load_task(ROOT,task) for short,task in zip(TASKS,p['tasks'])}
    csvout('task_metadata.csv',[dict(task=t,name=TASK_LABELS[t],candidate_count=len(pool.ids),objective_unit=pool.unit,
        axes=json.dumps(pool.axes),input_kinds=json.dumps(pool.kinds),pool_path=pool.source_path.relative_to(ROOT).as_posix())
        for t,pool in pools.items()])
    paths=[dict(task=r['task'],seed=r['seed'],method=r['method'],path=r['path']) for r in p['controls']]
    paths += [dict(task=t,seed=s,method=m,path=f'experiments/runs/grounded_baselines_20261105/runs/s{s}/{t}_{m}/state.json')
              for t in TASKS for s in range(4) for m in ('lgbo','rbo')]
    rows=[];traces=[];curves={};pending=[]
    for spec in paths:
        path=ROOT/spec['path']
        if not path.exists():pending.append(dict(**spec,observed=0));continue
        state=read(path);history=state['history']
        if len(history)!=20 or state['status']!='COMPLETE':pending.append(dict(**spec,observed=len(history)));continue
        raw=np.asarray([h['raw_value'] for h in history],float);best=np.maximum.accumulate(raw)
        assert len({h['candidate_id'] for h in history})==20
        row=dict(task=spec['task'],seed=spec['seed'],method=spec['method'],SearchAUC=float(best.mean()),
            final_best=float(best[-1]),first_value=float(raw[0]),first_final_best_query=int(np.flatnonzero(raw==best[-1])[0])+1,
            post_first_logical_LLM_calls=sum(c['query']>1 for c in state.get('calls',[])),
            objective_unit=pools[spec['task']].unit,
            ln_sum_cumulative_best=float(math.log(best.sum())) if best.sum()>0 else '',state_path=spec['path'])
        rows.append(row);curves[(spec['task'],spec['seed'],spec['method'])]=best
        for q,(h,value) in enumerate(zip(history,best),1):traces.append(dict(task=spec['task'],seed=spec['seed'],method=spec['method'],query=q,candidate_id=h['candidate_id'],raw_value=h['raw_value'],cumulative_best=float(value)))
    csvout('per_seed_results.csv',rows);csvout('trajectories.csv',traces)
    (OUT/'pending.json').write_text(json.dumps(pending,indent=2)+'\n',encoding='utf-8')
    summaries=[]
    for t in TASKS:
        for m in NAMES:
            rs=[r for r in rows if (r['task'],r['method'])==(t,m)]
            d=dict(task=t,method=m,completed=len(rs),planned=4)
            for f in ('SearchAUC','final_best'):
                d[f+'_mean']=statistics.mean(r[f] for r in rs) if rs else ''
                d[f+'_sample_SD']=statistics.stdev(r[f] for r in rs) if len(rs)>1 else ''
            summaries.append(d)
    csvout('summary.csv',summaries)
    paired=[]
    for r in rows:
        if r['method'] not in ('lgbo','rbo'):continue
        for reference in ('bo','first','hyp'):
            ref=next(x for x in rows if (x['task'],x['seed'],x['method'])==(r['task'],r['seed'],reference))
            paired.append(dict(task=r['task'],seed=r['seed'],method=r['method'],reference=reference,
                delta_SearchAUC=r['SearchAUC']-ref['SearchAUC'],delta_final_best=r['final_best']-ref['final_best']))
    if paired:csvout('paired_differences.csv',paired)
    lookup={(r['task'],r['seed'],r['method']):r for r in rows}
    decomposition=[]
    for r in rows:
        if r['method'] not in ('hyp','lgbo','rbo'):continue
        bo=lookup[(r['task'],r['seed'],'bo')];first=lookup[(r['task'],r['seed'],'first')]
        for metric in ('SearchAUC','final_best'):
            decomposition.append(dict(task=r['task'],seed=r['seed'],method=r['method'],metric=metric,
                shared_first_BO_minus_random_start_BO=first[metric]-bo[metric],
                continued_policy_minus_shared_first_BO=r[metric]-first[metric],total_policy_minus_random_start_BO=r[metric]-bo[metric]))
    csvout('paired_start_and_continuation_differences.csv',decomposition)
    paired_summary=[]
    for m in ('lgbo','rbo'):
        for reference in ('bo','first','hyp'):
            for metric in ('SearchAUC','final_best'):
                subset=[r for r in paired if r['method']==m and r['reference']==reference]
                values=[r['delta_'+metric] for r in subset]
                paired_summary.append(dict(method=m,reference=reference,metric=metric,completed_pairs=len(values),planned_pairs=32,
                    wins=sum(x>1e-9 for x in values),ties=sum(abs(x)<=1e-9 for x in values),losses=sum(x< -1e-9 for x in values)))
    csvout('paired_summary.csv',paired_summary)
    status='COMPLETE' if not pending else 'INCOMPLETE'
    text=[f'Matched grounded baselines: {status}',
        'Snapshot generated UTC: '+datetime.now(timezone.utc).isoformat(),
        'Progress and metrics below reflect this saved snapshot, not a live dashboard.', '',
        'Eight tasks,0 free observations+20 paid evaluations, exact-tie seeds0-3. Native GP/MC512 remain numerical seed0. All LLM arms share the frozen grounded first input and blinded initial scientific hypotheses. Initial priors were not resampled across seeds.',
        'New LGBO/RBO trajectories are compared with unchanged BO, same-first BO and hypothesis/KF/event trajectories. Each post-first request uses a fresh isolated gpt-6.1-terra medium subagent. Own observations and same enriched chemistry background are supplied; no other-arm outcomes or current GP ranks are supplied. RBO receives neutral GP top5 membership.',
        'SearchAUC = sum of20 post-query cumulative maxima /20, including the first point, zeros and all nonimprovements. ln_sum_cumulative_best is an explicitly defined optional diagnostic: ln(20*SearchAUC) per trajectory. It is NOT claimed to reproduce the paper-defined Log AUC. Do not exponentiate a mean log to obtain an arithmetic mean AUC.',
        'LGBO uses official regional mean tilting and full-pool native qLogEI; RBO uses the official Insight confidence extractor in a5-to1 interface. No KF/event lifecycle is inserted into either baseline. Necessary mixed-domain, fixed-pool, first-point, prompt and sequential adaptations are in protocol.json and docs/grounded_baselines_20261105.txt.',
        'The20 paid experimental evaluations are matched; LLM computation is not matched. Each new baseline calls the LLM19 times after the shared first point. The current hypothesis method refreshes only on events. Per-trajectory post-first logical call counts are retained in per_seed_results.csv; schema retries are recorded separately in requests.',
        'paired_start_and_continuation_differences.csv gives the exact accounting identity: (method - random-start BO) = (same-first BO - random-start BO) + (method - same-first BO), within the same task/seed. This distinguishes the starting-input comparison from subsequent trajectory differences; it does not identify a causal KF effect.',
        'The original LGBO marginal-SD reconstruction failed at singleton P0 prefixes before a new choice was committed. Exact native posterior sampling compatibility is documented in execution_compatibility.json. Frozen source, previous outcomes and existing LLM responses were retained; no performance rerolls.', '',
        'Task metadata:']
    for t,pool in pools.items():text.append(f'{TASK_LABELS[t]} | {len(pool.ids)} candidates | {pool.unit}')
    text+=['','Task | Method | Completed | SearchAUC mean (sample SD) | Final best mean (sample SD)']
    for r in summaries:
        def fmt(f):return 'not complete' if r[f+'_mean']=='' else f"{r[f+'_mean']:.4f} ({r[f+'_sample_SD']:.4f})" if r[f+'_sample_SD']!='' else f"{r[f+'_mean']:.4f} (SD unavailable)"
        text.append(f"{r['task']} | {NAMES[r['method']]} | {r['completed']}/4 | {fmt('SearchAUC')} | {fmt('final_best')}")
    text+=['','Paired seed results (method vs reference, wins/ties/losses; numerical metric equality tolerance1e-9):']
    for r in paired_summary:
        text.append(f"{NAMES[r['method']]} vs {NAMES[r['reference']]} | {r['metric']} | {r['wins']}/{r['ties']}/{r['losses']} ({r['completed_pairs']}/32 pairs)")
    text+=['','No task is dropped because of poor performance. Partial means are not final comparisons. Four seeds and one shared initial prior do not establish broad statistical superiority. Task objectives have different units and are not averaged together.']
    (OUT/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if not pending:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors=dict(bo='#228833',first='#999999',hyp='#AA3377',lgbo='#EE7733',rbo='#4477AA')
        styles=dict(bo='--',first=':',hyp='-',lgbo='-.',rbo=(0,(4,1,1,1)))
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,t in zip(axes.flat,TASKS):
            for m in NAMES:
                y=np.stack([curves[(t,s,m)] for s in range(4)]);mean=y.mean(0);sd=y.std(0,ddof=1)
                ax.plot(range(1,21),mean,label=NAMES[m],color=colors[m],ls=styles[m],lw=1.6,
                    marker='o' if m=='hyp' else None,markevery=4,ms=3)
                ax.fill_between(range(1,21),mean-sd,mean+sd,color=colors[m],alpha=.08)
            ax.set_title(TASK_LABELS[t]);ax.set_xlabel('Paid query');ax.set_ylabel(AXIS_LABELS[t],fontsize=9)
            ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
        fig.text(.5,.015,'Means and sample SD over4 exact-tie streams; shared fixed LLM first input/initial prior. Different task units.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.92))
        for suffix in ('png','pdf'):fig.savefig(OUT/f'cumulative_best.{suffix}',dpi=170)
    print(json.dumps(dict(status=status,completed_new=sum(r['method'] in ('lgbo','rbo') for r in rows),planned_new=64,pending=len(pending))))

if __name__=='__main__':main()
