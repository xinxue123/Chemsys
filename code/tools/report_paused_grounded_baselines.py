"""Organize completed and partial evidence after the user's explicit pause.

No experimental model, oracle, LLM service or worker is started by this tool.
Primary comparisons use the complete seed0 panel. Secondary means use exactly
the same available completed seeds for every method within each task.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv
import hashlib
import json
import shutil
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
REPORT=ROOT/'experiments/reports/grounded_baselines_20261105'
OUT=REPORT/'paused_snapshot'
OPS=ROOT/'experiments/operations/grounded_baselines_20261105'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
METHODS=('bo','first','hyp','lgbo','rbo')
NAMES=dict(bo='Pure BO',first='Same first + BO',hyp='Current hypothesis/KF/events',lgbo='LGBO adapter',rbo='Reasoning BO adapter')
LABELS=dict(c1='Reizman Case 1',c2='Reizman Case 2',c3='Reizman Case 3',c4='Reizman Case 4',perera='Perera Suzuki',doyle='Doyle Buchwald-Hartwig',snar='SnAr',hplc='HPLC')
UNITS=dict(c1='Simulated yield (%)',c2='Simulated yield (%)',c3='Simulated yield (%)',c4='Simulated yield (%)',perera='UV product area (%)',doyle='Reported yield (%)',snar='STY (kg m$^{-3}$ h$^{-1}$)',hplc='Peak area (a.u.)')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert not (OUT/'snapshot_manifest_sha256.json').exists(),'Existing paused snapshot is frozen'
    OUT.mkdir(parents=True,exist_ok=True)
    p=read(BASE/'protocol.json');audit=read(REPORT/'audit_partial.json')
    assert audit['status']=='PASS_PARTIAL_SAVED_EVIDENCE_AUDIT'
    with (REPORT/'per_seed_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['seed']=int(r['seed'])
        for field in ('SearchAUC','final_best','first_value'):r[field]=float(r[field])
    lookup={(r['task'],r['seed'],r['method']):r for r in rows}
    assert all((t,0,m) in lookup for t in TASKS for m in METHODS)
    csvout('completed_results.csv',rows)
    seed0=[lookup[(t,0,m)] for t in TASKS for m in METHODS]
    csvout('seed0_results.csv',seed0)
    inventory=[];prefixes=[];traces=[];state_copies=[]
    task_short=dict(zip(p['tasks'],TASKS))
    for task,seed,arm in p['jobs']:
        short=task_short[task];folder=BASE/'runs'/f's{seed}'/(short+'_'+arm);sf=folder/'state.json'
        if not sf.exists():
            inventory.append(dict(task=short,seed=seed,method=arm,status='NOT_STARTED',queries=0,total_queries=20,state_snapshot=''));continue
        state=read(sf);n=len(state['history'])
        dest=OUT/'checkpoints'/f's{seed}'/(short+'_'+arm);dest.mkdir(parents=True,exist_ok=True)
        for name in ('state.json','runtime_counters.json','initial_commit.json','completion.json'):
            src=folder/name
            if src.exists():shutil.copyfile(src,dest/name);state_copies.append((src,dest/name))
        status='COMPLETE' if n==20 and state['status']=='COMPLETE' else 'PAUSED_PARTIAL'
        inventory.append(dict(task=short,seed=seed,method=arm,status=status,queries=n,total_queries=20,state_snapshot=(dest/'state.json').relative_to(ROOT).as_posix()))
        best=-float('inf');total=0.
        for q,h in enumerate(state['history'],1):
            best=max(best,h['raw_value']);total+=best
            traces.append(dict(task=short,seed=seed,method=arm,query=q,candidate_id=h['candidate_id'],raw_value=h['raw_value'],cumulative_best=best,trajectory_status=status))
        if n<20:prefixes.append(dict(task=short,seed=seed,method=arm,queries=n,total_queries=20,best_so_far=best,
            prefix_mean_cumulative_best=total/n if n else '',full_budget_SearchAUC='NOT_AVAILABLE',reason='Incomplete; never compare this prefix with a completed20-query SearchAUC'))
    csvout('trajectory_inventory.csv',inventory);csvout('partial_prefixes.csv',prefixes);csvout('new_observed_trajectories.csv',traces)
    counts=Counter(r['status'] for r in inventory);queries=sum(r['queries'] for r in inventory)
    attempts=list(BASE.glob('requests/*/*/attempt*/receipt.json'))
    logical=list(BASE.glob('requests/*/*/validated.json'))
    assert len(logical)==audit['checked_fresh_LLM_requests'] and queries==audit['checked_committed_experiments']
    common={t:[s for s in range(4) if all((t,s,m) in lookup for m in METHODS)] for t in TASKS}
    balanced=[lookup[(t,s,m)] for t in TASKS for s in common[t] for m in METHODS]
    csvout('matched_completed_results.csv',balanced)
    summary=[]
    for t in TASKS:
        for m in METHODS:
            selected=[lookup[(t,s,m)] for s in common[t]]
            row=dict(task=t,method=m,available_seeds=json.dumps(common[t]),completed_seeds=len(selected),planned_seeds=4)
            for field in ('SearchAUC','final_best'):
                row[field+'_mean']=statistics.mean(r[field] for r in selected)
                row[field+'_sample_SD']=statistics.stdev(r[field] for r in selected) if len(selected)>1 else ''
            summary.append(row)
    csvout('matched_available_seed_summary.csv',summary)
    # Independently reconcile every reported complete metric, including controls.
    for r in rows:
        state=read(ROOT/r['state_path']);best=-float('inf');total=0.
        assert state['status']=='COMPLETE' and len(state['history'])==20
        for h in state['history']:best=max(best,h['raw_value']);total+=best
        assert abs(r['SearchAUC']-total/20)<1e-9 and r['final_best']==best
    paired=[]
    for t in TASKS:
        for s in common[t]:
            for m in ('hyp','lgbo','rbo'):
                for ref in ('bo','first'):
                    a,b=lookup[(t,s,m)],lookup[(t,s,ref)]
                    paired.append(dict(task=t,seed=s,method=m,reference=ref,delta_SearchAUC=a['SearchAUC']-b['SearchAUC'],delta_final_best=a['final_best']-b['final_best']))
    csvout('matched_paired_differences.csv',paired)
    comparisons=[]
    for m in ('hyp','lgbo','rbo'):
        for ref in ('bo','first'):
            for population in ('seed0','matched_available_seeds'):
                v=[r['delta_SearchAUC'] for r in paired if r['method']==m and r['reference']==ref and (population!='seed0' or r['seed']==0)]
                comparisons.append(dict(method=m,reference=ref,population=population,pairs=len(v),wins=sum(x>1e-9 for x in v),ties=sum(abs(x)<=1e-9 for x in v),losses=sum(x< -1e-9 for x in v)))
    csvout('SearchAUC_win_tie_loss.csv',comparisons)
    # Freeze all mutable checkpoints; immutable round evidence remains in place
    # with hashes, so a future resume can add files without losing this snapshot.
    immutable={};copied={str(src.relative_to(ROOT).as_posix()):dest.relative_to(ROOT).as_posix() for src,dest in state_copies}
    for path in BASE.rglob('*'):
        if not path.is_file() or path.suffix in ('.lock','.tmp','.pyc'):continue
        rel=path.relative_to(ROOT).as_posix()
        if rel in copied:continue
        immutable[rel]=sha(path)
    write(OUT/'immutable_run_evidence_sha256.json',dict(files=immutable,mutable_checkpoints_copied=copied))
    for name in ('audit_partial.json','per_seed_results.csv','paired_start_and_continuation_differences.csv','task_metadata.csv'):
        shutil.copyfile(REPORT/name,OUT/name)
    shutil.copyfile(OPS/'pause_process_receipt.json',OUT/'pause_process_receipt.json')
    for source in (Path(__file__),ROOT/'code/tools/report_grounded_baselines.py',ROOT/'code/tools/audit_grounded_baselines.py',ROOT/'docs/grounded_baselines_execution_notes_20261105.txt'):
        dest=OUT/'analysis_source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    pending=[]
    for ip in BASE.glob('requests/*/*/attempt*/invocation.json'):
        af=ip.parent
        if (af/'receipt.json').exists():continue
        assert not (af/'assignment.json').exists(),'Cannot freeze while an experimental child is assigned'
        pending.append(dict(invocation=ip.relative_to(ROOT).as_posix(),request=(af/'request.json').relative_to(ROOT).as_posix(),task_name=read(ip)['task_name']))
    status=dict(status='PAUSED_AT_USER_REQUEST',paused_UTC=datetime.now(timezone.utc).isoformat(),
        completed_new=counts['COMPLETE'],partial_new=counts['PAUSED_PARTIAL'],not_started=counts['NOT_STARTED'],planned_new=64,
        observed_queries=queries,planned_queries=1280,sealed_LLM_attempts=len(attempts),validated_logical_requests=len(logical),
        assigned_or_active_experimental_children=0,pending_unassigned_requests=len(pending),pending=pending,
        common_completed_seeds=common,protocol_sha256=p['protocol_sha256'],
        process_stop_receipt=(OPS/'pause_process_receipt.json').relative_to(ROOT).as_posix(),
        overall_four_seed_study_complete=False,independent_complete_metric_reconciliation=True,
        resume='Explicit user resume required; retain original scientific state statuses and all pending requests. Resume original compatibility runner and fresh isolated experimental LLM service, not a new protocol.')
    write(OUT/'pause_status.json',status);write(OPS/'finalization_status.json',status)
    text=['PAUSED MATCHED GROUNDED BASELINE COMPARISON',f"Snapshot UTC: {status['paused_UTC']}",'',
        f"User-requested pause: {counts['COMPLETE']}/64 new trajectories complete; {counts['PAUSED_PARTIAL']} partial; {counts['NOT_STARTED']} not started. {queries}/1280 paid table evaluations recorded. {len(logical)} logical LLM requests validated, {len(attempts)} actual response attempts sealed. No active experimental children or runner/finalizer processes remain.",
        'The four-seed study is unfinished. Existing BO, same-first BO and current hypothesis/KF/event controls remain unchanged. The original scientific state files retain resumable RUNNING/COMPLETE statuses; PAUSED is recorded externally, without invalidating committed state hashes.',
        'Settings: eight frozen candidate pools; zero free observations plus20 paid sequential queries. Numerical GP/MC512 seed0; all-query feature tie streams0..3. All LLM methods share the fixed chemically grounded first input and blinded initial knowledge. Later responses are fresh isolated gpt-6.1-terra medium children. Initial LLM knowledge was not resampled across seeds.',
        'SearchAUC is the mean of all20 post-query cumulative maxima, including the first point, zeros and nonimprovements. Incomplete prefixes have NO20-query SearchAUC; their prefix means are labeled separately and never padded or treated as finished results.',
        'PRIMARY: SEED0, COMPLETE EIGHT-TASK PANEL',
        'Each cell below is SearchAUC / final best. Units differ across tasks; Perera measures UV product area, not molar yield.',
        'Task | Pure BO | Same first + BO | Current hypothesis/KF/events | LGBO adapter | Reasoning BO adapter']
    for t in TASKS:text.append(LABELS[t]+' | '+' | '.join(f"{lookup[(t,0,m)]['SearchAUC']:.4f} / {lookup[(t,0,m)]['final_best']:.4f}" for m in METHODS))
    text+=['','SECONDARY: MATCHED AVAILABLE COMPLETED SEEDS',
        'Every method within a task uses exactly the same seeds. Sample SD is shown only for n>=2. These descriptive partial results do not establish four-seed stability.',
        'Task | Seeds | Method | SearchAUC mean (sample SD) | Final best mean (sample SD)']
    for r in summary:
        def fmt(field):return f"{r[field+'_mean']:.4f}"+(f" ({r[field+'_sample_SD']:.4f})" if r[field+'_sample_SD']!='' else ' (n=1)')
        text.append(f"{LABELS[r['task']]} | {r['available_seeds']} | {NAMES[r['method']]} | {fmt('SearchAUC')} | {fmt('final_best')}")
    text+=['','PAIRED SEARCHAUC COUNTS', 'Method | Reference | Population | Wins / ties / losses | Pairs']
    for r in comparisons:text.append(f"{NAMES[r['method']]} | {NAMES[r['reference']]} | {r['population']} | {r['wins']}/{r['ties']}/{r['losses']} | {r['pairs']}")
    text+=['','Interpretation: shared-first BO separates starting-input differences from continuation differences. These are observed trajectory comparisons; they do not identify a causal KF effect. Later LLM responses and tie-stream variation are not separately randomized.',
        'Experimental budgets match; LLM computation does not. New baselines use19 post-first logical calls per complete trajectory. The current method refreshes on events. Mean objectives are never pooled across tasks.',
        'Adapters: LGBO retains official regional posterior-mean tilting with unchanged covariance and full-pool native qLogEI, using finite uniform region quadrature capped512 and exact category membership. Reasoning BO retains the official prompt/confidence extractor in a sequential5-to1 interface, with shared initial text and own Insight history; the complete paper KG/vector retrieval/multi-agent batch system is not reproduced. These results are not a published-protocol reproduction or a literature leaderboard comparison.',
        'Numerical transport correction: the singleton P0/P1 SD reconstruction guard failed before a new LGBO choice was committed. The separately registered adapter calls native qLogEI on the same native covariance/jitter and official tilted posterior. Existing responses and prior outcomes were preserved; there was no performance reroll. See execution_compatibility.json and execution notes.',
        f"Saved-evidence audit passed all {queries} committed new choices and {len(logical)} successful independent LLM requests. Complete-row SearchAUC and final best were independently reconciled here. Full old-control hash verification is recorded separately before snapshot sealing.",
        'Resume only at a new explicit user request; no automatic scheduling or remaining experiments were started by this reporting tool.']
    (OUT/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(bo='#228833',first='#999999',hyp='#AA3377',lgbo='#EE7733',rbo='#4477AA')
    styles=dict(bo='--',first=':',hyp='-',lgbo='-.',rbo=(0,(4,1,1,1)))
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,t in zip(axes.flat,TASKS):
        for m in METHODS:
            state=read(ROOT/lookup[(t,0,m)]['state_path']);best=np.maximum.accumulate([h['raw_value'] for h in state['history']])
            ax.plot(range(1,21),best,label=NAMES[m],color=colors[m],ls=styles[m],lw=1.6,marker='o' if m=='hyp' else None,markevery=4,ms=3)
        ax.set_title(LABELS[t]);ax.set_xlabel('Paid query');ax.set_ylabel(UNITS[t],fontsize=9);ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,fontsize=9)
    fig.text(.5,.015,'Complete seed0 trajectories only; shared fixed LLM first input/knowledge. Paused four-seed study; no SD band.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.92));fig.savefig(OUT/'cumulative_best_seed0.png',dpi=180);plt.close(fig)
    print(json.dumps({k:status[k] for k in ('status','completed_new','partial_new','not_started','observed_queries','sealed_LLM_attempts','validated_logical_requests')}))


if __name__=='__main__':main()
