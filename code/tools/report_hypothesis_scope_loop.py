"""Inspectable English comparison and scientific figures for the scope-loop study."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hypothesis_scope_loop import study as s,knowledge as k

NAMES=dict(c1='Reizman Suzuki Case 1',c2='Reizman Suzuki Case 2',c3='Reizman Suzuki Case 3',c4='Reizman Suzuki Case 4',
    perera='Perera Suzuki',doyle='Doyle Buchwald-Hartwig',snar='SnAr',hplc='HPLC')
LABELS=dict(pure_BO='Pure BO',LLM_first_then_BO='LLM first, then BO',hypothesis_scope_loop='Hypotheses + refresh')
COLORS=dict(pure_BO='#444e63',LLM_first_then_BO='#be7c19',hypothesis_scope_loop='#087f8c')


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        lines.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return '\n'.join(lines)


def main():
    s.report();p=s.check(True);df=pd.read_csv(s.REPORT/'results.csv');rows=[];states={};hypotheses=[];events=[];snapshots=[]
    for task in s.native.TASKS:
        short=s.SHORT[task];state=s.read(s.folder_for(task)/'state.json');states[(short,'hypothesis_scope_loop')]=state
        for method,c in p['controls'][task].items():states[(short,method)]=s.read(ROOT/c['path'])
        f=df[df.task==short].set_index('method');new=f.loc['hypothesis_scope_loop'];bo=f.loc['pure_BO'];old=f.loc['LLM_first_then_BO']
        rows.append(dict(task=NAMES[short],BO_SearchAUC=bo.SearchAUC,LLM_first_SearchAUC=old.SearchAUC,new_SearchAUC=new.SearchAUC,
            delta_vs_BO=new.SearchAUC-bo.SearchAUC,delta_vs_same_first=new.SearchAUC-old.SearchAUC,
            BO_final=bo.final_best,LLM_first_final=old.final_best,new_final=new.final_best,
            BO_first_best_query=int(bo.first_final_best_query),LLM_first_best_query=int(old.first_final_best_query),
            new_first_best_query=int(new.first_final_best_query),LLM_calls=len(state['calls']),
            scoped_queries=int(new.scoped_queries),changed_choices=int(new.changed_from_native_BO),KF_updates=int(new.KF_updates)))
        pool=s.pool_for(task);scopes=k.Scopes(pool)
        for call in state['calls']:
            commit=s.read(s.folder_for(task)/'rounds'/f"{call['query']:02d}_commit.json")
            for note in commit['working_knowledge']['notes']:
                hypotheses.append(dict(task=short,query=call['query'],id=note['id'],semantic_key=note['semantic_key'],direction=note['direction'],
                    claim=note['claim'],scope=json.dumps(note['scope'],ensure_ascii=False),pool_scope_count=int(scopes.mask(note['scope']).sum()),
                    forecast_mean_raw=note['forecast']['mean_raw'],forecast_SD_raw=note['forecast']['sd_raw'],permission=note['permission'],
                    KF_reliability=note['KF']['reliability'],KF_updates=note['KF']['updates'],
                    review_action=note['review_action'],inherited_from=json.dumps(note['inherited_from']),
                    falsification=note['falsification'],request=call['request']))
            events.append(dict(task=short,query=call['query'],reasons=';'.join(call['event']['reasons']),
                observed_history_length=call['query']-1,preferred=json.dumps(call['preferred'],ensure_ascii=False),
                actual_selected_id=commit['decision']['selected_id'],projected_preferred_id=call['projection']['candidate_id'] if call['projection'] else None,
                edits=json.dumps(call['edits']),request=call['request']))
        for r in state['decisions']:
            feedback=s.read(s.folder_for(task)/'rounds'/f"{r['query']:02d}_feedback.json")
            for note in feedback['notes_after_feedback']:
                snapshots.append(dict(task=short,after_query=r['query'],id=note['id'],semantic_key=note['semantic_key'],version=note['version'],
                    permission=note['permission'],KF_updates=note['KF']['updates'],reliability=note['KF']['reliability'],
                    negative_streak=note['negative_streak'],guided_nonimprovements=note['deployed_nonimprovements'],
                    retests_used=note['retests_used'],pause_reasons=';'.join(note['pause_reasons'])))
    frame=pd.DataFrame(rows);frame.to_csv(s.REPORT/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(hypotheses).to_csv(s.REPORT/'hypotheses.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(events).to_csv(s.REPORT/'refreshes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(snapshots).to_csv(s.REPORT/'knowledge_states.csv',index=False,encoding='utf-8-sig')
    lines=['# Hypothesis-constrained qLogEI with feedback-triggered LLM refresh','',
        'Eight tasks, seed 0, zero free observations and 20 paid queries per trajectory. Frozen controls are unchanged. '
        'The new arm deliberately uses the same first physical input as the preceding independently grounded LLM-first control. '
        'Its new initial hypotheses are elicited without revealing that first outcome. This pairs the start to assess the '
        'added constraints and refresh; it is not an independent replication of end-to-end LLM initialization.','',
        'The LLM supplies a preferred input, an explanation, explicit physical favour/avoid scopes, an absolute raw-unit forecast, '
        'and a falsification criterion. From query 2 the native GP chooses the native qLogEI maximum inside eligible scopes. '
        'The GP mean, uncertainty, kernel fit and acquisition values are unchanged. Multiple favour regions are intersected; '
        'avoid regions are subtracted. There is no global acquisition-gap gate in this arm. Refresh preferred inputs are logged '
        'separately from the GP-selected executable experiment.','',
        'KF credibility uses only precommitted in-scope forecasts and actual feedback, scoring the probability of incumbent '
        'improvement against the GP forecast. Recommendation permission is a separate state: accurate failure prediction alone '
        'cannot restore a positive recommendation. Equivalent finite-input regions inherit negative evidence despite renaming. '
        'The program pauses under low relative skill, repeated predictive misses, repeated truly guidance-caused failures, or '
        'a favour forecast that no longer supports improvement. These are operational failures, not proof of universal chemical falsity.','',
        'Refreshes are triggered by paused or unusable guidance, an empty joint region, or three-query stagnation. Each refresh '
        'receives the full own prefix, original conditions and units, locked forecasts, KF feedback, permissions and event reasons. '
        'No current GP ranking, other trajectory or unmeasured objective is supplied. At most four logical LLM calls are allowed, '
        'with a three-query gap and a reserved late call. When knowledge is unavailable or contradictory, native full-pool BO resumes.','',
        'SearchAUC = mean of the 20 cumulative-best observed objectives after each paid query. Higher is better. Original units '
        'are preserved: the six reaction-yield tasks use percent, SnAr uses its space-time-yield unit, HPLC uses peak-area units. '
        'Scores must not be averaged across these different units.','',
        table(frame,['task','BO_SearchAUC','LLM_first_SearchAUC','new_SearchAUC','delta_vs_same_first','BO_final','LLM_first_final','new_final']),'',
        table(frame,['task','LLM_calls','scoped_queries','changed_choices','KF_updates','new_first_best_query']),'']
    wins=int((frame.delta_vs_BO>1e-9).sum());ties=int((abs(frame.delta_vs_BO)<=1e-9).sum())
    paired_wins=int((frame.delta_vs_same_first>1e-9).sum());paired_ties=int((abs(frame.delta_vs_same_first)<=1e-9).sum())
    lines+= [f'Observed SearchAUC versus pure BO: {wins} wins, {8-wins-ties} losses, {ties} ties. '
        f'Versus the paired LLM-first-only control: {paired_wins} wins, {8-paired_wins-paired_ties} losses, {paired_ties} ties. '
        'These are single-seed, single-LLM-realization development comparisons, not statistical evidence of stable superiority.','',
        'A scoped query can retain the same choice as full-pool BO. The changed_choices column counts only actual changes '
        'from unrestricted BO at the same measured prefix. It does not assert that a local improvement causes a whole-trajectory '
        'advantage. KF update counts are evidence exposure, not evidence that KF improved optimization.','',
        'The paired-control gains are substantial in Doyle (+21.691186) and Case 2 (+3.602602). '
        'Case 4 (+0.171114) and SnAr (+2.484974 in its much larger objective scale) have smaller AUC gains. '
        'Case 1 loses AUC despite reaching the pool maximum at query20; Case 4 loses final best yield despite its small AUC gain. '
        'Case 3, Perera and HPLC have identical selected trajectories to their LLM-first-only controls. '
        'Their recorded state/weight updates did not improve selection in this realization.','',
        'Across all tasks, 87 post-first queries used a nonempty eligible constrained region, but only 11 choices changed '
        'from unrestricted BO at their own measured prefixes. There were 43 GP-relative KF updates. These denominators '
        'separate nominal knowledge availability, actual influence and feedback exposure.','',
        'Doyle changes only query2, using the INITIAL aryl-iodide hypothesis. Its query2 yield is 40.302441%, below '
        'the incumbent 47.679377%; native BO on the resulting prefix reaches 93.101175% at query3. Its large trajectory '
        'benefit therefore does not demonstrate a causal benefit from KF or later refresh. Case 2 changes queries2,4,5,6,7; '
        'the query4 refreshed XPhos/RuPhos region raises the incumbent to 23.332539%, and the resulting trajectory reaches '
        '40.085316% at query10. The latter high point is a native BO choice on the altered prefix, not an LLM forecast '
        'directly identifying that point. SnAr changes only query20 through a refreshed avoidance rule and reaches '
        '11559.605122; its SearchAUC increment is only (11559.605122-11509.905633)/20 = 2.484974. '
        'Case 1 changes queries2 and4 to nonimproving points and query7 to a modest incumbent improvement; its full '
        'trajectory loses AUC versus the paired control. These are observed sequence differences, not module ablations.','',
        '## Task-level execution evidence','']
    for task in s.native.TASKS:
        short=s.SHORT[task];state=states[(short,'hypothesis_scope_loop')];f=frame[frame.task==NAMES[short]].iloc[0]
        changes=[r['query'] for r in state['decisions'] if r['decision'].get('changed_from_unrestricted_BO')]
        improvements=[r['query'] for r in state['decisions'] if r['decision'].get('changed_from_unrestricted_BO') and r['raw_value']>r['decision']['incumbent_before']]
        reasons=Counter(v for r in state['decisions'] for e in r['knowledge_feedback'] for v in e['reasons'])
        guidance_guards=Counter(v for e in state['permission_events'] for v in [e['reason']])
        lines+= [f'### {NAMES[short]}','',f'Refresh/initial queries: {[c["query"] for c in state["calls"]]}. '
            f'Changed choices: {changes}. Of those, actual incumbent-improving queries: {improvements}. '
            f'SearchAUC change versus the identical-start control: {f.delta_vs_same_first:+.6f}.',
            f'Feedback pause events: {dict(reasons)}. Recommendation guards: {dict(guidance_guards)}.','']
        for call in state['calls']:
            commit=s.read(s.folder_for(task)/'rounds'/f"{call['query']:02d}_commit.json")
            desc='; '.join(f"{n['id']} [{n['direction']}, {n['permission']}]: {n['claim']}" for n in commit['working_knowledge']['notes']) or 'LLM withdrew all hypotheses.'
            lines.append(f"- Query {call['query']}: {desc}")
        lines.append('')
    if (s.REPORT/'audit.json').exists():
        audit=s.read(s.REPORT/'audit.json')
        lines+= ['## Replay and custody audit','',
            f"All {audit['actual_queries']} decisions and KF transitions replay exactly; maximum posterior/acquisition array error "
            f"is {audit['maximum_numeric_error']}. {audit['actual_LLM_attempts']} actual independent LLM attempts have sealed receipts. "
            f"{audit['protected_artifacts']} registered historical artifacts remain unchanged. Audit status: {audit['status']}.",'']
        for incident in audit['agent_export_revision_incidents']:
            lines+= ['One scratch export changed after sealing: HPLC query4 preferred-prototype sample_flow changed by '
                '3.749689447829496e-10 mL/min. The experiment used the immutable sealed response; hypotheses, forecasts and '
                'selection inputs were unchanged. Post-first preferred prototypes do not override scoped qLogEI. Every actual '
                'selection and KF update still replays exactly. Both files, receipts and the initial failed strict-export audit '
                'are preserved. This is disclosed as an export-custody incident rather than an entirely clean transport audit.','']
    lines+= ['## Reproduction and inspection','',
        'Entry point: code/experiment_tools/run_hypothesis_scope_loop.py. Run `preflight`, `prepare`, `run --workers 8`, '
        'and `report` in that order for a fresh namespace. Existing responses and states resume immutably. LLM requests are '
        'serviced by actual fresh independent experimental subagents with isolated contexts; pending requests are exported '
        'and their actual assignment receipts are required before use. A CLI run waits for this LLM transport, rather than '
        'fabricating a local response. Model: gpt-6.1-terra, medium; provider usage is unavailable.','',
        'Each task has runs/<task>/rounds/NN_commit.json, NN_feedback.json, and (from query2) NN_candidates.csv and NN_GP.npz. '
        'Candidate tables contain physical inputs, native predictions/acquisition and scope/selection flags; they do not expose '
        'unmeasured objectives to the decision loop. requests/ retains full prompts, raw replies, validation and real receipts. '
        'hypotheses.csv, refreshes.csv, knowledge_states.csv and decision_trace.csv give compact inspection views.','',
        'The standalone audit reconstructs every prompt, scope, selected point, native posterior/acquisition and KF state. '
        'Execution correctness and preserved controls are separate from optimization efficacy. No conclusions about stable '
        'improvement or individual module causality follow from this one comparison.']
    (s.REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for kind in ('cumulative_best','observed_objectives'):
        fig,axs=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
        for ax,task in zip(axs.flat,s.native.TASKS):
            short=s.SHORT[task];pool=s.pool_for(task)
            for method in LABELS:
                y=np.asarray([r['raw_value'] for r in states[(short,method)]['history']])
                if kind=='cumulative_best':y=np.maximum.accumulate(y)
                ax.plot(range(1,21),y,label=LABELS[method],color=COLORS[method],linewidth=1.7,marker='.',markersize=3)
            state=states[(short,'hypothesis_scope_loop')]
            for call in state['calls'][1:]:ax.axvline(call['query'],color=COLORS['hypothesis_scope_loop'],alpha=.18,linewidth=1)
            ax.set_title(NAMES[short],fontsize=11);ax.set_xticks([1,4,8,12,16,20]);ax.set_xlabel('Paid query')
            ax.set_ylabel('Yield (%)' if pool.group in ('P0','P1') else pool.unit,fontsize=9);ax.grid(alpha=.16)
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
        fig.savefig(s.REPORT/(kind+'.png'),dpi=180);fig.savefig(s.REPORT/(kind+'.pdf'));plt.close(fig)
    print(frame.to_string(index=False))


if __name__=='__main__':main()
