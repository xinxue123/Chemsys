"""Scientific first-choice and cumulative-best evidence, using only sealed runs."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grounded_first_start import study as s


def main():
    assert s.read(s.REPORT/'audit.json')['status']=='PASS_ZERO_DATA_LLM_FIRST_AUDIT'
    p=s.check();results=pd.read_csv(s.REPORT/'results.csv');firsts=pd.read_csv(s.REPORT/'first_choices.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(4,2,figsize=(12,12));records=[]
    titles=['Case 1: yield (%)','Case 2: yield (%)','Case 3: yield (%)','Case 4: yield (%)',
        'Perera Suzuki: UV product area (%)','Doyle B-H: reported yield (%)',
        r'SnAr: STY (kg m$^{-3}$ h$^{-1}$)','HPLC: peak area (source units)']
    for task,title,ax in zip(s.native.TASKS,titles,axes.flat):
        states=[s.read(ROOT/p['controls'][task]['path']),s.read(s.folder_for(task)/'state.json')]
        for state,label,color,style in zip(states,['Pure BO, zero initial observations','Grounded LLM first, then BO'],['#4A4A4A','#326A9A'],['--','-']):
            raw=np.asarray([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(raw);q=np.arange(1,21)
            expected=results[(results.task==s.SHORT[task])&(results.method==('pure_BO' if state is states[0] else 'grounded_LLM_first_then_BO'))].iloc[0]
            assert abs(best.mean()-expected.SearchAUC)<1e-9 and abs(best[-1]-expected.final_best)<1e-9
            ax.plot(q,best,color=color,linestyle=style,lw=1.8,label=label)
            ax.scatter([1],[best[0]],s=26,color=color,zorder=4)
            records.extend(dict(task=s.SHORT[task],method=label,query=int(t),cumulative_best=float(y)) for t,y in zip(q,best))
        ax.set_title(title,loc='left',fontsize=11);ax.set_xlim(.7,20.3);ax.set_xticks([1,5,10,15,20])
        ax.set_xlabel('Paid experiment query');ax.grid(axis='y',color='#e5e5e5',lw=.6)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.958),ncol=2,frameon=False)
    fig.suptitle('A grounded LLM chooses the first experiment; native qLogEI chooses the next19',fontsize=14,y=.99)
    fig.text(.08,.012,'Seed 0; 0 +20 paid queries; one LLM realization per task. Dots mark the paid first choice. Each panel retains its own objective scale.',fontsize=9,color='#444444')
    fig.tight_layout(rect=[.025,.035,.99,.925]);fig.savefig(s.REPORT/'cumulative_best.png',dpi=170,facecolor='white');plt.close(fig)
    pd.DataFrame(records).to_csv(s.REPORT/'cumulative_best_plot_data.csv',index=False,encoding='utf-8-sig')
    summary=results.pivot(index='task',columns='method',values='SearchAUC')
    differences=summary['grounded_LLM_first_then_BO']-summary['pure_BO']
    text=['# Grounded LLM first experiment from zero observations','',
        'The LLM independently selects the first actual physical experiment before any observed outcome or task GP fit. Native full-pool qLogEI selects the following19 paid experiments. Seed0 and0+20 budgets are shared with the frozen pure-BO control.','',
        f'SearchAUC is higher on {int((differences>1e-9).sum())} tasks, lower on {int((differences< -1e-9).sum())}, and tied on {int((differences.abs()<=1e-9).sum())}. It averages20 post-query cumulative best values. These results do not compare directly to the preceding4+16 metric.','',
        '| Task | BO SearchAUC | LLM-first SearchAUC | BO final best | LLM-first final best |','|---|---:|---:|---:|---:|']
    for task in s.SHORT.values():
        f=results[results.task==task].set_index('method');b=f.loc['pure_BO'];a=f.loc['grounded_LLM_first_then_BO']
        text.append(f'| {task} | {b.SearchAUC:.6f} | {a.SearchAUC:.6f} | {b.final_best:.6f} | {a.final_best:.6f} |')
    text+=['','![Cumulative best objectives](cumulative_best.png)','',
        'All eight responses selected exact executable pool inputs; no projection confirmation request was needed in this realization. Input-only examples were supplied and all eight chosen first points occur among those examples. The contribution of the example design versus chemical reasoning is not separately identified. No GP scores or unmeasured yields were supplied.','',
        'Doyle selected the same first input as the pure-BO prior, and the following19 choices are identical. This is a useful same-input check; it is not an improvement. HPLC has lower SearchAUC despite a higher final best; terminal best and early discovery remain separate outcomes.','',
        'This study isolates the first experiment. KF and event refresh are not deployed on these trajectories. Subjective LLM forecasts are recorded in raw units but never treated as measured training labels. One seed and one independent response per task do not establish stability. The HPLC response has very broad uncertainty and no known detector-response scale at zero observations.','',
        '[First actual choices and reasons](first_choices.csv) · [All metrics](results.csv) · [Every decision](decision_trace.csv) · [Receipts and replay audit](audit.json)','',
        'Run `code/experiment_tools/run_grounded_first_start.py` for the registered experiment. Source, protocol, prompts, raw replies, first-choice commits and all GP arrays are preserved in the run namespace.']
    (s.REPORT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='FIGURE_AND_ANALYSIS_COMPLETE',figure=str(s.REPORT/'cumulative_best.png'))))


if __name__=='__main__':main()
