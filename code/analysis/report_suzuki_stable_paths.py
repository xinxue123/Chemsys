"""Report the refined stable-branch result and its conditional path diagram."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/suzuki_case1_recursive_paths_20261104'


def main():
    cert=json.loads((OUT/'stability_certificate.json').read_text(encoding='utf-8'))
    layer=json.loads((OUT/'layer2_summary.json').read_text(encoding='utf-8'))
    route=json.loads((OUT/'recorded_route_stability.json').read_text(encoding='utf-8'))
    assert cert['all_allowed_starts_evaluated']==3128
    assert cert['any_first_move_passes']==0
    transitions=route['transitions']
    assert [r['passes'] for r in transitions]==[0,5,5,5,5]
    text=[
        '# Case 1 recursive paths under stable-branch qualification','',
        'The endpoint is p03053, yield 89.56900024414062%. A state includes the complete observed set. '
        'The requested path-length cap is eight observations, including one low/middle initial point.',
        '',
        'The branch criterion used here requires a unique maximum acquisition value, a margin greater than 1e-10 logEI units, '
        'and preservation of that rank under five native MC512 sampler seeds. This is a numerical stability definition '
        'conditional on the fitted GP. It has not tested different GP fits or measurement noise.',
        '',
        f"All {cert['all_allowed_starts_evaluated']:,} low/middle singleton states were computed. "
        f"The largest acquisition spread was {cert['maximum_all_score_spread']:.3g}. "
        'Every state had a tied top value, so no first BO transition passed the necessary uniqueness and margin condition. '
        'Under this per-transition criterion, no fully qualified path of length two through eight can start from one allowed observation. '
        'This is a pruning certificate from exhaustive singleton evaluation; later tie-admissible states were not exhaustively enumerated.',
        '',
        f"The earlier, weaker tie-admissibility rule admitted {layer['two_step_paths']:,} two-step target connections "
        f"and {layer['outgoing_edges']:,} total successor edges. Those counts describe acquisition ties. "
        'They do not constitute certified stable paths.',
        '',
        'A recorded two-observation state does support a qualified tail to the endpoint. '
        'The first two observed yields are 8.868692398071289% and 39.636497497558594%. '
        'After that state, the next four native choices retain their unique lead across the five sampler seeds.',
        '',
        '| Query | Intended next yield (%) | Stable sampler evaluations | Minimum logEI margin |',
        '|---|---|---|---|']
    for r in transitions:
        text.append(f"| {r['query']} | {r['intended_yield']:.2f} | {r['passes']}/5 | {r['minimum_margin']:.8g} |")
    text.extend(['',
        '![Recorded entry and qualified tail](stable_path.png)','',
        'Each H node retains the previous observations. The displayed yield is the latest observation. '
        'The dashed entry is a recorded exploratory choice that failed stable qualification. '
        'The solid arrows form the tail qualified under the stated definition.',
        '',
        'The files all_initial_states.csv and all_two_step_paths.csv preserve the earlier exact-tie results. '
        'singleton_stability.csv and stability_certificate.json provide the refined first-transition test. '
        'recorded_route_stability.csv records all five acquisition evaluations per recorded step. '
        'No new chemical outcome or model response was requested.'])
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(13.5,3.8))
    ax.set(xlim=(-.1,5.1),ylim=(-1.05,1.3))
    ax.axis('off')
    labels=['H1 · 1 observed\n8.87%','H2 · 2 observed\n39.64%',
            'H3 · 3 observed\n46.70%','H4 · 4 observed\n44.63%',
            'H5 · 5 observed\n69.58%','H6 · endpoint\n89.57%']
    for i,label in enumerate(labels):
        ax.text(i,0,label,ha='center',va='center',fontsize=10.5,
                bbox=dict(boxstyle='round,pad=.65',fc='#EDF2F7',ec='#4B657D',lw=1.1))
    for i,r in enumerate(transitions):
        passed=r['all_qualified']
        color='#2463A1' if passed else '#8F6666'
        ax.annotate('',xy=(i+.72,0),xytext=(i+.28,0),
                    arrowprops=dict(arrowstyle='->',lw=1.8,color=color,linestyle='-' if passed else '--'))
        ax.text(i+.5,.35,f"q{i+2}: {r['passes']}/5",ha='center',va='bottom',color=color,fontsize=10)
    ax.text(2.5,1.05,'Case 1: stable continuation from a recorded two-observation state',
            ha='center',va='center',fontsize=15)
    ax.text(2.5,.68,'Every node retains the full observed history.',ha='center',color='#4C5663',fontsize=10)
    ax.text(2.5,-.58,'Dashed entry fails qualification. Solid transitions retain unique rank 1 across five sampler seeds.',
            ha='center',fontsize=10)
    ax.text(2.5,-.84,'From a single low/middle observation, all 3,128 starts fail the first-transition condition.',
            ha='center',fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/'stable_path.png',dpi=180)
    fig.savefig(OUT/'stable_path.svg')
    plt.close(fig)
    print('Report and conditional path diagram generated.')


if __name__=='__main__':
    main()
