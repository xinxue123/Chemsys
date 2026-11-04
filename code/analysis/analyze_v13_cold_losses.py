"""Read-only diagnosis of two frozen cold-start failures; no fits or LLM calls.

Recompute metrics and replay saved feedback against prospectively committed
predictions. All objective values come from existing six-arm run logs. This
does not call an oracle or construct a new optimization trajectory.
"""
from copy import deepcopy
from hashlib import sha256
import json
from math import erf, exp, sqrt
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/experiment_tools'), str(ROOT / 'code/chemist_system')]
from cold_start_v13 import adapter, control, policy, study
from knowledge_rank_v9 import engine

OUT = ROOT / 'experiments/reports/v13_loss_diagnosis_20261103'
SPECS = [
    ('Perera_seed2', 'Perera_Suzuki',
     'experiments/runs/cold_s123_20261102/s2/runs', 'ms2_perera',
     [(1, 5), (6, 9), (10, 14), (15, 20)]),
    ('HPLC_seed0', 'HPLC', 'experiments/runs/cold0_20261102/runs', 'hplc',
     [(1, 4), (5, 7), (8, 12), (13, 20)]),
]
READS = {}


def track(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    READS[path.relative_to(ROOT).as_posix()] = sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(track(path).decode('utf-8'))


def context_from_arrays(path):
    track(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    return dict(stats={k: arrays[k] for k in ('mu_y', 'sd_y', 'sd_predictive_y')},
                **{k: arrays[k] for k in ('base_logei', 'available', 'tie_priority')})


def metrics(state):
    values = np.asarray([r['raw_value'] for r in state['history']], float)
    assert len(values) == 20 and np.isfinite(values).all()
    best = np.maximum.accumulate(values)
    return dict(values=values.tolist(), cumulative_best=best.tolist(),
                SearchAUC=float(best.mean()), final_best=float(best[-1]),
                first_final_best_query=int(np.flatnonzero(values == best[-1])[0]) + 1)


def analyze(name, task, base, tag, phases):
    pool = study.pool_for(ROOT, task)  # public inputs only, no GP construction
    track(pool.source_path)
    states = {}
    commits = {}
    for arm in ('bo', 'v13', 'rbo'):
        folder = ROOT / base / f'{tag}_{arm}'
        state = read(folder / 'state.json')
        states[arm] = state
        pre = deepcopy(state)
        for key in ('history', 'notes', 'decisions', 'insights', 'edit_calls', 'review_queries'):
            pre[key] = []
        pre['status'] = 'READY'
        commits[arm] = []
        for q, observation in enumerate(state['history'], 1):
            c = read(folder / f'rounds/{q:02d}_commit.json')
            context = context_from_arrays(folder / f'rounds/{q:02d}_GP.npz')
            assert c['prestate_sha256'] == control.digest(pre)
            assert c['GP_sha256'] == engine.gp_hash(context)
            expected_available = ~np.isin(pool.ids, [r['candidate_id'] for r in pre['history']])
            np.testing.assert_array_equal(context['available'], expected_available)
            ordered, display = engine.candidates(pool, context, q)
            assert c['ordered_ids'] == [pool.ids[int(j)] for j in ordered]
            assert c['display_ids'] == [pool.ids[int(j)] for j in display]
            assert c['selected_id'] in c['display_ids']
            if q == 1:
                assert np.isnan(context['stats']['mu_y']).all() and not c['qLogEI_defined']
            else:
                for key in ('mu_y', 'sd_y', 'sd_predictive_y'):
                    assert np.isfinite(context['stats'][key]).all()
                assert np.isfinite(context['base_logei'][context['available']]).all()
            if arm == 'v13':
                assert c['refresh'] == policy.edit_event(pool, pre, ordered, q)
                if c['memory']:
                    assert c['memory']['plan'] == policy.apply_edits(c['memory']['raw'], pool, pre, q)
                else:
                    assert c['edited_notes'] == pre['notes']
                for forecast in c['decision']['forecasts']:
                    j = pool.ix[c['selected_id']]
                    mu = context['stats']['mu_y'][j]
                    sd = context['stats']['sd_predictive_y'][j]
                    target = max(r['raw_value'] for r in pre['history']) / pool.scale
                    expected = .5 * (1 + erf((mu-target) / sd / sqrt(2)))
                    assert abs(expected - forecast['p_GP']) < 1e-12
            if c['reasoning']:
                req = read(ROOT / c['reasoning']['request_relpath'])
                own = [(r['candidate_id'], r['objective']) for r in req['public']['own_observations']]
                assert own == [(r['candidate_id'], r['raw_value']) for r in pre['history']]
                assert {r['candidate_id'] for r in req['public']['candidate_menu']} == set(c['display_ids'])
            pre = study.feedback(pre, c, observation)
            commits[arm].append(c)
        assert pre == state

    ms = {a: metrics(s) for a, s in states.items()}
    differences = np.asarray(ms['v13']['cumulative_best']) - np.asarray(ms['bo']['cumulative_best'])
    phase_rows = [dict(first=lo, last=hi, SearchAUC_gap_contribution=float(differences[lo-1:hi].sum()/20))
                  for lo, hi in phases]
    assert abs(sum(r['SearchAUC_gap_contribution'] for r in phase_rows)
               - (ms['v13']['SearchAUC']-ms['bo']['SearchAUC'])) < 1e-10
    rows = []
    for q, c in enumerate(commits['v13'], 1):
        req = read(ROOT / c['reasoning']['request_relpath'])
        public = req['public']
        note_rows = public['retrieved_notes']['knowledge']
        row = dict(query=q, selected_id=c['selected_id'], objective=ms['v13']['values'][q-1],
                   BO_cumulative_best=ms['bo']['cumulative_best'][q-1],
                   V13_cumulative_best=ms['v13']['cumulative_best'][q-1],
                   RBO_cumulative_best=ms['rbo']['cumulative_best'][q-1],
                   gap_contribution=float(differences[q-1]/20),
                   own_BO_reference_id=c['BO_reference_id'], selected_equals_own_BO=c['selected_id']==c['BO_reference_id'],
                   qLogEI_gap=c['opportunity_gap'], EI_ratio_to_own_BO=None if q==1 else exp(-c['opportunity_gap']),
                   refresh=c['refresh'], knowledge_uses=c['decision']['knowledge_uses'],
                   retrieved_note_coverage=[dict(id=n['id'],version=n['version'],
                       menu_coverage=len(n['covered_candidate_ids']),permission=n['recommendation']['permission']) for n in note_rows],
                   forecasts=c['decision']['forecasts'], feedback=states['v13']['decisions'][q-1]['feedback'],
                   memory_actions=c['memory']['raw'] if c['memory'] else None,
                   memory_reviews=c['memory']['plan']['reviews'] if c['memory'] else None,
                   strategies=[h['strategy'] for h in c['reasoning']['raw']['insight']['hypotheses']],
                   commit_path=(ROOT/base/f'{tag}_v13'/f'rounds/{q:02d}_commit.json').as_posix())
        if q > 1:
            folder=ROOT/base/f'{tag}_v13'
            ctx=context_from_arrays(folder/f'rounds/{q:02d}_GP.npz')
            row['forecast_diagnostics']=[]
            for f in row['forecasts']:
                before_id=max(states['v13']['history'][:q-1],key=lambda h:h['raw_value'])['candidate_id']
                before=adapter.conditions(pool,before_id)
                focus=next(n['focus'] for n in c['edited_notes'] if (n['id'],n['version'])==(f['id'],f['version']))
                changed=[a for a,v in f['conditions'].items() if v!=before[a]]
                row['forecast_diagnostics'].append(dict(id=f['id'],version=f['version'],
                    changed_axes_from_incumbent=changed,
                    focus_changed_from_incumbent=any(a in changed for a in focus)))
            score=ctx['base_logei'][ctx['available']]
            row['exact_max_qLogEI_ties']=int((score==score.max()).sum())
        rows.append(row)
    equal_prefix=[]
    for q,(v,r) in enumerate(zip(commits['v13'],commits['rbo']),1):
        if states['v13']['history'][:q-1] == states['rbo']['history'][:q-1]:
            assert v['GP_sha256']==r['GP_sha256'] and v['display_ids']==r['display_ids']
            equal_prefix.append(dict(query=q,GP_and_menu_equal=True,V13=v['selected_id'],ReasoningBO=r['selected_id']))
    return dict(task=name,metrics=ms,rounds=rows,phase_decomposition=phase_rows,
                edit_calls=states['v13']['edit_calls'],review_queries=states['v13']['review_queries'],
                final_notes=states['v13']['notes'],same_prefix_comparisons=equal_prefix,
                checked_saved_decisions=60,replayed_GP_fits=0,new_oracle_calls=0,new_LLM_calls=0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for rel in ('experiments/runs/cold0_20261102/protocol.json',
                'experiments/runs/cold_s123_20261102/protocol.json'):
        protocol=read(ROOT/rel)
        for relative,expected in protocol['sources_sha256'].items():
            assert sha256(track(ROOT/relative)).hexdigest()==expected,relative
    results=[analyze(*s) for s in SPECS]
    positive_controls=[]
    for name,rel in (
        ('Case1_seed2','experiments/runs/cold_s123_20261102/s2/runs/ms2_c1'),
        ('Doyle_seed3','experiments/runs/cold_s123_20261102/s3/runs/ms3_doyle'),
        ('SnAr_seed0','experiments/runs/cold0_20261102/runs/snar'),
        ('HPLC_seed2','experiments/runs/cold_s123_20261102/s2/runs/ms2_hplc')):
        arms={a:read(ROOT/(rel+'_'+a)/'state.json') for a in ('bo','v13')}
        scores={a:metrics(s) for a,s in arms.items()}
        positive_controls.append(dict(task=name,
            first_values={a:s['history'][0]['raw_value'] for a,s in arms.items()},
            SearchAUC={a:m['SearchAUC'] for a,m in scores.items()},
            V13_minus_BO=scores['v13']['SearchAUC']-scores['bo']['SearchAUC'],
            V13_cumulative_best=scores['v13']['cumulative_best'],
            V13_feedback_queries=[d['query'] for d in arms['v13']['decisions'] if d['feedback']],
            final_permissions=[dict(id=n['id'],version=n['version'],permission=n['recommendation']['permission'])
                               for n in arms['v13']['notes']],
            verification='Contextual saved-state read, not an additional trajectory replay.'))
    evidence=dict(status='PASS_SCOPED_LOG_REPLAY',scope='2 tasks, 3 methods, 20 saved decisions per method',
                  no_new_trajectories=True,no_GP_refits=True,no_LLM_calls=True,no_new_oracle_queries=True,
                  caveat='Phase contributions are accounting, not causal module effects; no counterfactual trajectory is generated.',
                  results=results,positive_controls=positive_controls)
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(READS,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['Saved-log diagnosis, no experiments rerun','PASS_SCOPED_LOG_REPLAY','']
    for result in results:
        lines+=[result['task'],json.dumps({a:{k:v for k,v in m.items() if k not in ('values','cumulative_best')} for a,m in result['metrics'].items()}),
                'q | BO cumulative best | V13 value | V13 cumulative best | RBO cumulative best | cited notes | visible notes | refresh']
        for r in result['rounds']:
            lines.append(f"{r['query']:2} | {r['BO_cumulative_best']:.5f} | {r['objective']:.5f} | {r['V13_cumulative_best']:.5f} | {r['RBO_cumulative_best']:.5f} | {r['knowledge_uses']} | {r['retrieved_note_coverage']} | {r['refresh']}")
        lines+=['Phase accounting: '+json.dumps(result['phase_decomposition']), 'Same-prefix comparisons: '+json.dumps(result['same_prefix_comparisons']), '']
    (OUT/'round_trace.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    styles={'bo':('Pure BO','#29568b','-'),'v13':('V13','#c56321','-'),'rbo':('Reasoning BO','#24846b','--')}
    for ax,result in zip(axes,results):
        for arm,(label,color,style) in styles.items():
            ax.step(range(1,21),result['metrics'][arm]['cumulative_best'],where='post',label=label,color=color,ls=style,lw=2)
        if result['task']=='Perera_seed2':
            ax.axvspan(6,9,alpha=.08,color='#c56321')
            ax.text(7.5,4,'KF updates: q6-9',ha='center',fontsize=9,color='#8a4417')
            ax.set_ylim(0,103)
        else:
            ax.text(.97,.07,'No note forecasts / KF updates',ha='right',transform=ax.transAxes,fontsize=9,color='#555555')
            ax.set_ylim(0,2100)
        ax.set_title(result['task'].replace('_',' '))
        ax.set_xlabel('Paid query (0 initial observations)')
        ax.set_ylabel('Cumulative best product UV area (%)' if result['task'].startswith('Perera') else 'Cumulative best HPLC peak area')
        ax.set_xticks([1,4,8,12,16,20]);ax.set_xlim(1,20);ax.grid(alpha=.2)
        ax.legend(loc='lower right' if result['task'].startswith('Perera') else 'upper left',frameon=False,fontsize=9)
    fig.savefig(OUT/'cumulative_best.png',dpi=180)
    plt.close(fig)
    # Prove that the diagnosis did not edit any source read above.
    for relative,expected in READS.items():
        assert sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    print(json.dumps(dict(status=evidence['status'],saved_decisions_checked=120,
         source_files_verified_unchanged=len(READS),new_GP_fits=0,new_LLM_calls=0,
         new_oracle_calls=0,output=OUT.as_posix()),ensure_ascii=False))


if __name__=='__main__':
    main()
