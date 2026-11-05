"""Read-only gate attribution using saved candidate predictions; no GP/LLM calls."""
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp as native
from kernel_contrast_qlogei import PARAMETERS as P, policy

SOURCE = ROOT/'experiments/runs/kernel_contrast_qlogei_20261104'
OUT = ROOT/'experiments/reports/kernel_contrast_gate_diagnosis_20261104'
READS = {}


def read(path):
    b = Path(path).read_bytes(); READS[Path(path).relative_to(ROOT).as_posix()] = hashlib.sha256(b).hexdigest()
    return json.loads(b.decode('utf-8-sig'))


def csv(path):
    b = Path(path).read_bytes(); READS[Path(path).relative_to(ROOT).as_posix()] = hashlib.sha256(b).hexdigest()
    return pd.read_csv(__import__('io').BytesIO(b),encoding='utf-8-sig',float_precision='round_trip')


def geometric_masks(pool, history):
    n,d = len(pool.ids),len(pool.axes)
    anchors,_ = policy.contrasts(pool,history)
    exact = anchors >= 0
    two = np.zeros((n,d),bool); near05 = two.copy(); near10 = two.copy()
    minimum_changes = np.full(n,d+1)
    # Each bit means that at least one measured input is an anchor for this axis.
    for observed in history:
        point = pool.features[pool.ix[observed['candidate_id']]]
        changed = []; distance = []
        for axis,kind in enumerate(pool.kinds):
            if kind == 'continuous':
                diff = np.abs(np.asarray([f[axis] for f in pool.features],float)-point[axis])
                changed.append(diff > P['continuous_tolerance']); distance.append(diff)
            else:
                diff = np.asarray([f[axis] != point[axis] for f in pool.features])
                changed.append(diff); distance.append(diff.astype(float))
        changed = np.column_stack(changed); distance = np.column_stack(distance)
        count = changed.sum(axis=1); minimum_changes = np.minimum(minimum_changes,count)
        two |= changed & ((count <= 2)&(count >= 1))[:,None]
        for axis in range(d):
            other = [a for a in range(d) if a != axis]
            okay05 = np.ones(n,bool); okay10 = np.ones(n,bool)
            for a in other:
                okay05 &= distance[:,a] <= (.05 if pool.kinds[a]=='continuous' else 0.)
                okay10 &= distance[:,a] <= (.10 if pool.kinds[a]=='continuous' else 0.)
            near05[:,axis] |= changed[:,axis] & okay05
            near10[:,axis] |= changed[:,axis] & okay10
    return dict(exact=exact,two_axes=two,near05=near05,near10=near10,
        no_contrast_requirement=np.ones((n,d),bool)),minimum_changes


def pool_geometry(pool):
    from scipy.spatial import cKDTree
    columns = []
    for a,kind in enumerate(pool.kinds):
        values = [f[a] for f in pool.features]
        if kind == 'continuous':
            columns.append(np.asarray(values,float))
        else:
            # Integer labels are equality buckets only, never GP covariates.
            mapping = {v:i for i,v in enumerate(dict.fromkeys(values))}
            columns.append(np.asarray([mapping[v] for v in values],float))
    x = np.column_stack(columns); has = np.zeros(len(x),bool)
    for a,kind in enumerate(pool.kinds):
        other = [i for i in range(x.shape[1]) if i != a]
        pairs = cKDTree(x[:,other]).query_pairs(P['continuous_tolerance'],p=np.inf,output_type='ndarray')
        if len(pairs):
            pairs = pairs[np.abs(x[pairs[:,0],a]-x[pairs[:,1],a]) > P['continuous_tolerance']]
            has[pairs.reshape(-1)] = True
    corners = np.all(np.isclose(x[:,:3],0,atol=1e-8)|np.isclose(x[:,:3],1,atol=1e-8),axis=1) if pool.group=='P0' else None
    return dict(task=pool.task,candidates=len(x),axes=x.shape[1],
        points_with_any_exact_single_axis_neighbor=int(has.sum()),
        percent_with_exact_neighbor=100*float(has.mean()),
        distinct_values_each_axis=json.dumps([len(set(v)) for v in columns]),
        exact_neighbors_are_only_continuous_corners=bool(np.array_equal(has,corners)) if corners is not None else None)


def choose_saved(fits, means, sds, context, contrast, *, budget=True, mll_gap=.01, rho_gap=.05, score_gap=.5):
    best = max(m['MLL'] for m in fits)
    admitted = [i for i,m in enumerate(fits) if best-m['MLL'] <= mll_gap]
    allowed = context['budget'] if budget else context['available']
    score = np.zeros(len(allowed)); pair = [None]*len(allowed); axis = np.full(len(allowed),-1)
    disputed = np.zeros(len(allowed),bool)
    for i,j in combinations(admitted,2):
        gap = np.abs(np.asarray(fits[i]['rho'])-fits[j]['rho'])
        valid_axes = contrast & (gap >= rho_gap)[None,:]
        valid = allowed & valid_axes.any(axis=1)
        disputed |= valid
        values = np.abs(means[i]-means[j])/np.sqrt(np.maximum(sds[i]**2+sds[j]**2,1e-24))
        take = valid & (values >= score_gap) & (values > score)
        for k in np.flatnonzero(take):
            score[k] = values[k]; pair[k] = [i,j]
            axis[k] = int(np.argmax(np.where(valid_axes[k],gap,-1.)))
    qualified = allowed & (score >= score_gap) & (score > 0)
    indices = np.flatnonzero(qualified)
    selected = int(indices[np.lexsort((context['ties'][indices],-context['logei'][indices],-score[indices]))][0]) if len(indices) else context['bo']
    return dict(selected=selected,qualified=int(qualified.sum()),admitted=admitted,
        disputed=int(disputed.sum()),score=float(score[selected]),pair=pair[selected],axis=int(axis[selected]),
        changed=bool(selected != context['bo']))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = read(SOURCE/'protocol.json')
    for group in ('sources_sha256','inputs_sha256','protected_sha256'):
        for rel,expected in protocol[group].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected,rel
    stages,scenarios,model_pairs,excluded,pool_rows,prediction_checks = [],[],[],[],[],[]
    for task in native.TASKS:
        pool = native.load_task(ROOT,task,seed=0)
        short = dict(zip(native.TASKS,('c1','c2','c3','c4','perera','doyle','snar','hplc')))[task]
        folder = SOURCE/'runs'/short; state = read(folder/'state.json')
        pool_rows.append(pool_geometry(pool))
        READS[pool.source_path.relative_to(ROOT).as_posix()] = hashlib.sha256(pool.source_path.read_bytes()).hexdigest()
        for query in range(1,17):
            fits = read(folder/'rounds'/f'{query:02d}_models.json')
            commit = read(folder/'rounds'/f'{query:02d}_commit.json')['decision']
            table = csv(folder/'rounds'/f'{query:02d}_candidates.csv')
            gpfile = folder/'rounds'/f'{query:02d}_GP.npz'
            READS[gpfile.relative_to(ROOT).as_posix()] = hashlib.sha256(gpfile.read_bytes()).hexdigest()
            with np.load(gpfile,allow_pickle=False) as z:
                available,logei,ties,budget = (z[k].copy() for k in ('available','base_logei','tie_priority','budget'))
            indices = np.flatnonzero(available); high = indices[logei[indices] == logei[indices].max()]
            bo = int(high[np.argmin(ties[high])])
            context = dict(available=available,logei=logei,ties=ties,budget=budget,bo=bo)
            masks,min_changes = geometric_masks(pool,state['history'][:query+3])
            means = [table[f'model_{i}_mean_raw'].to_numpy() for i in range(len(fits))]
            sds = [table[f'model_{i}_latent_sd_raw'].to_numpy() for i in range(len(fits))]
            normal = choose_saved(fits,means,sds,context,masks['exact'])
            assert normal['qualified'] == commit['qualified_size']
            assert pool.ids[normal['selected']] == commit['selected_id'] or query == 16
            best = max(m['MLL'] for m in fits); plausible = normal['admitted']
            best_fit = max(fits,key=lambda m:m['MLL'])
            locked_scores = {r['id']:r['log_predictive_density'] for r in state['decisions'][query-1]['feedback']['prediction_scores']}
            for fit in fits:
                loss = best-fit['MLL']
                if loss > .01:
                    gain = locked_scores[fit['id']]-locked_scores[best_fit['id']]
                    prediction_checks.append(dict(task=short,query=query,source=fit['source'],MLL_loss=loss,
                        predictive_score_gain=gain,better=gain>1e-9,worse=gain< -1e-9,
                        loss_bin='(.01,.02]' if loss<=.02 else '(.02,.05]' if loss<=.05 else '(.05,.1]' if loss<=.1 else '>.1'))
            max_rho_all = max((max(abs(np.asarray(fits[i]['rho'])-fits[j]['rho'])) for i,j in combinations(range(len(fits)),2)),default=0.)
            max_rho_plausible = max((max(abs(np.asarray(fits[i]['rho'])-fits[j]['rho'])) for i,j in combinations(plausible,2)),default=0.)
            max_mean_disagreement = 0.; minimum_nontrivial_mll = float('inf')
            for i,j in combinations(range(len(fits)),2):
                gap = np.abs(np.asarray(fits[i]['rho'])-fits[j]['rho'])
                values = np.abs(means[i]-means[j])/np.sqrt(np.maximum(sds[i]**2+sds[j]**2,1e-24))
                is_plausible = i in plausible and j in plausible
                if is_plausible:
                    max_mean_disagreement = max(max_mean_disagreement,float(values[available].max()))
                if gap.max() >= .05:
                    minimum_nontrivial_mll = min(minimum_nontrivial_mll,max(best-fits[i]['MLL'],best-fits[j]['MLL']))
                model_pairs.append(dict(task=short,query=query,first=i,second=j,first_source=fits[i]['source'],second_source=fits[j]['source'],
                    first_MLL_loss=best-fits[i]['MLL'],second_MLL_loss=best-fits[j]['MLL'],both_admitted=is_plausible,
                    max_axis_rho_gap=float(gap.max()),max_disagreement_all=float(values[available].max()),
                    max_disagreement_budget=float(values[budget].max()),
                    disputed_axes=json.dumps([a for a,g in zip(pool.axes,gap) if g>=.05]),
                    maximum_mean_difference_raw=float(np.abs(means[i][available]-means[j][available]).max())))
            geom = masks['exact'].any(axis=1)
            stages.append(dict(task=short,query=query,available=int(available.sum()),budget=int(budget.sum()),
                budget_fraction=float(budget.sum()/available.sum()),epsilon=commit['epsilon'],
                single_axis_in_full_pool=int((available&geom).sum()),single_axis_in_budget=int((budget&geom).sum()),
                single_axis_min_logei_gap=float((logei[bo]-logei[available&geom]).min()) if (available&geom).any() else None,
                min_changed_axes_to_history=int(min_changes[available].min()),models=len(fits),admitted_models=len(plausible),
                max_rho_gap_all=max_rho_all,max_rho_gap_admitted=max_rho_plausible,
                max_disagreement_admitted_full_pool=max_mean_disagreement,
                minimum_MLL_loss_for_distinct_pair=minimum_nontrivial_mll if np.isfinite(minimum_nontrivial_mll) else None,
                qualified=normal['qualified'],changed=commit['differs_from_BO'],
                rule_chooses_BO=bool(normal['qualified'] and normal['selected']==bo),
                nearest_other_axis_5pct_in_budget=int((budget&masks['near05'].any(axis=1)).sum()),
                nearest_other_axis_10pct_in_budget=int((budget&masks['near10'].any(axis=1)).sum())))
            experiments = []
            for use_budget,use_mll,use_single in product((True,False),repeat=3):
                label = 'factorial_budget%d_MLL%d_single%d'%(use_budget,use_mll,use_single)
                experiments.append((label,dict(budget=use_budget,mll_gap=.01 if use_mll else np.inf),
                    masks['exact'] if use_single else masks['no_contrast_requirement']))
            experiments += [(f'MLL_gap_{gap:g}',dict(mll_gap=gap),masks['exact']) for gap in (.02,.05,.1)]
            experiments += [(f'rho_gap_{gap:g}',dict(rho_gap=gap),masks['exact']) for gap in (.01,.001,0.)]
            experiments += [(f'disagreement_{gap:g}',dict(score_gap=gap),masks['exact']) for gap in (.25,.1)]
            experiments += [(name,{},masks[name]) for name in ('two_axes','near05','near10')]
            experiments += [(f'{name}_MLL_{gap:g}',dict(mll_gap=gap),masks[name])
                for name in ('near05','near10','no_contrast_requirement') for gap in (.02,.05)]
            for label,options,contrast in experiments:
                result = choose_saved(fits,means,sds,context,contrast,**options)
                effective_changed = bool(query not in (1,16) and result['changed'])
                effective_qualified = bool(query not in (1,16) and result['qualified'])
                scenarios.append(dict(task=short,query=query,scenario=label,
                    qualified_candidates=result['qualified'],would_qualify=effective_qualified,
                    would_change=effective_changed,would_select_id=pool.ids[result['selected']],
                    differs_from_recorded_policy=bool(query not in (1,16) and pool.ids[result['selected']] != commit['selected_id']),
                    native_logEI_gap=float(logei[bo]-logei[result['selected']]),
                    score=result['score'],MLL_admitted=len(result['admitted']),
                    pair=json.dumps(result['pair']),axis=pool.axes[result['axis']] if result['axis']>=0 else None))
                if label in ('factorial_budget0_MLL1_single1','factorial_budget1_MLL0_single1') and result['qualified'] and query not in (1,16):
                    selected = result['selected']; pair = result['pair']
                    excluded.append(dict(task=short,query=query,relaxation=label,
                        proposed_id=pool.ids[selected],original_id=commit['selected_id'],BO_id=pool.ids[bo],
                        changed=result['changed'],native_logEI_gap=float(logei[bo]-logei[selected]),
                        original_epsilon=commit['epsilon'],score=result['score'],
                        pair_MLL_losses=json.dumps([best-fits[i]['MLL'] for i in pair]),
                        means=json.dumps([float(means[i][selected]) for i in pair]),
                        axis=pool.axes[result['axis']],conditions=json.dumps(pool.points[selected])))
    frames = dict(round_gate_details=pd.DataFrame(stages),saved_prefix_scenarios=pd.DataFrame(scenarios),
        fitted_model_pairs=pd.DataFrame(model_pairs),excluded_opportunities=pd.DataFrame(excluded),
        pool_geometry=pd.DataFrame(pool_rows),excluded_model_prediction_checks=pd.DataFrame(prediction_checks))
    for name,frame in frames.items():
        frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    scenario = frames['saved_prefix_scenarios']
    summary = scenario.groupby(['task','scenario']).agg(qualifying_rounds=('would_qualify','sum'),
        changed_rounds=('would_change','sum'),different_from_recorded_policy=('differs_from_recorded_policy','sum')).reset_index()
    summary.to_csv(OUT/'scenario_counts_by_task.csv',index=False,encoding='utf-8-sig')
    totals = scenario.groupby('scenario').agg(qualifying_rounds=('would_qualify','sum'),
        changed_rounds=('would_change','sum'),different_from_recorded_policy=('differs_from_recorded_policy','sum')).reset_index()
    totals.to_csv(OUT/'scenario_counts_all_tasks.csv',index=False,encoding='utf-8-sig')
    checks = frames['excluded_model_prediction_checks'].groupby('loss_bin').agg(
        model_rounds=('better','size'),better=('better','sum'),worse=('worse','sum'),
        median_predictive_log_score_gain=('predictive_score_gain','median')).reset_index()
    checks.to_csv(OUT/'excluded_model_prediction_summary.csv',index=False,encoding='utf-8-sig')
    for rel,expected in READS.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    (OUT/'read_only_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    audit = dict(status='PASS_SAVED_LOG_GATE_DIAGNOSIS',rounds=len(stages),scenario_evaluations=len(scenarios),
        original_selected_points_reproduced=128,new_GP_fits=0,new_experimental_queries=0,new_LLM_calls=0,
        saved_input_files_verified=len(READS),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Same-prefix hypothetical decisions only; not new trajectories or SearchAUC results.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(totals.to_string(index=False)); print(json.dumps(audit))


if __name__=='__main__':
    main()
