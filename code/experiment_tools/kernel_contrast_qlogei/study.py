"""Precommitted eight-task development test with archived BO controls."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native
from . import PARAMETERS as P, VARIANT
from . import models, policy

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/runs'/VARIANT
REPORT = ROOT/'experiments/reports'/VARIANT
DESIGN = ROOT/'docs'/f'{VARIANT}.txt'
SHORT = dict(zip(native.TASKS, ('c1','c2','c3','c4','perera','doyle','snar','hplc')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read(path) != value:
            raise ValueError('Immutable artifact differs: '+str(path))
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def check(protected=False):
    p = read(BASE/'protocol.json')
    if p['protocol_sha256'] != models.digest({k:v for k,v in p.items() if k != 'protocol_sha256'}):
        raise ValueError('Protocol digest changed')
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for path, value in p[group].items():
            if sha(ROOT/path) != value:
                raise ValueError('Frozen artifact changed: '+path)
    return p


def prepare():
    if (BASE/'protocol.json').exists():
        return check(True)
    prior_path = ROOT/'experiments/runs/v13_state_split_20261102/protocol.json'
    prior = read(prior_path)
    controls = {task:prior['controls'][task]['bo'] for task in native.TASKS}
    files = list(Path(__file__).parent.glob('*.py'))+[DESIGN,
        ROOT/'code/experiment_tools/run_kernel_contrast_qlogei.py', Path(native.__file__)]
    for name in ('mixed_kalman_suzuki_engine','pointwise_hte_engine',
                 'reaction_transfer_engine','continuous_benchmark_engine'):
        files.append(ROOT/f'code/chemist_system/chem/benchmarks/{name}.py')
    inputs, pools, protected = {}, [], {prior_path.relative_to(ROOT).as_posix():sha(prior_path)}
    for task in native.TASKS:
        pool = native.load_task(ROOT, task, seed=0)
        for path in (pool.source_path,pool.source_path.with_name('oracle.npz')):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        pools.append(dict(task=task, initial_ids=list(pool.initial_ids), count=len(pool.ids),
            axes=list(pool.axes), kinds=list(pool.kinds), unit=pool.unit, scale=pool.scale))
        control = controls[task]; path = ROOT/control['path']
        if sha(path) != control['sha256']:
            raise ValueError('Archived BO control changed')
        for path in path.parent.rglob('*'):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha(path)
    # Protect the complete previous kernel study and its original-source receipts.
    previous = ROOT/'experiments/reports/v13_kernel_conditions_20261104'
    for path in previous.iterdir():
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = sha(path)
    p = dict(variant=VARIANT, created_UTC=datetime.now(timezone.utc).isoformat(),
        tasks=list(native.TASKS), parameters=P, pools=pools, controls=controls,
        sources_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        inputs_sha256=inputs, protected_sha256=protected, new_LLM_calls=0,
        maximum_new_sequential_queries=8*16, development_not_held_out=True,
        source_protocol_sha256=prior['protocol_sha256'],
        metric='mean of16 post-query cumulative maxima including four initial observations')
    p['protocol_sha256'] = models.digest(p)
    write(BASE/'protocol.json',p)
    for f in files:
        target = BASE/'source_snapshot'/f.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(f,target)
    return check(True)


def initialize():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.chdir(ROOT/'experiments/chemist_system')
    native.warm()


def save_round(folder, query, context, fits, arrays, pool):
    stem = folder/'rounds'/f'{query:02d}'
    stem.parent.mkdir(parents=True,exist_ok=True)
    npz = stem.with_name(stem.name+'_GP.npz')
    if not npz.exists():
        with npz.open('xb') as f:
            np.savez_compressed(f, **{k:context['stats'][k] for k in ('mu_y','sd_y','sd_predictive_y')},
                **{k:context[k] for k in ('base_logei','available','tie_priority')},
                budget=arrays['budget'], qualified=arrays['qualified'], disagreement=arrays['score'])
    df = pd.DataFrame(dict(candidate_id=pool.ids, native_mean_raw=context['stats']['mu_y']*pool.scale,
        native_latent_sd_raw=context['stats']['sd_y']*pool.scale,
        native_logEI=context['base_logei'], available=context['available'],
        in_qLogEI_budget=arrays['budget'], qualified_contrast=arrays['qualified'],
        standardized_disagreement=arrays['score'], axis_correlation_gap=arrays['axis_correlation_gap'],
        isolated_axis=[pool.axes[a] if a >= 0 else None for a in arrays['chosen_axis']],
        conditions=[json.dumps(point,sort_keys=True) for point in pool.points]))
    for index,fit in enumerate(fits):
        df[f'model_{index}_mean_raw'] = fit['mu']*pool.scale
        df[f'model_{index}_latent_sd_raw'] = fit['sd']*pool.scale
        df[f'model_{index}_observation_sd_raw'] = fit['observation_sd']*pool.scale
    csv = stem.with_name(stem.name+'_candidates.csv')
    if not csv.exists():
        df.to_csv(csv,index=False,encoding='utf-8-sig')
    keep = ('id','source','parent','MLL','rho','parameters','checkpoint','optimizer_warnings')
    write(stem.with_name(stem.name+'_models.json'),
        [dict((k,m[k]) for k in keep) for m in fits])


def parents_from(fits):
    primary = fits[0]
    parents = [dict(role='native',id=primary['id'],parameters=primary['parameters'])]
    best = max(m['MLL'] for m in fits)
    alternative = [m for m in fits[1:] if best-m['MLL'] <= P['MLL_gap']]
    if alternative:
        chosen = max(alternative,key=lambda m:float(np.max(np.abs(np.array(m['rho'])-primary['rho']))))
        parents.append(dict(role='alternative',id=chosen['id'],parameters=chosen['parameters']))
    return parents


def run_one(task):
    started = time.perf_counter(); protocol = check()
    pool = native.load_task(ROOT,task,seed=0); folder = BASE/'runs'/SHORT[task]
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        initial = dict(task=task,initial_ids=list(pool.initial_ids),protocol_sha256=protocol['protocol_sha256'])
        initial['commit_sha256'] = models.digest(initial)
        write(folder/'initial_commit.json',initial)
        state_path = folder/'state.json'
        if state_path.exists():
            state = read(state_path)
            if state['protocol_sha256'] != protocol['protocol_sha256']:
                raise ValueError('Resume protocol changed')
        else:
            state = dict(task=task,protocol_sha256=protocol['protocol_sha256'],status='RUNNING',
                history=[native.read_observation(ROOT,pool,k) for k in pool.initial_ids],
                decisions=[],parents=[])
            write(state_path,state)
        if state['status'] == 'COMPLETE':
            return dict(task=task,status='COMPLETE',resumed=True)
        gp = native.GPAdapter(pool,BASE/'cache'/SHORT[task],seed=0)
        while len(state['decisions']) < P['queries']:
            query = len(state['decisions'])+1
            cp = folder/'rounds'/f'{query:02d}_commit.json'
            if cp.exists():
                commit = read(cp)
            else:
                context = gp.context(state['history'],query)
                gp_hash = models.state_hash(context['stats']['model'])
                fits = models.alternatives(gp,state['history'],context,state['parents'],folder/'alternatives')
                decision, arrays = policy.select(pool,state['history'],context,fits,query)
                if models.state_hash(context['stats']['model']) != gp_hash:
                    raise ValueError('Decision modified native GP')
                save_round(folder,query,context,fits,arrays,pool)
                commit = dict(query=query,decision=decision,prestate_sha256=models.digest(state),
                    next_parents=parents_from(fits),native_fit_parameters=context['stats']['fit_parameters'],
                    native_GP_hash=gp_hash,counters=gp.counters)
                commit['commit_sha256'] = models.digest(commit)
                write(cp,commit)
            if (commit['prestate_sha256'] != models.digest(state) or
                commit['commit_sha256'] != models.digest({k:v for k,v in commit.items() if k != 'commit_sha256'})):
                raise ValueError('Pre-outcome commitment changed')
            d = commit['decision']
            if d['selected_id'] in {r['candidate_id'] for r in state['history']}:
                raise ValueError('Repeated experimental query')
            observation = native.read_observation(ROOT,pool,d['selected_id'])
            feedback = policy.feedback(d,observation,state['history'])
            write(folder/'rounds'/f'{query:02d}_feedback.json',dict(observation=observation,feedback=feedback,
                commit_sha256=commit['commit_sha256']))
            before = max(h['raw_value'] for h in state['history'])
            state['history'].append(observation)
            state['decisions'].append(dict(query=query,**observation,decision=d,feedback=feedback,
                incumbent_before=before,incumbent_after=max(before,observation['raw_value'])))
            state['parents'] = commit['next_parents']
            state['status'] = 'COMPLETE' if query == P['queries'] else 'RUNNING'
            write(state_path,state,replace=True)
            print(json.dumps(dict(task=SHORT[task],round=query,selected=d['selected_id'],
                differs_from_BO=d['differs_from_BO'],budget=d['budget_size'],qualified=d['qualified_size'],
                raw_value=round(observation['raw_value'],5),best=round(state['decisions'][-1]['incumbent_after'],5))),flush=True)
            gp.release()
        return dict(task=task,status='COMPLETE',elapsed_seconds=time.perf_counter()-started,
            interventions=sum(d['decision']['differs_from_BO'] for d in state['decisions']))


def summary(state, method, task):
    raw = np.array([h['raw_value'] for h in state['history']])
    if len(raw) != 20:
        raise ValueError('A complete4+16 trajectory is required')
    best = np.maximum.accumulate(raw)
    return dict(task=SHORT[task],method=method,SearchAUC=float(best[4:].mean()),
        final_best=float(best[-1]),first_final_best_total=int(np.flatnonzero(raw == best[-1])[0])+1,
        first_final_best_query=max(0,int(np.flatnonzero(raw == best[-1])[0])-3),
        initial_best=float(best[3]))


def report():
    p = check(True); REPORT.mkdir(parents=True,exist_ok=True)
    summaries, rounds, replays = [], [], []
    for task in native.TASKS:
        pool = native.load_task(ROOT,task,seed=0)
        folder = BASE/'runs'/SHORT[task]; state = read(folder/'state.json')
        control_path = ROOT/p['controls'][task]['path']; control = read(control_path)
        if state['status'] != 'COMPLETE' or len(state['decisions']) != 16:
            raise ValueError('Incomplete study task')
        if state['history'][:4] != control['history'][:4]:
            # Historical extras may be present; the actual observations must agree.
            assert [(h['candidate_id'],h['y']) for h in state['history'][:4]] == [(h['candidate_id'],h['y']) for h in control['history'][:4]]
        a, b = summary(state,'kernel_contrast',task), summary(control,'pure_BO',task)
        a.update(delta_SearchAUC=a['SearchAUC']-b['SearchAUC'],
            relative_SearchAUC_gain=(a['SearchAUC']/b['SearchAUC']-1) if b['SearchAUC'] else None,
            delta_final_best=a['final_best']-b['final_best'],
            interventions=sum(d['decision']['differs_from_BO'] for d in state['decisions']),
            contrast_decisions=sum(d['decision']['reason']=='isolated_contrast_discriminates_plausible_kernels' for d in state['decisions']))
        summaries.extend([a,b])
        for q,d in enumerate(state['decisions'],1):
            cp = read(folder/'rounds'/f'{q:02d}_commit.json')
            assert cp['decision'] == d['decision']
            assert d['decision']['native_logei_gap'] <= d['decision']['epsilon']+1e-12
            assert d['candidate_id'] == cp['decision']['selected_id']
            if q == 16:
                assert not d['decision']['differs_from_BO']
            row = dict(task=SHORT[task],candidate_id=d['candidate_id'],raw_value=d['raw_value'],
                incumbent_after=d['incumbent_after'])
            row.update(d['decision'])
            row['pair_feedback'] = json.dumps(d['feedback']['pair_comparison'])
            row['observed_contrast_raw'] = (d['feedback']['observed_contrast_y']*pool.scale
                if d['feedback']['observed_contrast_y'] is not None else None)
            rounds.append(row)
            prefix = [(h['candidate_id'],h['y']) for h in state['history'][:q+3]]
            old = [(h['candidate_id'],h['y']) for h in control['history'][:q+3]]
            if prefix == old:
                with np.load(folder/'rounds'/f'{q:02d}_GP.npz') as now, np.load(control_path.parent/'rounds'/f'{q:02d}_GP.npz') as before:
                    errors = {k:float(np.max(np.abs(now[k].astype(float)-before[k].astype(float))))
                        for k in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority')}
                if max(errors.values()) > 1e-10:
                    raise ValueError('Shared-prefix native GP replay mismatch: '+str(errors))
                replays.append(dict(task=SHORT[task],query=q,maximum_absolute_error=max(errors.values())))
    frame = pd.DataFrame(summaries); frame.to_csv(REPORT/'results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rounds).to_csv(REPORT/'round_decisions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(replays).to_csv(REPORT/'shared_prefix_GP_replays.csv',index=False,encoding='utf-8-sig')
    ours = frame[frame.method == 'kernel_contrast']
    lines = ['# Kernel contrasts within qLogEI: development results','',
        'Eight frozen tasks, seed0, identical four initial conditions and sixteen queries. '
        'The primary GP and acquisition recipe are unchanged. Historical controls remain untouched. '
        'This is a development experiment on previously examined tasks, not held-out validation.','',
        'Eligible candidates satisfy logEI >= max(logEI)-log(4)*(16-t)/15. '
        'A single-factor contrast is preferred only when two same-data fits are within0.01 MLL '
        'of the best tested fit, differ in that axis correlation by at least0.05, and disagree '
        'in standardized predictive mean by at least0.5. Otherwise native BO is used. '
        'The final round uses the exact native BO maximum.','',
        '| Task | BO SearchAUC | Contrast SearchAUC | Difference | BO best | Contrast best | Changed actions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for _,a in ours.iterrows():
        b = frame[(frame.task == a.task)&(frame.method == 'pure_BO')].iloc[0]
        lines.append(f'| {a.task} | {b.SearchAUC:.5f} | {a.SearchAUC:.5f} | {a.delta_SearchAUC:+.5f} | {b.final_best:.5f} | {a.final_best:.5f} | {int(a.interventions)} |')
    win, loss = int((ours.delta_SearchAUC > 1e-9).sum()), int((ours.delta_SearchAUC < -1e-9).sum())
    lines += ['',f'SearchAUC: {win} better, {loss} worse, {8-win-loss} unchanged. '
        f'{int(ours.interventions.sum())} actions differ from the native reference on their own prefix. '
        'Acquisition-budget compliance is not a realized-yield or path-level guarantee.','',
        'Prediction scores compare locked, pre-observation Gaussian forecasts, including observation '
        'noise. Refitting the next measured prefix updates the kernel explanations. These scores '
        'are logged and are not multiplied back into training likelihoods. No LLM or KF was used.','',
        'See results.csv for final-best timing, round_decisions.csv for all decisions and observed '
        'contrasts, and shared_prefix_GP_replays.csv for archive fidelity. Native and alternative '
        'parameters, ancestry and warnings are in runs/*/rounds/*_models.json. Candidate tables '
        'contain no unqueried outcome labels. No statistical stability claim follows from seed0.']
    (REPORT/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    artifacts = {f.relative_to(ROOT).as_posix():sha(f) for f in BASE.rglob('*')
        if f.is_file() and f.suffix in ('.json','.npz','.pt','.csv')}
    write(REPORT/'artifact_hashes.json',artifacts,replace=True)
    audit = dict(status='PASS_KERNEL_CONTRAST_EXECUTION_AUDIT',tasks=8,initial=4,queries=16,
        changed_actions=int(ours.interventions.sum()),SearchAUC_wins=win,SearchAUC_losses=loss,
        shared_prefix_replays=len(replays),maximum_replay_error=max(r['maximum_absolute_error'] for r in replays),
        protected_files_verified=len(p['protected_sha256']),new_LLM_calls=0,
        operator_compiles_in_loop=0,native_GP_unmodified=True,development_not_held_out=True)
    write(REPORT/'audit.json',audit,replace=True)
    print(ours[['task','SearchAUC','delta_SearchAUC','final_best','delta_final_best','interventions']].to_string(index=False))
    print(json.dumps(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','report','status'))
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    if args.action == 'prepare':
        p = prepare(); print(json.dumps(dict(protocol_sha256=p['protocol_sha256'],tasks=len(p['tasks']),initial=4,queries=16)))
    elif args.action == 'run':
        check(True)
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
            futures = {executor.submit(run_one,t):t for t in native.TASKS}
            for future in as_completed(futures):
                print(json.dumps(future.result()),flush=True)
    elif args.action == 'report':
        report()
    else:
        print(json.dumps([dict(task=short,status=read(BASE/'runs'/short/'state.json')['status'])
            for short in SHORT.values() if (BASE/'runs'/short/'state.json').exists()]))


if __name__ == '__main__':
    main()
