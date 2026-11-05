"""Four staged arms with frozen BO and original kernel-contrast controls."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np
import pandas as pd
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native
from kernel_contrast_qlogei import study as old
from kernel_contrast_qlogei import models as old_models
from . import ARMS,PARAMETERS as P,VARIANT,models,policy

ROOT = old.ROOT
BASE = ROOT/'experiments/runs'/VARIANT
REPORT = ROOT/'experiments/reports'/VARIANT
DESIGN = ROOT/'docs'/f'{VARIANT}.txt'
SHORT = old.SHORT
read,write,sha = old.read,old.write,old.sha
JOBS = [(task,arm) for task in native.TASKS for arm in ARMS]


def check(protected=False):
    p = read(BASE/'protocol.json')
    if p['protocol_sha256'] != models.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Revision protocol changed')
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for path,value in p[group].items():
            if sha(ROOT/path)!=value:
                raise ValueError('Frozen artifact changed: '+path)
    return p


def prepare():
    if (BASE/'protocol.json').exists():
        return check(True)
    if (BASE/'runs').exists():
        raise ValueError('Registration must precede new experimental trajectories')
    previous = old.check(True)
    protected = dict(previous['sources_sha256'],**previous['protected_sha256'])
    protected.update(read(old.REPORT/'artifact_hashes.json'))
    for report in (old.REPORT,ROOT/'experiments/reports/kernel_contrast_gate_diagnosis_20261104'):
        for path in report.iterdir():
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha(path)
    files = list(Path(__file__).parent.glob('*.py'))+[DESIGN,ROOT/'code/experiment_tools/run_kernel_contrast_revision.py']
    files += [Path(old.__file__),Path(old_models.__file__),Path(native.__file__)]
    p = dict(variant=VARIANT,created_UTC=datetime.now(timezone.utc).isoformat(),parameters=P,
        jobs=JOBS,initial=4,queries=16,seed=0,controls=previous['controls'],
        previous_kernel_controls={task:dict(path=(old.BASE/'runs'/SHORT[task]/'state.json').relative_to(ROOT).as_posix(),
            sha256=sha(old.BASE/'runs'/SHORT[task]/'state.json')) for task in native.TASKS},
        sources_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files},
        protected_sha256=protected,inputs_sha256=previous['inputs_sha256'],
        maximum_new_queries=32*16,new_LLM_calls=0,native_GP_modified=False,
        extra_sidecar_optimizer_fits=0,development_not_held_out=True,
        acceptance='Retain every result; distinguish coverage, actual influence and SearchAUC benefit.')
    p['protocol_sha256'] = models.digest(p)
    write(BASE/'protocol.json',p)
    for source in files:
        target = BASE/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
    return check(True)


def initialize():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.chdir(ROOT/'experiments/chemist_system'); native.warm()


def folder_for(task,arm):
    return BASE/'runs'/SHORT[task]/arm


def save_round(folder,query,pool,context,fits,arrays):
    stem = folder/'rounds'/f'{query:02d}'; stem.parent.mkdir(parents=True,exist_ok=True)
    target = stem.with_name(stem.name+'_arrays.npz')
    if not target.exists():
        temp = target.with_suffix('.partial')
        with temp.open('wb') as f:
            np.savez_compressed(f,**{k:context['stats'][k] for k in ('mu_y','sd_y','sd_predictive_y')},
                **{k:context[k] for k in ('base_logei','available','tie_priority')},
                **{k:arrays[k] for k in ('score','raw_score','budget','qualified','weights','mixture_mean','within_variance')},
                expert_ids=np.asarray([m['id'] for m in fits]),
                expert_mu=np.stack([m['mu'] for m in fits]),expert_sd=np.stack([m['sd'] for m in fits]),
                expert_observation_sd=np.stack([m['observation_sd'] for m in fits]))
        temp.replace(target)
    best_quality = np.max([arrays['matches'].for_axes([a])['quality'] for a in range(len(pool.axes))],axis=0)
    table = pd.DataFrame(dict(candidate_id=pool.ids,available=context['available'],
        native_mean_raw=context['stats']['mu_y']*pool.scale,native_sd_raw=context['stats']['sd_y']*pool.scale,
        native_logEI=context['base_logei'],in_budget=arrays['budget'],qualified=arrays['qualified'],
        raw_disagreement=arrays['raw_score'],matched_disagreement=arrays['score'],
        best_single_axis_match_quality=best_quality,mixture_mean_raw=arrays['mixture_mean']*pool.scale,
        conditions=[json.dumps(point,sort_keys=True) for point in pool.points]))
    path = stem.with_name(stem.name+'_candidates.csv')
    if not path.exists():
        table.to_csv(path,index=False,encoding='utf-8-sig')
    metadata = []
    for item in fits:
        row = {k:v for k,v in item.items() if k not in ('mu','sd','observation_sd')}
        if row.get('checkpoint'):
            row['checkpoint'] = Path(row['checkpoint']).relative_to(ROOT).as_posix()
        metadata.append(row)
    write(stem.with_name(stem.name+'_models.json'),metadata)


def run_one(task,arm):
    started = time.perf_counter(); p = check(); pool = native.load_task(ROOT,task,seed=0)
    folder = folder_for(task,arm); folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        initial = dict(task=task,arm=arm,initial_ids=list(pool.initial_ids),protocol_sha256=p['protocol_sha256'])
        initial['commit_sha256'] = models.digest(initial); write(folder/'initial_commit.json',initial)
        sf = folder/'state.json'
        if sf.exists():
            state = read(sf)
            if state['protocol_sha256'] != p['protocol_sha256']:
                raise ValueError('Resume protocol mismatch')
        else:
            state = dict(task=task,arm=arm,status='RUNNING',protocol_sha256=p['protocol_sha256'],
                history=[native.read_observation(ROOT,pool,k) for k in pool.initial_ids],decisions=[],
                parents=[],log_weights=None,initial_log_weights=None)
            write(sf,state)
        if state['status']=='COMPLETE':
            return dict(task=SHORT[task],arm=arm,status='COMPLETE',resumed=True)
        gp = native.GPAdapter(pool,BASE/'cache'/SHORT[task],seed=0)
        while len(state['decisions'])<16:
            query = len(state['decisions'])+1; cp = folder/'rounds'/f'{query:02d}_commit.json'
            if cp.exists():
                commit = read(cp)
            else:
                context = gp.context(state['history'],query)
                primary_hash = models.state_hash(context['stats']['model'])
                fits = (old_models.alternatives(gp,state['history'],context,state['parents'],folder/'alternatives')
                    if arm=='geometry' else models.library(gp,state['history'],context,BASE/'libraries'/SHORT[task]))
                decision,arrays = policy.select(pool,state['history'],context,fits,arm,state['log_weights'],query,
                    initial_stored=state['initial_log_weights'])
                assert models.state_hash(context['stats']['model']) == primary_hash
                save_round(folder,query,pool,context,fits,arrays)
                commit = dict(query=query,decision=decision,prestate_sha256=models.digest(state),
                    native_state_sha256=primary_hash,native_parameters=context['stats']['fit_parameters'],
                    next_parents=old.parents_from(fits) if arm=='geometry' else [],counters=gp.counters,
                    primary_fit_key=context['stats']['fit_key'],primary_acquisition=context['acquisition_meta'])
                commit['commit_sha256'] = models.digest(commit); write(cp,commit)
            assert commit['prestate_sha256']==models.digest(state)
            assert commit['commit_sha256']==models.digest({k:v for k,v in commit.items() if k!='commit_sha256'})
            d = commit['decision']
            assert d['selected_id'] not in {h['candidate_id'] for h in state['history']}
            observation = native.read_observation(ROOT,pool,d['selected_id'])
            fb = policy.feedback(d,observation)
            write(folder/'rounds'/f'{query:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],
                observation=observation,feedback=fb))
            before = max(h['raw_value'] for h in state['history'])
            state['history'].append(observation)
            state['decisions'].append(dict(query=query,**observation,decision=d,feedback=fb,
                incumbent_before=before,incumbent_after=max(before,observation['raw_value'])))
            state['parents'] = commit['next_parents']; state['log_weights'] = fb['log_weights_after']
            if query==1:
                state['initial_log_weights'] = d['log_weights_before']
            state['status'] = 'COMPLETE' if query==16 else 'RUNNING'; write(sf,state,replace=True)
            print(json.dumps(dict(task=SHORT[task],arm=arm,query=query,selected=d['selected_id'],
                changed=d['differs_from_BO'],qualified=d['qualified_size'],ESS=round(d['effective_expert_count'],3),
                feedback_changes_choice=d['feedback_weights_change_selection'],
                value=round(observation['raw_value'],5),best=round(state['decisions'][-1]['incumbent_after'],5))),flush=True)
            gp.release()
        return dict(task=SHORT[task],arm=arm,status='COMPLETE',seconds=time.perf_counter()-started)


def report():
    p = check(True); REPORT.mkdir(parents=True,exist_ok=True)
    summaries,records,replays = [],[],[]
    for task in native.TASKS:
        reference_path = ROOT/p['controls'][task]['path']; bo = read(reference_path)
        old_state = read(ROOT/p['previous_kernel_controls'][task]['path'])
        b = old.summary(bo,'pure_BO',task); summaries += [b,old.summary(old_state,'original_kernel_contrast',task)]
        for arm in ARMS:
            folder = folder_for(task,arm); state = read(folder/'state.json')
            assert state['status']=='COMPLETE' and len(state['decisions'])==16
            assert [(h['candidate_id'],h['y']) for h in state['history'][:4]] == [(h['candidate_id'],h['y']) for h in bo['history'][:4]]
            s = old.summary(state,arm,task)
            s.update(delta_SearchAUC=s['SearchAUC']-b['SearchAUC'],delta_final_best=s['final_best']-b['final_best'],
                interventions=sum(d['decision']['differs_from_BO'] for d in state['decisions']),
                qualified_rounds=sum(d['decision']['qualified_size']>0 for d in state['decisions']),
                feedback_changes_choice=sum(d['decision']['feedback_weights_change_selection'] for d in state['decisions']))
            summaries.append(s)
            for q,d in enumerate(state['decisions'],1):
                packet = d['decision']; assert packet['native_logEI_gap'] <= packet['epsilon']+1e-12
                if q==16:
                    assert not packet['differs_from_BO']
                cp = read(folder/'rounds'/f'{q:02d}_commit.json')
                assert cp['counters']['operator_compiles_in_loop']==0
                row = dict(task=SHORT[task],arm=arm,query=q,selected_id=d['candidate_id'],raw_value=d['raw_value'],
                    best=d['incumbent_after'],changed=packet['differs_from_BO'],qualified=packet['qualified_size'],
                    score=packet['score'],max_budget_score=packet['maximum_budget_score'],
                    ESS=packet['effective_expert_count'],feedback_changes_choice=packet['feedback_weights_change_selection'],
                    matching=json.dumps(packet['matching']),weights=json.dumps(packet['weights']),
                    maximum_weight_change=d['feedback']['maximum_weight_change'])
                records.append(row)
                if [(h['candidate_id'],h['y']) for h in state['history'][:q+3]] == [(h['candidate_id'],h['y']) for h in bo['history'][:q+3]]:
                    with np.load(folder/'rounds'/f'{q:02d}_arrays.npz') as now,np.load(reference_path.parent/'rounds'/f'{q:02d}_GP.npz') as then:
                        error = max(float(np.max(np.abs(now[k].astype(float)-then[k].astype(float))))
                            for k in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority'))
                    assert error < 1e-10
                    replays.append(dict(task=SHORT[task],arm=arm,query=q,max_error=error))
    frame = pd.DataFrame(summaries); frame.to_csv(REPORT/'results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(REPORT/'round_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(replays).to_csv(REPORT/'native_GP_replays.csv',index=False,encoding='utf-8-sig')
    text = ['# Pool-compatible kernel tests: staged development results','',
        'All eight frozen pools; seed0; the same four initial observations and sixteen queries. '
        'The native GP, qLogEI budget and final-round BO reference are unchanged. '
        'Every task and all four predeclared development arms are retained. No LLM/KF was used.','',
        'G revises the contrast interface; H adds controlled kernel-sensitivity experts with hard '
        'MLL admission; S uses fixed initial soft MLL weights; F updates those same initial weights '
        'from every precommitted experimental forecast. These sidecars are conditional sensitivity '
        'experts, not a calibrated hyperparameter posterior or independently established chemistry.','',
        '| Task | BO | Original | G | H | S fixed | F feedback | F minus BO | F minus S |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in SHORT.values():
        f = frame[frame.task==task].set_index('method')
        names = ['pure_BO','original_kernel_contrast',*ARMS]; values = [f.loc[k,'SearchAUC'] for k in names]
        text.append('| '+task+' | '+' | '.join(f'{v:.5f}' for v in values)+f' | {values[-1]-values[0]:+.5f} | {values[-1]-values[-2]:+.5f} |')
    text += ['', 'Values are SearchAUC in each task\'s raw objective units; do not pool raw values '
        'across tasks. See results.csv for final best values, first attainment and action counts.','',
        'A local or joint match is not an identified single-variable causal effect. Its nuisance '
        'distance and quality penalty are committed before the observation. The fixed sensitivity '
        'library changes only focal component lengthscales; other parameters follow the native fit. '
        'Soft initial MLL is applied once; subsequent weights use one predictive density per paid '
        'outcome without multiplying the current training likelihood again.','',
        'The fixed-weight shadow on an F prefix asks whether accumulated feedback changes the '
        'current choice using the same current expert predictions. F versus S full trajectories '
        'is the separate optimization comparison. More interventions or correct weight propagation '
        'do not themselves establish a SearchAUC benefit. No threshold was retuned after results.','',
        'The tasks and seed were previously examined, so these are development results and do '
        'not establish stability across initialization or response randomness.','',
        'Reproduce: run_kernel_contrast_revision.py prepare; preflight; run --workers 12; report. '
        'Use runtime/environments/.venv-official/Scripts/python.exe. The entrypoint is under '
        'code/experiment_tools. All candidate-stage tables, locked expert predictions, model '
        'states, ancestry and weight updates are under the new experiments/runs directory.']
    (REPORT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    audit = dict(status='PASS_REVISION_EXECUTION_REPORT',tasks=8,arms=list(ARMS),decisions=len(records),
        shared_prefix_replays=len(replays),maximum_native_replay_error=max(r['max_error'] for r in replays),
        old_files_verified=len(p['protected_sha256']),new_LLM_calls=0,sidecar_optimizer_fits=0)
    write(REPORT/'audit.json',audit,replace=True)
    print(frame[['task','method','SearchAUC','final_best','delta_SearchAUC','interventions','feedback_changes_choice']].to_string(index=False))
    print(json.dumps(audit)); return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','run','status','report'))
    parser.add_argument('--workers',type=int,default=12)
    args = parser.parse_args()
    if args.action=='prepare':
        p = prepare(); print(json.dumps(dict(protocol_sha256=p['protocol_sha256'],jobs=len(JOBS),initial=4,queries=16)))
    elif args.action=='preflight':
        from .validation import preflight
        print(json.dumps(preflight()))
    elif args.action=='run':
        check(True)
        with ProcessPoolExecutor(max_workers=min(args.workers,len(JOBS)),initializer=initialize) as executor:
            futures = {executor.submit(run_one,*job):job for job in JOBS}
            for future in as_completed(futures):
                print(json.dumps(future.result()),flush=True)
    elif args.action=='report':
        report()
    else:
        print(json.dumps([dict(task=SHORT[t],arm=a,rounds=len(read(folder_for(t,a)/'state.json')['decisions']),
            status=read(folder_for(t,a)/'state.json')['status']) for t,a in JOBS if (folder_for(t,a)/'state.json').exists()]))


if __name__=='__main__':
    main()
