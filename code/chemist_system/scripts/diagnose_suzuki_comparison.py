"""Post-hoc explanation of the cross-protocol Suzuki difference.

Separate from all frozen primary experiments. No new real LLM replies and no
changes to official or legacy source files. Prepare before running controls.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import io
import json
from pathlib import Path
import statistics as st
import time
from unittest.mock import patch
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

ROOT=Path('data/runs/suzuki_diagnosis')
OUT=Path('research/suzuki_diagnosis')
LEGACY=Path('data/runs/chapter5')
NATIVE=Path('data/runs/suzuki_official')
CASES=(2,3,4)
SEEDS=tuple(range(5))

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    from chem.decision.replay import write_json_atomic
    write_json_atomic(p,v)
def digest(v):
    from chem.benchmarks.suzuki_domain import digest as h
    return h(v)
def filehash(p):return sha256(Path(p).read_bytes()).hexdigest()
def native_state(case,seed,method='LLINBO'):
    return read(NATIVE/'runs'/f'c{case}_s{seed}_{method}'/'state.json')
def old_result(case,seed,method='B0',regime='actual_assistant'):
    suite='assistant_test' if regime=='actual_assistant' else 'summit_test'
    folder=LEGACY/suite/'trajectories'/f'ReizmanSuzuki_case_{case}'/f's{seed}'
    if regime!='actual_assistant':folder/=regime
    return read(folder/method/'result.json')
def pointkey(p):return tuple([p['catalyst'],*[float(p[n]) for n in ('t_res','temperature','catalyst_loading')]])

def prepare():
    from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle,CATALYSTS,NUMERIC,LOW,HIGH
    if (OUT/'protocol.json').exists():
        p=read(OUT/'protocol.json');h=p.pop('protocol_sha256');assert h==digest(p)
        p['protocol_sha256']=h;assert p['driver_sha256']==filehash(__file__);return p
    OUT.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    p={'study':'posthoc_cross_protocol_diagnosis_v1','cases':list(CASES),'seeds':list(SEEDS),
       'initial_queries':4,'additional_queries':16,'posthoc':True,'confirmatory_significance_claim':False,
       'primary_records_read_only':True,'new_LLM_calls':0,
       'native_parent_protocol_sha256':read('research/suzuki_official/protocol.json')['protocol_sha256'],
       'legacy_engine_sha256':read(LEGACY/'assistant_test/freeze_receipt.json')['engine_sha256'],
       'legacy_controls':['old_pool_new_init','expanded_pool_old_init','expanded_pool_new_init'],
       'native_controls':['LLINBO_without_LLM','LGBO_neutral_prior'],
       'legacy_model':'Unchanged frozen run_campaign(method=B0), old sklearn GP, 11D onehot, Matern2.5, normalize_y=False, kappa2, 5 restarts',
       'expanded_candidates':'Old512 + 512 independently scrambled Sobol numeric triples per catalyst + all64 physical numeric corners. No values used to generate points.',
       'expanded_grid_seed_rule':'27102026+case*10000+catalyst_index',
       'initialization_factor':'Original pool random4 versus exact new LHS4 with saved original outcomes. Initial-only points may be appended to the task; remaining search choices retain the selected pool.',
       'LLINBO_without_LLM':'Execute same source justify loop and RNG steps; provider returns the current native GP-UCB proposal. No real or synthetic LLM inference claimed.',
       'LGBO_neutral_prior':'Original public API, full-domain region confidence0.5 => delta0; all native parameters retained. This baseline keeps the regional branch, whereas active guidance may trigger native point fallback.',
       'legacy_bridge_qualification':'Replay one complete legacy B0 case2 seed0 and require exact IDs and values before dispatching other legacy controls.',
       'planned_controls':75,'legacy_controls_count':45,'native_controls_count':30,
       'exploratory_diagnostic_oracle_queries_planned':1500,
       'extra_qualification_queries':20,'simulator_pool_inspection':'Offline evaluator-only inference, not supplied as observations to the policies.',
       'resources':{'legacy_workers':6,'native_workers':4,'threads_per_worker':1,'device':'CPU','compiled_before_loops':'prebuilt binaries; native imports and matrix warm-up precede loops'},
       'driver_sha256':filehash(__file__),
       'sources':{str(f):filehash(f) for f in [Path('chem/decision/evaluation.py'),Path('chem/bo/llm.py'),
          Path('chem/decision/hypothesis.py'),Path('chem/decision/domain_guard.py'),
          Path('chem/benchmarks/official_suzuki_adapter.py'),Path('research/suzuki_official/results.json'),
          Path('research/chapter5/summit_test_results.json'),Path('research/chapter5/assistant_test_results.json')]}}
    inputs=[];summaries=[]
    for c in CASES:
        folder=ROOT/'inputs'/f'case{c}';folder.mkdir(parents=True,exist_ok=True)
        task=read(LEGACY/'summit_cache'/f'case_{c}'/'task.json')
        base=list(task['candidates']);ids=list(task['candidate_ids']);seen={pointkey(x) for x in base}
        extra=[]
        for k,cat in enumerate(CATALYSTS):
            unit=qmc.Sobol(3,scramble=True,seed=27102026+c*10000+k).random_base2(9)
            corners=np.array([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)])
            for u in np.concatenate([unit,corners]):
                x={'catalyst':cat,**dict(zip(NUMERIC,map(float,LOW+u*(HIGH-LOW))))}
                if pointkey(x) not in seen:extra.append(x);seen.add(pointkey(x))
        expanded={**task,'candidate_ids':ids+[f'e{i:05d}' for i in range(len(extra))], 'candidates':base+extra}
        # Outcome evaluation occurs only after the candidate coordinates are fixed.
        oracle=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',c)
        base_y=np.load(LEGACY/'summit_cache'/f'case_{c}'/'oracle.npz')['values']
        parity=float(abs(oracle.predict(base)-base_y).max()*100);assert parity<1e-4
        y=np.concatenate([base_y,oracle.predict(extra)])
        write(folder/'expanded_task.json',expanded);np.savez_compressed(folder/'expanded_values.npz',values=y)
        summary={'case':c,'old_pool_size':len(base),'expanded_pool_size':len(y),
            'old_pool_max_pct':float(base_y.max()*100),'expanded_pool_max_pct':float(y.max()*100),
            'old_pool_oracle_parity_error_pct':parity,
            'old_pool_best':base[int(base_y.argmax())]}
        summaries.append(summary)
        inputs.extend([folder/'expanded_task.json',folder/'expanded_values.npz'])
    p['input_sha256']={str(f):filehash(f) for f in inputs}
    p['protocol_sha256']=digest(p)
    write(OUT/'protocol.json',p);write(OUT/'pool_inspection.json',summaries)
    static_report()
    return p

def static_report():
    groups=[]
    for regime in ('actual_assistant','relevant','single_harmful','joint_only_harmful','mixed'):
        for method in ('B0','B1','B2','B3','B4'):
            rows=[old_result(c,s,method,regime) for c in CASES for s in SEEDS]
            groups.append({'family':'legacy','regime':regime,'method':method,'n':len(rows),
                'search_auc_pct':st.mean(r['search_auc']*100 for r in rows),
                'final_best_pct':st.mean(r['final_best']*100 for r in rows),
                'initial_best_pct':st.mean(r['initial_best']*100 for r in rows),
                'stopped':sum(r['status']!='COMPLETE' for r in rows),
                'by_case':{str(c):{'search_auc_pct':st.mean(r['search_auc']*100 for r in rows if r['task_id'].endswith(str(c))),
                    'final_best_pct':st.mean(r['final_best']*100 for r in rows if r['task_id'].endswith(str(c))),
                    'initial_best_pct':st.mean(r['initial_best']*100 for r in rows if r['task_id'].endswith(str(c)))} for c in CASES}})
    official=read('research/suzuki_official/results.json')
    for method in ('LLINBO','LGBO','HEBO'):
        rows=[r for r in official['runs'] if r['case'] in CASES and r['method']==method]
        groups.append({'family':'official','regime':'current_assistant' if method!='HEBO' else 'no_LLM','method':method,'n':len(rows),
            **{k:st.mean(r[k] for r in rows) for k in ('search_auc_pct','final_best_pct','initial_best_pct')},
            'by_case':{str(c):{k:st.mean(r[k] for r in rows if r['case']==c) for k in ('search_auc_pct','final_best_pct','initial_best_pct')} for c in CASES}})
    # This is a factual ledger of within-prefix choices, not a causal gain estimate.
    choices=[]
    for c in CASES:
        for s in SEEDS:
            state=read(LEGACY/'assistant_test/trajectories'/f'ReizmanSuzuki_case_{c}'/f's{s}'/'B4/state.json')
            choices.extend(state['decisions'])
    write(OUT/'raw_reconciliation.json',{'cases':list(CASES),'groups':groups,
        'legacy_B4_actual_assistant':{'decisions':len(choices),
            'same_prefix_UCB_changed':sum(e['selection']['candidate_id']!=e['ucb_reference_candidate_id'] for e in choices),
            'mean_opportunity_loss_pct':st.mean(e['selection']['opportunity_loss'] for e in choices)*100,
            'positive_opportunity_loss_steps':sum(e['selection']['opportunity_loss']>0 for e in choices),
            'knowledge_changes_by_round':{str(t):sum(e['round']==t and e['selection']['candidate_id']!=e['ucb_reference_candidate_id'] for e in choices) for t in range(1,17)}}})

def worker_legacy():threadpool_limits(1)

def legacy_one(job):
    from chem.decision.evaluation import run_campaign,CampaignSettings,score_campaign,engine_fingerprint
    from chem.decision.replay import SynthesisTask,ReplayOracle
    from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle
    p=read(OUT/'protocol.json');assert engine_fingerprint()==p['legacy_engine_sha256']
    case,seed,recipe=job;folder=ROOT/'controls'/f'c{case}_s{seed}_{recipe}'
    if (folder/'result.json').exists():return read(folder/'result.json')
    original=read(LEGACY/'assistant_test/trajectories'/f'ReizmanSuzuki_case_{case}'/f's{seed}'/'B0/state.json')
    old_task=read(LEGACY/'assistant_test/inputs'/f'ReizmanSuzuki_case_{case}'/'bundle.json')['task']
    if recipe.startswith('expanded'):
        definition=read(ROOT/'inputs'/f'case{case}'/'expanded_task.json')
        values=np.load(ROOT/'inputs'/f'case{case}'/'expanded_values.npz')['values'].copy()
    else:
        definition=deepcopy(old_task)
        values=np.load(LEGACY/'summit_cache'/f'case_{case}'/'oracle.npz')['values'].copy()
    if recipe.endswith('new_init'):
        initial=native_state(case,seed)['history'][:4];init_ids=[]
        for i,h in enumerate(initial):
            key=f'w{i:04d}';init_ids.append(key)
            definition['candidate_ids'].append(key);definition['candidates'].append(h['point'])
            values=np.append(values,h['y'])
    else:init_ids=original['header']['initial_ids']
    task=SynthesisTask.from_dict(definition)
    setting=CampaignSettings(**{**original['header']['settings'],
        'refresh_after':tuple(original['header']['settings']['refresh_after']),
        'escape_rounds':tuple(original['header']['settings']['escape_rounds'])})
    state=run_campaign(task,ReplayOracle(task.candidate_ids,values),'B0',seed,setting,init_ids,
        output_dir=folder,cache_dir=ROOT/'legacy_predictions')
    result=score_campaign(state,task.candidate_ids,values)
    if recipe=='legacy_bridge_qualification':
        old=old_result(case,seed)
        assert state['oracle_ledger']==original['oracle_ledger']
        assert result['best_curve']==old['best_curve'] and result['search_auc']==old['search_auc']
    result.update(case=case,recipe=recipe,diagnostic_protocol_sha256=p['protocol_sha256'],
        pool_size=len(values),original_source_unchanged=True,LLM_calls=0)
    write(folder/'result.json',result)
    return result

def worker_native():
    from chem.benchmarks.official_suzuki_runner import _worker_init
    _worker_init()

def native_one(job):
    from chem.benchmarks import official_suzuki_adapter as a
    from chem.benchmarks import official_suzuki_runner as r
    from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle
    p=read(OUT/'protocol.json');case,seed,recipe=job
    folder=ROOT/'controls'/f'c{case}_s{seed}_{recipe}'
    if (folder/'result.json').exists():return read(folder/'result.json')
    initial=native_state(case,seed)['history'][:4];history=deepcopy(initial)
    oracle=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',case)
    log=io.StringIO();started=time.perf_counter();jit0=sum(r.JIT_CALLS.values())
    detail={}
    with redirect_stdout(log),redirect_stderr(log):
        if recipe=='LLINBO_without_LLM':
            up=a.modules('LLINBO');native_acq=up.optimize_acqf_ucb;slot={}
            def acq(*args,**kwargs):
                z=native_acq(*args,**kwargs);slot['z']=z.detach().reshape(-1).tolist();return z
            def assistant(prefix,prompt):
                assert prefix==[(tuple(h['z']),h['y']) for h in history]
                return json.dumps(slot['z'])
            def objective(z):
                z=list(map(float,z));point=a.decode_z(z);value=float(oracle.predict([point])[0])
                history.append({'query':len(history)+1,'point':point,'z':z,'y':value,'initial':False})
                return value
            with patch.object(up,'optimize_acqf_ucb',acq):
                trace=a.execute_llinbo(case=case,initial=initial,objective=objective,assistant=assistant,
                    seed=20261027+case*10000+seed*100,memo_directory=None)
            assert all(h['z']==g for h,g in zip(history[4:],trace['acquisition_points']))
            detail={'all_16_choices_native_GP_UCB':True,'native_fit_classes':trace['fit_classes']}
        else:
            text='Thinking: explicit no-LLM neutral-prior diagnostic, confidence0.5 maps to zero tilt.\nFinal Answer:\n[region, [[60,30,0.5,0],[600,110,2.5,1]],0.5]'
            plans=[]
            for t in range(1,17):
                z,d=a.execute_lgbo(history,text,20261027+case*10000+seed*100+t)
                assert d['plan']['delta']==0 and d['plan']['mode']=='region'
                point=a.decode_z(z);value=float(oracle.predict([point])[0])
                history.append({'query':len(history)+1,'point':point,'z':z,'y':value,'initial':False});plans.append(d)
            detail={'plans':plans,'all_deltas_zero':True,'region_branch_retained':True}
    curve=np.maximum.accumulate([h['y'] for h in history]).tolist()
    result={'recipe':recipe,'case':case,'seed':seed,'status':'COMPLETE','actual_queries':len(history),
        'initial_best':curve[3],'final_best':curve[-1],'search_auc':float(np.mean(curve[4:])),
        'best_curve':curve,'history':history,'details':detail,
        'diagnostic_protocol_sha256':p['protocol_sha256'],'LLM_calls':0,
        'operator_compiles_in_loop':sum(r.JIT_CALLS.values())-jit0,'seconds':time.perf_counter()-started}
    write(folder/'result.json',result)
    if log.getvalue():(folder/'native.log').write_text(log.getvalue(),encoding='utf8')
    return result

def run_controls(which):
    p=prepare()
    if which=='legacy':
        with threadpool_limits(1):legacy_one((2,0,'legacy_bridge_qualification'))
        jobs=[(c,s,m) for c in CASES for s in SEEDS for m in p['legacy_controls']]
        worker,init,workers=legacy_one,worker_legacy,6
    else:
        jobs=[(c,s,m) for c in CASES for s in SEEDS for m in p['native_controls']]
        worker,init,workers=native_one,worker_native,4
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            result=f.result()
            print(json.dumps({k:result[k] for k in ('case','seed','recipe','status','search_auc')},separators=(',',':')),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','legacy','native','summarize'])
    args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(),separators=(',',':')))
    elif args.action in ('legacy','native'):run_controls(args.action)
    else:raise NotImplementedError('summary is authored after the frozen diagnostic runs')

if __name__=='__main__':main()
