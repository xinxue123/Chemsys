"""Resumable, paper-scale matched study. Evaluation is outside the engine."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

from . import reasoning_bo_data as data
from . import reasoning_bo_engine as engine
from . import reasoning_bo_llm as llm
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=Path('data/runs/reasoning_bo_comparison')
OUT=data.OUT
MILESTONE=Path('research/milestones/M1_Reizman_seed0_20261028/milestone.json')
SEEDS=list(range(10))
API_WORKERS=2
CPU_WORKERS=6
_WARMED=False


def source_hashes():
    paths=[Path(__file__),Path(data.__file__),Path(engine.__file__),Path(llm.__file__),
        Path(engine.nominal.__file__),Path(llm.nominal.__file__),Path(data.__file__).with_name('hte_workbook.py'),
        Path(data.__file__).with_name('reaction_transfer_data.py'),
        Path(data.__file__).with_name('reasoning_bo_report.py'),
        Path('reasoning_bo_comparison.py'),Path('tests/test_reasoning_bo_comparison.py')]
    for row in data.read(MILESTONE)['files']:
        if row['path'].startswith('chem/') and row['path'].endswith('.py'):
            path=Path(row['path'])
            assert data.filehash(path)==row['sha256'],'M1 protected source changed: '+str(path)
            paths.append(path)
    return {str(p.as_posix()):data.filehash(p) for p in sorted(set(paths))}


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    quality=data.prepare()
    config=api.load_config(config_path)
    upstream=data.read(data.UPSTREAM/'SOURCE_PROVENANCE.json')
    assert all(data.filehash(data.UPSTREAM/r['path'])==r['local_sha256'] for r in upstream['files'])
    manifest=[{'task':task,'seed':seed,'method':method,
        'run_id':f'{task}_s{seed:02d}_{method}'} for task in data.SPECS for seed in SEEDS for method in engine.METHODS]
    p={'study':'M2_ReasoningBO_matched_full_tables_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scientific_sha256':source_hashes(),'upstream_commit':upstream['commit'],
        'upstream_provenance_sha256':data.filehash(data.UPSTREAM/'SOURCE_PROVENANCE.json'),
        'upstream_executed_AST':llm.SOURCE_RECEIPT,
        'data_quality_sha256':data.filehash(OUT/'data_quality.json'),
        'protected_M1_milestone_sha256':data.filehash(MILESTONE),
        'manifest_sha256':digest(manifest),'tasks':list(data.SPECS),'seeds':SEEDS,'methods':list(engine.METHODS),
        'budget':{'initial':3,'batches':30,'batch_size':3,'queries_per_trajectory':93,
            'trajectories':len(manifest),'logical_objective_queries':len(manifest)*93},
        'api':config.public(),'request_parameters':llm.nominal.PARAMETERS,
        'system_prompt_sha256':{'ReasoningBO':digest(llm.SYSTEM),'regional':digest(llm.REGION_SYSTEM)},
        'engine_parameters':engine.PARAMETERS,
        'initialization':'For every task/seed run official overview and initial_sampling once. Share the confidence-selected3 experiments among six methods. BO_q3_Sobol is a separately labeled baseline with3 seeded Sobol points.',
        'initial_knowledge':'One fresh observed-only typed-evidence regional knowledge plan after shared3 observations, copied identically into Gate_only, LGBO_region, New_R2. No oracle-constructed correct/wrong stress priors in this paper-style main study.',
        'common_bottom':'Unordered all-category MixedSingleTaskGP, double CPU, standardize observed y, 128 Sobol MC paths, fixed prefix-dependent fit/sampling seeds, qLogNEI, no within-batch feedback.',
        'decision_mechanisms':{
            'ReasoningBO':'Official unchanged confidence sort high>medium>low and stable hypothesis/parameter_sets order; choose3 from shared BO shortlist5. Full native visible insight history, notes disabled.',
            'BO_shared':'Same LLM initialization; choose the highest common joint qLogNEI of all10 size3 subsets of the BO shortlist5.',
            'Gate_only':'Same initial regional field, fixed amplitude1; maximize regional joint qLogNEI among subsets within the decaying base log-utility budget.',
            'LGBO_region':'Same nominal regional field and initial knowledge; maximize regional joint qLogNEI over all10 shortlist subsets, without opportunity gate/KF/B4/refresh.',
            'New_R2':'Gate + joint KF + original B4 on (batch,common witness) pairs + conflict base-batch fallback + at most3 observed-only event refreshes.',
            'BO_q3_shared':'Same LLM initialization; direct greedy optimize_acqf_discrete(q=3), no5-to3 filtering.',
            'BO_q3_Sobol':'Seeded nominal Sobol initialization3; same direct q=3 BO as BO_q3_shared.'},
        'batch_KF':'Joint update theta=[b,a] from all3 residuals using their prequery GP observation covariance +.0009I, Joseph covariance. Deploy clipped a only; b nuisance; no KF variance added to GP. One state per region geometry. q=1 algebra exactly reduces to M1 scalar update.',
        'batch_B4':'Unchanged domain_guard enumerates at most4 clauses. Expand10 batches into30 (batch,witness) items with repeated joint base score; one common point must satisfy all retained clauses. Any revoked clause/conflict triggers R2 base-best batch.',
        'batch_events':'At most3 refreshes, gap>=2 batches, last refresh28; stagnation3 batches (<.5pp incumbent gain each), or2 RMS Mahalanobis innovations>2.5, or B4 conflict. No outcome of the next batch is available.',
        'regional_integration':'Official LGBO covariance-tilt methods execute inherited unchanged; nominal uniform measured-region quadrature. Exact if<=512 witnesses, else fixed outcome-independent sample512. All5 possible slots, at most4 active clauses.',
        'official_adaptations':[
            'User-selected deepseek-flash replaces published QWQ-Plus; same provider for all knowledge/Reasoner calls, first valid response, bounded schema repairs. Alias is not an immutable model snapshot.',
            'Common categorical Mixed GP replaces official Ax SingleTaskGP, obeying the user requirement against arbitrary ordinal coding; finite measured-pool discrete optimization replaces Ax relaxed search.',
            'Use qLogNEI as imported by official models.py, despite prose qLogEI. Common128 MC paths and cache_root=False are explicit numerical settings.',
            'Full author-bundled Suzuki3696/Doyle3955, rather than inconsistent launcher/config bounds. Restore one omitted Suzuki ligand relative to launcher and one omitted Doyle additive relative to compass. Never query5 absent Doyle rows.',
            'Official script30 iterations is30 batches, not30 objective evaluations: ordinary initialization3 +90. Unmatched vanilla_first105/reasoning_first102 ablations are not compared at unequal budgets.',
            'Run original reviewed BaseReasoner/PromptManager/JSONL class/function AST without importing telemetry/credentials/DB constructors. Main no-notes variant, no GRPO/LoRA retraining or hidden-CoT database.',
            'Replace auxiliary JSON-cleaner LLM by deterministic syntax-only JSON pass-through; confidence ordering and scientific content unchanged.',
            'Enforce paper Algorithm1 P_t subset C_t and unique unused first3. Source code did not enforce subset/duplicates; invalid replies receive schema-only repair without querying the oracle.',
            'Replace append-only stale/duplicated trial CSV by exact completed prefix before each native Reasoner call; preserve complete visible insights. Prompts retain native iteration-index wording; logged batch index is authoritative.',
            'Original overview template executed; native initial_sampling template does not interpolate overview despite receiving that metadata. This native behavior is retained.',
            'Provider requests use JSON envelope containing the exact native formatted prompt and replicate label; full requests and visible responses saved. Hidden reasoning only count/hash retained.',
            'Our sequential regional/KF/B4 method needs the specified simultaneous batch extension; it is labeled New_R2_batch, not claimed byte-identical to the Reizman scalar executor.',
            'Regional LGBO/Gate/New_R2 are restricted to the common BO shortlist5 and its10 size3 subsets. This is an action-space adaptation, not an unchanged whole-pool LGBO executor; BO_q3_shared separately measures shortlist effects.',
            'Use explicit seeds0..9; repository sample launcher only specifies seed42 and does not publish all10 seeds.'
        ],
        'metrics':'Primary SearchAUC = mean of30 post-batch cumulative best yields (%); initial/batch curves saved. Final best, first batch attaining final best, simple regret to measured-table maximum, paired seed deltas. No arbitrary ordering of simultaneous outcomes is scored.',
        'failure_policy':'Preserve all planned task/seed/method slots. No result-based reseeding or response selection. Resource pauses remain unfinished and explicitly reported; never count them as complete.',
        'resources':{'available_logical_CPUs':os.cpu_count(),'GP_workers':CPU_WORKERS,'API_workers':API_WORKERS,
            'threads_per_GP_process':1,'device':'CPU small categorical exact GP (<94 observations)',
            'operator_compiles_in_loop':0,'prewarm':'One discarded synthetic-prefix acquisition per process, no JIT/torch.compile.',
            'secrets':'config.txt read only by API processes; not copied, printed or hashed.'},
        'references':['https://onlinelibrary.wiley.com/doi/10.1002/mgea.70086','https://arxiv.org/html/2505.12833v2',
            'https://github.com/little1d/Reasoning-BO/tree/'+upstream['commit']]}
    p['protocol_sha256']=digest(p)
    data.write(OUT/'protocol.json',p);data.write(ROOT/'protocol.json',p);data.write(ROOT/'manifest.json',manifest)
    return p


def check_protocol():
    p=data.read(OUT/'protocol.json');raw=deepcopy(p);expected=raw.pop('protocol_sha256')
    assert digest(raw)==expected
    assert p['scientific_sha256']==source_hashes(),'scientific code changed after protocol freeze'
    assert p['manifest_sha256']==digest(data.read(ROOT/'manifest.json'))
    assert p['protected_M1_milestone_sha256']==data.filehash(MILESTONE)
    assert p['upstream_provenance_sha256']==data.filehash(data.UPSTREAM/'SOURCE_PROVENANCE.json')
    for row in data.read(OUT/'data_quality.json')['tasks']:
        for name in ('pool','oracle','compass'):
            ext='npz' if name=='oracle' else 'json'
            assert row[name+'_sha256']==data.filehash(data.ROOT/row['task']/(name+'.'+ext))
    return p


def _setup_worker():
    global _WARMED
    if _WARMED:return
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    x=torch.tensor([[0.,0.],[0.,1.],[1.,0.]],dtype=torch.double)
    y=torch.tensor([[0.],[1.],[.4]],dtype=torch.double)
    model=MixedSingleTaskGP(x,y,cat_dims=[0,1]);model.eval()
    acq=qLogNoisyExpectedImprovement(model,x,cache_root=False,prune_baseline=False,
        sampler=SobolQMCNormalSampler(torch.Size([128]),seed=20261028))
    with torch.no_grad():acq(torch.tensor([[[1.,1.],[0.,1.],[1.,0.]]],dtype=torch.double))
    _WARMED=True


def sobol_ids(pool,seed):
    from scipy.stats import qmc
    sampler=qmc.Sobol(len(pool['axes']),scramble=True,seed=seed)
    ids=[]
    for row in sampler.random_base2(8):
        point={a:pool['levels'][a][min(int(row[j]*len(pool['levels'][a])),len(pool['levels'][a])-1)] for j,a in enumerate(pool['axes'])}
        try:key=pool['ids'][llm.index_point(pool,point)]
        except ValueError:continue
        if key not in ids:ids.append(key)
        if len(ids)==3:return ids
    raise RuntimeError('Sobol could not find3 unique measured points')


def bootstrap(task,seed,config_path='../config.txt'):
    _setup_worker();p=check_protocol();pool=data.load_pool(task)
    folder=ROOT/'initialization'/f'{task}_s{seed:02d}'
    if (folder/'ready.json').exists():return data.read(folder/'ready.json')
    client=api.APIClient(api.load_config(config_path))
    try:
        reasoner=llm.make_reasoner(client,pool,seed,folder/'native',p['protocol_sha256'])
        initial=llm.initial_reasoning(reasoner)
        data.write(folder/'initial_batch_committed.json',{'selected_ids':initial['selected_ids'],
            'protocol_sha256':p['protocol_sha256'],'requests':initial['requests'],'no_outcomes_in_selection':True})
        history=data.reveal(task,initial['selected_ids'])
        state={'task':task,'seed':seed,'method':'New_R2','history':history,'decisions':[],
            'knowledge':{'confidence':.7,'clauses':[]},'banks':{},'calls':[],
            'protocol_sha256':p['protocol_sha256']}
        cache=engine.BatchCache(pool,ROOT/'cache'/f'{task}_s{seed:02d}')
        d=engine.setup(cache,state,1)
        answer,context=llm.regional_call(client,pool,state,d,['initial_observed_prefix'],initial=True)
        result={'task':task,'seed':seed,'initial':initial,'history':history,
            'knowledge':answer['plan']['knowledge'],'knowledge_context_sha256':context,
            'protocol_sha256':p['protocol_sha256'],'status':'READY'}
        data.write(folder/'ready.json',result)
        print(json.dumps({'phase':'bootstrap','task':task,'seed':seed,'status':'READY'}),flush=True)
        return result
    finally:client.close()


def initial_state(spec,pool,p):
    init=data.read(ROOT/'initialization'/f"{spec['task']}_s{spec['seed']:02d}"/'ready.json')
    history=init['history'] if spec['method']!='BO_q3_Sobol' else data.reveal(spec['task'],sobol_ids(pool,spec['seed']))
    return {**spec,'protocol_sha256':p['protocol_sha256'],'history':deepcopy(history),
        'initial_insight':deepcopy(init['initial']['insight']),'knowledge':deepcopy(init['knowledge']),
        'initial_knowledge_context_sha256':init['knowledge_context_sha256'],
        'banks':{},'decisions':[],'calls':[],'status':'READY','runtime_seconds':0.}


def serializable_decision(d):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v.to_dict() if k=='diagnosis' else v)
        for k,v in d.items()}


def commit_and_reveal(folder,receipt):
    assert len(receipt['selected_ids'])==3 and len(set(receipt['selected_ids']))==3
    assert receipt['simultaneous_commit'] is True
    target=Path(folder)/'batches'/f"{receipt['iteration']:02d}_committed.json"
    data.write(target,receipt)
    return data.reveal(receipt['task'],receipt['selected_ids'])


def run_one(spec,config_path='../config.txt'):
    _setup_worker();p=check_protocol();pool=data.load_pool(spec['task'])
    folder=ROOT/'runs'/spec['run_id'];path=folder/'state.json'
    state=data.read(path) if path.exists() else initial_state(spec,pool,p)
    if state['status']=='COMPLETE':return {'run_id':spec['run_id'],'status':'COMPLETE','batches':30}
    cache=engine.BatchCache(pool,ROOT/'cache'/f"{spec['task']}_s{spec['seed']:02d}")
    needs_api=spec['method'] in ('ReasoningBO','New_R2')
    client=api.APIClient(api.load_config(config_path)) if needs_api else None
    reasoner=llm.make_reasoner(client,pool,spec['seed'],folder/'native',p['protocol_sha256']) if spec['method']=='ReasoningBO' else None
    started=time.perf_counter()
    try:
        while len(state['decisions'])<30:
            iteration=len(state['decisions'])+1
            d=engine.setup(cache,state,iteration)
            reasons=engine.event_reasons(state,d,iteration)
            if reasons:
                answer,context=llm.regional_call(client,pool,state,d,reasons)
                state['knowledge']=answer['plan']['knowledge']
                state['calls'].append({'iteration':iteration,'reasons':reasons,'context_sha256':context,
                    'validated':answer,'before_refresh_decision':serializable_decision(d)})
                data.write(path,state)
                d=engine.setup(cache,state,iteration)
            insight=None;request=None
            if reasoner:
                selected,insight,request=llm.iteration_reasoning(reasoner,state,d['shortlist'],iteration)
                chosen=set(selected)
                d['selected']=next(i for i,b in enumerate(d['batches']) if {pool['ids'][j] for j in b}==chosen)
            else:
                selected=[pool['ids'][j] for j in d['batches'][d['selected']]]
            assert len(selected)==3 and len(set(selected))==3
            assert not ({h['candidate_id'] for h in state['history']} & set(selected))
            indices=[pool['ids'].index(key) for key in selected]
            m=engine.measurement_inputs(cache,state['history'],d,indices)
            receipt={'task':spec['task'],'iteration':iteration,'selected_ids':selected,'history_sha256':digest(state['history']),
                'history_count':len(state['history']),'decision':serializable_decision(d),
                'prequery_measurement':m,'insight':insight,'request':request,
                'protocol_sha256':p['protocol_sha256'],'simultaneous_commit':True}
            # The only outcome access in the optimization loop is BELOW this
            # persisted commitment. No partial outcomes change another choice.
            observed=commit_and_reveal(folder,receipt)
            before=max(h['y'] for h in state['history'])
            measurement=None
            if spec['method']=='New_R2' and d['bank_key']:
                bank,measurement=engine.joint_kf(d['bank'],d['prior'],iteration,m,[r['y'] for r in observed])
                state['banks'][d['bank_key']]=bank
            after=max(before,max(r['y'] for r in observed))
            record={'iteration':iteration,'selected_ids':selected,'observed_y':[h['y'] for h in observed],
                'incumbent_before':before,'incumbent_after':after,'incumbent_gain':after-before,
                'committed_sha256':data.filehash(folder/'batches'/f'{iteration:02d}_committed.json'),
                'selected_batch_index':d['selected'],'base_best_batch_index':d['reference'],
                'base_logNEI':float(d['base'][d['selected']]),'guided_logNEI':float(d['score'][d['selected']]),
                'opportunity_loss':float(d['base'][d['reference']]-d['base'][d['selected']]),
                'eligible_batches':int(d['eligible'].sum()),'budget':d['budget'],
                'amplitude':d['amplitude'],'fallback':d['fallback'],'retained':d['retained'],
                'diagnosis':d['diagnosis'].to_dict(),'measurement':measurement,'knowledge':deepcopy(state['knowledge']),
                'insight':insight,'request':request}
            assert spec['method'] in ('LGBO_region','ReasoningBO') or record['opportunity_loss']<=d['budget']
            state['history']+=observed;state['decisions'].append(record)
            state['status']='COMPLETE' if iteration==30 else 'RUNNING'
            data.write(path,state)
            if iteration in (1,10,20,30):
                print(json.dumps({'run_id':spec['run_id'],'batch':iteration,'status':state['status']}),flush=True)
            cache.release()
    except Exception as error:
        state['status']='INTERRUPTED';state['error']={'kind':type(error).__name__,'message':str(error)[:260]}
        # No raw network exception/request header is printed.
        data.write(folder/'error_trace.json',{'trace':traceback.format_exc()[-3200:] if not needs_api else 'See secret-safe request transport/validation logs.'})
    finally:
        state['runtime_seconds']+=time.perf_counter()-started
        data.write(path,state)
        if client:client.close()
    return {'run_id':spec['run_id'],'status':state['status'],'batches':len(state['decisions'])}


def progress():
    p=check_protocol();manifest=data.read(ROOT/'manifest.json')
    states=[]
    for s in manifest:
        path=ROOT/'runs'/s['run_id']/'state.json'
        states.append(data.read(path) if path.exists() else {**s,'status':'NOT_STARTED','decisions':[],'history':[]})
    result={'protocol_sha256':p['protocol_sha256'],'planned':len(manifest),
        'ready_initializations':len(list((ROOT/'initialization').glob('*/ready.json'))),
        'completed':sum(s['status']=='COMPLETE' for s in states),
        'committed_batches_completed':sum(len(s['decisions']) for s in states),
        'logical_observations_received':sum(len(s['history']) for s in states),
        'by_method':{m:{'complete':sum(s['status']=='COMPLETE' for s in states if s['method']==m),
            'batches':sum(len(s['decisions']) for s in states if s['method']==m)} for m in engine.METHODS},
        'unfinished':[{'run_id':s['run_id'],'status':s['status'],'batches':len(s['decisions']),
            **({'error':s['error']} if 'error' in s else {})} for s in states if s['status']!='COMPLETE']}
    responses=list((ROOT/'requests').glob('*/attempt*/response.json'))
    result['api']={'validated':len(list((ROOT/'requests').glob('*/validated.json'))),
        'responses':len(responses),'usage':{}}
    for path in responses:
        for key,value in data.read(path).get('usage',{}).items():
            if type(value) in (int,float):result['api']['usage'][key]=result['api']['usage'].get(key,0)+value
    data.write(OUT/'progress.json',result)
    return result


def run(config_path='../config.txt',phase='all',limit=None):
    check_protocol();pairs=[(t,s) for t in data.SPECS for s in SEEDS]
    if limit is not None:pairs=pairs[:limit]
    if phase in ('all','bootstrap'):
        with ProcessPoolExecutor(max_workers=API_WORKERS,initializer=_setup_worker) as ex:
            jobs={ex.submit(bootstrap,t,s,config_path):(t,s) for t,s in pairs}
            for f in as_completed(jobs):
                try:f.result()
                except Exception as error:print(json.dumps({'phase':'bootstrap','task_seed':jobs[f],'error':type(error).__name__}),flush=True)
        if phase=='bootstrap':return progress()
    available={(t,s) for t,s in pairs if (ROOT/'initialization'/f'{t}_s{s:02d}'/'ready.json').exists()}
    specs=[s for s in data.read(ROOT/'manifest.json') if (s['task'],s['seed']) in available]
    if phase=='offline':specs=[s for s in specs if s['method'] not in ('ReasoningBO','New_R2')]
    if phase=='online':specs=[s for s in specs if s['method'] in ('ReasoningBO','New_R2')]
    with ProcessPoolExecutor(max_workers=API_WORKERS,initializer=_setup_worker) as api_ex, \
            ProcessPoolExecutor(max_workers=CPU_WORKERS,initializer=_setup_worker) as cpu_ex:
        jobs=[]
        for spec in specs:
            ex=api_ex if spec['method'] in ('ReasoningBO','New_R2') else cpu_ex
            jobs.append(ex.submit(run_one,spec,config_path))
        for f in as_completed(jobs):
            try:print(json.dumps(f.result()),flush=True)
            except Exception as error:print(json.dumps({'worker_error':type(error).__name__}),flush=True)
    return progress()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','check','bootstrap','run','offline','online','status','report','validate'])
    parser.add_argument('--config',default='../config.txt');parser.add_argument('--limit',type=int)
    a=parser.parse_args()
    if a.action=='prepare':r=prepare(a.config)
    elif a.action=='check':r=check_protocol()
    elif a.action in ('bootstrap','run','offline','online'):
        r=run(a.config,{'run':'all'}.get(a.action,a.action),a.limit)
    elif a.action=='status':r=progress()
    else:
        from . import reasoning_bo_report as report
        r=report.build() if a.action=='report' else report.validate()
    if a.action not in ('prepare','check'):
        print(json.dumps({k:v for k,v in r.items() if k!='unfinished'},ensure_ascii=False,indent=2),flush=True)
    else:print(json.dumps({'protocol_sha256':r['protocol_sha256'],'budget':r['budget']},indent=2))
