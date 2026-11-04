"""Frozen V13 continuation/holdout loop with explicit external interventions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import csv
import hashlib
import io
import json
import os
import shutil
import zipfile

import numpy as np
from chem.benchmarks import knowledge_gp as gpmod
from cold_start_v13 import study as frozen,control,adapter,policy,prompts
from cold_start_v13.gp import ColdGP
from knowledge_rank_v9 import kf as legacy
from . import VARIANT,MODEL,EFFORT,queue

ROOT=Path(__file__).resolve().parents[3]
FOCUS=('Reizman_Suzuki_Case_4','Perera_Suzuki','HPLC')
SHORT=dict(frozen.SHORT);SHORT['Direct_Arylation']='aryl'
ARMS=('bo','rbo','full','kf_fixed','no_refresh','coupled')
NATIVE_FEEDBACK=frozen.feedback
NOTE_FEEDBACK=control.feedback
CALIBRATION=control.calibration.update
EDIT_EVENT=policy.edit_event
NATIVE_DOMAIN=prompts.task_context
NATIVE_POOL=frozen.pool_for
DESIGN='docs/v13_late_validation_20261103.txt'
COUPLED='''This registered comparison reconnects predictive reliability to the
legacy KF lifecycle. Use the supplied recommendation permission. Two improving
AND positive-relative-skill events support an active direction. Paused knowledge
may become active after sufficient positive predictive skill, including accurate
failure predictions. A numeric active state alone does not prove chemical truth.
Keep adverse facts and obey exact scope. References use positive_support,
tentative_test, review_test or warning as permitted. Assess every applicable live
Note at the selected point using current-incumbent improvement probabilities.
Return insight,knowledge_uses,assessments in the supplied V13 JSON schema.
This control changes lifecycle coupling, not the GP or five-point menu.'''


def base():return (ROOT/'experiments/runs'/VARIANT).resolve()
def read(p):return queue.read(p)
def sha(p):return queue.file_sha(p)
def write(p,x):return queue.write_once(p,x)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()


def jobs():
    result=[]
    for task in FOCUS:
        for seed in (0,3):
            for arm in ARMS:
                for rep in ((0,) if arm=='bo' else (0,1)):
                    result.append(dict(id=f'p_{SHORT[task]}{seed}_{arm}{rep}',task=task,seed=seed,rep=rep,arm=arm,kind='prefix',prefix_queries=10))
    for seed in (0,1,2):
        for arm in ('bo','rbo','full'):
            result.append(dict(id=f'h_aryl{seed}_{arm}',task='Direct_Arylation',seed=seed,rep=0,arm=arm,kind='holdout',prefix_queries=0))
    assert len(result)==75
    return result


def source_state(task,seed):
    if seed==0:return ROOT/'experiments/runs/cold0_20261102/runs'/f'{SHORT[task]}_v13/state.json'
    return ROOT/'experiments/runs/cold_s123_20261102'/f's{seed}/runs/ms{seed}_{SHORT[task]}_v13/state.json'


def replay_prefix(task,seed):
    sf=source_state(task,seed);final=read(sf);s=deepcopy(final)
    s.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
    for q in range(1,11):
        cp=read(sf.parent/'rounds'/f'{q:02}_commit.json')
        if cp['prestate_sha256']!=control.digest(s):raise ValueError('Prefix replay prestate mismatch')
        s=NATIVE_FEEDBACK(s,cp,final['history'][q-1])
    if control.digest(s)!=read(sf.parent/'rounds/11_commit.json')['prestate_sha256']:
        raise ValueError('Prefix is not original exact query11 prestate')
    return s


def prepare_holdout():
    folder=base()/'holdout';archive=ROOT/'code/chemist_system/third_party/reasoning_bo_official/data/chemistry_data.zip'
    config=ROOT/'code/chemist_system/third_party/reasoning_bo_official/src/config/direct_arylation_config.json'
    with zipfile.ZipFile(archive) as z:data=z.read('direct_arylation.csv')
    write(base()/'holdout_selection_before_target_processing.json',dict(task='Direct_Arylation',archive_sha256=sha(archive),
        member='direct_arylation.csv',member_sha256=hashlib.sha256(data).hexdigest(),config_sha256=sha(config),
        selection_basis='public reaction config only; no performance screening',design_sha256=sha(ROOT/DESIGN)))
    axes=('Base_SMILES','Ligand_SMILES','Solvent_SMILES','Concentration','Temp_C')
    reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')));seen=set();points=[];values=[]
    raw_count=0
    for row in reader:
        raw_count+=1;point={a:str(row[a]) for a in axes};key=tuple(point[a] for a in axes)
        if key in seen:continue
        seen.add(key);points.append(point);values.append(float(row['yield']))
    if not np.isfinite(values).all() or len(points)<20:raise ValueError('Invalid holdout table')
    pool=dict(task_id='Direct_Arylation',reaction_class='Direct arylation',axes=list(axes),
        ids=[f'a{j:05d}' for j in range(len(points))],points=points,
        levels={a:sorted({p[a] for p in points}) for a in axes},fixed_substrates={},
        objective_unit='reported reaction yield percent',domain_definition='all measured five-category combinations; first exact-input replicate',
        source_archive_sha256=sha(archive),source_member='direct_arylation.csv')
    write(folder/'pool.json',pool)
    oracle=folder/'oracle.npz'
    if not oracle.exists():np.savez_compressed(oracle,values=np.asarray(values)/100.,raw_values=np.asarray(values))
    write(folder/'receipt.json',dict(raw_rows=raw_count,unique_inputs=len(points),duplicate_rows=raw_count-len(points),
        first_record_policy=True,pool_sha256=sha(folder/'pool.json'),oracle_sha256=sha(oracle),
        outcome_access='adapter conversion/evaluation only; decision agents see queried outcomes only'))


def pool_for(task):
    if task!='Direct_Arylation':return NATIVE_POOL(ROOT,task)
    path=base()/'holdout/pool.json';p=read(path)
    pool=gpmod.Pool(task,'P1',tuple(p['ids']),tuple(p['axes']),('categorical',)*5,
        tuple(p['points']),tuple(tuple(x[a] for a in p['axes']) for x in p['points']),
        tuple(p['ids'][:4]),p['objective_unit'],100.,p,ROOT,path)
    pool.initial_ids=()
    return pool


def domain(task,pool):
    if task!='Direct_Arylation':return NATIVE_DOMAIN(task,pool)
    return dict(reaction='Direct arylation condition optimization',objective='Maximize measured reaction yield (%)',
        variables=[dict(name=a,kind='categorical',levels=pool.raw_pool['levels'][a]) for a in pool.axes],
        interpretation='Five nominal experimental factors, including discrete concentration and temperature settings; no ordinal category distances',
        source_references=[])


def prepare():
    b=base();b.mkdir(parents=True,exist_ok=True)
    if (b/'protocol.json').exists():return check()
    write(b/'scope_before_new_outcomes.json',dict(design_sha256=sha(ROOT/DESIGN),jobs=jobs(),
        new_decisions_at_registration=0,new_LLM_calls_at_registration=0))
    prepare_holdout();prefixes=[]
    for task in FOCUS:
        for seed in (0,3):
            s=replay_prefix(task,seed);path=b/'prefixes'/f'{SHORT[task]}{seed}.json';write(path,s)
            prefixes.append(dict(task=task,seed=seed,snapshot=rel(path),snapshot_sha256=sha(path),
                source=rel(source_state(task,seed)),source_sha256=sha(source_state(task,seed))))
    graph={}
    for v in ('cold0_20261102','cold_s123_20261102'):
        graph.update(read(ROOT/'experiments/runs'/v/'protocol.json')['sources_sha256'])
    for path in Path(__file__).parent.glob('*.py'):graph[rel(path)]=sha(path)
    graph[DESIGN]=sha(ROOT/DESIGN)
    graph['code/experiment_tools/run_v13_late_validation.py']=sha(ROOT/'code/experiment_tools/run_v13_late_validation.py')
    protect={}
    for name in ('main.pdf','supplement.pdf','artifact_manifest.json'):
        protect['paper/v13/'+name]=sha(ROOT/'paper/v13'/name)
    protect['paper/v13_source.zip']=sha(ROOT/'paper/v13_source.zip')
    for v in ('v13_state_split_20261102','v13_seed123_20261102','cold0_20261102','cold_s123_20261102'):
        for sf in (ROOT/'experiments/runs'/v).rglob('state.json'):
            if 'cache' not in sf.parts:protect[rel(sf)]=sha(sf)
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs(),prefixes=prefixes,
        model=MODEL,effort=EFFORT,backend='fresh_subagent_per_request',fork_turns='none',
        total_trajectories=75,new_paid_queries=840,maximum_logical_selection_calls=720,
        GP_numerical_seed=0,MC_paths=512,query_budget=20,sources_sha256=graph,protected_sha256=protect,
        holdout=read(b/'holdout/receipt.json'),intervention_definitions=(ROOT/DESIGN).read_text(encoding='utf-8'))
    p['protocol_sha256']=control.digest(p);write(b/'protocol.json',p)
    return check()


def check(protected=False):
    p=read(base()/'protocol.json')
    if p['protocol_sha256']!=control.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Protocol changed')
    for name,h in p['sources_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Registered source changed: '+name)
    if protected:
        for name,h in p['protected_sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Historical evidence changed')
    return p


def freeze_update(state,p,g,s):
    proposed,e=CALIBRATION(state,p,g,s);actual=deepcopy(state);actual['updates']+=1
    e.update(applied=False,proposed_after=proposed,after=deepcopy(actual),intervention='freeze_after_common_prefix')
    return actual,e


def coupled_initial(note):
    st=legacy.initial();cal=note['calibration'];rec=note['recommendation']
    st.update(mean=cal['mean'],variance=cal['variance'],weight=cal['reliability'],probability_positive=cal['probability_positive'],
        updates=cal['updates'],local_updates=cal['updates'],status='paused' if rec['permission'] in ('suppressed','reviewing') else 'active',
        paused_round=rec['suppressed_query'])
    for e in note['events']:
        z=e['calibration']['Brier_gain'];st['bad_streak']=st['bad_streak']+1 if z<-.02 else 0;st['good_streak']=st['good_streak']+1 if z>.02 else 0
    return st


def coupled_feedback(note,packet,history,observation):
    out,event=NOTE_FEEDBACK(note,packet,history,observation)
    proposed=deepcopy(event)
    before=note.get('_coupled_filter',coupled_initial(note))
    after,le=legacy.update(before,packet['p_LLM'],packet['p_GP'],event['improved'],packet['query'])
    out['_coupled_filter']=after
    out['calibration'].update(mean=after['mean'],variance=after['variance'],reliability=after['weight'],
        probability_positive=after['probability_positive'],updates=after['updates'])
    events=note['events']+[event]
    positives=[e['event_id'] for e in events if e['improved'] and e['calibration']['Brier_gain']>1e-12]
    permission='not_applicable' if note['effect']=='avoid' else 'withdrawn' if after['status']=='retracted' else 'suppressed' if after['status']=='paused' else 'allowed' if len(positives)>=2 else 'tentative'
    rec=out['recommendation'];rec.update(permission=permission,positive_tests=positives,review=None)
    if permission=='suppressed':rec['suppressed_query']=after['paused_round']
    event.update(recommendation_after=permission,reason='legacy_coupled_'+le['reason'],
        split_state_proposal=proposed,legacy_filter=le)
    event['calibration']['after']=deepcopy(out['calibration'])
    out['events'][-1]=event
    return out,event


@contextmanager
def redirected(job,p):
    b=base()/job['id'];arm='bo' if job['arm']=='bo' else 'reasoning_bo' if job['arm']=='rbo' else 'v13'
    def gp(pool,path,seed=0):return ColdGP(pool,base()/'cache'/pool.task,seed=0)
    def edit(pool,state,ordered,q):
        if job['arm']=='no_refresh':return dict(allowed=False,reasons=['post_prefix_refresh_disabled'])
        return EDIT_EVENT(pool,state,ordered,q)
    priorities=frozen.native_gp.semantic_priorities
    def tie(pool,q,seed=0):return priorities(pool,q,job['seed'] if job['kind']=='holdout' and q==1 else 0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(frozen,'check',return_value=p))
        stack.enter_context(patch.object(frozen,'directory',return_value=b))
        stack.enter_context(patch.object(frozen,'pool_for',side_effect=lambda r,t:pool_for(t)))
        stack.enter_context(patch.object(frozen,'run_path',return_value=b/'runs'/job['id']))
        stack.enter_context(patch.object(frozen,'tag',return_value=job['id']))
        stack.enter_context(patch.object(frozen,'ColdGP',side_effect=gp))
        stack.enter_context(patch.object(frozen.transport,'request',side_effect=queue.request))
        stack.enter_context(patch.object(frozen.native_gp,'semantic_priorities',side_effect=tie))
        stack.enter_context(patch.object(prompts,'task_context',side_effect=domain))
        stack.enter_context(patch.object(policy,'edit_event',side_effect=edit))
        if job['arm']=='kf_fixed':stack.enter_context(patch.object(control.calibration,'update',side_effect=freeze_update))
        if job['arm']=='coupled':
            stack.enter_context(patch.object(control,'feedback',side_effect=coupled_feedback))
            header=prompts.REASONING_SYSTEM.split('Knowledge has TWO separate channels.')[0]
            stack.enter_context(patch.object(prompts,'REASONING_SYSTEM',header+COUPLED))
            stack.enter_context(patch.object(prompts,'MEMORY_SYSTEM',prompts.MEMORY_SYSTEM+'\n'+COUPLED))
        queue.CURRENT=job['id']
        yield b,arm
        queue.CURRENT=None


def initialize(job,p,b,arm):
    folder=b/'runs'/job['id'];sf=folder/'state.json'
    if sf.exists():return
    pool=pool_for(job['task'])
    initial=dict(task=job['task'],arm=arm,initial_ids=[],history=[],initial=0,queries=20,
        protocol_sha256=p['protocol_sha256'],input_sha256=sha(pool.source_path))
    initial['commit_sha256']=control.digest(initial);write(folder/'initial_commit.json',initial)
    if job['kind']=='prefix':
        s=read(base()/'prefixes'/f'{SHORT[job["task"]]}{job["seed"]}.json')
        s.update(arm=arm,protocol_sha256=p['protocol_sha256'],initial_commit_sha256=initial['commit_sha256'])
        if job['arm']=='coupled':
            for n in s['notes']:n['_coupled_filter']=coupled_initial(n)
    else:s=frozen.initial_state(job['task'],arm,p,initial)
    frozen.warm.write(sf,s)
    write(folder/'branch_receipt.json',dict(job=job,observed_prefix_count=len(s['history']),
        prefix_candidate_ids=[h['candidate_id'] for h in s['history']],prefix_counts_toward_total_budget=True,
        initial_prestate_sha256=control.digest(s)))


def run_job(job,p):
    os.chdir(ROOT/'experiments/chemist_system')
    with redirected(job,p) as (b,arm):
        initialize(job,p,b,arm)
        return frozen.run_one(ROOT,job['task'],arm)


def preflight():
    p=check(True);proofs=[]
    os.chdir(ROOT/'experiments/chemist_system')
    copied=0
    for task in FOCUS:
        destination=base()/'cache'/task;destination.mkdir(parents=True,exist_ok=True)
        for variant in ('cold0_20261102','cold_s123_20261102'):
            for path in (ROOT/'experiments/runs'/variant/'cache'/task).glob('*'):
                if path.suffix not in ('.pt','.npz'):continue
                target=destination/path.name
                if not target.exists():shutil.copy2(path,target);copied+=1
                elif sha(path)!=sha(target):
                    if path.suffix!='.npz':raise ValueError('Conflicting native model cache')
                    with np.load(path,allow_pickle=False) as a,np.load(target,allow_pickle=False) as b:
                        if a.files!=b.files:raise ValueError('Cache schema conflict')
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    for context in p['prefixes']:
        task,seed=context['task'],context['seed'];s=read(ROOT/context['snapshot']);pool=pool_for(task)
        gp=ColdGP(pool,base()/'cache'/task,seed=0);ctx=gp.context(s['history'],11)
        old=source_state(task,seed).parent/'rounds/11_GP.npz'
        with np.load(old,allow_pickle=False) as a:
            for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_array_equal(a[key],ctx['stats'][key])
            for key in ('base_logei','available','tie_priority'):np.testing.assert_array_equal(a[key],ctx[key])
        gp.release();proofs.append(dict(task=task,seed=seed,exact_first_branch_GP=True))
    pool=pool_for('Direct_Arylation');gp=ColdGP(pool,base()/'preflight_cache'/'aryl',seed=0)
    synthetic=[dict(candidate_id=k,y=y,raw_value=y*100) for k,y in zip(pool.ids,(.2,.3,.4,.1))]
    for n in (0,1,2,4):
        ctx=gp.context(synthetic[:n],n+1);assert ctx['fitted_GP']==bool(n);gp.release()
    result=dict(status='PASS',common_prefixes=proofs,holdout_synthetic_counts=[0,1,2,4],
        no_real_holdout_oracle_queries=True,operator_compiles_in_loop=0,protocol_sha256=p['protocol_sha256'],
        exact_prefix_cache_files_copied=copied)
    write(base()/'preflight.json',result);return result


def batch(workers=6):
    p=check(True);assert read(base()/'preflight.json')['status']=='PASS'
    errors=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Deterministic controls first; parallel numeric work then hides LLM wait.
        ordered=sorted(p['jobs'],key=lambda j:(j['arm']!='bo',j['id']))
        fs={ex.submit(run_job,j,p):j for j in ordered}
        for f in as_completed(fs):
            try:f.result()
            except Exception as e:errors.append(dict(job=fs[f]['id'],error_type=type(e).__name__,message=str(e)))
    frozen.warm.write(base()/'batch_receipt.json',dict(status='COMPLETE' if not errors else 'INCOMPLETE',errors=errors))
    return dict(status='COMPLETE' if not errors else 'INCOMPLETE',errors=errors)
