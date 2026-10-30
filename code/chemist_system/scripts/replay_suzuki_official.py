"""Independently call unchanged native entrypoints against recorded prefixes.

No objective/simulator evaluations, new LLM decisions, cache hits, or numerical
precision repairs. All 960 selected additional conditions are checked.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import time
from hashlib import sha256
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from chem.benchmarks import official_suzuki_adapter as a
from chem.benchmarks import official_suzuki_runner as r
from chem.benchmarks.suzuki_domain import same_point,digest
from chem.decision.replay import write_json_atomic


def replay_one(run_id):
    torch.set_num_threads(1)
    state=r.read(r.ROOT/'runs'/run_id/'state.json');n=len(state['history'])-4
    result={'run_id':run_id,'method':state['method'],'checked_decisions':0,'fresh_oracle_queries':0,
        'state_sha256':digest(state),'auditor_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    start=time.perf_counter();log=io.StringIO()
    try:
        with threadpool_limits(1),redirect_stdout(log),redirect_stderr(log):
            if state['method']=='LLINBO':
                mod=a.modules('LLINBO');count=0;prompts=0
                a.seed_all(20261027+state['case']*10000+state['seed']*100)
                warm=[(tuple(h['z']),h['y']) for h in state['history'][:4]]
                def warmstart(self,num_warmstart=1,objective_function=None):
                    assert num_warmstart==4;return list(warm)
                def query(self,prompt,**kwargs):
                    nonlocal prompts
                    i=len(self.history)-4
                    q=r.read(r.ROOT/'requests'/state['responses'][i]['context_sha256']/'request.json')
                    assert prompt==q['user_prompt']
                    assert self.history==[(tuple(h['z']),h['y']) for h in state['history'][:i+4]]
                    prompts+=1
                    return state['responses'][i]['upstream_reply']
                def objective(z):
                    nonlocal count
                    expected=state['history'][count+4];raw=[float(v) for v in z]
                    assert raw==expected['z'],{'step':count+1,'native_z':raw,'recorded_z':expected['z']}
                    count+=1
                    return expected['y']
                # GP fitting, max variance grid, optimize_acqf, UCB and the
                # acceptance condition remain completely unwrapped here.
                with patch.object(mod.LLAMAGENT_L,'llm_warmstarting',warmstart),patch.object(mod.LLAMAGENT_L,'query_llm',query):
                    agent=mod.LLMIBO_BFO('justify',objective,dim=4,desc=a.description(state['case']),T=n,T_ini=4,T_rep=1,verbose=False)
                    histories,regrets=agent.run()
                assert histories[0]==[(tuple(h['z']),h['y']) for h in state['history']]
                assert prompts==count==n
                # The source shuffles the LLM prompt after the final native
                # acquisition call. Reconstruct that last source-only step
                # from the recorded post-acquisition RNG before comparing.
                final_rng=a.rng_hash(a.rng_state())
                # A paused state may already have fit/proposed the next
                # unqueried step. Select the last *consumed* acquisition.
                last=[e for e in state['memo_last_invocation'] if e['function']=='optimize_acqf_ucb'][n-1]
                saved=torch.load(r.ROOT/'runs'/run_id/'memo'/(last['key']+'.pt'),map_location='cpu',weights_only=False)
                a.restore_rng(saved['rng_after'])
                with patch.object(mod.LLAMAGENT_L,'query_llm',lambda *args,**kwargs:state['responses'][-1]['upstream_reply']):
                    mod.LLAMAGENT_L([(tuple(h['z']),h['y']) for h in state['history'][:-1]],dim=4,
                        func_desc=a.description(state['case'])).sample_candidate_points()
                assert final_rng==a.rng_hash(a.rng_state())
                result.update(checked_decisions=count,exact_raw_vectors=count,
                    original_prompts_equal=prompts,final_rng_equal=True,
                    GP_and_acquisition_wrappers=0,memo_used=False)
            elif state['method']=='LGBO':
                core,prompt=a.modules('LGBO')
                space=core.ContinuousSpace.from_parameters(a.PARAMETERS)
                branches={}
                for i in range(n):
                    history=state['history'][:i+4]
                    parsed=prompt.parse_assistant_response(state['responses'][i]['upstream_reply'])
                    _,plan,z=core.propose_lgbo_batch(
                        X_norm=torch.tensor([h['z'] for h in history],dtype=torch.double),
                        y=torch.tensor([[h['y']] for h in history],dtype=torch.double),
                        parsed_preference=parsed,space=space,goal='max',batch_q=1,
                        seed=20261027+state['case']*10000+state['seed']*100+i+1)
                    assert z[0].tolist()==state['history'][i+4]['z'],{'step':i+1}
                    assert core.serialize_plan(plan)==state['decisions'][i]['plan']
                    branches[plan['mode']]=branches.get(plan['mode'],0)+1
                    result['checked_decisions']+=1
                result.update(exact_raw_vectors=n,native_branches=branches,public_API_called_directly=True)
            else:
                from chem.benchmarks.suzuki_policies import hebo_select
                for i in range(n):
                    history=state['history'][:i+4]
                    point,detail=hebo_select([h['point'] for h in history],[h['y'] for h in history],
                        20261027+state['case']*10000+state['seed']*100+i+1)
                    expected=state['history'][i+4]['point']
                    assert point==expected,{'step':i+1,'native':point,'recorded':expected}
                    assert detail['gp_epochs']==100 and detail['evolution_population']==100 and detail['evolution_generations']==100
                    result['checked_decisions']+=1
                result.update(exact_physical_vectors=n,unmodified_HEBO_suggest=True)
        result['status']='PASS'
    except Exception as exc:
        import traceback
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-start
    result['numerical_log']={'abandoned_fit_epochs':log.getvalue().count('give up fitting GP'),
        'random_prediction_fallbacks':log.getvalue().count('output random predictions')}
    out=r.ROOT/'native_replay'/run_id
    write_json_atomic(out/'receipt.json',result)
    if log.getvalue():(out/'upstream.log').write_text(log.getvalue(),encoding='utf8')
    return result


def main(method=None,workers=4):
    p=r.check_protocol();jobs=[];rows=[]
    for spec in r.read(r.ROOT/'manifest.json'):
        if method and spec['method']!=method:continue
        key=spec['run_id'];file=r.ROOT/'native_replay'/key/'receipt.json'
        previous=r.read(file) if file.exists() else {}
        state=r.read(r.ROOT/'runs'/key/'state.json')
        if (previous.get('status')=='PASS' and previous.get('state_sha256')==digest(state)
            and previous.get('auditor_sha256')==sha256(Path(__file__).read_bytes()).hexdigest()): rows.append(previous)
        else:jobs.append(key)
    # Match the experiment's import/warm-up phase before setting per-step
    # seeds. Importing HEBO's transitive modules after seeding consumes RNG.
    with ProcessPoolExecutor(max_workers=workers,initializer=r._worker_init) as pool:
        futures={pool.submit(replay_one,k):k for k in jobs}
        for f in as_completed(futures):
            v=f.result();rows.append(v)
            print(json.dumps({k:v[k] for k in ('run_id','status','checked_decisions')},separators=(',',':')),flush=True)
    rows.sort(key=lambda v:v['run_id'])
    value={'status':'PASS' if all(v['status']=='PASS' for v in rows) else 'FAIL',
        'protocol_sha256':p['protocol_sha256'],'replayed_runs':len(rows),
        'replayed_additional_decisions':sum(v['checked_decisions'] for v in rows),
        'fresh_oracle_queries':0,'new_assistant_decisions':0,'rows':rows,'workers':workers,
        'worker_import_and_warmup':'same frozen _worker_init before per-step seeds',
        'scope':'unchanged native entrypoints, recorded original LLM replies, exact raw selected vectors; no memoized native function results'}
    write_json_atomic(r.RESEARCH/(f'native_replay_{method.lower()}.json' if method else 'native_replay.json'),value)
    return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['LLINBO','LGBO','HEBO'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    main(args.method,args.workers)
