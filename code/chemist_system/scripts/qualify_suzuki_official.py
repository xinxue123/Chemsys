"""Qualify source identity, Suzuki I/O, and unchanged upstream execution paths."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from chem.decision.replay import write_json_atomic
from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle,initial_design
from chem.benchmarks.official_suzuki_adapter import (modules,encode_point,decode_z,execute_llinbo,
    execute_lgbo,AwaitingAssistant,seed_all,rng_hash,rng_state,description,check_sources)
from chem.benchmarks.suzuki_policies import hebo_select


def main():
    torch.set_num_threads(1)
    out=Path('research/suzuki_official')
    out.mkdir(parents=True,exist_ok=True)
    check_sources()
    oracle=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',1)
    points=initial_design(1,0)
    values=oracle.predict(points)
    initial=[{'z':encode_point(p),'y':float(v)} for p,v in zip(points,values)]
    z_answers=[[.5,.95,.95,.1875],[.75,.95,.9,.4375]]
    upstream=modules('LLINBO')
    started=time.perf_counter()
    direct_queries=[]
    def obj(z):
        y=float(oracle.predict([decode_z(z)])[0]);direct_queries.append((list(z),y));return y
    def warm(self,num_warmstart=1,objective_function=None):
        assert num_warmstart==4
        return [(tuple(r['z']),r['y']) for r in initial]
    def direct_llm(self,prompt,model='gpt-3.5-turbo',max_tokens=2000):
        return json.dumps(z_answers[len(self.history)-4])
    seed=20271027
    with patch.object(upstream.LLAMAGENT_L,'llm_warmstarting',warm),patch.object(upstream.LLAMAGENT_L,'query_llm',direct_llm):
        seed_all(seed)
        a=upstream.LLMIBO_BFO('justify',obj,dim=4,desc=description(1),T=2,T_ini=4,T_rep=1,verbose=False)
        native_h,native_regret=a.run()
    direct_rng=rng_hash(rng_state())
    direct_seconds=time.perf_counter()-started
    replay_queries=[]
    fresh_oracle_calls=0
    budget_available=0
    def replay_obj(z):
        nonlocal fresh_oracle_calls
        i=replay_obj.index;replay_obj.index+=1
        if i<len(replay_queries):
            assert np.array_equal(np.array(z),np.array(replay_queries[i][0]))
            return replay_queries[i][1]
        y=float(oracle.predict([decode_z(z)])[0]);fresh_oracle_calls+=1
        replay_queries.append((list(z),y));return y
    def replay_llm(history,prompt):
        t=len(history)-4
        if t>=budget_available: raise AwaitingAssistant('qualification pause')
        return json.dumps(z_answers[t])
    memo=Path('data/runs/suzuki_official/qualification_memo')
    traces=[]
    for available in (0,1,2):
        budget_available=available;replay_obj.index=0
        try:
            result=execute_llinbo(case=1,initial=initial,objective=replay_obj,assistant=replay_llm,
                                  seed=seed,steps=2,memo_directory=memo)
            traces.append(result)
        except AwaitingAssistant as exc:
            traces.append(exc.trace)
    final_rng=rng_hash(rng_state())
    assert direct_queries==replay_queries
    assert [[(tuple(x),y) for x,y in h] for h in result['histories']]==native_h
    assert np.array_equal(native_regret,np.array(result['raw_upstream_regrets']))
    assert direct_rng==final_rng
    assert fresh_oracle_calls==2
    text='Thinking:\nInterface qualification fixture only.\nFinal Answer:\n[region, [[200,70,1.2,0.0],[600,110,2.5,0.25]],0.7]'
    lg_z,lg_trace=execute_lgbo(initial,text,seed)
    decode_z(lg_z)
    _,hebo_trace=hebo_select(points,values,seed)
    parity=[]
    for case in (1,2,3,4):
        task=json.loads(Path(f'data/runs/chapter5/summit_cache/case_{case}/task.json').read_text(encoding='utf8'))
        official=np.load(f'data/runs/chapter5/summit_cache/case_{case}/oracle.npz')['values']
        current=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',case).predict(task['candidates'])
        error=float(np.max(np.abs(current-official))*100)
        assert error<1e-4
        parity.append({'case':case,'checked_points':512,'max_error_percentage_points':error})
    receipt={'status':'PASS','scope':'source/interface/resume qualification only; fixed LLM stubs are not scientific LLM runs',
             'byte_identical_upstream_files':13,'LLINBO_direct_upstream_vs_paused_memoized_query_equality':True,
             'LLINBO_full_history_equality':True,'LLINBO_final_rng_equality':True,'LLINBO_new_oracle_calls_across_three_resumes':fresh_oracle_calls,
             'LLINBO_qualification_additional_budget':2,'LLINBO_direct_seconds':direct_seconds,
             'LLINBO_memo_hits_in_final_resume':sum(r['cache_hit'] for r in traces[-1]['memo_events']),
             'LGBO_original_qLogEI_path':lg_trace,'HEBO_original_path':hebo_trace,'emulator_parity':parity,
             'environment':{'torch':torch.__version__,'botorch':__import__('botorch').__version__,'gpytorch':__import__('gpytorch').__version__},
             'source_manifest_sha256':sha256(Path('research/suzuki_official/sources.json').read_bytes()).hexdigest()}
    write_json_atomic(out/'qualification.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(1): main()
