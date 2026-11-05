"""Independent native GP/score reconstruction for each distinct new branch."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/analysis')]
from matched_catalyst_control import study
import audit_llm_initialization_reliability as independent


def initialize():
    independent.BASE=study.BASE/'fresh_native_audit'
    independent.initialize()


def audit_job(job):
    folder=study.BASE/'runs'/job['id']
    state=study.read(folder/'state.json')
    rows=[]
    for q in range(job['replacement']['query'],9):
        prefix=sorted(state['history'][:q-1],key=lambda h:independent.FEATURES[independent.IX[h['candidate_id']]])
        x=independent.X[[independent.IX[h['candidate_id']] for h in prefix]]
        y=torch.tensor([[h['y']] for h in prefix],dtype=torch.double)
        torch.manual_seed(20261104+len(prefix))
        model=independent.native_model(x,y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if len(prefix)>=4:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
            model.eval()
            acq=qLogExpectedImprovement(model,best_f=float(y.max()),
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=38112000+q))
            with torch.no_grad():
                score=torch.cat([acq(part.unsqueeze(1)) for part in independent.X.split(256)]).numpy()
                posterior=model.posterior(independent.X)
                mu=posterior.mean.reshape(-1).numpy()
                variance=posterior.variance.reshape(-1).numpy()
        available=~np.isin(independent.IDS,[h['candidate_id'] for h in prefix])
        priority=independent.priorities(q)
        js=np.flatnonzero(available & (score==score[available].max()))
        bo=int(js[np.argmin(priority[js])])
        decision=state['decisions'][q-1]['decision']
        actual=independent.IX[decision['selected_id']]
        assert decision['BO_reference_id']==independent.IDS[bo]
        if q==job['replacement']['query']:
            anchor=independent.IX[decision['anchor_id']]
            # Independent matching through physical continuous coordinates and
            # nominal label inequality, without importing the production matcher.
            match=available & np.all(independent.X[:,:3].numpy()==independent.X[anchor,:3].numpy(),axis=1)
            match &= independent.X[:,3].numpy()!=float(independent.X[anchor,3])
            js=np.flatnonzero(match)
            js=js[score[js]==score[js].max()]
            chosen=int(js[np.argmin(priority[js])])
            assert actual==chosen and decision['design_override']
        else:
            assert actual==bo and not decision['design_override']
        with np.load(folder/'rounds'/f'{q:02}_GP.npz') as old:
            errors={k:float(np.max(np.abs(v-old[k]))) for k,v in
                    (('mu',mu),('variance',variance),('logei',score))}
            assert np.array_equal(priority,old['priorities'])
            assert np.array_equal(available,old['available'])
            assert max(errors.values())<1e-8,errors
        assert sum(independent.COMPILES.values())==0
        rows.append(dict(job=job['id'],query=q,selected_id=independent.IDS[actual],
            errors=errors,native_logei_loss=float(score[bo]-score[actual]),
            hyperparameters_fitted=len(prefix)>=4,warnings=[str(w.message) for w in caught]))
    print(json.dumps(dict(job=job['id'],prefixes=len(rows),max_error=max(max(r['errors'].values()) for r in rows))),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    p=study.check(True)
    old=independent.read(ROOT/'experiments/runs/llm_initialization_audit_20261104/registration.json')
    assert study.sha(independent.__file__)==old['audit_source_sha256']
    study.write(study.BASE/'fresh_native_audit/registration.json',dict(GP_parameters=p['GP_parameters'],
        constructor_source_sha256=study.sha(independent.__file__),protocol_sha256=p['protocol_sha256']))
    jobs=[j for j in p['jobs'] if j['physical_primary']==j['id']]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
        fs=[executor.submit(audit_job,j) for j in jobs]
        for f in as_completed(fs):
            rows.extend(f.result())
    study.check(True)
    result=dict(status='PASS',distinct_physical_branches=len(jobs),checked_prefixes=len(rows),
        new_GP_fits=sum(r['hyperparameters_fitted'] for r in rows),
        maximum_mean_error=max(r['errors']['mu'] for r in rows),
        maximum_variance_error=max(r['errors']['variance'] for r in rows),
        maximum_logei_error=max(r['errors']['logei'] for r in rows),
        prefixes_with_native_warnings=sum(bool(r['warnings']) for r in rows),
        old_models_or_score_caches_used_for_computation=False,new_oracle_queries=0,new_LLM_calls=0,
        operator_compiles_in_loop=0,original_artifacts_unchanged=True,checks=rows)
    study.write(study.REPORT/'fresh_native_check.json',result,replace=True)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':
    main()
