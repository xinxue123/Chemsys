"""Fresh native GP/qLogEI audit, without loading optimization model caches."""
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
from llm_bo_handoff import study
import audit_llm_initialization_reliability as independent


def initialize():
    independent.BASE=study.BASE/'fresh_native_audit'
    independent.initialize()


def audit_job(job):
    state=study.read(study.BASE/'runs'/job['id']/'state.json')
    rows=[]
    for q in range(2,9):
        ordered=sorted(state['history'][:q-1],
            key=lambda h:independent.FEATURES[independent.IX[h['candidate_id']]])
        x=independent.X[[independent.IX[h['candidate_id']] for h in ordered]]
        y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
        torch.manual_seed(20261104+len(ordered))
        model=independent.native_model(x,y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if len(ordered)>=4:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
            model.eval()
            acq=qLogExpectedImprovement(model,best_f=float(y.max()),
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=38112000+q))
            with torch.no_grad():
                scores=torch.cat([acq(part.unsqueeze(1)) for part in independent.X.split(256)]).numpy()
                posterior=model.posterior(independent.X)
                mu=posterior.mean.reshape(-1).numpy()
                variance=posterior.variance.reshape(-1).numpy()
        valid=~np.isin(independent.IDS,[h['candidate_id'] for h in ordered])
        priority=independent.priorities(q)
        maximum=float(scores[valid].max())
        leaders=np.flatnonzero(valid & (scores==maximum))
        native=int(leaders[np.argmin(priority[leaders])])
        decision=state['decisions'][q-1]['decision']
        actual=independent.IX[decision['selected_id']]
        assert independent.IDS[native]==decision['BO_reference_id']
        assert maximum-scores[actual]<=decision['effective_logei_gap']+1e-8
        if q==8:
            assert actual==native
        with np.load(study.BASE/'runs'/job['id']/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as old:
            errors={name:float(np.max(np.abs(value-old[name]))) for name,value in
                    (('mu',mu),('variance',variance),('logei',scores))}
            for name,e in errors.items():
                assert e<1e-8,(job['id'],q,name,e)
        assert sum(independent.COMPILES.values())==0
        rows.append(dict(job=job['id'],query=q,native_BO_reference=independent.IDS[native],
            selected_id=decision['selected_id'],native_logei_loss=maximum-float(scores[actual]),
            permitted_logei_loss=decision['effective_logei_gap'],errors=errors,
            hyperparameters_fitted=q>=5,warnings=[str(w.message) for w in caught]))
    print(json.dumps(dict(job=job['id'],prefixes=7,max_logei_error=max(r['errors']['logei'] for r in rows))),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    p=study.check(True)
    registration=ROOT/'experiments/runs/llm_initialization_audit_20261104/registration.json'
    if not registration.exists():
        registration=ROOT/'results/llm_initialization_audit_20261104/source_registration.json'
    old=independent.read(registration)
    assert study.sha(independent.__file__)==old['audit_source_sha256']
    study.write(study.BASE/'fresh_native_audit/registration.json',dict(GP_parameters=p['GP_parameters'],
        constructor_source_sha256=study.sha(independent.__file__),original_protocol_sha256=p['protocol_sha256']))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as executor:
        fs=[executor.submit(audit_job,j) for j in p['jobs']]
        for f in as_completed(fs):
            rows.extend(f.result())
    study.check(True)
    result=dict(status='PASS',prefixes=len(rows),full_pool_native_BO_references_matched=len(rows),
        selected_candidate_budget_checks=len(rows),terminal_native_BO_choices_matched=10,
        maximum_mean_error=max(r['errors']['mu'] for r in rows),
        maximum_variance_error=max(r['errors']['variance'] for r in rows),
        maximum_logei_error=max(r['errors']['logei'] for r in rows),
        new_GP_fits=sum(r['hyperparameters_fitted'] for r in rows),
        prefixes_with_native_fitting_warnings=sum(bool(r['warnings']) for r in rows),
        old_models_or_scores_used_for_computation=False,new_LLM_calls=0,new_oracle_queries=0,
        original_artifacts_unchanged=True,operator_compiles_in_loop=0,checks=rows)
    study.write(study.REPORT/'fresh_native_check.json',result,replace=True)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':
    main()
