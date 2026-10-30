"""Unchanged official _run_justify with4+16 nominal-domain I/O hooks."""
from contextlib import ExitStack
from hashlib import sha256
import json
from unittest.mock import patch

import numpy as np
import torch

from .official_suzuki_adapter import modules
from .llinbo_guarded_engine import MatchedThreshold
from . import pointwise_hte_engine as engine


def execute(pool,state,cache,propose,objective):
    native=modules('LLINBO');ids=pool['ids'];lookup={key:i for i,key in enumerate(ids)}
    by_vector={tuple(x):i for i,x in enumerate(cache.X.tolist())}
    warm=[(tuple(cache.X[lookup[h['candidate_id']]].tolist()),h['y']) for h in state['history'][:4]]
    active={};gates=[]
    def convert(history):return [{'candidate_id':ids[by_vector[tuple(x)]],'y':float(y)} for x,y in history]
    def warmstart(self,num_warmstart=1,objective_function=None):
        assert num_warmstart==4;return list(warm)
    def fit(history):
        prefix=convert(history);scores,_,meta=cache.score(prefix)
        active['model']={'history':prefix,'scores':scores,'metadata':meta};return active['model']
    def maximum(model,bounds,**unused):return MatchedThreshold(4)
    def argmax(model,bounds,beta):
        available=~np.isin(ids,[h['candidate_id'] for h in model['history']])
        j=engine.nominal.argmax(model['scores'],available,ids);t=len(model['history'])-3
        active.update(iteration=t,reference=j,source_beta=float(beta))
        engine.nominal.seed_all(40102026+cache.task_seed*100+t)
        return cache.X[j:j+1]
    def query(self,prompt,model='gpt-3.5-turbo',max_tokens=2000):
        projection,response=propose(convert(self.history),prompt,active['iteration'])
        active.update(proposal=projection,response=response)
        return json.dumps(cache.X[lookup[projection['candidate_id']]].tolist())
    def acquisition(model,beta):
        gate={'iteration':active['iteration'],'budget':float(np.log(4)*(16-active['iteration'])/15),'values':[]}
        gates.append(gate)
        class Lookup:
            def __call__(self,X):
                j=by_vector[tuple(X.detach().reshape(-1).tolist())];value=float(model['scores'][j])
                gate['values'].append(value);return torch.tensor(value,dtype=torch.double)
        return Lookup()
    def evaluate(vector):
        j=by_vector[tuple(vector)];gate=gates[-1];best,proposed=gate['values']
        rejected=best>proposed+gate['budget'];expected=ids[active['reference']] if rejected else active['proposal']['candidate_id']
        assert ids[j]==expected
        return objective(ids[j],{'reference_id':ids[active['reference']],
            'proposed_id':active['proposal']['candidate_id'],'accepted':not rejected,
            'proposal_base_logei_loss':best-proposed,'budget':gate['budget'],
            'projection':active['proposal'],'response_context_sha256':active['response']['context_sha256'],
            'source_beta_computed_but_unused':active['source_beta']},active['model']['history'])
    desc=('Maximize '+pool['reaction_class']+' experimental yield over the supplied nominal feature domain. '
        'All listed axes, including substrates, vary. Numeric coordinates are unordered I/O labels only. '
        'The named category-ID/SMILES provider prompt supersedes legacy continuous bounds and distance wording. '
        'Use only this trajectory observations; no remembered benchmark outcomes or other trajectories.')
    with ExitStack() as stack:
        for obj,name,fn in [(native.LLAMAGENT_L,'llm_warmstarting',warmstart),(native.LLAMAGENT_L,'query_llm',query),
                (native,'train_gp',fit),(native,'find_max_variance_bound',maximum),
                (native,'optimize_acqf_ucb',argmax),(native,'UpperConfidenceBound',acquisition)]:
            stack.enter_context(patch.object(obj,name,fn))
        histories,regrets=native.LLMIBO_BFO('justify',evaluate,dim=len(pool['axes']),desc=desc,
            T=16,T_ini=4,T_rep=1,verbose=False).run()
    return {'source_entrypoint':'unchanged LLMIBO_BFO._run_justify',
        'source_function_bytecode_sha256':sha256(native.LLMIBO_BFO._run_justify.__code__.co_code).hexdigest(),
        'native_gate_calls':len(gates)}
