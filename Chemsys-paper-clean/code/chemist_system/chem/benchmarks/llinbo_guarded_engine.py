"""Run the unchanged official Justify body through explicit matched-study hooks.

This is LLINBO-Matched, not the original UCB experiment. Vendor files are never
edited. The GP, acquisition, threshold, finite-pool transport and provider are
deliberate adaptations, recorded by the companion frozen protocol.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
from unittest.mock import patch

import numpy as np
import torch

from . import guarded_suzuki_engine as shared
from .official_suzuki_adapter import modules, encode_point
from .suzuki_domain import CATALYSTS, NUMERIC, LOW, HIGH, validate_point


def project_proposal(point, pool, queried):
    """Nearest unqueried point of the SAME catalyst; coordinates only."""
    point=validate_point(point)
    z=np.asarray(encode_point(point))
    available=np.array([key not in queried and p['catalyst']==point['catalyst']
                        for key,p in zip(pool['ids'],pool['points'])])
    if not available.any():raise ValueError('no unused candidate for this catalyst')
    xyz=np.asarray(pool['z'])[:,:3]
    distance=((xyz-z[:3])**2).sum(1)
    chosen=min(np.flatnonzero(available),key=lambda i:(float(distance[i]),pool['ids'][i]))
    return {'candidate_id':pool['ids'][chosen], 'z':pool['z'][chosen],
            'point':pool['points'][chosen], 'normalized_numeric_distance':float(np.sqrt(distance[chosen])),
            'raw_point':point,'raw_z':z.tolist()}


@dataclass(frozen=True)
class MatchedThreshold:
    """Replace the max-var/t schedule at its original division hook."""
    ratio: int
    def __truediv__(self, iteration):
        return shared.budget_at(self.ratio, int(iteration))


def description(advice):
    clauses=[]
    for row in advice['clauses']:
        j=row['coordinate'];lo,hi=row['interval']
        if j<3:
            label=NUMERIC[j];bounds=[float(LOW[j]+lo*(HIGH[j]-LOW[j])),float(LOW[j]+hi*(HIGH[j]-LOW[j]))]
        else:
            label='catalyst';bounds=[c for k,c in enumerate(CATALYSTS) if lo<=(k+.5)/8<=hi]
        clauses.append({'id':row['id'],'variable':label,'preferred':bounds})
    return ('a historical Suzuki reaction emulator; maximize yield/100. '
        'x=[normalized residence time,normalized temperature,normalized catalyst loading,catalyst interval code]. '
        'Physical bounds: residence60..600s, temperature30..110C, loading0.5..2.5mol%. '
        f'Catalyst labels in interval order: {list(CATALYSTS)}. Labels are anonymous; do not invent ligand structures. '
        'A supplied preference is uncertain and may be questioned using this trajectory observations: '
        +json.dumps({'confidence':advice.get('confidence'),'clauses':clauses},separators=(',',':'))+'. '
        'Use only this trajectory observed evaluations; do not transfer benchmark outcomes from other trajectories. '
        'The provider transports your proposal to the nearest unused same-catalyst member of the common fixed pool. '
        'The shared BO score is hidden from you; propose a point before its budget review.')


def execute(*,pool,initial,advice,case,seed,ratio,cache,assistant,objective,steps=16):
    """Execute the actual upstream run -> _run_justify, including its if/else.

    The source's train_gp/argmax/UCB hooks now return the common GP score table.
    Its max_var / (t+1) hook returns the preselected common logEI-loss schedule.
    sample_candidate_points still constructs/parses the original source prompt.
    """
    upstream=modules('LLINBO')
    ids=pool['ids'];z=pool['z'];index={key:i for i,key in enumerate(ids)}
    by_z={tuple(v):i for i,v in enumerate(z)}
    warm=[(tuple(z[index[row['candidate_id']]]),row['y']) for row in initial]
    gates=[];state={'current':None,'iteration':0,'proposal':None}

    def convert(history):
        return [{'candidate_id':ids[by_z[tuple(x)]],'y':float(y)} for x,y in history]

    def warmstart(self,num_warmstart=1,objective_function=None):
        assert num_warmstart==4
        return list(warm)

    def fit(history):
        h=convert(history);scores,meta=cache.score(h)
        value={'history':h,'scores':scores,'metadata':meta}
        state['current']=value
        return value

    def maximum(model,bounds,dim=4,resolution=10):
        return MatchedThreshold(ratio)

    def argmax(model,bounds,beta):
        queried={v['candidate_id'] for v in model['history']}
        available=np.array([key not in queried for key in ids])
        i=shared.argmax(model['scores'],available,ids)
        t=len(model['history'])-3;state['iteration']=t
        # Upstream shuffles its prompt history. Explicit per-prefix seed makes
        # prompts independent of cache hits and pause/replay placement.
        shared.seed_all(40102026+case*10000+seed*100+t)
        state['reference_index']=i;state['source_beta']=float(beta)
        return torch.tensor([z[i]],dtype=torch.double)

    def query(self,prompt,model='gpt-3.5-turbo',max_tokens=2000):
        projection,response=assistant(convert(self.history),prompt,state['iteration'])
        state['proposal']={'projection':projection,'response':response}
        return json.dumps(projection['z'],separators=(',',':'))

    def acquisition(model,beta):
        row={'iteration':state['iteration'],'source_beta_computed_but_unused':float(beta),
             'budget_logei':shared.budget_at(ratio,state['iteration']),'scores':[]}
        gates.append(row)
        class Lookup:
            def __call__(self,x):
                vector=tuple(x.detach().reshape(-1).tolist());i=by_z[vector]
                value=float(model['scores'][i]);row['scores'].append(value)
                return torch.tensor(value,dtype=torch.double)
        return Lookup()

    def evaluate(vector):
        i=by_z[tuple(vector)];row=gates[-1]
        best,proposed=row['scores'];rejected=best>proposed+row['budget_logei']
        proposal=state['proposal']['projection'];chosen=ids[i]
        expected=ids[state['reference_index']] if rejected else proposal['candidate_id']
        assert chosen==expected
        gap=best-float(state['current']['scores'][i])
        assert gap<=row['budget_logei'],(gap,row['budget_logei'])
        event={'iteration':row['iteration'],'reference_id':ids[state['reference_index']],
            'proposed_id':proposal['candidate_id'],'selected_id':chosen,'accepted':not rejected,
            'proposal_equals_reference':proposal['candidate_id']==ids[state['reference_index']],
            'proposal_base_logei':proposed,'maximum_base_logei':best,'selected_base_logei':best-gap,
            'proposal_logei_loss':best-proposed,'selected_logei_loss':gap,'budget_logei':row['budget_logei'],
            'source_beta_computed_but_unused':state['source_beta'],
            'base_acquisition':state['current']['metadata'],'projection':proposal,
            'response_context_sha256':state['proposal']['response']['context_sha256']}
        return objective(chosen,event)

    with ExitStack() as stack:
        for obj,name,replacement in [(upstream.LLAMAGENT_L,'llm_warmstarting',warmstart),
            (upstream.LLAMAGENT_L,'query_llm',query),(upstream,'train_gp',fit),
            (upstream,'find_max_variance_bound',maximum),(upstream,'optimize_acqf_ucb',argmax),
            (upstream,'UpperConfidenceBound',acquisition)]:
            stack.enter_context(patch.object(obj,name,replacement))
        agent=upstream.LLMIBO_BFO('justify',evaluate,dim=4,desc=description(advice),T=steps,T_ini=4,T_rep=1,verbose=False)
        histories,regrets=agent.run()
    return {'histories':histories,'source_entrypoint':'LLMIBO_BFO.run -> unchanged _run_justify',
            'native_gate_calls':len(gates)}
