"""Suzuki I/O adapters that execute byte-identical upstream algorithm bodies.

No acquisition, GP, confidence calibration, kernel, or selection rule is
implemented here. LLINBO runs the actual upstream _run_justify loop. Its pause
support memoizes deterministic upstream calls with full RNG receipts; equality
to an uninterrupted upstream run is qualified separately.
"""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import importlib
import json
from pathlib import Path
import pickle
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from chem.decision.replay import write_json_atomic
from .suzuki_domain import CATALYSTS, NUMERIC, LOW, HIGH, validate_point, digest

PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / 'third_party/suzuki_official'
PARAMETERS = [*({'name':n,'type':'continuous','bounds':[float(l),float(h)]} for n,l,h in zip(NUMERIC,LOW,HIGH)),
              {'name':'catalyst_code','type':'continuous','bounds':[0.,1.],
               'description':'fixed categorical interval code; decode floor(8*z), with z=1 mapped to label7'}]


class AwaitingAssistant(RuntimeError):
    pass


def check_sources():
    data=json.loads((PROJECT/'research/suzuki_official/sources.json').read_text(encoding='utf8'))
    for row in data['files']:
        raw=(PROJECT/row['path']).read_bytes()
        blob=sha256(raw).hexdigest()
        if blob != row['sha256']:
            raise ValueError(f"upstream source changed: {row['path']}")
    return data


def modules(method):
    check_sources()
    folder=VENDOR/('llinbo' if method=='LLINBO' else 'lgbo')
    if str(folder) not in sys.path:
        sys.path.insert(0,str(folder))
    if method=='LLINBO':
        module=importlib.import_module('LLM_agent_BBFO')
        assert Path(module.__file__).resolve()==folder/'LLM_agent_BBFO.py'
        return module
    core=importlib.import_module('lgbo_core')
    prompt=importlib.import_module('prompt')
    assert Path(core.__file__).resolve()==folder/'lgbo_core.py'
    assert Path(prompt.__file__).resolve()==folder/'prompt.py'
    return core,prompt


def encode_point(point):
    p=validate_point(point)
    return [*((p[n]-l)/(h-l) for n,l,h in zip(NUMERIC,LOW,HIGH)),(CATALYSTS.index(p['catalyst'])+.5)/8]


def validate_z(z):
    values=np.asarray(z,dtype=float).reshape(-1)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('official continuous input must be a finite 4-vector inside [0,1]^4')
    return values


def decode_z(z):
    x=validate_z(z)
    category=min(int(np.floor(8*x[3])),7)
    return validate_point({'catalyst':CATALYSTS[category],
                           **{n:float(v) for n,v in zip(NUMERIC,LOW+x[:3]*(HIGH-LOW))}})


def continuous_point(z):
    x=validate_z(z)
    return {**{n:float(v) for n,v in zip(NUMERIC,LOW+x[:3]*(HIGH-LOW))},'catalyst_code':float(x[3])}


def description(case):
    return (f'pretrained Reizman-Suzuki Case {case}; maximize yld/100. '
            'x=[normalized residence time,normalized temperature,normalized catalyst loading,catalyst interval code]. '
            'Physical bounds are residence60..600s,temperature30..110C,loading0.5..2.5mol%. '
            f'Catalyst code z uses fixed intervals [k/8,(k+1)/8) for labels k=0..7: {list(CATALYSTS)}; z=1 uses label7. '
            'The label order is a transport convention, not evidence of chemical similarity. '
            'Catalyst labels are anonymous: do not invent ligand structures or transfer a known ranking from other trajectories. '
            'Use only this observed history; the benchmark is a deterministic historical emulator, not a laboratory experiment.')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def rng_hash(state):
    return sha256(pickle.dumps((state['python'],state['numpy'],state['torch'].numpy().tobytes()),protocol=4)).hexdigest()


class UpstreamMemo:
    """Read-through execution cache. A hit restores the upstream post-call RNG."""
    def __init__(self,path):
        self.path=Path(path)
        self.path.mkdir(parents=True,exist_ok=True)
        self.events=[]

    def call(self,name,payload,original,args,kwargs=None):
        before=rng_state()
        key=digest({'function':name,'inputs':payload,'rng_before':rng_hash(before)})
        file=self.path/(key+'.pt')
        if file.exists():
            saved=torch.load(file,map_location='cpu',weights_only=False)
            if saved['name']!=name or saved['payload']!=payload or saved['rng_before']!=rng_hash(before):
                raise ValueError('memo input lineage mismatch')
            if name=='train_gp':
                # Same upstream constructor, no replacement GP or refit. Restore
                # all fitted tensors including the official outcome transform.
                from botorch.models import SingleTaskGP
                x=torch.tensor([list(a) for a,b in args[0]],dtype=torch.float64)
                y=torch.tensor([[b] for a,b in args[0]],dtype=torch.float64)
                result=SingleTaskGP(x,y)
                result.load_state_dict(saved['value']['state_dict'])
                result.train(saved['value']['training'])
            else:
                result=saved['value']
            restore_rng(saved['rng_after'])
            hit=True
        else:
            result=original(*args,**(kwargs or {}))
            after=rng_state()
            value=({'state_dict':result.state_dict(),'training':result.training} if name=='train_gp' else result)
            saved={'name':name,'payload':payload,'rng_before':rng_hash(before),'rng_after':after,'value':value}
            temporary=file.with_suffix('.tmp')
            torch.save(saved,temporary)
            temporary.replace(file)
            hit=False
        self.events.append({'function':name,'key':key,'cache_hit':hit,'file_sha256':sha256(file.read_bytes()).hexdigest(),
                            'rng_before':rng_hash(before),'rng_after':rng_hash(saved['rng_after'])})
        return result


def execute_llinbo(*,case,initial,objective,assistant,seed,steps=16,memo_directory=None):
    """Direct upstream loop. Only initialization, LLM transport and logging differ."""
    upstream=modules('LLINBO')
    seed_all(seed)
    warm=[(tuple(row['z']),row['y']) for row in initial]
    memo=UpstreamMemo(memo_directory) if memo_directory is not None else None
    original_fit=upstream.train_gp
    original_maxvar=upstream.find_max_variance_bound
    original_acq=upstream.optimize_acqf_ucb
    original_ucb=upstream.UpperConfidenceBound
    gates=[]
    metadata={'last_fit_inputs':None,'psi0':None,'acquisition_calls':0,'acquisition_points':[], 'fit_classes':[]}

    def warmstart(self,num_warmstart=1,objective_function=None):
        assert num_warmstart==len(warm)
        return [(tuple(x),y) for x,y in warm]

    def query(self,prompt,model='gpt-3.5-turbo',max_tokens=2000):
        # Existing upstream sample_candidate_points builds and parses its own
        # exact prompt. Only the provider transport is replaced by the user-
        # selected current assistant. It may pause without consuming a query.
        return assistant(self.history,prompt)

    def fit(history):
        payload=[[[float(v) for v in x],float(y)] for x,y in history]
        metadata['last_fit_inputs']=payload
        result=memo.call('train_gp',payload,original_fit,(history,)) if memo else original_fit(history)
        metadata['fit_classes'].append({'gp':type(result).__module__+'.'+type(result).__name__,
                                        'kernel':type(result.covar_module).__name__,'train_rows':len(history)})
        return result

    def maximum(model,bounds,dim=2,resolution=10):
        payload={'history':metadata['last_fit_inputs'],'bounds':bounds.tolist(),'dim':dim,'resolution':resolution}
        result=(memo.call('find_max_variance_bound',payload,original_maxvar,(model,bounds),{'dim':dim,'resolution':resolution})
                if memo else original_maxvar(model,bounds,dim=dim,resolution=resolution))
        metadata['psi0']=float(result)
        return result

    def acq(model,bounds,beta):
        payload={'history':metadata['last_fit_inputs'],'bounds':bounds.tolist(),'beta':float(beta)}
        metadata['acquisition_calls']+=1
        result=(memo.call('optimize_acqf_ucb',payload,original_acq,(model,bounds,beta)) if memo else original_acq(model,bounds,beta))
        metadata['acquisition_points'].append(result.detach().reshape(-1).tolist())
        return result

    def ucb_factory(*args,**kwargs):
        native=original_ucb(*args,**kwargs)
        row={'beta':float(kwargs['beta']),'values':[]}
        gates.append(row)
        class Probe:
            def __call__(self,*a,**k):
                output=native(*a,**k)
                row['values'].append(float(output.item()))
                return output
        return Probe()

    with ExitStack() as stack:
        stack.enter_context(patch.object(upstream.LLAMAGENT_L,'llm_warmstarting',warmstart))
        stack.enter_context(patch.object(upstream.LLAMAGENT_L,'query_llm',query))
        stack.enter_context(patch.object(upstream,'train_gp',fit))
        stack.enter_context(patch.object(upstream,'find_max_variance_bound',maximum))
        stack.enter_context(patch.object(upstream,'optimize_acqf_ucb',acq))
        stack.enter_context(patch.object(upstream,'UpperConfidenceBound',ucb_factory))
        agent=upstream.LLMIBO_BFO('justify',objective,dim=4,desc=description(case),T=steps,T_ini=4,T_rep=1,verbose=False)
        try:
            histories,raw_regrets=agent.run()
        except AwaitingAssistant as exc:
            exc.trace={'gates':gates,'psi0':metadata['psi0'],'memo_events':memo.events if memo else [],
                       'acquisition_points':metadata['acquisition_points'],'fit_classes':metadata['fit_classes']}
            raise
    return {'histories':[[[list(x),float(y)] for x,y in h] for h in histories],
            'raw_upstream_regrets':raw_regrets.tolist(),'gates':gates,'psi0':metadata['psi0'],
            'memo_events':memo.events if memo else [],'acquisition_points':metadata['acquisition_points'],
            'fit_classes':metadata['fit_classes'],
            'upstream_method':str(Path(upstream.__file__).relative_to(PROJECT))+':LLMIBO_BFO._run_justify'}


def lgbo_prompts(case,history):
    core,prompt=modules('LGBO')
    observations=[{'x':continuous_point(h['z']),'y':h['y']} for h in history]
    user=prompt.build_user_prompt(background=description(case),parameters=PARAMETERS,
        objective='Maximize normalized yield yld/100.',constraints='Stay inside all declared bounds; catalyst_code is decoded by its fixed interval.',
        history=observations,batch_q=1,y_key='y',extra_request='Use this trajectory history only, in the declared physical parameter order.')
    return prompt.DEFAULT_SYSTEM_PROMPT,user


def execute_lgbo(history,text,seed):
    core,prompt=modules('LGBO')
    parsed=prompt.parse_assistant_response(text)
    if parsed['mode'] not in ('point','region'):
        raise ValueError('official LGBO parser did not recognize assistant preference')
    space=core.ContinuousSpace.from_parameters(PARAMETERS)
    x=torch.tensor([h['z'] for h in history],dtype=torch.double)
    y=torch.tensor([[h['y']] for h in history],dtype=torch.double)
    boo=importlib.import_module('boo')
    original=boo.qLogExpectedImprovement
    calls=[]
    def factory(*args,**kwargs):
        calls.append({'class':'botorch.acquisition.logei.qLogExpectedImprovement',
                      'mc_sample_shape':list(kwargs['sampler'].sample_shape)})
        return original(*args,**kwargs)
    started=time.perf_counter()
    with patch.object(boo,'qLogExpectedImprovement',factory):
        points,plan,z=core.propose_lgbo_batch(X_norm=x,y=y,parsed_preference=parsed,
            space=space,goal='max',batch_q=1,seed=seed)
    # The source intentionally converts undersampled small boxes into a point
    # prior. That native branch uses WeightedQLogEI.ask, not this region ctor.
    # Observe the chosen path; never reject a legal upstream selection.
    if plan['mode'].startswith('region') and (len(calls)!=1 or calls[0]['mc_sample_shape']!=[512]):
        raise ValueError('official regional acquisition logging mismatch')
    return z[0].tolist(),{'upstream_function':'lgbo_core.propose_lgbo_batch',
        'plan':core.serialize_plan(plan),'parsed_preference':parsed,'acquisition_calls':calls,
        'cand_size':8192,'region_grid_size':512,'guidance_scale':3.,
        'native_branch':plan['mode'],'seconds':time.perf_counter()-started,
        'returned_physical_coordinates':points[0]}
