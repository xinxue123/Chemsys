"""One reviewed event proposal, otherwise exactly the full-pool qLogEI maximum."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import ndtr
from chem.benchmarks import knowledge_gp
from fast_hit_bo.policy import choose
from llm_screen_qlogei import knowledge
from . import PARAMETERS as P


def validate(raw,pool,old,query):
    if not isinstance(raw,dict) or set(raw)!={'proposal','hypotheses','summary'}:
        raise ValueError('Exactly proposal,hypotheses,summary required')
    notes=knowledge.install({k:raw[k] for k in ('hypotheses','summary')},pool,old,query)
    proposal=raw['proposal']
    if proposal is not None:
        if not isinstance(proposal,dict) or set(proposal)!={'conditions','basis_ids','rationale'}:
            raise ValueError('Proposal needs conditions,basis_ids,rationale')
        point=proposal['conditions']
        if not isinstance(point,dict) or set(point)!=set(pool.axes):
            raise ValueError('All real input axes required')
        for axis,kind in zip(pool.axes,pool.kinds):
            domain=[p[axis] for p in pool.points]
            value=point[axis]
            if kind=='categorical':
                if value not in domain:
                    raise ValueError('Exact real catalyst label required')
            elif type(value) not in (int,float) or not math.isfinite(value) or not min(domain)<=value<=max(domain):
                raise ValueError('Continuous proposal must lie inside physical domain')
        basis=proposal['basis_ids']
        if (not isinstance(basis,list) or not basis or len(set(basis))!=len(basis)
            or set(basis)-{n['id'] for n in notes}):
            raise ValueError('Cite supplied favour hypothesis IDs')
        for key in basis:
            note=next(n for n in notes if n['id']==key)
            if note['direction']!='favour':
                raise ValueError('Only favour knowledge can recommend a proposal')
            for axis,condition in note['scope'].items():
                if pool.kinds[pool.axes.index(axis)]=='categorical':
                    if point[axis] not in condition:
                        raise ValueError('Proposal must match its cited scope')
                else:
                    if not condition[0]<=point[axis]<=condition[1]:
                        raise ValueError('Proposal must match its cited scope')
        if not isinstance(proposal['rationale'],str) or not 1<=len(proposal['rationale'])<=1800:
            raise ValueError('Concise rationale required')
    return dict(notes=notes,proposal=deepcopy(proposal))


def random_first(pool,seed):
    ordered=sorted(range(len(pool.ids)),key=lambda j:pool.features[j])
    rng=np.random.default_rng(np.random.SeedSequence([20261104,seed,971]))
    return int(ordered[int(rng.integers(len(ordered)))])


def project(pool,proposal,notes,available,query):
    point=proposal['conditions']
    valid=available.copy()
    valid &= np.asarray([p['catalyst']==point['catalyst'] for p in pool.points])
    for key in proposal['basis_ids']:
        note=next(n for n in notes if n['id']==key)
        valid &= knowledge.scope_mask(pool,note['scope'])
    js=np.flatnonzero(valid)
    if not len(js):
        return None,None
    distances=np.zeros(len(js))
    for axis,kind in zip(pool.axes,pool.kinds):
        if kind=='continuous':
            values=np.asarray([p[axis] for p in pool.points])
            distances+=((values[js]-point[axis])/(values.max()-values.min()))**2
    distances=np.sqrt(distances)
    priorities=knowledge_gp.semantic_priorities(pool,query,0)
    k=int(np.lexsort((priorities[js],distances))[0])
    return int(js[k]),float(distances[k])


def decide(pool,context,notes,proposal,query,seed,arm):
    available=np.ones(len(pool.ids),dtype=bool) if context is None else context['available']
    baseline=random_first(pool,seed) if query==1 else choose(context['logei'],available,context['priorities'])
    selected=baseline
    proposed=None
    distance=None
    reason='unrestricted_uniform_random_initial' if query==1 else 'full_pool_qLogEI'
    acceptance=False
    if proposal is not None:
        proposed,distance=project(pool,proposal,notes,available,query)
        basis=[next(n for n in notes if n['id']==key) for key in proposal['basis_ids']]
        permission=all(n['permission'] in ('provisional','active') and n['KF']['reliability']>=P['minimum_reliability'] for n in basis)
        if proposed is None:
            reason='event_rejected_no_unqueried_scoped_point'
        elif distance>P['maximum_projection_distance']:
            reason='event_rejected_projection_too_far'
        elif not permission:
            reason='event_rejected_basis_paused_or_low_confidence'
        elif query==1:
            selected=proposed
            acceptance=True
            reason='unrestricted_LLM_initial'
        elif context['logei'][proposed]>=context['logei'][baseline]-P['logei_budget']:
            selected=proposed
            acceptance=True
            reason='confidence_event_proposal_passed_qLogEI_review'
        else:
            reason='event_rejected_qLogEI_budget'
    forecasts=[]
    if query>1:
        p_gp=float(ndtr((context['mu'][selected]-context['incumbent'])/np.sqrt(max(context['variance'][selected],1e-18))))
        for n in notes:
            if knowledge.scope_mask(pool,n['scope'])[selected]:
                forecasts.append(dict(id=n['id'],version=n['version'],incumbent_y=context['incumbent'],
                    p_LLM=knowledge.forecast(n,context['incumbent']),p_GP=p_gp,
                    yield_mean=n['yield_mean'],yield_sd=n['yield_sd'],
                    promoted_selection=bool(acceptance and n['id'] in proposal['basis_ids'])))
    return dict(selected_id=pool.ids[selected],BO_reference_id=None if query==1 else pool.ids[baseline],
        rule=reason,query=query,rollout_used=False,continuous_screening_used=False,
        proposal=deepcopy(proposal),projected_id=None if proposed is None else pool.ids[proposed],
        projection_distance=distance,proposal_accepted=acceptance,
        forecasts=forecasts,qLogEI_opportunity_loss=None if query==1 else float(context['logei'][baseline]-context['logei'][selected]),
        random_initial_shadow=pool.ids[baseline] if query==1 else None)
