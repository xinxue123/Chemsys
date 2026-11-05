"""Independent chemical hypotheses before screening and before new outcomes."""
from copy import deepcopy
from chem.benchmarks.knowledge_domain_context import task_context

SYSTEM='''You advise sequential Suzuki-Miyaura reaction optimization.
Use only the supplied domain and this trajectory's observed experiments.
Opaque catalyst labels have no molecular identities or ordering. Do not infer
ligand sterics/electronics from their names, invent substrate identities, recall
benchmark yields, infer outcomes from IDs or seek any unmeasured outcome.

Return exactly {"hypotheses":[...],"summary":"concise public explanation"}.
There may be zero to four hypotheses; abstain when no useful distinction exists.
Each hypothesis is exactly {"id":"short stable ID","claim":"local uncertain
chemical/process hypothesis and its evidence basis","direction":"favour or
avoid","scope":{real axis:condition,...},"yield_mean":number,"yield_sd":number}.
A continuous condition is [physical_lower_bound,physical_upper_bound], inclusive
and inside the supplied domain. A categorical condition is a nonempty list of
exact real labels, treated as nominal. Missing scope axes are unrestricted.
Use a scope that covers at least four actual executable conditions.
yield_mean is your uncertain local yield forecast in percent [0,100]. yield_sd
is its uncertainty in percentage points [5,50], not a confidence score. It defines
a Normal forecast throughout this local scope; changing the incumbent changes
its implied probability of improvement. Avoid unsupported numerical precision.
The program checks your forecasts prospectively against actual outcomes and GP
improvement probabilities; these are hypotheses, never synthetic observations.

Initial request: propose locally useful distinctions independently of GP scores.
Refresh request: use the given new observations, failed forecasts, KF relative
predictive skill and recommendation permissions. Retain, substantively revise or
remove failed claims. Return the complete desired hypothesis set. Reusing the
same direction and physical scope inherits its KF evidence and paused state,
even after renaming, rewording or changing its numerical forecast. A paused
recommendation cannot be restored merely because it accurately predicts a
failure. A different physical scope may be proposed provisionally with explicit
evidence and uncertainty; old adverse observations remain available to you.

The program uses these rules to screen a bounded qLogEI set, then selects the
unchanged qLogEI maximum. You do not pick the next experiment, alter the GP or
observe any future result. Concise public rationales only, no private reasoning.
'''


def public(pool,state,query,event):
    domain=task_context(pool.task,pool)
    for key in ('task','group','source_references'):
        domain.pop(key,None)
    observations=[dict(query=i,candidate_id=h['candidate_id'],
        conditions=pool.points[pool.ix[h['candidate_id']]],yield_percent=h['raw_value'])
        for i,h in enumerate(state['history'],1)]
    # IDs are removed from the input-only coverage examples.
    sample=[]
    for label in sorted({p['catalyst'] for p in pool.points}):
        points=sorted((p for p in pool.points if p['catalyst']==label),
                      key=lambda p:tuple(p[a] for a in pool.axes))
        for ix in (0,len(points)//2,len(points)-1):
            sample.append(points[ix])
    return dict(domain=domain,query=query,request_kind='initial' if not state['knowledge_calls'] else 'refresh',
        observations=observations,current_incumbent_percent=max(h['raw_value'] for h in state['history']),
        own_knowledge=deepcopy(state['notes']),archived_knowledge=deepcopy(state['knowledge_archive']),
        trigger=event,input_only_condition_examples=sample,
        unseen_yields_not_supplied=True,GP_scores_not_supplied=True)
