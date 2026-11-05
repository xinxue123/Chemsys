"""No GP scores or unobserved yields are exposed to a decision responder."""
from copy import deepcopy
from chem.benchmarks.knowledge_domain_context import task_context

SYSTEM='''You provide sparse proposals for sequential Suzuki-Miyaura optimization.
The FIRST request has zero observations: propose ONE informative, promising first
experiment using general chemistry and the supplied input domain. Later requests
occur ONLY when experiment feedback has reduced confidence in earlier knowledge.
At a refresh, retain, revise or withdraw knowledge and optionally suggest ONE
unqueried next experiment. Between these requests, native qLogEI acts alone.

Use only the supplied domain and own measured observations. Catalyst codes are
opaque nominal labels, with no known ligand mapping or intrinsic ordering. Do
not invent substrate or ligand structures, recall benchmark yields, infer target
values from identifiers, seek unobserved outcomes, or use other trajectories.

Return exactly {"proposal":object or null,"hypotheses":list,"summary":string}.
proposal, when present, is exactly {"conditions":{all four real axis names:their
physical values},"basis_ids":[one or more hypothesis IDs],"rationale":string}.
Continuous values may be anywhere inside the supplied physical input bounds;
catalyst must be an exact real label. The program projects onto the nearest
unqueried supplied pool point, with the same catalyst and matching cited scopes.
All pool conditions are eligible to be a FIRST point, with no yield grouping.
The actual projection is logged. On refresh, a proposed point can be executed
only if its native qLogEI meets the fixed acquisition budget review; otherwise
the full-pool qLogEI maximum is executed. You do not see GP scores or future yields.
Null is a valid proposal abstention. All cited basis_ids must occur in hypotheses,
be favour hypotheses, and match the proposed physical conditions.

hypotheses has zero to four objects, each exactly {"id":string,"claim":string,
"direction":"favour or avoid","scope":{axis:condition,...},"yield_mean":number,
"yield_sd":number}. A continuous condition is an inclusive [lo,hi] in physical
units, inside the supplied input domain. A categorical condition is a nonempty
list of exact labels. Omitted axes are unrestricted. Cover at least four actual
pool points in each scope. yield_mean is an uncertain yield forecast in [0,100]
percent, and yield_sd is its SD in [5,50] percentage points. These are hypotheses,
never measurements. Do not imply unsupported numerical precision.

Before each measured choice the program records applicable improvement forecasts.
The KF scores their relative Brier skill against that round's unmodified GP.
Confidence and recommendation permissions are separate. Accurate failure
prediction never automatically restores a paused recommendation. At refresh,
renamed/reworded same-direction/same-scope hypotheses inherit KF and permissions,
even when their numerical forecast changes. Do not reset negative evidence.
Substantively different scopes may be proposed provisionally with uncertainty.
Use the supplied adverse evidence when doing so. Give concise public rationales,
not private reasoning. The summary must explain the evidence and uncertainty.
'''


def public(pool,state,query,event):
    domain=task_context(pool.task,pool)
    for k in ('task','group','source_references'):
        domain.pop(k,None)
    observations=[dict(query=i,candidate_id=h['candidate_id'],
        conditions=pool.points[pool.ix[h['candidate_id']]],yield_percent=h['raw_value'])
        for i,h in enumerate(state['history'],1)]
    examples=[]
    for catalyst in sorted({point['catalyst'] for point in pool.points}):
        points=sorted((point for point in pool.points if point['catalyst']==catalyst),
                      key=lambda point:tuple(point[a] for a in pool.axes))
        examples.extend(points[k] for k in (0,len(points)//2,len(points)-1))
    return dict(domain=domain,request_kind='initial' if query==1 else 'confidence_refresh',
        query=query,observations=observations,input_only_examples=examples,
        current_incumbent_percent=max((h['raw_value'] for h in state['history']),default=None),
        own_knowledge=deepcopy(state['notes']),archived_knowledge=deepcopy(state['knowledge_archive']),
        trigger=event,GP_scores_not_supplied=True,unobserved_yields_not_supplied=True)
