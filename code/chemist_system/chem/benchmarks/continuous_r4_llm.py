"""An observation-only proposer; its API accepts no GP/selection object."""
from copy import deepcopy
import math
from filelock import FileLock
from . import continuous_benchmark_llm as common
from . import continuous_benchmark_data as data
from . import reaction_transfer_llm as transport
from . import continuous_r4_engine as e
from .suzuki_domain import digest

SYSTEM=common.COMMON+'''
Independently propose ONE physical experiment using ONLY the domain and your own
observed experiments. You are intentionally not given GP predictions, acquisition
scores, the BO reference, a candidate ranking, a gate label, or other trajectories.
Do not invent those quantities. A separate executor later reviews your proposal.

Return EXACT JSON:
{"action":"propose","point":{"physical_axis":0.0},"prediction":100.0,
 "hypothesis":{"reference_index":1,"focus_axes":["physical_axis"],
   "statement":"a testable local contrast relative to this observation",
   "falsification":"observed response that would contradict the contrast"},
 "pause_ids":[],"reason":"short evidence-grounded rationale","support":[1]}.
For abstain, action="abstain",point=null,prediction=null,hypothesis={}.
point must contain EVERY physical axis, with finite values within the supplied
domain. prediction is a nonnegative RAW objective forecast <=2*objective_scale.
reference_index and support cite OWN observed 1-based indices; include the reference
in support (1..3 indices). Use one or two focus_axes; keep other conditions close
to the reference when possible and acknowledge any confounding. Predictions are
uncertain, not new observations. Never infer a chemical law from one mixed change.

Your physical proposal is projected to the nearest unqueried member of the original
fixed input pool. This may change conditions; the exact projected point is logged.
The local numerical model uses an affine contrast from the observed reference to
your projected proposal along focus_axes, with compact 32-neighbor support, bounded
extrapolation and background changes <0.25 of each input range. It is an observational
joint contrast, NOT independently identified causal slopes. If projection removes
the focus contrast, the software explicitly falls back to a local constant model.

You may pause existing hypothesis IDs if observations contradict them. Pause stops
deployment but retains shadow monitoring and counterevidence. Later prospective
evidence can restore a paused model. There is no irreversible deletion command.
New hypotheses never immediately change the GP mean. Two positive prospective
checks at different experiments are needed first. The software may allocate one of
the SAME sixteen experiments to local validation instead of your fresh proposal.
Up to four calls are event scheduled; spend them on useful contrasts rather than
inventing a new plateau after every result. reason/statement<=1200 characters,
falsification<=800. Concise public rationale only; no private reasoning trace.'''


def payload(pool,state,t):
    """Whitelist only physical domain, observations and past declarations.

    In particular, do not reuse common.public/observations: those functions
    attach prequery GP residuals and uncertainty to historical observations.
    """
    ix={key:j for j,key in enumerate(pool['ids'])}
    observations=[{'index':i,'point':deepcopy(pool['points'][ix[h['candidate_id']]]),
        'objective':float(h['raw_value'])} for i,h in enumerate(state['history'],1)]
    hypotheses=[{'id':key,'created_query':h['created_query'],
        'proposed_projected_point':deepcopy(pool['points'][ix[h['candidate_id']]]),
        'prediction_raw':h['prediction_raw'],'reference_index':h['basis']['reference_index'],
        'focus_axes':h['focus_axes'],'statement':h['statement'],'falsification':h['falsification']}
        for key,h in state['hypotheses'].items()]
    return {'description':pool['description'],'domain':deepcopy(pool['bounds']),
        'objective_name':pool['objective_name'],'objective_unit':pool['objective_unit'],
        'objective_scale':pool['objective_scale'],'query_round':t,'remaining_queries':17-t,
        'observations':observations,
        'initial_uncertain_preferences_normalized_0_to_1':deepcopy(state['initial_knowledge']),
        'own_past_hypothesis_declarations':hypotheses}


def validate(raw,pool,state):
    required={'action','point','prediction','hypothesis','pause_ids','reason','support'}
    if not isinstance(raw,dict) or set(raw)!=required:raise ValueError('exact blind-proposal schema required')
    if raw['action'] not in ('propose','abstain'):raise ValueError('invalid action')
    common.support_indices(raw,state['history'])
    if len(raw['support'])>3:raise ValueError('at most3 own supports')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1200:raise ValueError('concise rationale required')
    pause=raw['pause_ids']
    if not isinstance(pause,list) or len(pause)!=len(set(pause)) or any(k not in state['hypotheses'] for k in pause):
        raise ValueError('pause existing distinct hypothesis IDs only')
    if raw['action']=='abstain':
        if raw['point'] is not None or raw['prediction'] is not None or raw['hypothesis']!={}:raise ValueError('abstain requires null point/prediction and empty hypothesis')
        return deepcopy(raw)
    p=raw['point']
    if not isinstance(p,dict) or set(p)!=set(pool['axes']):raise ValueError('all physical axes required')
    for a,v in p.items():
        if type(v) not in (int,float) or not math.isfinite(v) or not pool['bounds'][a][0]<=v<=pool['bounds'][a][1]:raise ValueError('finite in-domain physical point required')
    v=raw['prediction']
    if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=2*pool['objective_scale']:raise ValueError('finite raw forecast required')
    h=raw['hypothesis']
    if not isinstance(h,dict) or set(h)!={'reference_index','focus_axes','statement','falsification'}:raise ValueError('structured contrast fields required')
    if type(h['reference_index'])!=int or h['reference_index'] not in raw['support']:raise ValueError('reference must be a cited own observation')
    axes=h['focus_axes']
    if not isinstance(axes,list) or not 1<=len(axes)<=2 or len(axes)!=len(set(axes)) or any(a not in pool['axes'] for a in axes):raise ValueError('one or two physical focus axes required')
    for k,maximum in [('statement',1200),('falsification',800)]:
        if not isinstance(h[k],str) or not 1<=len(h[k])<=maximum:raise ValueError('concise testable statement/falsification required')
    return deepcopy(raw)


def request(client,pool,state,t):
    public=payload(pool,state,t)
    context={'public':public,'observed_prefix':deepcopy(state['history']),'task':pool['task'],
        'phase':'blind_physical_proposal','protocol_sha256':state['protocol_sha256'],'system_sha256':digest(SYSTEM)}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=e.ROOT/'requests'/key
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.request.lock'),timeout=900):
        if (folder/'context.json').exists():assert digest(data.read(folder/'context.json'))==digest(context)
        else:data.write(folder/'context.json',context)
        answer=transport.get(client,context,folder,SYSTEM,lambda raw:validate(raw,pool,state))
    return answer,key
