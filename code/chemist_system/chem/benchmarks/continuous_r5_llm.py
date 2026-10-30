"""GP-blind proposals of actual inputs with executable direction contracts."""
from copy import deepcopy
import math
from filelock import FileLock
from . import continuous_benchmark_llm as common
from . import continuous_benchmark_data as data
from . import reaction_transfer_llm as transport
from . import continuous_r5_engine as e
from .suzuki_domain import digest

SYSTEM=common.COMMON+'''
Propose ONE REAL unqueried experiment from candidate_menu, or abstain. The menu
uses physical input geometry and your OWN observed results only. No GP predictions,
uncertainty, BO reference, acquisition scores, gate labels or other trajectories
are supplied. Menu order and IDs are not rankings.

Return EXACT JSON:
{"action":"propose","candidate_id":"...","prediction":100.0,
 "hypothesis":{"reference_index":1,"focus_axes":["physical_axis"],
   "directions":{"physical_axis":"increase"},
   "statement":"testable local observational contrast relative to reference",
   "falsification":"observed response that would contradict this contrast"},
 "pause_ids":[],"reason":"compare high-reference regions briefly","support":[1]}.
For abstain use action="abstain",candidate_id=null,prediction=null,hypothesis={}.
Select exactly one displayed candidate_id. Its EXACT displayed conditions are
executed if approved; there is NO nearest-point substitution. prediction is a
finite nonnegative RAW forecast <=2*objective_scale for that real candidate.

reference_index must be from high_references and cited in support (1..3 own
observed indices). Compare the different high-performing regimes; do not always
anchor on the global best. Declare1..2 focus_axes and exactly those direction keys
(increase/decrease), matching the real candidate relative to the reference.
Other axes must change by <=0.25 of their domain range. Acknowledge confounding:
this is an observational joint contrast, NOT identified independent causal slopes.
Never treat predictions as measurements or invent outcome/variable definitions.

Software independently checks direction/background consistency. A mismatch is
retained as a rejected proposal, never silently converted to an opposite experiment.
The local numeric model is a compact32-neighbor affine contrast with zero effect
outside the declared direction/background contract. Two distinct prospective
informative positive observations are required before next-round mean correction.
All proposed and validation experiments use the SAME sixteen query budget and
original GP review gate. A high forecast cannot override review.

pause_ids can pause existing models. Monitoring and negative evidence are retained;
repeating a pause does not reset its recovery epoch. No irreversible deletion.
At most4 calls are event scheduled, at least2 opportunities reserved for rounds11..14.
Use late observations to reconsider multiple high-performing regions. reason and
statement<=1200 chars, falsification<=800. Concise public rationale only; no private
reasoning trace. Output only the exact JSON.'''


def payload(pool,state,t):
    ix={key:j for j,key in enumerate(pool['ids'])}
    observations=[{'index':i,'point':deepcopy(pool['points'][ix[h['candidate_id']]]),'objective':float(h['raw_value'])}
        for i,h in enumerate(state['history'],1)]
    declarations=[{'id':key,'created_query':h['created_query'],'actual_point':deepcopy(pool['points'][ix[h['candidate_id']]]),
        'prediction_raw':h['prediction_raw'],'reference_index':h['basis']['reference_index'],'focus_axes':h['focus_axes'],
        'directions':h['semantic_contract']['directions'],'statement':h['statement'],'falsification':h['falsification']}
        for key,h in state['hypotheses'].items()]
    return {'description':pool['description'],'domain':deepcopy(pool['bounds']),'objective_name':pool['objective_name'],
        'objective_unit':pool['objective_unit'],'objective_scale':pool['objective_scale'],'query_round':t,
        'remaining_queries':17-t,'observations':observations,
        'initial_uncertain_preferences_normalized_0_to_1':deepcopy(state['initial_knowledge']),
        'high_references':e.high_references(pool,state),'candidate_menu':e.candidate_menu(pool,state,t),
        'own_past_hypothesis_declarations':declarations}


def validate(raw,pool,state,t):
    required={'action','candidate_id','prediction','hypothesis','pause_ids','reason','support'}
    if not isinstance(raw,dict) or set(raw)!=required:raise ValueError('exact R5 schema required')
    if raw['action'] not in ('propose','abstain'):raise ValueError('invalid action')
    common.support_indices(raw,state['history'])
    if len(raw['support'])>3:raise ValueError('at most3 own supports')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1200:raise ValueError('concise rationale required')
    pause=raw['pause_ids']
    if not isinstance(pause,list) or len(pause)!=len(set(pause)) or any(k not in state['hypotheses'] for k in pause):raise ValueError('pause existing distinct IDs only')
    if raw['action']=='abstain':
        if raw['candidate_id'] is not None or raw['prediction'] is not None or raw['hypothesis']!={}:raise ValueError('abstain needs null ID/prediction, empty hypothesis')
        return deepcopy(raw)
    if raw['candidate_id'] not in {r['candidate_id'] for r in e.candidate_menu(pool,state,t)}:raise ValueError('select a displayed actual candidate ID')
    v=raw['prediction']
    if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=2*pool['objective_scale']:raise ValueError('finite raw forecast required')
    h=raw['hypothesis']
    if not isinstance(h,dict) or set(h)!={'reference_index','focus_axes','directions','statement','falsification'}:raise ValueError('structured direction contract required')
    if type(h['reference_index'])!=int or h['reference_index'] not in {r['index'] for r in e.high_references(pool,state)} or h['reference_index'] not in raw['support']:raise ValueError('cite a displayed high reference')
    axes=h['focus_axes']
    if not isinstance(axes,list) or not 1<=len(axes)<=2 or len(axes)!=len(set(axes)) or any(a not in pool['axes'] for a in axes):raise ValueError('one or two physical focus axes required')
    directions=h['directions']
    if not isinstance(directions,dict) or set(directions)!=set(axes) or any(v not in ('increase','decrease') for v in directions.values()):raise ValueError('increase/decrease for each focus axis required')
    for key,maximum in [('statement',1200),('falsification',800)]:
        if not isinstance(h[key],str) or not 1<=len(h[key])<=maximum:raise ValueError('concise statement/falsification required')
    # Semantic mismatches are retained scientific rejections, not API retries.
    return deepcopy(raw)


def request(client,pool,state,t):
    context={'public':payload(pool,state,t),'observed_prefix':deepcopy(state['history']),'task':pool['task'],
        'phase':'blind_real_input_proposal','protocol_sha256':state['protocol_sha256'],'system_sha256':digest(SYSTEM)}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=e.ROOT/'requests'/key
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.request.lock'),timeout=900):
        if (folder/'context.json').exists():assert digest(data.read(folder/'context.json'))==digest(context)
        else:data.write(folder/'context.json',context)
        answer=transport.get(client,context,folder,SYSTEM,lambda raw:validate(raw,pool,state,t))
    return answer,key
