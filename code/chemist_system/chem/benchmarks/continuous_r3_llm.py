"""Observed-only R3 candidate proposals with explicit local numeric hypotheses."""
from copy import deepcopy
import math
from filelock import FileLock
from . import continuous_benchmark_llm as old
from . import continuous_benchmark_data as data
from . import reaction_transfer_llm as transport
from . import continuous_r3_engine as e
from .suzuki_domain import digest

SYSTEM=old.COMMON+'''
Propose an experiment and a TESTABLE LOCAL numeric hypothesis, or abstain.
The base GP independently scores the entire unchanged input pool. The candidate
card is a context-limited menu: BO top16, own incumbent neighbors16, and nearby
validation opportunities for up to two unresolved hypotheses. Menu order and IDs
are not rankings. You may select ONE displayed candidate_id only. No unseen
benchmark outcomes or other trajectories exist. Review uses BASE qLogEI loss:
if loss>epsilon, the proposed experiment is rejected, even if your forecast is high.

Return EXACT JSON:
{"action":"propose","candidate_id":"...","prediction":100.0,
 "hypothesis":{"statement":"uncertain approximately constant response level nearby",
               "focus_axes":["axis_name"],"falsification":"observed result that would contradict the claim"},
 "withdraw_ids":[],"reason":"concise candidate rationale","support":[1,2]}.
For abstain use action="abstain", candidate_id=null,prediction=null,hypothesis={}.
withdraw_ids may name existing hypotheses to retire without requiring a positive
forecast gain. Withdrawing affects that model only. support must reference 1..3
OWN observation indices; do not claim a GP prediction was an observation.
prediction is a finite nonnegative forecast in RAW units, at most2*objective_scale.
It is NOT an observed yield or a probability. Focus on 1 or2 named continuous axes.

The numerical local model is intentionally simple: your expected response level
is approximately constant around the nominated point. The software derives a
normalized RMS neighborhood using the 32nd nearest FIXED input, including distance
ties; a compact kernel makes its effect exactly zero outside that neighborhood.
This does not assert a causal slope on focus_axes. Other axes can differ within
the neighborhood. Explain confounding; avoid global conclusions from mixed changes.
Do not overstate confidence. A plausible reason alone does not authorize a mean shift.

New hypotheses cannot modify the GP mean before future experimental feedback.
Each own observed experiment inside scope receives a FROZEN prequery shadow check.
Its bounded feature is kernel*(your forecast-GP mean), clipped to one GP predictive SD.
A quarter-strength shadow is scored against GP with its uncertainty; an informative
positive rolling validation score can authorize a small LOCAL mean correction next
round. Two positives allow a larger cap. Actual failures suspend only that hypothesis;
shadow checks continue, so later evidence can restore it. All forecasts used in a
check are committed before observing the experimental result.

Validated local guidance has decision priority if it strictly changes the optimum
inside the original base acquisition gate. Otherwise a GP-approved proposal is a
budgeted validation trial. Abstaining leaves the validated policy/base BO in control.
All trials use one of the SAME sixteen additional experiments. No extra tests.
reason,statement<=1200 chars; falsification<=800 chars. No private reasoning.'''


def validate(raw,pool,state,card):
    required={'action','candidate_id','prediction','hypothesis','withdraw_ids','reason','support'}
    if not isinstance(raw,dict) or set(raw)!=required:raise ValueError('exact candidate/local-hypothesis schema required')
    if raw['action'] not in ('propose','abstain'):raise ValueError('action must be propose or abstain')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1200:raise ValueError('concise reason required')
    old.support_indices(raw,state['history'])
    if len(raw['support'])>3:raise ValueError('cite at most3 own observations')
    withdrawal=raw['withdraw_ids']
    if not isinstance(withdrawal,list) or len(withdrawal)!=len(set(withdrawal)) or any(k not in state['hypotheses'] for k in withdrawal):
        raise ValueError('withdraw_ids must name existing distinct hypothesis IDs')
    if raw['action']=='abstain':
        if raw['candidate_id'] is not None or raw['prediction'] is not None or raw['hypothesis']!={}:raise ValueError('abstain uses null candidate/prediction and empty hypothesis')
        return deepcopy(raw)
    menu={row['candidate_id'] for row in card}
    if raw['candidate_id'] not in menu:raise ValueError('propose one displayed unqueried candidate_id')
    value=raw['prediction']
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=2*pool['objective_scale']:
        raise ValueError('forecast must be finite, RAW and within[0,2*objective_scale]')
    hypothesis=raw['hypothesis']
    if not isinstance(hypothesis,dict) or set(hypothesis)!={'statement','focus_axes','falsification'}:raise ValueError('statement,focus_axes,falsification required')
    axes=hypothesis['focus_axes']
    if not isinstance(axes,list) or not 1<=len(axes)<=2 or len(axes)!=len(set(axes)) or any(a not in pool['axes'] for a in axes):
        raise ValueError('one or two known continuous focus_axes required')
    for key,maximum in [('statement',1200),('falsification',800)]:
        if not isinstance(hypothesis[key],str) or not 1<=len(hypothesis[key])<=maximum:raise ValueError('concise local statement and falsification required')
    return deepcopy(raw)


def payload(pool,state,d,t,card):
    result=old.public(pool,state,t)
    result.update(candidate_card=card,epsilon=d['budget'],BO_reference=pool['ids'][d['reference']],
        local_model_definition=e.PARAMETERS['local_forecast'],
        protected_prefix_queries=e.BOOTSTRAP[pool['task']],
        own_legacy_prefix_knowledge=state['legacy_context'],
        hypothesis_registry=[{'id':key,'created_query':h['created_query'],'anchor_candidate_id':h['candidate_id'],
            'prediction_raw':h['prediction_raw'],'scope':h['scope'],'statement':h['statement'],
            'focus_axes':h['focus_axes'],'status':h['status'],'evidence':h['evidence'],
            'last_actual_failure_query':h['last_actual_failure_query'],
            'recent_shadow_checks':h['shadow_records'][-4:],'recent_actual_checks':h['actual_records'][-2:]}
            for key,h in state['hypotheses'].items()])
    return result


def request(client,pool,state,d,t,card):
    public=payload(pool,state,d,t,card)
    context={'public':public,'observed_prefix':deepcopy(state['history']),'task':pool['task'],
             'phase':'candidate_and_local_hypothesis','protocol_sha256':state['protocol_sha256'],'system_sha256':digest(SYSTEM)}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=e.ROOT/'requests'/key
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.request.lock'),timeout=900):
        if (folder/'context.json').exists():assert digest(data.read(folder/'context.json'))==digest(context)
        else:data.write(folder/'context.json',context)
        answer=transport.get(client,context,folder,SYSTEM,lambda raw:validate(raw,pool,state,card))
    return answer,key
