"""R1 observed-only region repair and bounded joint-condition experiment advice."""
from copy import deepcopy
import math
import numpy as np
from . import continuous_benchmark_llm as old
from . import continuous_benchmark_data as data
from . import revised_continuous_engine as engine
from . import reaction_transfer_llm as transport
from .suzuki_domain import digest

ROOT=engine.ROOT
REGION_SYSTEM=old.COMMON+'''
Update an uncertain preferred region using at most FOUR normalized intervals.
The JSON schema and grounded factual support are:
{"actions":[{"id":"H1","operation":"revise","bounds":[0.2,0.8],"reason":"concise uncertain preference",
"support":[{"index":1,"axis":"axis_name","value":0.0,"objective":0.0}]}],"summary":"..."}.
Use supplied fixed_slots. add for a vacant slot, revise for an active slot, retain/revoke for an active slot.
retain/revoke require bounds=[]; account for every active slot. Vacant slots may be omitted.
Intervals are normalized to[0,1], must have 0<=lower<upper<=1 and width<1. A revise must change geometry.
Support references ONLY this request's OWN displayed observations with exact PHYSICAL axis value and objective.
The joint region MUST contain at least ONE UNQUERIED fixed-pool candidate. The software checks this.
The supplied unqueried witness examples contain physical and normalized coordinates. You may widen,
revise or revoke unsupported preferences. An empty knowledge set is allowed. Do not overfit one best observation.
GP predictions are not observed facts. B4 diagnoses acquisition-budget compatibility, not chemical falsity;
its equally sized subset tie-break uses fixed hypothesis slots, not chemical credibility.
The same official regional covariance field modifies ONLY GP mean. The revised projected SCALAR KF
uses y-mu_GP = a*(gate*raw_field)+noise, with no nuisance b; a and deployment are bounded in[0,1].
The innovation is exactly actual observed minus the DEPLOYED prequery predicted mean. Paired squared-error
gain compares this deployed prediction against GP alone; a negative gain must not increase a.
Closed gates give zero measurement feature and zero deployed gain. Shadow predictions are separate,
frozen evidence for the original region-trust recovery latch only, not deployed-success claims.
A genuinely new geometry starts conservatively at a=0.5; renaming a blocked geometry cannot restore trust.
At most THREE region refreshes occur; at most TWO before query9. At least one allowance remains for
queries9..14. A refresh consumes an LLM allowance, not an extra experiment. Prefer a substantial,
testable revision or revoke when the knowledge adds nothing beyond GP. Do not merely chase GP top points.
Every objective field ending_y is normalized by objective_scale; other displayed objective values use raw units.
reason<=1000 chars, summary<=1600 chars. Concise scientific hypotheses only; output exactly actions,summary.'''

INTERACTION_SYSTEM=old.COMMON+'''
Select ONE existing unqueried experiment from the COMPLETE supplied joint_test_pool.
Every option already satisfies the unchanged base-qLogEI opportunity-loss gate. This test occupies one
of the same16 additional queries, with at most two tests per trajectory; no out-of-gate exception is allowed.
Each point changes TWO focused variables together from the own best observed experiment. Test interaction
or compensation, rather than extrapolating a single-axis result globally or always retaining the same bias.
For exact_two_axes, all other variables match the anchor exactly. For matched_background_joint_conditions,
the other variables may ALSO differ; inspect all_changed_axes and background_RMS_normalized. This is an
OBSERVATIONAL JOINT CONDITION CHALLENGE, not an isolated causal interaction experiment. Never claim it
identifies an independent causal effect. No off-pool point or synthetic observation may be executed.
Compare chemistry/process plausibility, GP uncertainty, deployed predictions, previous own joint results
and possible improvement within the remaining budget. No hidden outcomes or previous benchmark runs exist.
Return EXACT JSON {"candidate_id":"...","prediction":0.0,"reason":"concise testable rationale",
"support":[1,...]}. prediction is a finite nonnegative expected objective in supplied RAW units.
support references own observed1-based indices. Do not output rankings for unseen benchmark outcomes.
Give a concise rationale, not private reasoning; do not add conditions absent from domain.'''

def request(client,pool,state,t,phase,system,payload,validator):
    context={'public':payload,'observed_prefix':deepcopy(state['history']),'task':pool['task'],'phase':phase,
        'protocol_sha256':state['protocol_sha256'],'system_sha256':digest(system)}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=ROOT/'requests'/key
    data.write(folder/'context.json',context)
    return transport.get(client,context,folder,system,validator),key

def validate_region(raw,pool,state):
    plan=old.validate_region(raw,pool,state)
    admission=engine.admission(pool,plan['knowledge'],state['history'])
    if not admission['admitted']:raise ValueError('joint region has ZERO UNQUERIED executable candidates; revise/widen/revoke, using the supplied unqueried witnesses')
    return {**plan,'admission':admission}

def public(pool,state,t):
    payload=old.public(pool,state,t)
    for row in payload['observations']:
        i=row['index']
        if i>4:
            m=state['decisions'][i-5]['measurement']
            row['deployment_evidence']={k:m[k] for k in ('actual_deployed_correction_y','actual_paired_SE_gain','coefficient_before','coefficient_after')}
    payload['previous_joint_tests']=deepcopy(state['interaction_tests'])
    return payload

def region_call(client,pool,state,t,d,reasons,quota):
    p=public(pool,state,t);p.update(current_knowledge=state['knowledge'],reasons=reasons,refresh_allowance=quota,
        current_admission=d['admission'],B4_diagnosis=d['diagnosis'].to_dict(),calibration_bank=d['bank'],
        calibration_forecast=d['prior'],deployed_amplitude=d['amplitude'],trust_gate=d['trust'])
    js=sorted(np.flatnonzero(d['available']),key=lambda j:(-d['base'][j],d['tie_priority'][j]))[:16]
    p['unqueried_witness_examples']=[{'candidate_id':pool['ids'][j],'point':pool['points'][j],
        'normalized':dict(zip(pool['axes'],pool['z'][j])),'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
        'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'],'base_qLogEI':float(d['base'][j])} for j in js]
    # Full tied classes remain visible even if the physical witness list is capped.
    classes=[];covered=0;X=np.asarray(pool['z'])
    for score in np.unique(d['base'][d['available']])[::-1]:
        jj=np.flatnonzero(d['available']&(d['base']==score));classes.append({'base_qLogEI':float(score),'candidate_count':len(jj),
            'normalized_ranges':{a:[float(X[jj,k].min()),float(X[jj,k].max())] for k,a in enumerate(pool['axes'])}})
        covered+=len(jj)
        if covered>=16:break
    p['complete_highest_score_classes']=classes
    return request(client,pool,state,t,'region_refresh',REGION_SYSTEM,p,lambda raw:validate_region(raw,pool,state))

def validate_interaction(raw,prepared,state):
    options={r['candidate_id']:r for r in prepared['candidates']}
    if not isinstance(raw,dict) or set(raw)!={'candidate_id','prediction','reason','support'} or raw['candidate_id'] not in options:raise ValueError('select exactly one candidate from the eligible joint test pool')
    value=raw['prediction']
    if type(value) not in (float,int) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative expected raw objective required')
    old.support_indices(raw,state['history'])
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=1600:raise ValueError('concise rationale required')
    row=options[raw['candidate_id']]
    return {**deepcopy(raw),**{k:deepcopy(row[k]) for k in ('focus_axes','all_changed_axes','background_RMS_normalized','geometry')},
        'normal_gate_eligible':True}

def interaction_call(client,pool,state,t,d,prepared):
    p=public(pool,state,t);p.update(joint_test_pool=prepared,current_knowledge=state['knowledge'],
        knowledge_admission=d['admission'],deployed_amplitude=d['amplitude'])
    return request(client,pool,state,t,'interaction_test',INTERACTION_SYSTEM,p,lambda raw:validate_interaction(raw,prepared,state))
