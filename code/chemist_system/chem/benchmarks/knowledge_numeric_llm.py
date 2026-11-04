"""V8 numerical local hypotheses; Terra transport, new schema/context cache.

Only current own observations and public input/constraint fields are exposed.
No current or birth GP means/variances, unqueried objectives, beta residuals or
other-arm results are included. A new SYSTEM makes old cached replies distinct.
"""
from copy import deepcopy
import json
from pathlib import Path

from filelock import FileLock

from . import knowledge_actionability as support
from . import knowledge_transfer_llm as old
from . import knowledge_transfer_engine as constraints
from . import knowledge_numeric_policy as policy
from . import knowledge_openai_client as api
from .knowledge_domain_context import task_context

digest,write,make_menu=old.digest,old.write,old.make_menu

SYSTEM='''You assist sequential chemistry/process experiments with prospective,
quantitative LOCAL hypotheses. Use public domain/input information and ONLY the
supplied own measured observations. Do not retrieve or recall benchmark target
values, invent observations or molecular identities behind opaque labels, use
other experimental arms, or interpret candidate IDs as rankings or distances.
Yield, space-time yield and detector response have different physical units.

Return exactly one JSON object with actions and summary. actions contains zero
to three objects, each with exactly op,id,plus_id,minus_id,focus,background,
numeric_prior,reason. At most four hypotheses remain live, including dormant
ones. op is add,retain,revise,retract. Add uses id:null; others cite an existing
id. Reasons must be public scientific rationales under 600 characters, prefixed
mechanistic_prior, observed_association or both. Summary is under 1000 characters.

For add/revise, plus_id is a displayed UNQUERIED candidate; minus_id is one of
your own already measured observations. They must differ physically. focus names
one or two actual axes. Continuous focus is {"direction":"increase" or
"decrease","bounds":[lo,hi]} in the supplied normalized 0..1 coordinates,
containing both anchors and agreeing with their change. Categorical focus is
{"direction":"prefer_plus","allowed":[real labels]}, containing the different
plus and reference labels. The shorthand "prefer_plus" is allowed. All other
focus categories remain unordered: never assign numeric ranks or distances.
Within the original example, nonfocus categories must match and nonfocus
continuous values may differ by at most .25. This checks geometry, not causality.

background explicitly declares pre-experiment transfer to NONFOCUS axes:
continuous {"bounds":[lo,hi]} or categorical {"allowed":[real labels]}, always
containing both examples. Omitted axes use the intersection of example-centered
.25 neighborhoods or their common category. State plausible interactions that
could reverse the proposed effect. Once declared, the scope and feature are
fixed. Later predictions may transfer within that scope across backgrounds;
software does not reanchor to a different highest measured reference each round.

numeric_prior is {"delta_mean_raw":positive number,"delta_sd_raw":positive
number} or null. Both numbers use the RAW objective unit shown in the task.
delta_mean_raw is your expectation of Y_plus minus the FIXED actually measured
reference objective. delta_sd_raw describes uncertainty about that expected
contrast; it is not experimental noise, a confidence label, or an ordinal
strength level. Supply an honest numerical prior only when defensible. null
is explicit numerical abstention: no new expert is created and a revised old
expert is not silently replaced. Empty actions is also valid abstention. No
precise target lookup or remembered benchmark answer is allowed. For retain or
retract use plus_id:null,minus_id:null,focus:{},background:{},numeric_prior:null.

The fixed feature equals one at plus, zero at reference and zero outside scope.
Inside scope each continuous component is clip((x-reference)/(plus-reference),
-1,1); a categorical component is the indicator of the plus label. One or two
components are averaged equally. This is a local model, not a universal causal
law. Changing an anchor, scope, feature or numerical prior requires a new version.

Software translates your predicted raw plus objective (measured reference plus
delta_mean_raw) into a residual relative to the GP frozen at hypothesis birth.
A positive FULL predicted contrast can therefore have a NEGATIVE residual if
that GP was more optimistic. Do not confuse residual sign with reversal of the
physical contrast. Only later, unique own experiments calibrate that coefficient;
birth observations are not counted again. The fixed birth covariance and noise
are not refitted inside this expert. GP conditioning removes feature information
already explained by those data. Even where the original feature is zero,
conditional correlations can leave a nonzero residual feature; deployment is
nevertheless restricted to the predeclared scope by a separate decision policy.

Forward evidence has four separate safety accounts: full expert prediction
versus current GP; full expert versus the matched birth-GP null; mean-only gains
against current GP; and mean-only gains against the birth-GP null. They assess
incremental predictive utility, not chemical truth, and are not independent
likelihoods to multiply. Variance-only improvements cannot establish a mean
benefit. Applicable negative evidence survives revisions once per experiment;
positive credit is not inherited. Retain, revise or retract from the prospective
feedback, not by changing scope to erase a failed observation. Empty current
coverage means dormant, not scientifically false.

proposal_support provides up to four known geometrically compatible references
per candidate, ordered by measured objective then physical features. Include all
required_focus_axes and choose other focus axes only from additional_changed_axes.
The list can be truncated and is not a minus_id whitelist. The cost flag is
GP-derived execution side information, NOT a chemistry-quality label, a
validation result or a truth probability. optional_window_open reports only
the time/quota window, not a promise of verification. Prefer a defensible low
cost example while that window is open; abstain rather than invent a mechanism
or numerical effect to satisfy the flag. No GP means, uncertainty values,
acquisition scores or BO identity are provided. The proposals are constraint
informed, not claimed to be independent of the GP. An original example is not
a forced experiment; current own observations can also test an active expert.
'''


def validate(raw):
    if not isinstance(raw,dict) or set(raw)!={'actions','summary'}:
        raise ValueError('Expected actions and summary')
    if not isinstance(raw['summary'],str) or not 1<=len(raw['summary'].strip())<=1000:
        raise ValueError('Invalid summary')
    if not isinstance(raw['actions'],list) or len(raw['actions'])>policy.MAX_ACTIONS:
        raise ValueError('At most three actions')
    for action in raw['actions']:
        if not isinstance(action,dict) or set(action)!=policy.ACTION_FIELDS:
            raise ValueError('Invalid numeric action keys')
        if action['op'] not in ('add','retain','revise','retract'):raise ValueError('Invalid operation')
        if not isinstance(action['focus'],dict) or not isinstance(action['background'],dict):
            raise ValueError('Focus and background must be objects')
        prior=policy.validate_numeric_prior(action['numeric_prior'])
        if action['op'] in ('retain','retract') and (action['plus_id'] is not None or
                action['minus_id'] is not None or action['focus']!={} or action['background']!={} or prior is not None):
            raise ValueError('Retain/retract cannot rewrite numerical semantics')
        if not isinstance(action['reason'],str) or not 1<=len(action['reason'].strip())<=600:
            raise ValueError('Invalid public reason')
    return deepcopy(raw)


def public_payload(pool,state,t,menu_ids,coverage,*,context,parameters=None):
    parameters=constraints.PARAMETERS if parameters is None else parameters
    domain=task_context(pool.task,pool);domain.pop('source_references',None)
    known={row['candidate_id'] for row in state['history']}
    if set(menu_ids)&known:raise ValueError('Candidate menu must contain only unqueried points')
    def point(key):
        j=pool.ix[key]
        return dict(id=key,point=deepcopy(pool.points[j]),normalized_continuous={
            axis:float(pool.features[j][k]) for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds)) if kind=='continuous'})
    ledger=[]
    for h in state['hypotheses']:
        row={key:deepcopy(h[key]) for key in ('id','version','parent','focus','background','scope',
            'categorical_preference','status','reason','created_round') if key in h}
        row['origin_example']=dict(plus_id=h['plus_id'],minus_id=h['minus_id'])
        row['numeric_prior']={key:h['numeric_prior'][key] for key in ('delta_mean_raw','delta_sd_raw')}
        row['evidence']=policy.evidence_summary(h)
        row['tested_predictions']=[dict(query_round=e['query_round'],point_id=e['point_id'],
            reference_id=e.get('reference_id',h['minus_id']),
            **{key:float(e.get(key,0.)) for key in policy.GAIN_FIELDS}) for e in h['ledger']['events']]
        row['inherited_counterevidence']=[dict(event_id=e['event_id'],point_id=e['point_id'],
            reference_id=e['reference_id'],query_round=e['query_round'],applicable=e['applicable'],
            **{key:float(e.get(key,0.)) for key in policy.GAIN_FIELDS}) for e in h.get('inherited_negative_evidence',[])]
        ledger.append(row)
    references=support.candidate_support(pool,state['history'],menu_ids,maximum_references=4)
    bo=int(context['bo_index']);ceiling=min(float(context['epsilon']),parameters['verification_maximum_gap'])
    for row in references:
        j=pool.ix[row['point_id']]
        row['passes_narrow_cost_gate']=bool(context['eligible'][j] and float(context['base_logei'][bo]-context['base_logei'][j])<=ceiling)
    safe_coverage=('id','version','weight','eligible_inside_scope','executable_probes','executable_points',
                   'can_change_rank_at_cap','eligibility_state')
    return dict(task=pool.task,objective_unit=pool.unit,query=t,remaining_queries=17-t,
        domain_context=domain,proposal_interface='numeric_birth_contrast_v1',
        menu_policy='Same GP-filtered menu and physical neighbors; no GP score, posterior or BO label',
        axes=[dict(name=a,type=k) for a,k in zip(pool.axes,pool.kinds)],
        own_observations=[dict(point(r['candidate_id']),objective=r['raw_value']) for r in state['history']],
        candidate_menu=[point(key) for key in menu_ids],knowledge_ledger=ledger,
        actionable_coverage=[{key:deepcopy(row[key]) for key in safe_coverage if key in row} for row in coverage],
        proposal_support=dict(schema='known_reference_cost_flags_v1',
            optional_window_open=bool(t<=parameters['last_verification_round'] and state['verification_used']<parameters['maximum_optional_verifications']),
            maximum_references_per_candidate=4,
            reference_order='Own measured objective descending, physical-feature tie break; never ID order.',
            focus_rule='One or two changed axes, including all required axes; other focus axes from additional_changed_axes.',
            geometry_note='Geometry supports the original example only; causal transfer requires a predeclared scope.',
            cost_flag_note='GP-derived narrow cost gate only, including BO if displayed; not a quality label or complete admission test.',
            candidate_support=references),
        feedback_note='Raw-unit positive full contrast and SD are declared before feedback. Birth-relative GP residual can have a different sign. Four score accounts are utility constraints, not independent likelihoods; no inherited positive credit.')


def request(client,payload,folder):
    """New exact context key, fixed existing Terra/medium transport, no fallback."""
    context=dict(public=payload,system=SYSTEM,model=api.MODEL,reasoning_effort=api.REASONING_EFFORT)
    key=digest(context);directory=Path(folder)/key;directory.mkdir(parents=True,exist_ok=True)
    with FileLock(str(directory/'.lock'),timeout=300):
        saved=directory/'validated.json'
        if saved.exists():
            result=json.loads(saved.read_text(encoding='utf-8'))
            if result['context_sha256']!=key:raise ValueError('Request cache context mismatch')
            validate(result['proposal']);return result
        write(directory/'context.json',context);correction=None
        for attempt in range(1,3):
            user=payload if correction is None else dict(same_experiment_context=payload,schema_correction=correction)
            path=directory/f'response_{attempt:02d}.json'
            if path.exists():response=json.loads(path.read_text(encoding='utf-8'))
            else:
                response=client.complete(SYSTEM,user,json_output=True);write(path,response)
            try:proposal=validate(json.loads(response['output_text']))
            except (ValueError,TypeError):
                correction='Return the exact numeric-prior action schema. No experiment has occurred.';continue
            result=dict(context_sha256=key,proposal=proposal,attempt=attempt,response_sha256=digest(response),
                        resolved_model=response['resolved_model'])
            write(saved,result);return result
    raise ValueError('Two response schema attempts exhausted')
