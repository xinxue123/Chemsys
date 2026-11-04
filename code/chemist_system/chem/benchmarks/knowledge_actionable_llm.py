"""V7 proposal support: known references and current narrow-cost eligibility.

This changes the proposal interface, not V5 rule review, forecasts or selection.
The GP-derived boolean is explicit side information; proposals are not claimed
to be independent of the GP. No mean, uncertainty, acquisition value, BO label,
unqueried outcome or external benchmark result is sent to the model.
"""
import json
from pathlib import Path

from filelock import FileLock

from . import knowledge_actionability as support
from . import knowledge_transfer_llm as base
from . import knowledge_transfer_engine as engine
from . import knowledge_openai_client as api

digest, write, validate, make_menu = base.digest, base.write, base.validate, base.make_menu
SYSTEM = base.SYSTEM + '''

The proposal_support object now makes execution constraints explicit. Its
passes_narrow_cost_gate flag is computed from the current GP acquisition budget;
it is NOT a chemistry-quality label, an observation, a rule-validation result or
a posterior truth probability. optional_window_open describes only whether the
existing optional-validation time/quota window is open. Further rule scope,
information, counterevidence and ranking tests are still applied by software.

For each displayed candidate, references list up to four physically comparable
own measured experiments, ordered by measured objective then physical features.
The list may be truncated. Each reference gives required_focus_axes and
additional_changed_axes. A geometrically legal focus contains ALL required axes,
may include additional changed axes, and has one or two axes in total. This is
only geometric compatibility, not proof of causality or chemical comparability.
Use domain reasoning to decide what relation is defensible. Do not blindly turn
geometric differences into a positive chemical prediction.

When the optional window is open, prefer a chemically defensible example that
passes the narrow cost gate and has a useful comparable known reference. If no
such hypothesis is defensible, abstain or give another well-motivated local rule;
do not invent a mechanism just to satisfy a flag. The reference hints are not a
minus_id whitelist, and passing them does not replace the original review. Your
focus and background declarations must still be explicit and pre-experiment.
Software continues to use the highest compatible own observed reference within
the final rule scope, which may differ from your example reference. It may test
other in-scope plus points; the original example is not a forced experiment.

A closed window or false cost flag is not evidence that a chemical rule is false.
Normal BO observations can still test a rule, and already supported knowledge
can still affect weighted selection. Retain/revise/retract based on scientific
meaning and the supplied forward evidence; do not reset credit or expand scope
merely to manufacture current eligibility. Output the unchanged action schema.
'''


def public_payload(pool,state,t,menu_ids,coverage,*,context):
    payload = base.public_payload(pool,state,t,menu_ids,coverage)
    references = support.candidate_support(pool,state['history'],menu_ids,maximum_references=4)
    bo = int(context['bo_index'])
    ceiling = min(float(context['epsilon']),engine.PARAMETERS['verification_maximum_gap'])
    for row in references:
        j = pool.ix[row['point_id']]
        row['passes_narrow_cost_gate'] = bool(context['eligible'][j]
            and float(context['base_logei'][bo]-context['base_logei'][j]) <= ceiling)
    payload['proposal_support'] = dict(
        schema='known_reference_cost_flags_v1',
        optional_window_open=bool(t <= engine.PARAMETERS['last_verification_round']
            and state['verification_used'] < engine.PARAMETERS['maximum_optional_verifications']),
        maximum_references_per_candidate=4,
        reference_order='Own observed objective descending, then physical features; never candidate ID order.',
        focus_rule='Choose one or two changed axes, including all required_focus_axes; any other focus must come from additional_changed_axes.',
        geometry_note='Nonfocus categories must be equal; nonfocus normalized continuous differences may not exceed .25 plus 1e-12 numerical tolerance. Geometry alone does not establish a chemical effect.',
        cost_flag_note='Only the current narrow acquisition-cost gate, including the base-BO point if displayed. No guarantee of scope, incremental information, rank effect or rule validity.',
        candidate_support=references)
    return payload


def request(client,payload,folder):
    """V5 transport/schema/cache, keyed by this explicit interface and prompt."""
    context = dict(public=payload,system=SYSTEM,model=api.MODEL,reasoning_effort=api.REASONING_EFFORT)
    key = digest(context)
    directory = Path(folder)/key
    directory.mkdir(parents=True,exist_ok=True)
    with FileLock(str(directory/'.lock'),timeout=300):
        saved = directory/'validated.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            if result['context_sha256'] != key:
                raise ValueError('Request cache context mismatch')
            validate(result['proposal'])
            return result
        write(directory/'context.json',context)
        correction = None
        for attempt in range(1,3):
            user = payload if correction is None else dict(same_experiment_context=payload,schema_correction=correction)
            path = directory/f'response_{attempt:02d}.json'
            if path.exists():
                response = json.loads(path.read_text(encoding='utf-8'))
            else:
                response = client.complete(SYSTEM,user,json_output=True)
                write(path,response)
            try:
                proposal = validate(json.loads(response['output_text']))
            except (ValueError,TypeError):
                correction = 'Return the exact required action schema. No experiment has occurred.'
                continue
            result = dict(context_sha256=key,proposal=proposal,attempt=attempt,
                response_sha256=digest(response),resolved_model=response['resolved_model'])
            write(saved,result)
            return result
    raise ValueError('two response schema attempts exhausted')
