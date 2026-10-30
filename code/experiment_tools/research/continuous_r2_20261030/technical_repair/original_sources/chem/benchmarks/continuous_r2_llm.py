"""R2 observed-only proposals, shared-prefix response caching, optional joint test."""
from copy import deepcopy
import math
import numpy as np
from filelock import FileLock
from . import continuous_benchmark_llm as original
from . import revised_continuous_llm as previous
from . import reaction_transfer_llm as transport
from . import continuous_benchmark_data as data
from . import continuous_r2_engine as e
from .suzuki_domain import digest

ROOT = e.ROOT
REGION_SYSTEM = previous.REGION_SYSTEM.replace(
    'A genuinely new geometry starts conservatively at a=0.5;',
    'A genuinely new geometry starts no stronger than a=0.5 and inherits lower coefficients from overlapping failed geometries;') + '''
R2 deployment scheduling replaces immediate deployment: this refresh creates a PENDING proposal.
This round executes the unchanged base-BO recommendation. The proposal's prediction is frozen
before feedback and evaluated as a SHADOW prediction on that BO experiment. It can be promoted
only on a later round after positive prospective paired-error evidence; an empty proposal may
withdraw preferences. This shadow evidence is not claimed as actual-deployment KF success.
Minor interval changes cannot erase actual negative calibration evidence. Do not make confident
single-variable claims from comparisons where other variables also changed. Explain uncertainty.
No held-out candidate outcomes, other trajectories, or previous benchmark results are available.'''

JOINT_SYSTEM = original.COMMON + '''
Review whether the scheduled later joint-condition challenge should replace base BO.
You may ABSTAIN and keep the supplied BO_reference, even when it is not in joint_test_pool.
Return EXACT JSON {"action":"test" or "keep_BO","candidate_id":"...","prediction":0.0,
"reason":"concise rationale","support":[1,...]}.
For keep_BO, candidate_id MUST equal BO_reference.candidate_id. For test, select ONE existing
candidate from the COMPLETE joint_test_pool. Every test already satisfies the original base
qLogEI opportunity-loss gate and uses one of the SAME sixteen experiment queries.
Compare BO's acquisition, mean and uncertainty explicitly against the joint challenge. High GP
mean or low uncertainty alone is not enough to justify overriding an exploration opportunity.
Use test when it has a clear scientific discrimination purpose beyond ordinary BO. You are
not required to spend a joint test. No out-of-gate point, synthetic observation or new variable.
Exact two-axis neighbors still do not identify an interaction without the two single-axis
controls. Scattered matched-background points change other variables too; they are observational
joint-condition challenges and cannot identify independent causal interactions.
prediction is finite, nonnegative and in RAW objective units; it is a forecast, not a fact.
support contains own observed1-based indices only. reason<=1600 characters. No private reasoning.'''


def request(client, pool, state, t, phase, system, payload, validator):
    # Method labels and ablation switches are deliberately absent. Identical own
    # prefix + prompt reuses one response across main and ablation trajectories.
    context = {'public': payload, 'observed_prefix': deepcopy(state['history']),
               'task': pool['task'], 'phase': phase, 'protocol_sha256': state['protocol_sha256'],
               'system_sha256': digest(system)}
    context['context_sha256'] = digest(context); key = context['context_sha256']
    folder = ROOT/'requests'/key; folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder/'.request.lock'), timeout=900):
        if (folder/'context.json').exists():
            assert data.read(folder/'context.json') == context
        else:
            data.write(folder/'context.json', context)
        answer = transport.get(client, context, folder, system, validator)
    return answer, key


def region_payload(pool, state, t, d, reasons, quota):
    payload = previous.public(pool, state, t)
    payload.update(current_knowledge=deepcopy(state['knowledge']), reasons=list(reasons),
        refresh_allowance=deepcopy(quota), current_admission=d['admission'],
        B4_diagnosis=d['diagnosis'].to_dict(), calibration_bank=d['bank'],
        calibration_forecast=d['prior'], deployed_amplitude=d['amplitude'], trust_gate=d['trust'],
        active_suspended=state['active_suspended'],
        prior_pending_assessments=[{'created_query': p['created_query'], 'knowledge': p['knowledge'],
            'status': p['status'], 'checks': p['checks']} for p in state.get('proposal_archive', [])],
        negative_geometry_evidence=deepcopy(state['geometry_ledger']))
    if state.get('pending'):
        payload['current_pending'] = {k: state['pending'][k] for k in ['created_query','knowledge','status','checks']}
    js = sorted(np.flatnonzero(d['available']), key=lambda j: (-d['base'][j], d['tie_priority'][j]))[:16]
    payload['unqueried_witness_examples'] = [{'candidate_id':pool['ids'][j], 'point':pool['points'][j],
        'normalized':dict(zip(pool['axes'],pool['z'][j])), 'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
        'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'], 'base_qLogEI':float(d['base'][j])} for j in js]
    classes = []; covered = 0; X = np.asarray(pool['z'])
    for score in np.unique(d['base'][d['available']])[::-1]:
        jj = np.flatnonzero(d['available'] & (d['base'] == score))
        classes.append({'base_qLogEI':float(score), 'candidate_count':len(jj),
                        'normalized_ranges':{a:[float(X[jj,k].min()),float(X[jj,k].max())] for k,a in enumerate(pool['axes'])}})
        covered += len(jj)
        if covered >= 16:
            break
    payload['complete_highest_score_classes'] = classes
    return payload


def region_call(client, pool, state, t, d, reasons, quota):
    return request(client,pool,state,t,'region_refresh',REGION_SYSTEM,
        region_payload(pool,state,t,d,reasons,quota),lambda raw:previous.validate_region(raw,pool,state))


def reference(pool, d):
    j = d['reference']; scale = pool['objective_scale']
    return {'candidate_id':pool['ids'][j], 'point':pool['points'][j],
            'GP_mean':float(d['stats']['mu_y'][j])*scale,
            'GP_latent_SD':float(d['stats']['sd_y'][j])*scale,
            'base_qLogEI':float(d['base'][j]), 'normal_gate_eligible':True}


def validate_joint(raw, prepared, state, bo_reference):
    if not isinstance(raw,dict) or set(raw) != {'action','candidate_id','prediction','reason','support'}:
        raise ValueError('action,candidate_id,prediction,reason,support required')
    if raw['action'] not in ('test','keep_BO'):
        raise ValueError('action must be test or keep_BO')
    if type(raw['prediction']) not in (int,float) or not math.isfinite(raw['prediction']) or raw['prediction'] < 0:
        raise ValueError('finite nonnegative RAW objective forecast required')
    original.support_indices(raw,state['history'])
    if not isinstance(raw['reason'],str) or not 1 <= len(raw['reason']) <= 1600:
        raise ValueError('concise rationale required')
    if raw['action'] == 'keep_BO':
        if raw['candidate_id'] != bo_reference['candidate_id']:
            raise ValueError('keep_BO must select the actual BO_reference even if outside joint pool')
        return {**deepcopy(raw),'normal_gate_eligible':True,'geometry':'BO_abstention',
                'focus_axes':[],'all_changed_axes':[],'background_RMS_normalized':0.}
    plan = previous.validate_interaction({k:v for k,v in raw.items() if k!='action'},prepared,state)
    return {'action':'test',**plan}


def joint_payload(pool, state, t, d, prepared):
    payload = previous.public(pool,state,t)
    payload.update(joint_test_pool=prepared, BO_reference=reference(pool,d),
                   current_knowledge=deepcopy(state['knowledge']), knowledge_admission=d['admission'],
                   deployed_amplitude=d['amplitude'], keep_BO_allowed=True)
    return payload


def joint_call(client,pool,state,t,d,prepared):
    ref = reference(pool,d)
    return request(client,pool,state,t,'optional_joint_test',JOINT_SYSTEM,
        joint_payload(pool,state,t,d,prepared),lambda raw:validate_joint(raw,prepared,state,ref))
