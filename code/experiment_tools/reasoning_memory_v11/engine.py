"""Official Reasoning BO alone selects; KF only updates future memory context."""
from copy import deepcopy
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import engine as previous,policy,kf
from . import llm

PARAMETERS=dict(kf.PARAMETERS,shortlist=5,maximum_live_notes=4,maximum_memory_calls=4,
    maximum_reasoning_calls=16,minimum_edit_gap=3,late_edit_round=11,GP_modified=False,
    selection='unchanged_official_stable_confidence_sort_n1',KF_use='advisory_memory_only',
    Borda_fusion=False,KG_queries=False,additional_opportunity_gate=False)


def plan(root,pool,state,context,t,memory_receipt,reasoning_receipt,refresh):
    before=previous.gp_hash(context);ordered,display=previous.candidates(pool,context,t)
    hs=memory_receipt['plan']['hypotheses'] if memory_receipt else deepcopy(state['hypotheses'])
    selected=pool.ix[reasoning_receipt['plan']['selected_id']]
    if selected not in ordered:raise ValueError('Official selection escaped BO candidates')
    if llm.official_extract(root,reasoning_receipt['raw']['insight'])!=pool.ids[selected]:raise ValueError('Official selection changed')
    probabilities=reasoning_receipt['plan']['probabilities'];predictions=[]
    for h in hs:
        policy.validate_existing(h)
        if h['kf']['status']=='retracted' or selected not in policy.scope_indices(pool,h,[selected]):continue
        p=dict(id=h['id'],version=h['version'],query=t,point_id=pool.ids[selected],semantic_sha256=h['semantic_sha256'],
            reference_id=h['reference_id'],reference_y=h['reference_y'],p_LLM=probabilities[h['id']][pool.ids[selected]],
            p_GP=previous.gp_probability(context,selected,h['reference_y']),history_sha256=digest(state['history']),GP_sha256=before)
        predictions.append(dict(prediction=p,prediction_sha256=digest(p)))
    if before!=previous.gp_hash(context):raise ValueError('Memory mutated GP arrays')
    hypotheses=reasoning_receipt['raw']['insight']['hypotheses'];priority={'high':0,'medium':1,'low':2}
    winner=min(range(len(hypotheses)),key=lambda i:priority[hypotheses[i]['confidence'].lower()])
    c=dict(query=t,arm='rbo_memory',protocol_sha256=state['protocol_sha256'],history_sha256=digest(state['history']),
        GP_sha256=before,GP_sha256_after_ranking=previous.gp_hash(context),GP_fit_key=context['stats']['fit_key'],
        selected_id=pool.ids[selected],BO_reference_id=pool.ids[int(ordered[0])],mode='base_bo' if selected==ordered[0] else 'ReasoningBO_memory',
        ordered_ids=[pool.ids[int(j)] for j in ordered],display_ids=[pool.ids[int(j)] for j in display],
        raw_qLogEI=[float(context['base_logei'][j]) for j in ordered],winning_hypothesis_index=winner,
        opportunity_gap=float(context['base_logei'][ordered[0]]-context['base_logei'][selected]),
        hypotheses_at_commit=hs,reviews=memory_receipt['plan']['reviews'] if memory_receipt else [],predictions=predictions,
        memory_context_sha256=memory_receipt['context_sha256'] if memory_receipt else None,
        memory_raw=memory_receipt['raw'] if memory_receipt else None,LLM_context_sha256=reasoning_receipt['context_sha256'],
        LLM_raw=deepcopy(reasoning_receipt['raw']['insight']),reasoning_raw=deepcopy(reasoning_receipt['raw']),
        memory_references=reasoning_receipt['plan']['memory_references'],forecast_probabilities=probabilities,
        refresh=refresh,KF_changes_selection_from_fixed_weights=False,
        selection_contract='Official confidence extraction only. No weight fusion, veto or additional query channel.',
        prestate=deepcopy(state),prestate_sha256=digest(state))
    c['commit_sha256']=digest(c);return c


def feedback(state,commit,observation):
    updated=deepcopy(state);previous.feedback(updated,commit,observation)
    if commit['refresh']['allowed']:updated['edit_calls'].append(commit['query'])
    updated['decisions'][-1].update(memory_references=deepcopy(commit['memory_references']),
        winning_hypothesis_index=commit['winning_hypothesis_index'])
    state.clear();state.update(updated)
