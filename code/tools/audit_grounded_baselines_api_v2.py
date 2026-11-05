"""Original complete scientific audit plus honest API/subagent transport custody."""
from contextlib import redirect_stdout
import argparse
import io
import json
from pathlib import Path
import audit_grounded_baselines as old
from chem.benchmarks.knowledge_openai_client import _requested_family

ROOT, BASE, OUT = old.ROOT, old.BASE, old.OUT
read, sha, digest, require = old.read, old.sha, old.digest, old.require
ORIGINAL_CHECK_LLM = old.check_llm


def check_llm(c, pool, state, prior, background, examples, extract):
    af = (ROOT/c['LLM']['request']).parent
    receipt = read(af/'receipt.json')
    if receipt.get('transport_type') != 'openai_responses_api':
        return ORIGINAL_CHECK_LLM(c, pool, state, prior, background, examples, extract)
    saved = c['LLM']
    req, assignment, completion = [read(af/n) for n in ('request.json','assignment.json','completion.json')]
    reg_paths = [BASE/'api_transport_registration.json',BASE/'api_transport_registration_v2.json']
    reg_path = next(p for p in reg_paths if sha(p) == receipt['registration_sha256'])
    require(receipt == saved['receipt'], 'committed API receipt')
    require(req['request_sha256'] == digest({k:v for k,v in req.items() if k != 'request_sha256'}), 'API request digest')
    require(receipt['request_sha256'] == assignment['request_sha256'] == req['request_sha256'], 'API request identity')
    require(receipt['assignment_sha256'] == sha(af/'assignment.json') and receipt['completion_sha256'] == sha(af/'completion.json'), 'API assignment and completion custody')
    require(receipt['raw_sha256'] == sha(af/'response.raw.json') and saved['raw'] == read(af/'response.raw.json'), 'API raw output custody')
    require(receipt['registration_sha256'] == assignment['registration_sha256'] == sha(reg_path), 'registered API continuation')
    inv = read(af/'invocation.json')
    require(assignment['invocation_sha256'] == sha(af/'invocation.json'), 'original planned invocation retained')
    require(inv['model'] == req['model'] == 'gpt-6.1-terra' and inv['reasoning_effort'] == req['effort'] == 'medium', 'requested model and effort unchanged')
    call = assignment['agent_id']
    require(call.startswith('openai-responses/') and call == receipt['agent_id'] == completion['agent_id'], 'actual HTTP call identifier, not fabricated agent')
    require(assignment['identity_kind'] == 'actual_stateless_HTTP_call_not_a_collaboration_agent', 'honest identity semantics')
    result = read(af/'api_result.json')
    require(receipt['api_result_sha256'] == completion['api_result_sha256'] == sha(af/'api_result.json'), 'actual public API result custody')
    require(result['status'] == 'completed' and result['requested_model'] == 'gpt-6.1-terra' and _requested_family(result['resolved_model']), 'API model/status verified')
    require(result['id'] == receipt['response_id'] == completion['response_id'], 'provider response ID')
    require(json.loads(result['output_text']) == saved['raw'] and result['usage'] == receipt['usage'], 'API text/usage reconciliation')
    started = read(af/'api_call_started.json')
    parameters = dict(read(reg_path)['wire_parameters'])
    if 'json_output_rule' in read(reg_path) and 'json' in (req['system']+json.dumps(req['public'],ensure_ascii=False)).casefold():
        parameters['text'] = {'format': {'type': 'json_object'}}
    require(started['call_id'] == call and started['parameters'] == parameters, 'actual stateless request parameters')
    require(not (af/'api_transport_error.json').exists(), 'no silent HTTP rerolls')
    export = read(af/'export.json')
    require(sha(ROOT/export['input']) == export['input_sha256'] == assignment['input_sha256'] == started['input_sha256'], 'exact API input custody')
    require(sha(ROOT/export['output']) == completion['output_sha256'], 'completed API output unchanged')
    ctx = read(af.parent/'context.json')
    public = ctx['public']
    require(digest(ctx) == saved['context_sha256'] == req['context_sha256'], 'exact scientific context key')
    require(ctx['system'] == req['system'] == old.system_constants()[state['arm'].upper()+'_SYSTEM'], 'frozen method system instructions')
    payload = public if req['attempt'] == 1 else dict(original=public,schema_repair_only=read(af.parent/'attempt1/error.json'))
    require(req['public'] == payload and read(ROOT/export['input']) == dict(system=req['system'],public=payload), 'exact stateless API scientific input')
    observations = [dict(query=i+1,candidate_id=h['candidate_id'],conditions=pool.points[pool.ix[h['candidate_id']]],objective_raw=h['raw_value']) for i,h in enumerate(state['history'])]
    require(public['own_observations'] == observations, 'complete own prefix only')
    require(public['background'] == background and public['blinded_initial_hypotheses'] == prior, 'same background and blind priors')
    require(public['experimental_budget'] == dict(initial=0,total_paid_queries=20,observed=len(observations),remaining=20-len(observations)), 'unchanged paid-query budget')
    require(public['incumbent_raw'] == max(h['raw_value'] for h in state['history']) and not public['current_GP_predictions_or_ranking_supplied'], 'no numerical GP ranks exposed')
    common_keys = {'background','objective_unit','experimental_budget','own_observations','blinded_initial_hypotheses','incumbent_raw','current_GP_predictions_or_ranking_supplied'}
    if state['arm'] == 'lgbo':
        require(set(public) == common_keys|{'input_only_examples','previous_preferences','all_unqueried_pool_inputs_allowed'}, 'authorized LGBO context only')
        require(public['previous_preferences'] == state['preferences'] and public['all_unqueried_pool_inputs_allowed'] is True, 'own preferences and full input domain')
        seen = {h['candidate_id'] for h in state['history']}
        require(public['input_only_examples'] == [r for r in examples if r['candidate_id'] not in seen], 'input-only unscored examples')
    else:
        require(set(public) == common_keys|{'candidate_menu','axes','candidate_source','official_optimization_prompt'}, 'authorized RBO context only')
        ids = c['decision']['display_ids']
        require([p['candidate_id'] for p in public['candidate_menu']] == ids, 'RBO GP-top5 menu')
        for row in public['candidate_menu']:
            require(row['conditions'] == pool.points[pool.ix[row['candidate_id']]], 'RBO chemical conditions')
        template = read(ROOT/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json')['optimization_loop']['en']
        expected = template.format(iteration=len(state['history']),target=background['objective'],retrieved_context=json.dumps(prior),
            insight_history=json.dumps(state['insights'],ensure_ascii=False),trial_data=json.dumps(observations,ensure_ascii=False),
            bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
        require(public['official_optimization_prompt'] == expected, 'official RBO prompt and own Insight history')
        with redirect_stdout(io.StringIO()):
            selected = extract(None,json.dumps(saved['raw']),n=1)
        require(selected == [{'candidate_id':c['decision']['selected_id']}], 'official stable confidence selection')
    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    reg = read(BASE/'api_transport_registration_v2.json')
    require(reg['original_protocol_sha256'] == read(BASE/'protocol.json')['protocol_sha256'], 'API registration protocol')
    for path,h in reg['source_sha256'].items():
        require(sha(ROOT/path) == h, 'registered API source '+path)
    # Recheck immutable paused evidence, including checkpoint copies.
    snap = ROOT/'experiments/reports/grounded_baselines_20261105/paused_snapshot'
    manifest = read(snap/'snapshot_manifest_sha256.json')
    require(sha(snap/'snapshot_manifest_sha256.json') == reg['pause_manifest_sha256'], 'pause manifest preserved')
    for path,h in manifest['files'].items():
        require(sha(ROOT/path) == h, 'pause snapshot preserved '+path)
    for path,h in reg['existing_response_receipts_sha256'].items():
        require(sha(ROOT/path) == h, 'pre-API response receipt unchanged '+path)
    for name in ('api_transport_registration.json','api_transport_registration_v2.json'):
        for path,h in read(BASE/name)['source_sha256'].items():
            require(sha(ROOT/path) == h, 'all transport versions preserved '+path)
    for path,h in reg['archived_rejected_HTTP_files_sha256'].items():
        require(sha(ROOT/path) == h, 'rejected HTTP evidence preserved '+path)
    old.check_llm = check_llm
    value = old.audit(args.final)
    backend_counts = {}
    for p in BASE.glob('requests/*/*/attempt*/receipt.json'):
        r = read(p)
        backend = r.get('transport_type','independent_collaboration_subagent')
        backend_counts[backend] = backend_counts.get(backend,0)+1
    value.update(transport='Explicitly recorded subagent-to-API continuation; not a pure-API replication',
                 actual_response_attempts_by_backend=backend_counts,paused_snapshot_unchanged=True,
                 all_pre_API_responses_preserved=True,API_source_files_verified=len(reg['source_sha256']))
    path = OUT/('audit.json' if args.final else 'audit_partial.json')
    full = read(path)
    full.update(value)
    path.write_text(json.dumps(full,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
