"""Keep original numerical reporting and add per-round transport provenance."""
from pathlib import Path
import csv
import json
import report_grounded_baselines as old

ROOT, BASE, OUT = old.ROOT, old.BASE, old.OUT


def main():
    if (BASE/'completion_receipt.json').exists():
        print('Completed results are frozen; no rewrite.')
        return
    old.main()
    rows = []
    groups = []
    for sf in sorted(BASE.glob('runs/*/*/state.json')):
        state = old.read(sf)
        counts = dict(independent_collaboration_subagent=0,openai_responses_api=0)
        for call in state['calls']:
            af = (ROOT/call['request']).parent
            r = old.read(af/'receipt.json')
            backend = r.get('transport_type','independent_collaboration_subagent')
            counts[backend] += 1
            rows.append(dict(task=sf.parent.name.rsplit('_',1)[0],seed=state['seed'],method=state['arm'],
                query=call['query'],backend=backend,requested_model=r['model'],effort=r['effort'],
                resolved_model=r.get('resolved_model','not_provider_attested'),
                response_or_agent_id=r.get('response_id',r['agent_id']),request=call['request']))
        groups.append(dict(task=sf.parent.name.rsplit('_',1)[0],seed=state['seed'],method=state['arm'],
            observed=len(state['history']),status=state['status'],**counts))
    old.csvout('LLM_backend_by_round.csv',rows)
    old.csvout('LLM_backend_by_trajectory.csv',groups)
    sub = sum(g['independent_collaboration_subagent'] for g in groups)
    api = sum(g['openai_responses_api'] for g in groups)
    p = OUT/'report.txt'
    text = p.read_text(encoding='utf-8')
    inaccurate = 'Each post-first request uses a fresh isolated gpt-6.1-terra medium subagent.'
    accurate = f'LLM transport was changed on explicit user instruction after the preserved pause: {sub} committed post-first requests use independent gpt-6.1-terra medium subagents and {api} use independent stateless Responses API calls with the same requested model and effort. The exact split is recorded in LLM_backend_by_round.csv and LLM_backend_by_trajectory.csv. This is a resumed mixed-backend study, NOT a pure API replication.'
    assert inaccurate in text
    text = text.replace(inaccurate,accurate)
    text += '\nAPI RESUMPTION DISCLOSURE\nThe original protocol, scientific prompt payloads, GP, candidate pools, tie streams, paid first inputs, all prior responses and controls are unchanged. The original protocol backend field describes the initial execution. api_transport_registration.json is the append-only continuation addendum. API calls request model=gpt-6.1-terra, reasoning.effort=medium, store=false, JSON-object output, no tools and no previous-response/session state. Output token cap8192 is a transport bound, not an experimental-query change. Returned model/response identifiers and token usage are retained; this does not prove backend equivalence to the prior subagents. Four seeds remain conditional on fixed initial knowledge and the shared first point.\n'
    p.write_text(text,encoding='utf-8')
    print(json.dumps(dict(committed_subagent_requests=sub,committed_API_requests=api,
                         backend_provenance_by_round=True)))


if __name__ == '__main__':
    main()
