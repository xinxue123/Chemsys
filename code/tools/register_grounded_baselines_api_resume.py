"""Verify the pause, then append an explicit user-authorized backend addendum."""
from datetime import datetime,timezone
from pathlib import Path
import json
import grounded_baselines_api_dispatch as d

ROOT,BASE,OPS = d.ROOT,d.BASE,d.OPS


def main():
    target = BASE/'api_transport_registration.json'
    if target.exists():
        print('API continuation is already registered; retain it.')
        return
    p = d.read(BASE/'protocol.json')
    pause = ROOT/'experiments/reports/grounded_baselines_20261105/paused_snapshot'
    manifest = d.read(pause/'snapshot_manifest_sha256.json')
    frozen = dict(manifest['files'])
    frozen.update(p['sources_sha256'])
    frozen.update(p['inputs_sha256'])
    frozen.update(p['protected_sha256'])
    for path,h in frozen.items():
        assert d.sha(ROOT/path) == h, path
    for name,field in [('execution_compatibility.json','new_source_sha256'),('delegated_transport_registration.json','source_sha256')]:
        for path,h in d.read(BASE/name)[field].items():assert d.sha(ROOT/path) == h,path
    states = [d.read(path) for path in BASE.glob('runs/*/*/state.json')]
    receipts = {path.relative_to(ROOT).as_posix():d.sha(path) for path in BASE.glob('requests/*/*/attempt*/receipt.json')}
    assert len(receipts) == 552 and sum(len(s['history']) for s in states) == 588
    assert sum(s['status']=='COMPLETE' for s in states) == 28
    pending,sealed,complete = d.inventory()
    assert len(pending) == 8 and sealed == 552 and complete == 28
    assert all(not (af/'assignment.json').exists() and not (ROOT/d.read(af/'export.json')['output']).exists() for af in pending)
    preflight = d.read(OPS/'preflight_network.json')
    assert preflight['status'] == 'PASS' and preflight['response']['resolved_model'] == 'gpt-6.1-terra'
    paths = ['code/tools/'+n for n in ('grounded_baselines_api_dispatch.py','audit_grounded_baselines_api.py',
             'report_grounded_baselines_api.py','register_grounded_baselines_api_resume.py','test_grounded_baselines_api_transport.py')]
    paths += ['code/chemist_system/chem/benchmarks/knowledge_openai_client.py']
    reg = dict(status='REGISTERED_USER_AUTHORIZED_API_CONTINUATION',created_UTC=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=p['protocol_sha256'],
        authorization='User: 使用openai-apikey的 6.1 terra medium继续之前的实验',
        credential_file='openai-key.txt',public_config=preflight['public_config'],
        model='gpt-6.1-terra',effort='medium',API_endpoint='Responses API',
        maximum_concurrent_API_calls=8,
        wire_parameters=dict(model='gpt-6.1-terra',reasoning={'effort':'medium'},store=False,
            max_output_tokens=8192,text={'format':{'type':'json_object'}},tools=[],previous_response_id=None),
        continuity=dict(completed_trajectories=28,partial_trajectories=8,not_started=28,
            committed_paid_queries=588,sealed_subagent_attempts=552,
            remaining_paid_queries=692,remaining_logical_post_first_requests=664,
            pending_unassigned_requests=8),
        scientific_prompts_GP_candidates_initial_points_budgets_and_controls_unchanged=True,
        completed_trajectories_not_rerun=True,old_responses_not_replaced=True,
        original_invocation_files='Retained as the original planned subagent invocation; actual API transport is identified by assignment/completion/receipt transport_type fields.',
        identity_semantics='Legacy agent_id key is an actual unique HTTP call identifier for API receipts, never a claim of a collaboration agent.',
        no_tools_browsing_or_other_trajectory_state=True,no_previous_response_or_conversation_id=True,
        interpretation='Mixed subagent/API continuation, not a pure-API replication or evidence that the two backends are equivalent.',
        pause_manifest_sha256=d.sha(pause/'snapshot_manifest_sha256.json'),
        preflight_sha256=d.sha(OPS/'preflight_network.json'),
        existing_response_receipts_sha256=receipts,
        source_sha256={path:d.sha(ROOT/path) for path in paths})
    d.write_once(target,reg)
    for path in paths:
        dest = BASE/'api_source_snapshot'/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/path).read_bytes())
    d.write_once(OPS/'resume_verification.json',dict(status='PASS_PAUSE_AND_FROZEN_CONTROL_VERIFICATION',
        checked_files=len(frozen),snapshot_files=len(manifest['files']),protected_prior_files=len(p['protected_sha256']),
        old_response_receipts=len(receipts),registration_sha256=d.sha(target)))
    print(json.dumps(dict(status=reg['status'],continuity=reg['continuity'],checked_files=len(frozen))))


if __name__=='__main__':main()
