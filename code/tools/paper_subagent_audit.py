"""Offline numerical replay and local provenance for the separate Codex cohort.

This auditor never requests model output, fits a GP, or reads an oracle table.
Saved collaboration receipts establish local custody, not server-side model,
system-prompt, token-accounting, or HTTP-response attestation.
"""
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools')]
import paper_audit as native
from audit_knowledge_rank_v9 import require
from audit_v12_llm_repeats import validate_shared_prefixes

BACKEND = 'codex_subagent'
CONTEXT_POLICY = 'fresh_agent_per_request_attempt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def completed_input_snapshot(root, run):
    """Bind a completed replay to its numerical and response bytes.

    Mutable scheduler/status files are excluded. A completed run's native
    observations, commitments, arrays and per-request receipts are immutable.
    """
    root, run = Path(root), Path(run)
    base = run.parent.parent
    paths = {run / 'state.json', run / 'initial_commit.json', base / 'launch.json'}
    paths.update(path for path in (run / 'rounds').iterdir() if path.suffix in ('.json', '.npz'))
    for folder in (base / 'requests', base / 'queue'):
        paths.update(path for path in folder.rglob('*') if path.is_file())
    for receipt in (base / 'queue').rglob('export.json'):
        exported = read(receipt)
        paths.update(root / exported[key] for key in ('packet_relpath', 'output_relpath'))
    return {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(paths)}


def verify_input_snapshot(root, snapshot):
    for name, expected in snapshot.items():
        path = Path(root) / name
        require(path.is_file() and file_sha(path) == expected, 'completed audited input changed: ' + name)


def validate_assignment(assignment, protocol):
    """Check declared invocation settings against the saved collaboration receipt."""
    require(assignment['backend'] == BACKEND, 'separate Codex backend identity')
    require(assignment['requested_model'] == protocol['model'], 'assigned requested model')
    require(assignment['requested_reasoning_effort'] == protocol['reasoning_effort'], 'assigned effort')
    require(assignment['fork_turns'] == 'none', 'fresh no-history request agent')
    require(assignment['context_policy'] == CONTEXT_POLICY, 'one agent per request attempt')
    require(isinstance(assignment['agent_id'], str) and assignment['agent_id'], 'actual agent identity')
    invocation = assignment['spawn_request']
    receipt = assignment['spawn_result']
    require(invocation['model'] == protocol['model'], 'saved spawn model override')
    require(invocation['reasoning_effort'] == protocol['reasoning_effort'], 'saved spawn effort override')
    require(invocation['fork_turns'] == 'none', 'saved spawn has no inherited conversation')
    require(receipt.get('task_name',receipt.get('agent_id')) == assignment['agent_id'], 'saved spawn result agent identity')
    require(isinstance(invocation['task_name'], str) and invocation['task_name'], 'saved spawn task identity')
    return assignment['agent_id']


def validate_public_export(root, queue_attempt, packet, assignment, protocol):
    """Verify only the frozen public prompt and registered transport text were sent."""
    root=Path(root).resolve();receipt=read(queue_attempt/'export.json')
    require(receipt['exposure_policy']=='opaque_exact_system_public_packet','opaque public exposure')
    require(receipt['request_sha256']==packet['request_sha256'],'export exact queue identity')
    expected=dict(system=packet['system'],public=packet['public'])
    path=(root/receipt['packet_relpath']).resolve();output=(root/receipt['output_relpath']).resolve()
    opaque=native.study.digest(dict(purpose='isolated_public_packet',request_sha256=packet['request_sha256']))
    folder=root/'experiments/agent_io/paper_v12_subagent'/opaque
    require(path==folder/'input.json' and output==folder/'output.json','opaque paths contain no task/seed/arm')
    require(read(path)==expected and receipt['packet_payload_sha256']==native.study.digest(expected),'only exact system/public exported')
    require(file_sha(path)==receipt['packet_sha256'],'sealed public bytes')
    template=root/protocol['decision_instructions_path']
    require(file_sha(template)==protocol['decision_instructions_sha256'],'registered isolation instructions')
    message=template.read_text(encoding='utf-8').format(input_path=path.as_posix(),output_path=output.as_posix())
    require(assignment['spawn_request']['message']==message,'no unregistered content in child invocation')
    if (queue_attempt/'response.raw.json').exists():
        require(file_sha(output)==file_sha(queue_attempt/'response.raw.json'),'exact authored opaque output submitted')


def validate_unique_records(responses, historical_ids=()):
    """Local response and agent identities must each be fresh for every attempt."""
    ids = [r['response_id'] for r in responses]
    agents = [r['agent_id'] for r in responses]
    require(all(isinstance(x, str) and x.startswith('subagent:') for x in ids), 'local transport response namespace')
    require(len(ids) == len(set(ids)), 'no reused local response identity')
    require(not set(ids).intersection(historical_ids), 'no historical response identity reuse')
    require(len(agents) == len(set(agents)), 'fresh independent agent for each attempt')
    return len(ids)


def validate_continuation_record(before, after, followup, assignment, packet, assignment_sha, output_sha):
    """A continuation completes the same unanswered request, not a new sample."""
    require(before['original_agent'] == after['agent_id'] == followup['target'] == assignment['agent_id'],
            'continuation uses the same sole-request agent')
    require(before['request_sha256'] == packet['request_sha256'], 'continuation keeps the original request')
    require(before['assignment_sha256'] == assignment_sha and after['assignment_unchanged'] is True,
            'continuation does not replace assignment or model')
    require(followup['message'] == assignment['spawn_request']['message'],
            'continuation repeats only the identical registered transport instruction')
    require(before['observed_tool_status'] == 'interrupted', 'continuation follows an interrupted agent')
    require(all(before[k] is False for k in ('output_exists', 'response_raw_exists', 'provenance_exists')),
            'no authored response existed before continuation')
    require(after['scientific_response_submitted_yet'] is False and after['output_exists'] is True,
            'continuation output recorded before first submission')
    require(after['output_sha256'] == output_sha, 'continued first output is unchanged')


def audit_continuations(root, study):
    """Retain transport interruptions explicitly, including false-alarm reports."""
    folder=Path(root)/'experiments/reports/paper_v12_subagent/transport_continuations'
    records=[]
    for path in sorted(folder.glob('*/before.json')):
        before=read(path);directory=path.parent;after=read(directory/'after.json')
        followup=read(directory/'followup_request.json');result=read(directory/'followup_result.json')
        request=(Path(root)/before['request_path']).resolve()
        require(request.is_relative_to(study.directory(root).resolve()), 'continuation remains in registered cohort')
        assignment=read(request.parent/'assignment.json');packet=read(request)
        export=read(request.parent/'export.json');output=Path(root)/export['output_relpath']
        validate_continuation_record(before,after,followup,assignment,packet,
                                     file_sha(request.parent/'assignment.json'),file_sha(output))
        require(result['tool']=='collaboration.followup_task' and result['result_content']=='',
                'actual empty followup tool result recorded without invented payload')
        resolution=Path(root)/'experiments/reports/paper_v12_subagent/incident_r000176_resolution.json'
        require(file_sha(resolution)==before['resolution_sha256'], 'retained transport-resolution evidence')
        records.append(dict(agent_id=assignment['agent_id'],request_path=before['request_path'],
                            request_sha256=packet['request_sha256'],output_sha256=file_sha(output),
                            continuation_instruction='identical_original_transport_message',
                            response_existed_before=False,new_scientific_request=False,
                            receipt_sha256={p.relative_to(root).as_posix():file_sha(p) for p in sorted(directory.glob('*.json'))},
                            resolution_sha256=file_sha(resolution)))
    if (Path(root)/'experiments/reports/paper_v12_subagent/incident_r000176_resolution.json').exists():
        require(any(r['agent_id'].endswith('/r000176') for r in records), 'known interrupted request continuation is retained')
    return records


def scan_assignments(root, study, protocol):
    """Include pending attempts when checking isolation and the registered budget."""
    namespace = study.directory(root)
    bases = {study.base(root, *job).resolve() for job in study.JOBS}
    require(all(path.parent.resolve() in bases for path in namespace.rglob('launch.json')),
            'no unregistered trajectory launches')
    agents, contexts = [], set()
    attempts = 0
    for job in study.JOBS:
        base = study.base(root, *job)
        for path in (base / 'queue').glob('*/attempt*/request.json'):
            packet = read(path)
            require(packet['job'] == list(job), 'pending queue registered trajectory')
            require(packet['request_sha256'] == native.study.digest({k: v for k, v in packet.items() if k != 'request_sha256'}),
                    'pending queue request hash')
            require(packet['context_sha256'] == path.parent.parent.name and
                    path.parent.name == f"attempt{packet['attempt']}" and
                    type(packet['attempt']) is int and 1 <= packet['attempt'] <= 3, 'pending request context/attempt path')
            contexts.add((tuple(job), packet['context_sha256']))
            attempts += 1
            assigned = path.parent / 'assignment.json'
            if assigned.exists():
                assignment = read(assigned)
                require(assignment['request_sha256'] == packet['request_sha256'], 'pending assignment exact request')
                agents.append(validate_assignment(assignment, protocol))
                validate_public_export(root,path.parent,packet,assignment,protocol)
    require(len(agents) == len(set(agents)), 'no request agent reused, including pending attempts')
    budget = sum(16 + (0 if arm == 'reasoning_bo' else 1 if arm == 'no_refresh' else 4)
                 for _, _, arm in study.JOBS)
    require(len(contexts) <= budget and attempts <= 3 * len(contexts), 'bounded registered request budget')
    return dict(queue_contexts=len(contexts), queue_attempts=attempts, assigned_request_agents=len(agents),
                maximum_logical_requests=budget)


def exact_attempt_request(context, number, prior_error=None):
    require(type(number) is int and 1 <= number <= 3, 'bounded schema-repair attempt')
    if number == 1:
        require(prior_error is None, 'first request has no schema repair')
        payload = context['public']
    else:
        require(isinstance(prior_error, str) and prior_error, 'repair follows recorded schema error')
        payload = dict(original=context['public'], schema_repair_only=prior_error)
    return dict(system=context['system'], public=payload)


def audit_attempt(root, native_attempt, queue_attempt, context, number, protocol, prior_error=None, job=None):
    """Join frozen request/response files to exact queue bytes and spawn receipts."""
    root, native_attempt, queue_attempt = map(Path, (root, native_attempt, queue_attempt))
    expected = exact_attempt_request(context, number, prior_error)
    require(read(native_attempt / 'request.json') == expected, 'exact frozen first/repair payload')
    packet = read(queue_attempt / 'request.json')
    assignment = read(queue_attempt / 'assignment.json')
    provenance = read(queue_attempt / 'provenance.json')
    response = read(native_attempt / 'response.json')
    agent_id = validate_assignment(assignment, protocol)
    if protocol.get('public_exposure_policy')=='opaque_exact_system_public_packet':
        validate_public_export(root,queue_attempt,packet,assignment,protocol)
    require(packet['system'] == expected['system'] and packet['public'] == expected['public'], 'exact queued payload')
    require(packet['schema_version'] == 1 and packet['backend'] == BACKEND, 'queue schema/backend')
    require(packet['attempt'] == number, 'queue attempt identity')
    require(packet['json_output'] is True and packet['max_output_tokens'] == 6000, 'frozen generation request settings')
    require(packet['fork_turns'] == 'none' and packet['context_policy'] == CONTEXT_POLICY, 'queue context isolation')
    if job is not None:
        require(packet['job'] == list(job), 'queue registered trajectory identity')
    request_hash = native.study.digest({k: v for k, v in packet.items() if k != 'request_sha256'})
    require(packet['request_sha256'] == request_hash, 'queued request content hash')
    require(provenance['request_sha256'] == request_hash, 'response tied to exact request')
    require(assignment['request_sha256'] == request_hash, 'assignment tied to exact request')
    require(packet['context_sha256'] == native.study.digest(context), 'queued frozen context hash')
    require(packet['requested_model'] == protocol['model'], 'queue requested model')
    require(packet['requested_reasoning_effort'] == protocol['reasoning_effort'], 'queue requested effort')
    require(provenance['backend'] == BACKEND, 'response backend provenance')
    require(provenance['agent_id'] == agent_id, 'response assigned agent')
    for key in ('requested_model', 'requested_reasoning_effort', 'fork_turns', 'context_policy'):
        require(provenance[key] == assignment[key], 'response and assignment ' + key)
    raw_path = queue_attempt / 'response.raw.json'
    raw_text = raw_path.read_bytes().decode('utf-8')
    raw_hash = file_sha(raw_path)
    assignment_hash = file_sha(queue_attempt / 'assignment.json')
    require(provenance['raw_output_sha256'] == raw_hash, 'literal agent output byte hash')
    require(provenance['assignment_sha256'] == assignment_hash, 'sealed actual-spawn assignment hash')
    require(provenance['context_sha256'] == packet['context_sha256'], 'response frozen context')
    require(provenance['response_id_kind'] == 'local_content_receipt', 'local identity is not an HTTP ID')
    identity = 'subagent:' + native.study.digest(dict(request_sha256=request_hash,
                raw_output_sha256=raw_hash, assignment_sha256=assignment_hash))
    require(provenance['response_id'] == identity, 'content-derived local response identity')
    require(provenance['usage_available'] is False, 'token accounting unavailable')
    require(provenance['model_verification'] == assignment['model_verification'] ==
            'local_orchestrator_spawn_receipt_not_provider_attestation', 'model evidence boundary')
    require(response['output_text'] == raw_text, 'native response preserves literal agent text')
    require(response['id'] == provenance['response_id'], 'native response local identity')
    require(response['backend'] == BACKEND and response['usage_available'] is False, 'native transport labels')
    require(response['transport_provenance'] == provenance, 'native response provenance unchanged')
    require(response.get('usage', {}) == {}, 'unavailable token usage is not fabricated')
    require(response.get('resolved_model') is None, 'no unverified resolved-model claim')
    require(response.get('requested_model', protocol['model']) == protocol['model'], 'response requested model')
    return dict(path=(native_attempt / 'response.json').relative_to(root).as_posix(),
                queue_path=queue_attempt.relative_to(root).as_posix(),
                response_id=response['id'], agent_id=agent_id, attempt=number,
                context_sha256=packet['context_sha256'], request_sha256=request_hash,
                raw_output_sha256=raw_hash, assignment_sha256=assignment_hash,
                provenance_sha256=file_sha(queue_attempt / 'provenance.json'),
                requested_model=protocol['model'], requested_reasoning_effort=protocol['reasoning_effort'])


def schema_validator(root, pool, commit, kind, context, arm):
    """Use frozen schema validation to reject unjustified schema-repair rerolls."""
    if kind == 'memory':
        return lambda raw: native.study.v12.llm.validate_memory(
            raw, pool, commit['prestate'], commit['display_ids'], commit['query'])
    if arm == 'reasoning_bo':
        return lambda raw: native.study.v9.llm.validate_rbo(root, raw, commit['display_ids'])
    return lambda raw: native.study.v12.llm.validate_reasoning(
        root, raw, pool, commit['hypotheses_at_commit'], commit['display_ids'],
        context['public']['retrieved_notes'])


def audit_transport(root, run, protocol, arm, pool):
    """Check every committed context and every schema-only attempt, without reruns."""
    run = Path(run)
    base = run.parent.parent
    contexts, prefixes, _ = native.request_and_prefix_evidence(Path(root), run, arm)
    launch = read(base / 'launch.json')
    job = (launch['task'], launch['initialization_seed'], launch['arm'])
    responses = []
    expected_keys = {row['context_sha256'] for row in contexts}
    require({p.name for p in (base / 'queue').iterdir() if p.is_dir()} == expected_keys,
            'queue contains exactly committed contexts')
    for row in contexts:
        key = row['context_sha256']
        folder = base / 'requests' / key
        context = read(folder / 'context.json')
        valid = read(folder / 'validated.json')
        commit = read(run / 'rounds' / f"{row['query']:02d}_commit.json")
        validator = schema_validator(root, pool, commit, row['kind'], context, arm)
        require(native.study.digest(context) == key, 'exact context digest')
        count = valid['attempts']
        require(type(count) is int and 1 <= count <= 3, 'bounded schema-only repair count')
        expected_attempts = {f'attempt{i}' for i in range(1, count + 1)}
        for parent in (folder, base / 'queue' / key):
            require({p.name for p in parent.glob('attempt*') if p.is_dir()} == expected_attempts,
                    'no discarded or extra attempts after validation')
        prior_error = None
        for number in range(1, count + 1):
            attempt = folder / f'attempt{number}'
            record = audit_attempt(root, attempt, base / 'queue' / key / f'attempt{number}',
                                   context, number, protocol, prior_error, job)
            responses.append(record)
            response = read(attempt / 'response.json')
            validation = read(base / 'queue' / key / f'attempt{number}' / 'validation.json')
            require(validation['request_sha256'] == record['request_sha256'] and
                    validation['raw_output_sha256'] == record['raw_output_sha256'], 'queue validation content identity')
            if number < count:
                prior_error = read(attempt / 'schema_error.json')['error']
                require(validation['status'] == 'SCHEMA_REJECTED' and validation['error'] == prior_error,
                        'queue rejection matches native schema error')
                try:
                    validator(json.loads(response['output_text']))
                except (ValueError, TypeError, KeyError) as exc:
                    require(str(exc) == prior_error, 'schema-only retry follows the actual frozen validator error')
                else:
                    raise AssertionError('A valid response was discarded for an unjustified retry')
            else:
                require(not (attempt / 'schema_error.json').exists(), 'validated attempt has no schema error')
                require(response['id'] == valid['response_id'], 'accepted response is last bounded attempt')
                require(json.loads(response['output_text']) == valid['raw'], 'accepted raw output unchanged')
                require(validator(valid['raw']) == valid['plan'], 'accepted frozen schema result unchanged')
                require(validation['status'] == 'VALIDATED' and validation['plan_sha256'] == native.study.digest(valid['plan']),
                        'queue accepted schema-plan custody')
                require(valid['backend'] == BACKEND and valid['usage_available'] is False and valid['usage'] == {},
                        'validated response backend and unavailable token accounting')
                require(valid['transport_provenance'] == response['transport_provenance'], 'validated provenance unchanged')
    validate_unique_records(responses)
    return contexts, prefixes, responses


def audit_completed(root, study, protocol, task, seed, arm):
    """Replay native numerical transitions with a public input pool only."""
    root = Path(root)
    run = study.run_path(root, task, seed, arm)
    input_snapshot = completed_input_snapshot(root, run)
    pool = study.pool_for(root, task, seed, protocol)
    control = native.control_run(root, task, seed)
    launch = read(run.parent.parent / 'launch.json')
    require((launch['task'], launch['initialization_seed'], launch['arm'], launch['native_arm']) ==
            (task, seed, arm, native.study.native_arm(arm)), 'registered launch identity')
    require(launch['protocol_sha256'] == protocol['protocol_sha256'], 'new cohort launch protocol')
    require(launch['started_utc'] >= protocol['created_utc'], 'new protocol predates launch')
    require(launch['LLM_cache_initially_empty'] and launch['response_files_copied'] == 0, 'fresh response namespace')
    row = native.audit_llm(root, run, protocol, pool, control, arm)
    state = read(run / 'state.json')
    pf = native.study.directory(root) / 'preflight' / f's{seed}' / native.study.SHORT[task]
    require(state['history'][:4] == read(pf / 'receipt.json')['initial_history'], 'registered initial observations')
    with np.load(pf / '01_GP.npz', allow_pickle=False) as a, np.load(run / 'rounds/01_GP.npz', allow_pickle=False) as b:
        require(set(a.files).issubset(b.files), 'registered first GP arrays present')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    contexts, prefixes, responses = audit_transport(root, run, protocol, arm, pool)
    require(completed_input_snapshot(root, run) == input_snapshot, 'completed inputs stable during numerical and transport replay')
    return dict(row=dict(row, initialization_seed=seed), contexts=contexts, prefixes=prefixes,
                responses=responses, control=control, pool=pool, input_snapshot=input_snapshot)


def main():
    auditor_start_sha = file_sha(__file__)
    from paper_v12_subagent import study
    protocol = study.check(ROOT, verify_controls=True)
    require(protocol['backend'] == BACKEND and protocol['context_policy'] == CONTEXT_POLICY, 'registered request transport')
    require(protocol['fork_turns'] == 'none', 'registered no-history agent requests')
    rows, pending, failures, contexts, prefixes, responses = [], [], [], [], [], []
    controls = {}
    completed_inputs = {}
    for task, seed, arm in study.JOBS:
        run = study.run_path(ROOT, task, seed, arm)
        state = read(run / 'state.json') if (run / 'state.json').exists() else None
        if not state or state['status'] != 'COMPLETE':
            pending.append(dict(task=task, initialization_seed=seed, arm=arm,
                                queries=len(state['decisions']) if state else 0))
            continue
        try:
            result = audit_completed(ROOT, study, protocol, task, seed, arm)
            rows.append(result['row'])
            contexts.extend(dict(x, initialization_seed=seed) for x in result['contexts'])
            prefixes.extend(dict(x, initialization_seed=seed, arm=arm) for x in result['prefixes'])
            responses.extend(dict(x, task=task, initialization_seed=seed, arm=arm) for x in result['responses'])
            controls.setdefault(result['control'], result['pool'])
            completed_inputs.update(result['input_snapshot'])
        except Exception as exc:
            failures.append(dict(task=task, initialization_seed=seed, arm=arm,
                                 kind=type(exc).__name__, message=str(exc)))
    unique = shared = None
    assignment_counts = {}; continuations=[]
    try:
        assignment_counts = scan_assignments(ROOT, study, protocol)
        continuations = audit_continuations(ROOT, study)
        historical_ids = set()
        original = native.study.check(ROOT, True)
        for name in original['protected_controls_sha256']:
            if '/requests/' in name and '/attempt' in name and name.endswith('/response.json'):
                value = read(ROOT / name)
                historical_ids.add(value.get('id', value.get('response_id')))
        unique = validate_unique_records(responses, historical_ids)
        for control, pool in controls.items():
            control_state = read(control / 'state.json')
            native.audit_bo(control, pool, dict(protocol_sha256=control_state['protocol_sha256']))
            _, control_prefixes, _ = native.request_and_prefix_evidence(ROOT, control, 'bo')
            prefixes.extend(control_prefixes)
        shared = validate_shared_prefixes(prefixes)
        verify_input_snapshot(ROOT, completed_inputs)
    except Exception as exc:
        failures.append(dict(stage='cross_run_custody', kind=type(exc).__name__, message=str(exc)))
    if file_sha(__file__) != auditor_start_sha:
        failures.append(dict(stage='auditor_source_custody',kind='SourceChangedDuringAudit',message='Auditor source changed while replay was running'))
    status = ('FAIL' if failures else ('PASS' if not pending else
              ('PASS_COMPLETED_TRAJECTORIES_WITH_PENDING_CELLS' if rows else 'NO_COMPLETED_TRAJECTORIES')))
    result = dict(status=status, completed=rows, pending=pending, failures=failures,
                  completed_trajectories=len(rows), registered_trajectories=len(study.JOBS),
                  audited_queries=sum(row['rounds'] for row in rows),
                  unique_local_response_records=unique, unique_request_agents=unique,
                  shared_history_groups_checked=shared, pure_BO_controls_replayed=len(controls),
                  assignment_custody=assignment_counts,
                  transport_continuations=continuations,
                  completed_input_files=len(completed_inputs), completed_input_sha256=completed_inputs,
                  protocol_sha256=protocol['protocol_sha256'], auditor_sha256=auditor_start_sha,
                  new_oracle_queries=0, new_API_calls=0, new_GP_fits=0,
                  backend=BACKEND, context_policy=CONTEXT_POLICY,
                  boundary='Completed trajectories only: native independent numerical and chronology replay; registered removal transitions; exact public request and schema-repair payloads; local raw-byte and collaboration-spawn receipt custody. No HTTP IDs, independently attested resolved model, effective system-prompt equivalence, token costs, server weights/cache, independent GP refits, or external timestamp attestation. Pending cells are not outcomes. Original API cohorts remain separate.')
    output = ROOT / 'experiments/reports/paper_v12_subagent'
    native.study.write(output / 'request_evidence.json', dict(contexts=contexts, prefixes=prefixes, responses=responses))
    native.study.write(output / 'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('completed', 'pending', 'completed_input_sha256')}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
