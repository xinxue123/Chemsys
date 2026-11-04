"""Immutable local queue for one fresh, isolated decision agent per request."""
from pathlib import Path
import json
import time

from paper_v12_subagent.transport import write_once, write_once_bytes, file_sha, read, digest
from . import BACKEND, MODEL, EFFORT, VARIANT

INSTRUCTIONS = '''Answer one experimental-decision request as the language model.
This is a fresh independent request. Read only the assigned UTF-8 JSON input in full:
{input_path}
It contains exactly system and public. Follow system and respond to public using
only the supplied own observations and domain information. Do not inspect other
files, search the repository or internet, recall benchmark outcome tables, use
candidate identifiers as ranks, or seek unmeasured outcomes. Do not delegate.
Write your own requested JSON response without Markdown fences to:
{output_path}
Tools may only read that input and write that output. Do not run experiments,
inspect validation, edit an existing response, or implement a scoring heuristic
or response generator. Include concise public rationales, not private reasoning.
After writing the output, finish with a short completion message.'''


def base(root):
    return Path(root).resolve() / 'experiments/runs' / VARIANT


def verify_request(path):
    path = Path(path)
    value = read(path)
    if (value['request_sha256'] != digest({k: v for k, v in value.items() if k != 'request_sha256'}) or
            value['backend'] != BACKEND or value['model'] != MODEL or value['effort'] != EFFORT or
            value['fork_turns'] != 'none'):
        raise ValueError('Request receipt changed')
    return value


def record_spawn(root, request_path, spawn_result):
    path = Path(request_path).resolve()
    if not path.is_relative_to(base(root)):
        raise ValueError('Request outside V13 namespace')
    verify_request(path)
    invocation = read(path.parent / 'invocation.json')
    # The result is copied verbatim from the collaboration tool, not invented.
    agent_id = spawn_result.get('task_name', spawn_result.get('agent_id'))
    if agent_id != '/root/' + invocation['task_name']:
        raise ValueError('Spawn result does not match the actual prepared invocation')
    assignment = dict(agent_id=agent_id, spawn_request=invocation, spawn_result=spawn_result,
                      model_verification='local_orchestrator_receipt_not_provider_attestation')
    write_once(path.parent / 'assignment.json', assignment)
    return dict(agent_id=agent_id, request=str(path))


def seal(root, request_path):
    path = Path(request_path).resolve()
    if not path.is_relative_to(base(root)):
        raise ValueError('Request outside V13 namespace')
    req = verify_request(path)
    assignment = read(path.parent / 'assignment.json')
    invocation = read(path.parent / 'invocation.json')
    if assignment['spawn_request'] != invocation:
        raise ValueError('Assignment drift')
    export = read(path.parent / 'export.json')
    packet = Path(root) / export['input']
    if file_sha(packet) != export['input_sha256']:
        raise ValueError('Public packet changed')
    data = (Path(root) / export['output']).read_bytes()
    if not 0 < len(data) <= 2_000_000:
        raise ValueError('Invalid response size')
    data.decode('utf-8')
    write_once_bytes(path.parent / 'response.raw.json', data)
    receipt = dict(request_sha256=req['request_sha256'], raw_sha256=file_sha(path.parent / 'response.raw.json'),
                   assignment_sha256=file_sha(path.parent / 'assignment.json'), backend=BACKEND,
                   usage_available=False, agent_id=assignment['agent_id'])
    write_once(path.parent / 'response_receipt.json', receipt)
    return dict(status='SEALED', request=str(path))


def request(root, task_short, kind, query, system, public, validator, *, selected_validator=None):
    context = dict(system=system, public=public, model=MODEL, effort=EFFORT, backend=BACKEND)
    key = digest(context)
    folder = base(root) / 'requests' / task_short / key
    write_once(folder / 'context.json', context)
    error, prior_raw, selected_lock = None, None, None
    for attempt in range(1, 4):
        af = folder / f'attempt{attempt}'
        payload = public if error is None else dict(original=public, schema_repair_only=error,
            original_response=prior_raw, selected_id_must_remain=selected_lock)
        value = dict(system=system, public=payload, backend=BACKEND, model=MODEL, effort=EFFORT,
                     fork_turns='none', task=task_short, kind=kind, query=query, attempt=attempt, context_sha256=key)
        value['request_sha256'] = digest(value)
        write_once(af / 'request.json', value)
        opaque = digest(dict(request_sha256=value['request_sha256'], namespace=VARIANT))
        io = Path(root) / 'experiments/agent_io' / VARIANT / opaque
        write_once(io / 'input.json', dict(system=system, public=payload))
        write_once(af / 'export.json', dict(input=(io / 'input.json').relative_to(root).as_posix(),
                   output=(io / 'output.json').relative_to(root).as_posix(), input_sha256=file_sha(io / 'input.json')))
        invocation = dict(task_name=f'v13_{task_short}_{query:02d}_{kind}_{attempt}', fork_turns='none',
                          model=MODEL, reasoning_effort=EFFORT,
                          message=INSTRUCTIONS.format(input_path=(io / 'input.json').as_posix(),
                                                      output_path=(io / 'output.json').as_posix()))
        write_once(af / 'invocation.json', invocation)
        while not (af / 'response_receipt.json').exists():
            time.sleep(.25)
        receipt = read(af / 'response_receipt.json')
        if (receipt['request_sha256'] != value['request_sha256'] or
                receipt['raw_sha256'] != file_sha(af / 'response.raw.json') or
                receipt['assignment_sha256'] != file_sha(af / 'assignment.json')):
            raise ValueError('Response custody changed')
        try:
            raw = read(af / 'response.raw.json')
            if selected_validator:
                selected = selected_validator(raw)
                if selected_lock is not None and selected != selected_lock:
                    raise ValueError('Schema repair changed a previously valid official selected point')
                selected_lock = selected
            plan = validator(raw)
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)
            prior_raw = raw if 'raw' in locals() else None
            write_once(af / 'validation.json', dict(status='SCHEMA_REJECTED', error=error, selected_lock=selected_lock))
            continue
        result = dict(context_sha256=key, raw=raw, plan=plan, attempts=attempt,
                      response_receipt=receipt, request_relpath=(af / 'request.json').relative_to(root).as_posix())
        write_once(af / 'validation.json', dict(status='VALIDATED', plan_sha256=digest(plan)))
        write_once(folder / 'validated.json', result)
        return result
    raise ValueError('LLM exhausted the three schema-only attempts; no oracle call was made')


def pending(root):
    answer = []
    for path in sorted((base(root) / 'requests').glob('*/*/attempt*/request.json')):
        af = path.parent
        if (af / 'response_receipt.json').exists() or not (af / 'invocation.json').exists():
            continue
        export = read(af / 'export.json')
        answer.append(dict(request=str(path), invocation=read(af / 'invocation.json'),
                           assigned=(af / 'assignment.json').exists(), output_exists=(Path(root) / export['output']).exists()))
    return answer
