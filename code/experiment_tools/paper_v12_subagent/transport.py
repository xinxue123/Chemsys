"""File queue around the unchanged native three-attempt request function.

This module never reads credentials or contacts a network. ``id`` and ``usage``
exist only to satisfy the frozen caller: the ID is a labelled local receipt and
the empty usage object means unavailable, never zero tokens.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import json
import os
import time
import uuid

from filelock import FileLock

from . import BACKEND, CONTEXT_POLICY, MODEL, REASONING_EFFORT


class ReceiptDrift(RuntimeError):
    """Evidence changed; never route this through a schema-repair retry."""


class PendingRequest(RuntimeError):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__('Subagent response pending: ' + str(self.path))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once_bytes(path, data):
    """Atomic publication, exact immutable resume, and no overwrite fallback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.write.lock', timeout=30):
        if path.exists():
            if path.read_bytes() != data:
                raise ReceiptDrift('Immutable evidence differs: ' + str(path))
            return
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


def write_once(path, value):
    write_once_bytes(path, canonical(value) + b'\n')


def request_record(path):
    path = Path(path).resolve()
    if path.name != 'request.json' or path.parent.name not in ('attempt1', 'attempt2', 'attempt3'):
        raise ValueError('Expected an attempt request.json')
    request = read(path)
    if (request.get('backend') != BACKEND or request.get('context_policy') != CONTEXT_POLICY
            or request.get('requested_model') != MODEL
            or request.get('requested_reasoning_effort') != REASONING_EFFORT
            or request.get('fork_turns') != 'none'
            or request.get('request_sha256') != digest({k: v for k, v in request.items() if k != 'request_sha256'})
            or request.get('context_sha256') != path.parent.parent.name
            or request.get('attempt') != int(path.parent.name[-1])):
        raise ReceiptDrift('Queue request identity or settings changed')
    return request


def assign(request_path, agent_id, evidence):
    """Persist the actual collaboration spawn invocation and result, supplied by caller."""
    request_path = Path(request_path).resolve()
    request = request_record(request_path)
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError('Actual spawned agent ID is required')
    if not isinstance(evidence, dict) or set(evidence) != {'spawn_request', 'spawn_result'}:
        raise ValueError('Supply saved spawn_request and spawn_result evidence')
    invocation, result = evidence['spawn_request'], evidence['spawn_result']
    if (not isinstance(invocation, dict) or not isinstance(result, dict)
            or invocation.get('model') != MODEL or invocation.get('reasoning_effort') != REASONING_EFFORT
            or invocation.get('fork_turns') != 'none' or not invocation.get('task_name')
            or result.get('agent_id', result.get('task_name')) != agent_id):
        raise ValueError('Spawn evidence does not match the registered fresh-agent request')
    assignment = dict(backend=BACKEND, agent_id=agent_id,
                      requested_model=MODEL, requested_reasoning_effort=REASONING_EFFORT,
                      fork_turns='none', context_policy=CONTEXT_POLICY,
                      request_sha256=request['request_sha256'], **evidence,
                      model_verification='local_orchestrator_spawn_receipt_not_provider_attestation')
    namespace = next((p for p in request_path.parents if p.name == 'paper_v12_subagent'),
                     request_path.parents[3])
    with FileLock(str(namespace / '.assignment.lock'), timeout=30):
        for path in namespace.rglob('assignment.json'):
            if path != request_path.parent / 'assignment.json' and read(path).get('agent_id') == agent_id:
                raise ValueError('Every request attempt requires a fresh agent ID')
        write_once(request_path.parent / 'assignment.json', assignment)
    return assignment


def _assignment(path, request):
    value = read(path)
    # Recheck the actual evidence, without publishing or mutating it.
    invocation, result = value.get('spawn_request', {}), value.get('spawn_result', {})
    if (value.get('backend') != BACKEND or value.get('request_sha256') != request['request_sha256']
            or value.get('requested_model') != MODEL or value.get('requested_reasoning_effort') != REASONING_EFFORT
            or value.get('fork_turns') != 'none' or value.get('context_policy') != CONTEXT_POLICY
            or invocation.get('model') != MODEL or invocation.get('reasoning_effort') != REASONING_EFFORT
            or invocation.get('fork_turns') != 'none' or not invocation.get('task_name')
            or result.get('agent_id', result.get('task_name')) != value.get('agent_id') or not value.get('agent_id')):
        raise ReceiptDrift('Assignment changed or lacks actual spawn evidence')
    return value


def response_identity(request_sha256, raw_output_sha256, assignment_sha256):
    return 'subagent:' + digest(dict(request_sha256=request_sha256,
                                    raw_output_sha256=raw_output_sha256,
                                    assignment_sha256=assignment_sha256))


def export_request(root, request_path):
    """Publish only exact system/public under an opaque path for the decision agent."""
    root = Path(root).resolve()
    request_path = Path(request_path).resolve()
    namespace = root / 'experiments/runs/paper_v12_subagent'
    if not request_path.is_relative_to(namespace.resolve()):
        raise ValueError('Export request must belong to the subagent namespace')
    request = request_record(request_path)
    opaque = digest(dict(purpose='isolated_public_packet', request_sha256=request['request_sha256']))
    folder = root / 'experiments/agent_io/paper_v12_subagent' / opaque
    if not folder.resolve().is_relative_to(root / 'experiments/agent_io/paper_v12_subagent'):
        raise ValueError('Opaque export escapes its designated namespace')
    packet = dict(system=request['system'], public=request['public'])
    path = folder / 'input.json'
    write_once(path, packet)
    receipt = dict(backend=BACKEND, request_sha256=request['request_sha256'],
                   exposure_policy='opaque_exact_system_public_packet',
                   packet_relpath=path.relative_to(root).as_posix(), packet_sha256=file_sha(path),
                   packet_payload_sha256=digest(packet),
                   output_relpath=(folder / 'output.json').relative_to(root).as_posix())
    write_once(request_path.parent / 'export.json', receipt)
    return receipt


def submit(request_path, raw_path, agent_id):
    """Seal exact authored output bytes; malformed JSON remains a native schema attempt."""
    request_path = Path(request_path).resolve()
    request = request_record(request_path)
    folder = request_path.parent
    assignment = _assignment(folder / 'assignment.json', request)
    if assignment['agent_id'] != agent_id:
        raise ValueError('Only the assigned fresh agent can submit this attempt')
    data = Path(raw_path).read_bytes()
    if not data or len(data) > 2_000_000:
        raise ValueError('Public output must be nonempty and at most 2 MB')
    data.decode('utf-8')
    raw_sha = sha256(data).hexdigest()
    assignment_sha = file_sha(folder / 'assignment.json')
    identity = response_identity(request['request_sha256'], raw_sha, assignment_sha)
    with FileLock(str(folder / '.submission.lock'), timeout=30):
        write_once_bytes(folder / 'response.raw.json', data)
        path = folder / 'provenance.json'
        if path.exists():
            value = read(path)
            if (value['response_id'] != identity or value['agent_id'] != agent_id
                    or value['raw_output_sha256'] != raw_sha):
                raise ReceiptDrift('Submitted response already sealed with different evidence')
            return value
        value = dict(backend=BACKEND, response_id=identity, response_id_kind='local_content_receipt',
                     context_sha256=request['context_sha256'], request_sha256=request['request_sha256'],
                     raw_output_sha256=raw_sha, assignment_sha256=assignment_sha,
                     agent_id=agent_id, requested_model=MODEL, requested_reasoning_effort=REASONING_EFFORT,
                     fork_turns='none', context_policy=CONTEXT_POLICY, usage_available=False,
                     model_verification='local_orchestrator_spawn_receipt_not_provider_attestation',
                     submitted_utc=datetime.now(timezone.utc).isoformat())
        write_once(path, value)
    return value


def read_response(request_path):
    """Verify bytes and assignment before returning the minimal native client shape."""
    request_path = Path(request_path).resolve()
    request = request_record(request_path)
    folder = request_path.parent
    with FileLock(str(folder / '.submission.lock'), timeout=30):
        if not (folder / 'provenance.json').exists():
            raise PendingRequest(request_path)
        assignment = _assignment(folder / 'assignment.json', request)
        provenance = read(folder / 'provenance.json')
        output = (folder / 'response.raw.json').read_bytes()
        raw_sha = sha256(output).hexdigest()
        assignment_sha = file_sha(folder / 'assignment.json')
        identity = response_identity(request['request_sha256'], raw_sha, assignment_sha)
        expected = dict(backend=BACKEND, response_id=identity, response_id_kind='local_content_receipt',
                        context_sha256=request['context_sha256'], request_sha256=request['request_sha256'],
                        raw_output_sha256=raw_sha, assignment_sha256=assignment_sha,
                        agent_id=assignment['agent_id'], requested_model=MODEL,
                        requested_reasoning_effort=REASONING_EFFORT, fork_turns='none',
                        context_policy=CONTEXT_POLICY, usage_available=False,
                        model_verification='local_orchestrator_spawn_receipt_not_provider_attestation')
        if any(provenance.get(k) != v for k, v in expected.items()):
            raise ReceiptDrift('Response provenance does not match immutable request/output/assignment')
        return dict(output_text=output.decode('utf-8'), id=identity, usage={}, backend=BACKEND,
                    usage_available=False, requested_model=MODEL,
                    requested_reasoning_effort=REASONING_EFFORT, transport_provenance=provenance)


class FileQueueClient:
    """A persistent runner waits; each complete call requires a new decision agent."""
    def __init__(self, base, job, *, wait_seconds=-1, poll_seconds=0.25):
        self.base = Path(base).resolve()
        self.job = list(job)
        self.wait_seconds = float(wait_seconds)
        self.poll_seconds = float(poll_seconds)
        if self.wait_seconds < 0 and self.wait_seconds != -1:
            raise ValueError('Use -1 for persistent waiting or a nonnegative duration')
        if not 0 < self.poll_seconds <= 5:
            raise ValueError('Polling duration must be in (0,5] seconds')
        self.active = None

    def close(self):
        pass

    def bind(self, context, attempt, payload):
        key = digest(context)
        value = dict(schema_version=1, backend=BACKEND, job=self.job,
                     context_sha256=key, attempt=attempt, system=context['system'], public=payload,
                     requested_model=MODEL, requested_reasoning_effort=REASONING_EFFORT,
                     json_output=True, max_output_tokens=6000, fork_turns='none', context_policy=CONTEXT_POLICY)
        value['request_sha256'] = digest(value)
        path = self.base / 'queue' / key / f'attempt{attempt}' / 'request.json'
        write_once(path, value)
        self.active = path
        return path

    def complete(self, system, public, *, json_output=False, max_output_tokens=6000):
        if self.active is None:
            raise RuntimeError('FileQueueClient must be called through its native request adapter')
        request = request_record(self.active)
        if (system != request['system'] or public != request['public']
                or json_output is not True or max_output_tokens != 6000):
            raise ReceiptDrift('Generation request differs from frozen caller')
        started = time.monotonic()
        print(json.dumps(dict(status='WAITING_FOR_SUBAGENT', backend=BACKEND,
                              request=str(self.active), request_sha256=request['request_sha256'])), flush=True)
        while True:
            try:
                return read_response(self.active)
            except PendingRequest:
                if self.wait_seconds >= 0 and time.monotonic() - started >= self.wait_seconds:
                    raise
                time.sleep(self.poll_seconds)


def request(client, system, public, directory, validator):
    """Run the frozen request function, changing only persistence and transport.

    Its context hash, validators, three-attempt bound, and schema repair payloads
    are unchanged. The native write hook becomes immutable and adds provenance.
    """
    from knowledge_rank_v9 import llm as native
    if not isinstance(client, FileQueueClient):
        raise TypeError('Subagent requests require the credential-free file client')
    directory = Path(directory).resolve()
    if directory != client.base / 'requests':
        raise ValueError('Request path escaped its trajectory namespace')
    context = dict(system=system, public=public, model=MODEL, reasoning_effort=REASONING_EFFORT)
    key = digest(context)
    folder = directory / key
    target = folder / 'validated.json'

    def bind_attempt(attempt, payload):
        client.bind(context, attempt, payload)

    def immutable_native_write(path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Native request wrote outside its exact context')
        if path.name == 'request.json' and path.parent.name.startswith('attempt'):
            bind_attempt(int(path.parent.name[7:]), value['public'])
        if path == target:
            response = read_response(client.active)
            if value['response_id'] != response['id'] or value['usage'] != {}:
                raise ReceiptDrift('Native compatibility receipt changed')
            value = dict(value, backend=BACKEND, usage_available=False,
                         transport_provenance=response['transport_provenance'])
        if path.name == 'schema_error.json':
            response = read_response(client.active)
            write_once(client.active.parent / 'validation.json',
                       dict(status='SCHEMA_REJECTED', request_sha256=request_record(client.active)['request_sha256'],
                            raw_output_sha256=response['transport_provenance']['raw_output_sha256'], error=value['error']))
        write_once(path, value)

    def checked_validator(raw):
        response = read_response(client.active)
        native_response = folder / client.active.parent.name / 'response.json'
        if native_response.exists() and read(native_response) != response:
            raise ReceiptDrift('Native cached response differs from sealed transport receipt')
        if json.loads(response['output_text']) != raw:
            raise ReceiptDrift('Native cached response differs from sealed subagent output')
        plan = validator(raw)
        write_once(client.active.parent / 'validation.json',
                   dict(status='VALIDATED', request_sha256=request_record(client.active)['request_sha256'],
                        raw_output_sha256=response['transport_provenance']['raw_output_sha256'], plan_sha256=digest(plan)))
        return plan

    # Cache-hit paths do not invoke the native write hook; bind their exact attempt first.
    if target.exists():
        saved = read(target)
        attempt = saved['attempts']
        native_request = read(folder / f'attempt{attempt}' / 'request.json')
        bind_attempt(attempt, native_request['public'])
        expected_response = read_response(client.active)
        if (saved.get('backend') != BACKEND or saved.get('usage_available') is not False
                or saved.get('response_id') != expected_response['id']
                or saved.get('usage') != {}
                or saved.get('transport_provenance') != expected_response['transport_provenance']):
            raise ReceiptDrift('Cached validation lacks matching subagent provenance')
    with patch.object(native, 'write', side_effect=immutable_native_write):
        _NATIVE_REQUEST(client, system, public, directory, checked_validator)
    return read(target)


# Captured before study patches exported request names; the scientific wrapper stays native.
from knowledge_rank_v9.llm import request as _NATIVE_REQUEST
