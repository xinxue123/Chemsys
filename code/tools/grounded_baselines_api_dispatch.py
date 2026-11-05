"""Resume registered LLM requests via stateless, secret-safe Responses calls.

Transport only: never imports an optimizer, an oracle, or scientific validators.
Original request/invocation files and already sealed responses are immutable.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/runs/grounded_baselines_20261105'
OPS = ROOT/'experiments/operations/grounded_baselines_20261105/api_resume_20261105'
REG = BASE/'api_transport_registration.json'
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks.knowledge_openai_client import APIClient, load_config


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.write.lock', timeout=60):
        if path.exists():
            assert read(path) == value, 'Immutable API evidence differs: '+str(path)
            return
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   indent=2)+'\n', encoding='utf-8')
        temp.replace(path)


def status(value):
    OPS.mkdir(parents=True, exist_ok=True)
    p = OPS/'status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(updated_UTC=utc(), **value), indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def serve(af, config, registration, client_factory=APIClient):
    """One independent HTTP request, followed by a complete custody chain."""
    with FileLock(str(af/'.api_worker.lock'), timeout=1):
        if (af/'receipt.json').exists():
            return dict(status='ALREADY_SEALED')
        req = read(af/'request.json')
        inv = read(af/'invocation.json')
        export = read(af/'export.json')
        ip, op = ROOT/export['input'], ROOT/export['output']
        assert sha(ip) == export['input_sha256'], 'Input changed before HTTP call'
        inp = read(ip)
        assert inp == dict(system=req['system'], public=req['public'])
        assert req['model'] == inv['model'] == registration['model'] == 'gpt-6.1-terra'
        assert req['effort'] == inv['reasoning_effort'] == registration['effort'] == 'medium'
        ap = af/'assignment.json'
        if not ap.exists():
            assert not op.exists(), 'Unassigned output must not be overwritten'
            call_id = 'openai-responses/'+uuid.uuid4().hex
            write_once(ap, dict(agent_id=call_id, transport_type='openai_responses_api',
                identity_kind='actual_stateless_HTTP_call_not_a_collaboration_agent',
                request_sha256=req['request_sha256'], invocation_sha256=sha(af/'invocation.json'),
                input_sha256=sha(ip), registration_sha256=sha(REG),
                assigned_UTC=utc(), provider_attestation='response.model_checked_by_APIClient'))
        assignment = read(ap)
        assert assignment['transport_type'] == 'openai_responses_api'
        assert assignment['request_sha256'] == req['request_sha256']
        assert assignment['registration_sha256'] == sha(REG)
        assert assignment['input_sha256'] == sha(ip)
        result_path = af/'api_result.json'
        if not result_path.exists():
            assert not (af/'api_call_started.json').exists(), 'Interrupted HTTP call needs explicit recovery; no silent resampling'
            assert not op.exists()
            write_once(af/'api_call_started.json', dict(call_id=assignment['agent_id'],
                started_UTC=utc(), request_sha256=req['request_sha256'],
                input_sha256=sha(ip), parameters=registration['wire_parameters']))
            try:
                with client_factory(config, timeout=240.) as client:
                    result = client.complete(inp['system'], inp['public'], json_output=True,
                        max_output_tokens=registration['wire_parameters']['max_output_tokens'])
                # Only public output, model identifiers and allowlisted token counts.
                write_once(result_path, result)
            except Exception as exc:
                write_once(af/'api_transport_error.json', dict(type=type(exc).__name__,
                    message=str(exc).replace(config.api_key, '[REDACTED]'), failed_UTC=utc(),
                    automatic_scientific_resampling=False))
                raise
        result = read(result_path)
        raw = json.loads(result['output_text'])
        assert isinstance(raw, dict) and result['status'] == 'completed'
        assert result['requested_model'] == 'gpt-6.1-terra'
        write_once(op, raw)
        cp = af/'completion.json'
        if not cp.exists():
            write_once(cp, dict(agent_id=assignment['agent_id'],
                transport_type='openai_responses_api',
                source='Actual completed Responses API result, model and status verified',
                response_id=result['id'], resolved_model=result['resolved_model'],
                api_result_sha256=sha(result_path), output_sha256=sha(op),
                acknowledged_UTC=utc()))
        completion = read(cp)
        assert completion['output_sha256'] == sha(op)
        write_once(af/'response.raw.json', raw)
        write_once(af/'receipt.json', dict(agent_id=assignment['agent_id'],
            transport_type='openai_responses_api', request_sha256=req['request_sha256'],
            raw_sha256=sha(af/'response.raw.json'), assignment_sha256=sha(ap),
            model='gpt-6.1-terra', effort='medium', resolved_model=result['resolved_model'],
            response_id=result['id'], usage_available=True, usage=result['usage'],
            api_result_sha256=sha(result_path), registration_sha256=sha(REG),
            completion_sha256=sha(cp), sealed_UTC=utc()))
        return dict(status='SEALED', request=af.relative_to(ROOT).as_posix(),
                    response_id=result['id'], usage=result['usage'])


def inventory():
    pending = []
    sealed = 0
    for p in BASE.glob('requests/*/*/attempt*/invocation.json'):
        af = p.parent
        if (af/'receipt.json').exists():
            sealed += 1
        else:
            pending.append(af)
    return sorted(pending), sealed, len(list(BASE.glob('runs/*/*/completion.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='openai-key.txt')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    registration = read(REG)
    assert 1 <= args.workers <= registration['maximum_concurrent_API_calls']
    assert registration['original_protocol_sha256'] == read(BASE/'protocol.json')['protocol_sha256']
    for path, expected in registration['source_sha256'].items():
        assert sha(ROOT/path) == expected, 'Registered API source changed: '+path
    config = load_config(ROOT/args.config)
    assert config.public() == registration['public_config']
    inflight = {}
    processed = 0
    last_print = None
    with FileLock(str(OPS/'.dispatcher.lock'), timeout=1), ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            pending, sealed, complete = inventory()
            active = set(inflight.values())
            stop = (OPS/'STOP').exists()
            if not stop:
                for af in pending:
                    if af in active:
                        continue
                    if len(inflight) >= args.workers:
                        break
                    if (af/'assignment.json').exists():
                        assert read(af/'assignment.json').get('transport_type') == 'openai_responses_api', 'A subagent still owns this request'
                    inflight[executor.submit(serve, af, config, registration)] = af
            value = dict(status='DRAINING' if stop else 'RUNNING', completed_new=complete,
                planned_new=64, sealed_attempts=sealed, pending_requests=len(pending),
                inflight=len(inflight), API_attempts_sealed_this_session=processed)
            status(value)
            key = (complete, sealed//10, stop)
            if key != last_print:
                print(json.dumps(value), flush=True)
                last_print = key
            if (complete == 64 or stop) and not inflight:
                value['status'] = 'COMPLETE' if complete == 64 else 'PAUSED'
                status(value)
                print(json.dumps(value), flush=True)
                return
            if inflight:
                done, _ = wait(inflight, timeout=1., return_when=FIRST_COMPLETED)
                for future in done:
                    af = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        status(dict(status='TRANSPORT_ERROR_REQUIRES_INSPECTION', request=af.relative_to(ROOT).as_posix(),
                            type=type(exc).__name__, message=str(exc).replace(config.api_key, '[REDACTED]')))
                        raise
                    processed += result['status'] == 'SEALED'
            else:
                time.sleep(1.)


if __name__ == '__main__':
    main()
