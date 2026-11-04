"""Persistent stdio helper for metadata operations; never runs a decision model.

This reuses the unchanged dispatcher functions and all their custody checks.
It does not launch trajectory workers, invoke agents, read response content for
selection, alter a scientific prompt, or cache a generated response.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys
import time

import paper_subagent_dispatch as dispatch

ROOT = dispatch.ROOT


def execute(command):
    action = command['action']
    if action == 'progress':
        result = dispatch.progress()
        if command.get('compact') is True:
            result['rows'] = [row for row in result['rows'] if row['status'] != 'COMPLETE']
        return result
    if action == 'prepare':
        return dispatch.prepare(Path(command['request']), command['task_name'])
    if action == 'record':
        return dispatch.record_spawn(Path(command['request']), command['spawn_result'])
    if action == 'submit':
        return dispatch.submit(Path(command['request']))
    if action == 'check':
        request = Path(command['request']).resolve()
        packet = dispatch.transport.request_record(request)
        export = dispatch.transport.read(request.parent / 'export.json')
        assigned = request.parent / 'assignment.json'
        if assigned.exists():
            from paper_subagent_audit import validate_assignment, validate_public_export
            protocol = dispatch.study.check(ROOT)
            assignment = dispatch.transport.read(assigned)
            validate_assignment(assignment, protocol)
            validate_public_export(ROOT, request.parent, packet, assignment, protocol)
        opaque = dispatch.transport.digest(dict(purpose='isolated_public_packet', request_sha256=packet['request_sha256']))
        expected = ROOT / 'experiments/agent_io/paper_v12_subagent' / opaque / 'input.json'
        if (ROOT / export['packet_relpath']).resolve() != expected.resolve():
            raise ValueError('Opaque request path does not match its frozen identity')
        return dict(request_sha256=packet['request_sha256'], context_sha256=packet['context_sha256'],
                    packet_relpath=export['packet_relpath'], output_relpath=export['output_relpath'],
                    assigned=assigned.exists(), submitted=(request.parent / 'provenance.json').exists())
    raise ValueError('Unsupported metadata action')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands', type=Path, help='Read JSON lines from a file and exit at EOF; otherwise use live stdin.')
    parser.add_argument('--serve-files', action='store_true', help='Persistent atomic file inbox; avoids Windows terminal line wrapping.')
    args = parser.parse_args()
    if args.commands and args.serve_files:
        parser.error('Choose either a command file or the persistent inbox')
    protocol = dispatch.study.check(ROOT)
    sources = [Path(__file__).resolve(), Path(dispatch.__file__).resolve()]
    hashes = {str(path): sha256(path.read_bytes()).hexdigest() for path in sources}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'experiments/logs/paper_v12_subagent/metadata_broker' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    if args.serve_files:
        (folder/'inbox').mkdir();(folder/'outbox').mkdir()
    ready = dict(status='READY_METADATA_ONLY', log_folder=folder.relative_to(ROOT).as_posix(),
                 source_sha256=hashes, frozen_protocol_sha256=protocol['protocol_sha256'])
    dispatch.transport.write_once(folder / 'startup.json', ready)
    print(json.dumps(ready), flush=True)
    stream = args.commands.open(encoding='utf-8') if args.commands else sys.stdin
    def entries():
        if not args.serve_files:
            for line in stream:
                if line.strip():
                    yield None,line
            return
        while True:
            waiting=[p for p in sorted((folder/'inbox').glob('*.json')) if not (folder/'outbox'/p.name).exists()]
            if not waiting:
                time.sleep(.1)
            for path in waiting:
                yield path,path.read_text(encoding='utf-8-sig')
    try:
        for input_file,line in entries():
            started = time.perf_counter()
            command = None
            try:
                command = json.loads(line)
                if any(sha256(path.read_bytes()).hexdigest() != hashes[str(path)] for path in sources):
                    raise RuntimeError('Metadata helper source changed; start a new helper before another action')
                if command['action'] == 'exit':
                    result = dict(status='CLOSING_METADATA_HELPER')
                else:
                    result = execute(command)
                response = dict(id=command.get('id'), action=command['action'], ok=True,
                                seconds=time.perf_counter()-started, result=result)
            except Exception as error:
                response = dict(id=command.get('id') if isinstance(command, dict) else None, ok=False,
                                seconds=time.perf_counter()-started, error_type=type(error).__name__, error=str(error))
            with (folder / 'commands.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(dict(command=command, response=response), ensure_ascii=False) + '\n')
            if input_file:
                dispatch.transport.write_once(folder/'outbox'/input_file.name,response)
            else:
                print(json.dumps(response, ensure_ascii=False), flush=True)
            if isinstance(command, dict) and command.get('action') == 'exit':
                break
    finally:
        if args.commands:
            stream.close()


if __name__ == '__main__':
    main()
