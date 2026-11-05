"""Receipt-only transport supporting actual nested experimental LLM children.

No scientific runtime, oracle, optimizer or response generation is imported.
The registered scientific requests and isolated child instructions stay intact.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/runs/grounded_baselines_20261105'
PARENT = '/root/grounded_experimental_llm_transport'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    with FileLock(str(path)+'.write.lock', timeout=60):
        if path.exists():
            assert read(path) == value, 'Immutable receipt differs: '+str(path)
            return
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   indent=2)+'\n', encoding='utf-8')
        temp.replace(path)


def dispatch(command):
    spawned = {r['task_name']: r for r in command.get('spawned', [])}
    completed = set(command.get('completed', []))
    for name in set(spawned) | completed:
        assert name.startswith(PARENT+'/hsm_glr_'), name
        assert name.count('/') == 3, name
    for inv_path in BASE.glob('requests/*/*/attempt*/invocation.json'):
        af = inv_path.parent
        if (af/'receipt.json').exists():
            continue
        inv = read(inv_path)
        name = PARENT+'/'+inv['task_name']
        if name in spawned:
            assert inv['fork_turns'] == 'none'
            assert inv['model'] == 'gpt-6.1-terra'
            assert inv['reasoning_effort'] == 'medium'
            for other in BASE.glob('requests/*/*/attempt*/assignment.json'):
                if other != af/'assignment.json':
                    assert read(other)['agent_id'] != name
            write(af/'assignment.json', dict(agent_id=name, spawn_request=inv,
                spawn_result=spawned[name], provider_attestation=False))
            spawned.pop(name)
        if name not in completed:
            continue
        assignment = read(af/'assignment.json')
        assert assignment['agent_id'] == name
        assert assignment['spawn_request'] == inv
        export = read(af/'export.json')
        ip, op = ROOT/export['input'], ROOT/export['output']
        assert sha(ip) == export['input_sha256']
        cp = af/'completion.json'
        if not cp.exists():
            write(cp, dict(agent_id=name,
                source='Actual collaboration FINAL_ANSWER notification acknowledged by coordinator',
                output_sha256=sha(op), acknowledged_UTC=datetime.now(timezone.utc).isoformat()))
        completion = read(cp)
        assert completion['agent_id'] == name and completion['output_sha256'] == sha(op)
        req = read(af/'request.json')
        write(af/'response.raw.json', read(op))
        write(af/'receipt.json', dict(agent_id=name, request_sha256=req['request_sha256'],
            raw_sha256=sha(af/'response.raw.json'), assignment_sha256=sha(af/'assignment.json'),
            model='gpt-6.1-terra', effort='medium', usage_available=False,
            completion_sha256=sha(cp), sealed_UTC=datetime.now(timezone.utc).isoformat()))
        completed.remove(name)
    assert not spawned, 'No request for actual spawned child: '+str(spawned)
    assert not completed, 'No pending request for completion: '+str(completed)
    return dict(status='OK')


print('READY_RECEIPT_ONLY_DELEGATED_TRANSPORT', flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command.get('exit'):
            break
        print(json.dumps(dispatch(command)), flush=True)
    except Exception as exc:
        print(json.dumps(dict(error=type(exc).__name__, message=str(exc))), flush=True)
