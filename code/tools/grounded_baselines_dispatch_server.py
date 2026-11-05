"""Persistent coordinator; actual agent completions are acknowledged explicitly."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import json
from grounded_baselines import transport
print('DISPATCH_READY',flush=True)
for line in sys.stdin:
    try:
        command=json.loads(line)
        if command.get('exit'):break
        for item in transport.pending():
            name=item['invocation']['task_name']
            if name in command.get('spawned',[]) and not item['assigned']:
                transport.assign(item['request'],dict(task_name='/root/'+name))
            if name in command.get('completed',[]):transport.complete(item['request']);transport.seal(item['request'])
        pending=transport.pending()
        print(json.dumps([dict(name=r['invocation']['task_name'],assigned=r['assigned'],ready=r['output_exists'],
            io=str(Path(r['export']['input']).parent).replace('\\','/')) for r in pending]),flush=True)
    except Exception as exc:print(json.dumps(dict(error=type(exc).__name__,message=str(exc))),flush=True)
