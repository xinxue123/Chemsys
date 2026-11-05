"""Compact operational dispatcher; original receipt logic stays unchanged."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from grounded_baselines import transport
print('READY',flush=True)
for line in sys.stdin:
    try:
        command=json.loads(line)
        if command.get('exit'):break
        rows=transport.pending()
        for item in rows:
            name=item['invocation']['task_name']
            if name in command.get('spawned',[]) and not item['assigned']:
                transport.assign(item['request'],dict(task_name='/root/'+name))
            if name in command.get('completed',[]):transport.complete(item['request']);transport.seal(item['request'])
        rows=transport.pending()
        selected=[r for r in rows if not r['assigned']]
        selected.sort(key=lambda r:r['invocation']['task_name'])
        print(json.dumps(dict(pending=len(rows),assigned=sum(r['assigned'] for r in rows),
            next=[dict(name=r['invocation']['task_name'],io=Path(r['export']['input']).parent.name)
                  for r in selected[:command.get('limit',3)]])),flush=True)
    except Exception as exc:print(json.dumps(dict(error=type(exc).__name__,message=str(exc))),flush=True)
