"""Read-only compact dispatch inventory without importing scientific runtimes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
items=[];assigned=0;sealed=0
for path in (BASE/'requests').glob('*/*/attempt*/invocation.json'):
    folder=path.parent
    if (folder/'receipt.json').exists():sealed+=1;continue
    if (folder/'assignment.json').exists():assigned+=1;continue
    inv=json.loads(path.read_text(encoding='utf-8'))
    export=json.loads((folder/'export.json').read_text(encoding='utf-8'))
    items.append(dict(name=inv['task_name'],io=Path(export['input']).parent.name))
items.sort(key=lambda r:r['name'])
result=dict(sealed=sealed,assigned=assigned,pending=len(items),completed=len(list(BASE.glob('runs/*/*/completion.json'))))
if '--summary' not in sys.argv:result['next']=items[:3]
print(json.dumps(result))
