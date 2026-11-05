"""Only request metadata and actual responder receipts are handled here."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from llm_screen_qlogei import study
from cold_start_v13 import transport


def main():
    action=sys.argv[1]
    with study.transport_scope():
        if action=='pending':
            print(json.dumps(transport.pending(ROOT)))
        elif action=='assign':
            payload=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
            results=[transport.record_spawn(ROOT,row['request'],row['spawn_result']) for row in payload]
            print(json.dumps(results))
        elif action=='seal':
            payload=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
            results=[transport.seal(ROOT,path) for path in payload]
            print(json.dumps(results))
        elif action=='status':
            rows=[]
            for j in study.read(study.BASE/'protocol.json')['jobs']:
                sf=study.BASE/'runs'/j['id']/'state.json'
                s=study.read(sf) if sf.exists() else {}
                rows.append(dict(job=j['id'],queries=len(s.get('history',[])),status=s.get('status','WAITING'),
                    LLM_calls=s.get('knowledge_calls',[])))
            print(json.dumps(rows))
        else:
            raise ValueError(action)


if __name__=='__main__':
    main()
