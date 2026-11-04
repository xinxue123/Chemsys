"""Lightweight local-file client; no Torch import, network or decision model."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('progress','prepare','record','submit','check','exit'))
    parser.add_argument('--broker',required=True,type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--task-name')
    parser.add_argument('--result-file',type=Path)
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--timeout',type=float,default=30.)
    args=parser.parse_args()
    folder=args.broker.resolve()
    namespace=(ROOT/'experiments/logs/paper_v12_subagent/metadata_broker').resolve()
    if not folder.is_relative_to(namespace) or not (folder/'startup.json').is_file() or not (folder/'inbox').is_dir():
        raise ValueError('Expected the live metadata broker folder returned at startup')
    command=dict(id=uuid.uuid4().hex,action=args.action)
    if args.request:command['request']=str(args.request.resolve())
    if args.task_name:command['task_name']=args.task_name
    if args.result_file:command['spawn_result']=json.loads(args.result_file.read_text(encoding='utf-8-sig'))
    if args.compact:command['compact']=True
    path=folder/'inbox'/(command['id']+'.json');temporary=path.with_suffix('.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(command,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    if path.exists():raise ValueError('Command identity already exists')
    temporary.replace(path)
    answer=folder/'outbox'/path.name;started=time.monotonic()
    while not answer.exists():
        if time.monotonic()-started>args.timeout:
            print(json.dumps(dict(status='OBSERVATION_TIMEOUT_NOT_OPERATION_FAILURE',command_file=str(path),response_file=str(answer),
                                  note='Check this exact command response and broker handle before retrying; native operations may already have completed.')))
            raise SystemExit(2)
        time.sleep(.1)
    response=json.loads(answer.read_text(encoding='utf-8'))
    print(json.dumps(response,ensure_ascii=False,indent=2))
    raise SystemExit(0 if response['ok'] else 1)


if __name__=='__main__':main()
