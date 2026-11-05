"""Dispatch receipts for observed spawn results; never fabricate model outputs.

The runs directory is a Windows junction. Canonicalize the containment root in
this operator only; preserve the frozen scientific runner and all request hashes.
"""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chemical_hypothesis_bo import transport as t
t.BASE=t.BASE.resolve()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spawned',nargs='*',default=[],help='Exact task names returned by actual collaboration.spawn_agent calls.')
    args=parser.parse_args()
    for p in t.pending():
        name=p['invocation']['task_name']
        if name in args.spawned and not p['assigned']:
            # This tool returns only task_name, reproduced exactly here after
            # the root observed it. The invocation supplies the full parameters.
            t.assign(p['request'],dict(task_name='/root/'+name))
        if (Path(p['request']).parent/'assignment.json').exists() and p['output_exists']:
            print(json.dumps(t.seal(p['request'])))
    pending=[]
    for p in t.pending():
        export=t.read(Path(p['request']).parent/'export.json')
        pending.append(dict(name=p['invocation']['task_name'],request=p['request'],
            input=export['input'],output=export['output'],assigned=p['assigned'],ready=p['output_exists']))
    print(json.dumps(dict(pending=pending)))


if __name__=='__main__':main()
