"""Compact dispatch metadata only; actual agent spawning stays in collaboration."""
from pathlib import Path
import argparse
import json
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from state_split_v13_experiment import study,transport

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('next','record','seal','finish'))
    p.add_argument('names',nargs='*',help='Exact task_name values returned by the collaboration tool')
    a=p.parse_args()
    rows=transport.pending(ROOT)
    if a.action!='next':
        by_name={'/root/'+r['invocation']['task_name']:r for r in rows}
        original_base=transport.base
        # experiments/runs is a Windows junction. Normalize BOTH sides of the
        # metadata-only namespace check. Scientific workers remain untouched.
        canonical_base=original_base(ROOT).resolve()
        transport.write_once(canonical_base/'dispatch_path_adapter.json',dict(
            purpose='canonicalize_both_sides_of_Windows_junction_namespace_check',
            lexical_path=str(original_base(ROOT)),canonical_path=str(canonical_base),
            helper_sha256=transport.file_sha(__file__),scientific_source_modified=False))
        with patch.object(transport,'base',side_effect=lambda root:original_base(root).resolve()):
            for name in a.names:
                r=by_name[name]
                if a.action in ('record','finish'):
                    transport.record_spawn(ROOT,r['request'],dict(task_name=name))
                if a.action in ('seal','finish'):
                    transport.seal(ROOT,r['request'])
    status=study.status(ROOT)
    print(json.dumps(dict(complete=status['complete'],queries=status['queries'],rows=status['rows'],
        pending=[dict(task=r['invocation']['task_name'],request=r['request'],
                      input=str((ROOT/transport.read(Path(r['request']).parent/'export.json')['input']).as_posix()),
                      assigned=r['assigned'],output_exists=r['output_exists']) for r in status['pending']]),ensure_ascii=False))
if __name__=='__main__':main()
