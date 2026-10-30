"""Print only public queued contexts for the assistant; never loads an oracle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem.decision.assistant_suite import assistant_status, request_view
from chem.decision.hypothesis import canonical_hash
from chem.decision.replay import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite')
    parser.add_argument('--case', type=int)
    args = parser.parse_args()
    views = []
    for filename in assistant_status(args.suite)['pending_requests']:
        path = Path(filename)
        view = request_view(json.loads(path.read_text(encoding='utf-8')))
        if args.case is not None and not view['task_id'].endswith(f'_{args.case}'):
            continue
        write_json_atomic(path.with_name('presentation.json'), view)
        views.append(view)
    shown_options = set()
    for v in sorted(views, key=lambda r: (r['task_id'], r['expected_version'], r['context_sha256'])):
        print('CONTEXT', v['context_sha256'], v['task_id'], 'v', v['expected_version'])
        print('OBS [id,catalyst,seconds,degC,mol%,normalized_yield]')
        print(json.dumps([[r['candidate_id'], r['params']['catalyst'], round(r['params']['t_res'], 1),
                           round(r['params']['temperature'], 1), round(r['params']['catalyst_loading'], 2),
                           round(r['value'], 4)] for r in v['observations']], separators=(',', ':')))
        print('PREVIOUS', json.dumps(v['previous_batch'], ensure_ascii=False, separators=(',', ':')))
        options_hash = canonical_hash(v['candidate_options'])
        print('OPTIONS_SHA', options_hash)
        if options_hash not in shown_options:
            shown_options.add(options_hash)
            print(json.dumps([[r['id'], r['catalyst'], round(r['t_res'], 1),
                               round(r['temperature'], 1), round(r['catalyst_loading'], 2)]
                              for r in v['candidate_options']], separators=(',', ':')))
    print('UNIQUE_PENDING_CONTEXTS', len(views))


if __name__ == '__main__':
    main()
