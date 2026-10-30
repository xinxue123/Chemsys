"""Exact transcription only. Never generates or ranks scientific proposals.

Input answers are manually authored by the current assistant. This adapter binds
them to already-created pending contexts and copies IDs / explicitly retained
preferences. It never loads an oracle, GP, optimizer, or unrevealed response.
"""
import argparse
import json
from pathlib import Path

ROOT = Path('data/runs/suzuki_baselines')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pending():
    result = {}
    for p in sorted((ROOT / 'runs').glob('*/state.json')):
        state = read(p)
        if state['status'] == 'WAITING_LLM':
            request = read(ROOT / 'requests' / state['pending_context'] / 'request.json')
            result[state['run_id']] = request
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', choices=['full', 'latest'])
    parser.add_argument('--answers', type=Path)
    args = parser.parse_args()
    requests = pending()
    if args.show:
        for run_id, request in requests.items():
            history = request['history']
            selected = history if args.show == 'full' else [history[-1]]
            row = {'id': run_id, 't': request['iteration'],
                   'H': [[h['query'], h['point']['catalyst'], *[round(h['point'][n], 3) for n in
                          ('t_res', 'temperature', 'catalyst_loading')], round(h['yield_percent'], 3)] for h in selected]}
            if args.show == 'latest':
                best = max(history, key=lambda h: h['yield_percent'])
                row['best'] = [best['query'], best['point'], round(best['yield_percent'], 3)]
            print(json.dumps(row, separators=(',', ':')))
        return
    answers = read(args.answers)
    responses = []
    for run_id, answer in answers['answers'].items():
        request = requests[run_id]
        # No coordinates, region bounds, confidence, rationale, or citations are
        # generated here; every one must be explicitly supplied by the assistant.
        response = dict(answer)
        response.update({k: request[k] for k in ('run_id', 'iteration', 'context_sha256')})
        responses.append(response)
    output = args.answers.with_suffix('.expanded.json')
    output.write_text(json.dumps({'responses': responses}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'expanded_file': str(output), 'responses': len(responses)}))


if __name__ == '__main__':
    main()
