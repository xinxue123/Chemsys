"""Transcribe explicit assistant decisions to retain an earlier authored batch.

No outcomes, priors or rankings are read. Every context ID and candidate queue
must be supplied literally by the assistant after reviewing that context.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem.decision.hypothesis import canonical_hash
from chem.decision.replay import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite')
    parser.add_argument('decisions')
    parser.add_argument('output')
    args = parser.parse_args()
    decisions_path = Path(args.decisions)
    directives = json.loads(decisions_path.read_text(encoding='utf-8'))
    answers = []
    for directive in directives:
        if set(directive) != {'context_sha256', 'decision', 'candidate_ids'}:
            raise ValueError('explicit context, retention decision and candidate queue are required')
        if directive['decision'] != 'retain_previous_authored_predicates':
            raise ValueError('new hypotheses must be authored literally in the complete answer format')
        key = directive['context_sha256']
        request = json.loads((Path(args.suite) / 'requests' / key / 'request.json').read_text(encoding='utf-8'))
        if canonical_hash(request['context']) != key:
            raise ValueError('request changed')
        batch = deepcopy(request['context']['previous_batch'])
        batch['version'] = request['context']['expected_version']
        answers.append({'context_sha256': key, 'batch': batch, 'candidate_ids': directive['candidate_ids']})
    output = Path(args.output)
    if output.exists() and json.loads(output.read_text(encoding='utf-8')) != answers:
        raise ValueError('transcribed answers already exist and differ')
    write_json_atomic(output, answers)
    write_json_atomic(output.with_suffix('.transcription.json'), {
        'purpose': 'mechanical copy of prior assistant-authored predicates, explicitly selected per context',
        'chemical_inference_in_script': False, 'unqueried_outcomes_read': False,
        'directives_sha256': sha256(decisions_path.read_bytes()).hexdigest(),
        'answers_sha256': sha256(output.read_bytes()).hexdigest(),
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'count': len(answers)})
    print(json.dumps({'transcribed_explicit_retention_decisions': len(answers)}))


if __name__ == '__main__':
    main()
