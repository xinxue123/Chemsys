"""Inspect frozen prompts and feedback; never request an experiment or an LLM."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/runs/v13_origin_ablation_20261103'
OUT = ROOT / 'experiments/reports/v13_kf_injection_audit_20261103'
CELLS = (('doyle', 0), ('doyle', 1), ('hplc', 3), ('perera', 2))
ARMS = ('full', 'fixed', 'no_refresh', 'neither')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    content = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError('Preserve earlier analysis before changing its inputs: ' + name)
    path.write_text(content, encoding='utf-8')


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in (differences(a[k], b[k], path + '/' + k)
                          if k in a and k in b else [path + '/' + k])]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + '/length']
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in differences(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


def guard():
    protocol = read(BASE / 'targeted/protocol.json')
    origin = read(BASE / 'protocol.json')
    groups = (protocol['sources_sha256'], protocol['cancelled_phase_states_sha256'],
              origin['inputs_sha256'], origin['protected_sha256'])
    for group in groups:
        for name, expected in group.items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen source/evidence changed: ' + name)
    milestone = ROOT / 'experiments/milestones/v13_targeted_ablation_20261103'
    receipt = read(milestone / 'receipt.json')
    if sha(milestone / 'evidence.zip') != receipt['archive_sha256']:
        raise ValueError('Sealed evidence archive changed')
    return protocol


def load(task, seed, arm):
    folder = BASE / 'targeted/runs' / f't_{task}{seed}_{arm}'
    state = read(folder / 'state.json')
    rounds = []
    for q in range(1, 21):
        path = folder / 'rounds' / f'{q:02}_commit.json'
        commit = read(path)
        request = ROOT / commit['reasoning']['request_relpath']
        context_path = request.parent / 'context.json'
        rounds.append(dict(commit=commit, context=read(context_path),
                           commit_path=path.resolve().relative_to(ROOT).as_posix(),
                           context_path=context_path.resolve().relative_to(ROOT).as_posix()))
    return dict(state=state, rounds=rounds)


def main():
    protocol = guard()
    data = {(t, s, a): load(t, s, a) for t, s in CELLS for a in ARMS}
    rows, summaries, examples, divergences = [], [], [], []
    for (task, seed, arm), run in data.items():
        stats = Counter()
        for q, entry in enumerate(run['rounds'], 1):
            c = entry['commit']
            public = entry['context']['public']
            notes = public['retrieved_notes']['knowledge']
            raw = c['reasoning']['raw']
            selected = c['selected_id']
            uses = raw['knowledge_uses']
            hypotheses = raw['insight']['hypotheses']
            selected_notes = [n for n in notes if selected in n['covered_candidate_ids']]
            row = dict(task=task, seed=seed, arm=arm, query=q, selected_id=selected,
                       raw_objective=run['state']['history'][q-1]['raw_value'],
                       GP_reference_id=c['BO_reference_id'], visible_notes=len(notes),
                       has_feedback_note=int(any(n['prediction_reliability']['updates'] for n in notes)),
                       has_changed_KF_score=int(any(abs(n['prediction_reliability']['reliability']-.4)>1e-12 for n in notes)),
                       has_partial_scope=int(any(0 < len(n['covered_candidate_ids']) < 5 for n in notes)),
                       has_uniform_scope=int(any(len(n['covered_candidate_ids']) == 5 for n in notes)),
                       selected_covered=len(selected_notes), declared_uses=json.dumps(uses),
                       first_hypothesis_confidence=hypotheses[0]['confidence'],
                       all_hypotheses_low=int(all(h['confidence']=='low' for h in hypotheses)),
                       one_distinct_proposed_point=int(len({p['candidate_id'] for h in hypotheses for p in h['parameter_sets']})==1),
                       commit_path=entry['commit_path'], context_path=entry['context_path'])
            rows.append(row)
            for k in ('has_feedback_note', 'has_changed_KF_score', 'has_partial_scope',
                      'has_uniform_scope', 'all_hypotheses_low', 'one_distinct_proposed_point'):
                stats[k] += row[k]
            stats['visible_note_rounds'] += bool(notes)
            stats['selected_note_rounds'] += bool(selected_notes)
            stats['KF_feedback_events'] += len(run['state']['decisions'][q-1]['feedback'])
            stats['non_GP_choice_rounds'] += selected != c['BO_reference_id']
            stats['declared_knowledge_use_rounds'] += bool(uses)
            stats.update('use_' + u['use'] for u in uses)
            if (task=='hplc' and q in (5,6,13)) or (task=='perera' and q in (6,7,8)):
                examples.append(dict(**row, public_notes=notes, raw_response=raw,
                    committed_forecasts=c['decision']['forecasts'],
                    feedback=run['state']['decisions'][q-1]['feedback']))
        summaries.append(dict(task=task, seed=seed, arm=arm, queries=20, **dict(stats)))
    for task, seed in CELLS:
        full = data[task, seed, 'full']
        for arm in ('fixed','no_refresh','neither'):
            other = data[task, seed, arm]
            first_prompt = first_choice = None
            for q, (a,b) in enumerate(zip(full['rounds'],other['rounds']),1):
                if first_prompt is None and a['context'] != b['context']:
                    ac = {k:v for k,v in a['context'].items() if k!='public'}
                    bc = {k:v for k,v in b['context'].items() if k!='public'}
                    ap = {k:v for k,v in a['context']['public'].items() if k!='official_optimization_prompt'}
                    bp = {k:v for k,v in b['context']['public'].items() if k!='official_optimization_prompt'}
                    first_prompt = dict(query=q, context_metadata_identical=ac==bc,
                        non_embedded_public_difference_paths=differences(ap,bp),
                        own_observations_identical=ap['own_observations']==bp['own_observations'],
                        menus_identical=ap['candidate_menu']==bp['candidate_menu'],
                        previous_Insights_identical=full['state']['insights'][:q-1]==other['state']['insights'][:q-1],
                        full_context_path=a['context_path'], ablation_context_path=b['context_path'])
                if first_choice is None and a['commit']['selected_id'] != b['commit']['selected_id']:
                    first_choice = dict(query=q,
                        own_experiments_identical=full['state']['history'][:q-1]==other['state']['history'][:q-1],
                        GP_arrays_identical=a['commit']['GP_sha256']==b['commit']['GP_sha256'],
                        menus_identical=a['commit']['display_ids']==b['commit']['display_ids'],
                        previous_Insights_identical=full['state']['insights'][:q-1]==other['state']['insights'][:q-1],
                        full_selected=a['commit']['selected_id'], ablation_selected=b['commit']['selected_id'])
            divergences.append(dict(task=task, seed=seed, arm=arm,
                                    first_prompt_difference=first_prompt, first_choice_difference=first_choice))
    guard()
    OUT.mkdir(parents=True,exist_ok=True)
    write('summary.json',dict(status='READ_ONLY_LOG_AUDIT_COMPLETE', cells=4, trajectories=16,
        scientific_queries=0, LLM_calls=0, GP_fits=0, frozen_evidence_unchanged=True,
        targeted_protocol_sha256=protocol['protocol_sha256'], stats=summaries,
        inference_limit='Changed-prompt response sampling is not independently controlled; no new policy is evaluated.'))
    write('first_differences.json',divergences)
    write('selected_round_evidence.json',examples)
    with (OUT/'round_injection.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='COMPLETE', queries=0, stats=[x for x in summaries if x['arm']=='full'],
        first_differences=[x for x in divergences if x['arm']=='fixed']),ensure_ascii=False))


if __name__=='__main__':
    main()
