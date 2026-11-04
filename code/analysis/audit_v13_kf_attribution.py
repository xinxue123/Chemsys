"""Compile existing knowledge into candidate tables and inspect attribution.

This is a pre-existing-menu diagnostic, not a new selector or a policy replay.
"""
from collections import Counter
import csv
import json
import math

from audit_v13_kf_injection import CELLS, OUT, guard, load, write


def main():
    guard()
    events, snapshots, totals = [], [], []
    for task, seed in CELLS:
        for arm in ('full', 'fixed'):
            run = load(task, seed, arm)
            stats = Counter()
            for q, entry in enumerate(run['rounds'], 1):
                c = entry['commit']
                public = entry['context']['public']
                notes = public['retrieved_notes']['knowledge']
                by_key = {(n['id'],n['version']):n for n in notes}
                menu = {p['candidate_id']:p for p in public['candidate_menu']}
                selected = menu[c['selected_id']]
                incumbent = public['current_incumbent']
                previous = next((p for p in public['own_observations']
                                 if p['candidate_id']==incumbent['candidate_id']), None) if incumbent else None
                for f, e in zip(c['decision']['forecasts'],run['state']['decisions'][q-1]['feedback']):
                    if f['commit_sha256'] != e['commit_sha256']:
                        raise ValueError('Feedback identity mismatch')
                    y = float(e['improved'])
                    z = (y-f['p_GP'])**2-(y-f['p_LLM'])**2
                    if not math.isclose(z,e['calibration']['Brier_gain'],abs_tol=1e-12):
                        raise ValueError('Calibration evidence differs')
                    note = by_key[f['id'],f['version']]
                    changed = [a for a in selected['conditions']
                               if selected['conditions'][a]!=previous['conditions'][a]]
                    focused = [a for a in note['focus'] if a in changed]
                    tested = f['use'] in ('tentative_test','positive_support','review_test')
                    row = dict(task=task,seed=seed,arm=arm,query=q,
                        note_id=note['id'],version=note['version'],effect=note['effect'],
                        selected_id=c['selected_id'],incumbent_id=incumbent['candidate_id'],
                        declared_use=f['use'],direction_test=e['direction_test'],improved=e['improved'],
                        p_LLM=f['p_LLM'],p_GP=f['p_GP'],Brier_gain=z,
                        focus_axes=note['focus'],changed_axes_from_incumbent=changed,
                        changed_focus_axes=focused,all_five_candidates_covered=len(note['covered_candidate_ids'])==5,
                        isolated_focus_change=bool(focused) and set(changed)<=set(note['focus']),
                        permission_before=e['recommendation_before'],permission_after=e['recommendation_after'],
                        r_before=e['calibration']['before']['reliability'],r_after=e['calibration']['after']['reliability'])
                    events.append(row)
                    stats['forecasts']+=1
                    stats['declared_direction_tests']+=tested
                    stats['improved_direction_tests']+=tested and e['improved']
                    stats['direction_tests_changing_focus']+=tested and bool(focused)
                    stats['direction_tests_with_uniform_menu_scope']+=tested and row['all_five_candidates_covered']
                    stats['isolated_focus_changes']+=row['isolated_focus_change']
                if (task=='hplc' and q==6) or (task=='perera' and q==8):
                    for cid in menu:
                        covered = [n for n in notes if cid in n['covered_candidate_ids']]
                        snapshots.append(dict(task=task,seed=seed,arm=arm,query=q,candidate_id=cid,
                            observed_choice=cid==c['selected_id'],GP_reference=cid==c['BO_reference_id'],
                            notes=json.dumps([dict(id=n['id'],r=n['prediction_reliability']['reliability'],
                                permission=n['recommendation']['permission']) for n in covered]),
                            positive_support_note_ids=json.dumps([n['id'] for n in covered if n['recommendation']['positive_support_allowed']]),
                            tentative_test_note_ids=json.dumps([n['id'] for n in covered if n['recommendation']['permission']=='tentative']),
                            adverse_warning_note_ids=json.dumps([n['id'] for n in covered if n['adverse_facts']]),
                            eligible_review_note_ids=json.dumps([n['id'] for n in covered
                                if any(r['candidate_id']==cid for r in n['eligible_review_candidates'])]),
                            original_GP_point_remains_executable=True,
                            diagnostic_only_no_new_rank_or_selection=True))
            totals.append(dict(task=task,seed=seed,arm=arm,**dict(stats)))
    guard()
    write('attribution.json',dict(status='READ_ONLY_COMPLETE',scientific_queries=0,
        definition='direction_test records declared positive use; positive_tests additionally requires observed improvement; neither requires a focused-variable contrast',
        totals=totals,events=events))
    with (OUT/'candidate_action_snapshots.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(snapshots[0]));writer.writeheader();writer.writerows(snapshots)
    print(json.dumps(dict(status='COMPLETE',scientific_queries=0,totals=totals,
        perera_full_events=[e for e in events if e['task']=='perera' and e['arm']=='full']),ensure_ascii=False))


if __name__=='__main__':
    main()
