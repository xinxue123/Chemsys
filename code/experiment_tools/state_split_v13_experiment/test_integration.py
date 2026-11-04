from copy import deepcopy
from pathlib import Path
import unittest

from knowledge_information_v10.tests.test_pipeline import fixture, context
from reasoning_memory_v13 import control
from reasoning_memory_v13.tests.test_state_split import raw_response
from state_split_v13_experiment import policy, study

ROOT = Path(__file__).resolve().parents[3]


def initial():
    pool, state = fixture()
    state.update(notes=[], review_queries=[])
    return pool, state


def action(**kw):
    a = dict(op='add', id='H001', claim='Moderate x may help', focus=['x'], effect='promote',
             scope={}, reference_id='p3', reason='Tentative input trend', evidence_queries=[])
    a.update(kw)
    return a


def edited(pool, state, actions):
    return policy.apply_edits(dict(summary='Test', actions=actions), pool, state, len(state['decisions'])+1)


class Integration(unittest.TestCase):
    def test_editor_uses_true_categories_and_rejects_unknown_labels(self):
        pool, state = initial()
        result = edited(pool, state, [action(scope={'category': {'allowed': ['1']}})])
        self.assertFalse(result['reviews'][0]['accepted'])
        self.assertEqual(result['notes'], [])

    def test_retaining_suppressed_note_cannot_restore_permission(self):
        pool, state = initial()
        note = edited(pool, state, [action()])['notes'][0]
        state['notes'] = [control.suppress(note, 1, 'test')]
        result = edited(pool, state, [action(op='retain', claim=None, focus=[], effect=None, scope={}, reference_id=None)])
        self.assertEqual(result['notes'][0]['recommendation']['permission'], 'suppressed')
        self.assertEqual(result['notes'][0]['calibration'], note['calibration'])

    def test_global_review_cap_masks_and_rejects_review_use(self):
        pool, state = initial()
        note = edited(pool, state, [action()])['notes'][0]
        state['review_queries'] = [3, 5]
        ids = list(pool.ids[4:9])
        pub = policy.public_context(pool, [note], state['history'], ids, 1, state['review_queries'])
        self.assertEqual(pub['review_query_budget_remaining'], 0)
        self.assertEqual(pub['knowledge'][0]['eligible_review_candidates'], [])
        raw = raw_response(ids, [note], [dict(id='H001', version=1, use='review_test')])
        with self.assertRaisesRegex(ValueError, 'trajectory-wide'):
            policy.validate_decision(ROOT, pool, [note], state, context(pool, state), ids, 1, raw)

    def test_public_context_contains_claim_and_separate_calibration(self):
        pool, state = initial()
        note = edited(pool, state, [action()])['notes'][0]
        pub = policy.public_context(pool, [note], state['history'], list(pool.ids[4:9]), 1, [])
        self.assertEqual(pub['knowledge'][0]['claim'], note['claim'])
        self.assertIn('prediction_reliability', pub['knowledge'][0])
        self.assertNotIn('kf', pub['knowledge'][0])

    def test_atomic_full_loop_16_queries_matches_frozen_adapter(self):
        pool, state = initial()
        state['notes'] = edited(pool, state, [action()])['notes']
        for query in range(1, 17):
            ctx = context(pool, state)
            ids = [k for k in pool.ids if k not in {r['candidate_id'] for r in state['history']}][:5]
            raw = raw_response(ids, state['notes'])
            packet = policy.validate_decision(ROOT, pool, state['notes'], state, ctx, ids, query, raw)
            c = dict(query=query, edited_notes=deepcopy(state['notes']), prestate_sha256=control.digest(state),
                     decision=packet, BO_reference_id=ids[0], refresh=dict(allowed=query==1))
            c['commit_sha256'] = control.digest(c)
            before = deepcopy(state)
            observation = dict(candidate_id=packet['selected_id'], y=.2, raw_value=20.)
            state = study.feedback(state, c, observation)
            with self.assertRaises(ValueError):
                study.feedback(state, c, observation)
            self.assertEqual(len(before['history'])+1, len(state['history']))
        self.assertEqual(state['status'], 'COMPLETE')
        self.assertEqual(len({r['candidate_id'] for r in state['history']}), 20)
        self.assertEqual(state['notes'][0]['recommendation']['permission'], 'suppressed')


if __name__ == '__main__':
    unittest.main()
