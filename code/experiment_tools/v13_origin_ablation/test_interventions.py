"""Synthetic intervention and coupling checks; no real outcomes or LLM calls."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from cold_start_v13 import control
from . import queue,study


class InterventionTests(unittest.TestCase):
    def test_fixed_state_does_not_expose_unapplied_weights(self):
        before=control.calibration.initial();snapshot=deepcopy(before)
        after,event=study.fixed_update(before,.8,.2,True)
        self.assertEqual(before,snapshot)
        self.assertEqual(after,{**before,'updates':1})
        self.assertGreater(event['Brier_gain'],0)
        self.assertEqual(event['after'],after)
        self.assertNotIn('proposed_after',event)
        self.assertNotIn('intervention',event)
        self.assertEqual(event['kalman_gain'],0)

    def test_numerical_freeze_keeps_failure_suppression(self):
        n=control.new_note('H001',1,'promote',{'x':{'bounds':[0.,1.]}},['x'])
        history=[dict(candidate_id='old',raw_value=50.,y=.5)]
        packet=control.commit_forecast(n,history,2,'new',{'x':.5},.1,.4,use='tentative_test')
        observation=dict(candidate_id='new',raw_value=30.,y=.3)
        with patch.object(control.calibration,'update',side_effect=study.fixed_update):
            after,event=control.feedback(n,packet,history,observation)
        self.assertEqual(after['calibration']['reliability'],.4)
        self.assertTrue(event['accurate_failure_forecast'])
        self.assertEqual(after['recommendation']['permission'],'suppressed')

    def test_refresh_mask_retains_initial_editor_only(self):
        state=dict(edit_calls=[])
        with patch.object(study,'EDIT_EVENT',return_value=dict(allowed=True,reasons=['synthetic'])):
            for arm in ('full','fixed','no_refresh','neither'):
                self.assertTrue(study.edit_event(arm,None,state,[],2)['allowed'])
            self.assertFalse(study.edit_event('no_notes',None,state,[],2)['allowed'])
            state['edit_calls']=[2]
            for arm in ('no_refresh','neither','no_notes'):
                self.assertFalse(study.edit_event(arm,None,state,[],5)['allowed'])
            for arm in ('full','fixed'):
                self.assertTrue(study.edit_event(arm,None,state,[],5)['allowed'])

    def test_response_coupling_requires_same_input_and_start(self):
        temporary_parent=(study.ROOT/'runtime/tmp').resolve()
        self.assertTrue(temporary_parent.is_relative_to(study.ROOT))
        temporary_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='oa_test_',dir=temporary_parent) as d:
            root=Path(d);raw=root/'synthetic.json';raw.write_text(json.dumps({'selected_id':'s0'}),encoding='utf-8')
            self.assertTrue(root.resolve().is_relative_to(temporary_parent))
            original=dict(system='synthetic only',public=dict(query=1,menu=['s0']))
            queue.seed(root,'synthetic_cell','choose',original,raw,{'synthetic':True})
            validator=lambda x:dict(selected_id=x['selected_id'])
            try:
                queue.CURRENT=dict(id='synthetic_full',cell='synthetic_cell',arm='full')
                a=queue.request(root,'unused','choose',1,original['system'],original['public'],validator)
                queue.CURRENT=dict(id='synthetic_fixed',cell='synthetic_cell',arm='fixed')
                b=queue.request(root,'unused','choose',1,original['system'],original['public'],validator)
                self.assertEqual(a['raw'],b['raw'])
                self.assertEqual(a['response_receipt']['raw_sha256'],b['response_receipt']['raw_sha256'])
                queue.CURRENT=dict(id='synthetic_other',cell='other_start',arm='full')
                with self.assertRaisesRegex(ValueError,'unarchived'):
                    queue.request(root,'unused','choose',1,original['system'],original['public'],validator)
                queue.CURRENT=dict(id='synthetic_changed',cell='synthetic_cell',arm='full')
                with self.assertRaisesRegex(ValueError,'unarchived'):
                    queue.request(root,'unused','choose',1,original['system'],dict(query=1,menu=['s1']),validator)
            finally:queue.CURRENT=None


if __name__=='__main__':unittest.main()
