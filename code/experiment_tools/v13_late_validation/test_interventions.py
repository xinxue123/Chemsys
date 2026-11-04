import unittest
from copy import deepcopy
from unittest.mock import patch
from . import study


class Interventions(unittest.TestCase):
    def test_freeze_retains_numeric_state_and_directional_failure(self):
        c=study.control;n=c.new_note('H001',1,'promote',{},['x'])
        history=[dict(candidate_id='initial',raw_value=4.)]
        p=c.commit_forecast(n,history,2,'next',{'x':.2},.1,.6,use='tentative_test')
        with patch.object(c.calibration,'update',side_effect=study.freeze_update):
            after,e=c.feedback(n,p,history,dict(candidate_id='next',raw_value=2.))
        self.assertEqual(after['calibration']['reliability'],.4)
        self.assertEqual(after['calibration']['mean'],0.)
        self.assertEqual(after['recommendation']['permission'],'suppressed')
        self.assertFalse(e['calibration']['applied'])
        self.assertGreater(e['calibration']['proposed_after']['reliability'],.4)

    def test_coupling_reconnects_accurate_failure_to_old_positive_authority(self):
        c=study.control;n=c.new_note('H001',1,'promote',{},['x'])
        h=[dict(candidate_id='initial',raw_value=4.)]
        for value in (5.,6.):
            q=len(h)+1;p=c.commit_forecast(n,h,q,f'p{q}',{'x':.2},.8,.5,use='tentative_test')
            obs=dict(candidate_id=f'p{q}',raw_value=value)
            n,_=c.feedback(n,p,h,obs);h.append(obs)
        self.assertEqual(n['recommendation']['permission'],'allowed')
        p=c.commit_forecast(n,h,4,'failure',{'x':.2},.1,.6,use='positive_support')
        obs=dict(candidate_id='failure',raw_value=3.)
        split,_=c.feedback(n,p,h,obs);coupled,e=study.coupled_feedback(n,p,h,obs)
        self.assertEqual(split['recommendation']['permission'],'suppressed')
        self.assertEqual(coupled['recommendation']['permission'],'allowed')
        self.assertGreater(coupled['calibration']['reliability'],n['calibration']['reliability'])
        self.assertTrue(e['accurate_failure_forecast'])

    def test_shared_prefix_is_exact_for_all_six_contexts(self):
        for task in study.FOCUS:
            for seed in (0,3):
                state=study.replay_prefix(task,seed)
                self.assertEqual(len(state['history']),10)
                self.assertEqual(len(state['decisions']),10)


if __name__=='__main__':unittest.main()
