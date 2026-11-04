"""Regression checks for real cold starts and unchanged split-state evidence."""
from copy import deepcopy
import unittest
import numpy as np

from reasoning_memory_v13 import control as warm_control
from . import control, policy
from .validation import metrics


def row(key,value):
    return dict(candidate_id=key,raw_value=value,y=value/100)


class ColdStartTests(unittest.TestCase):
    def test_no_incumbent_forecast_without_observation(self):
        note = control.new_note('H001',1,'promote',{},['temperature'],created_query=2)
        with self.assertRaises(ValueError):
            control.commit_forecast(note,[],1,'a',{'temperature':.5},.5,.5,use='tentative_test')

    def test_two_successes_keep_original_calibration_and_permissions(self):
        warm_history = [row('w0',0.),row('w1',1.),row('w2',2.),row('a',10.)]
        cold_history = [row('a',10.)]
        old = warm_control.new_note('H001',1,'promote',{},['temperature'],created_query=1)
        new = control.new_note('H001',1,'promote',{},['temperature'],created_query=2)
        for i,value in enumerate((12.,15.)):
            obs = row('p'+str(i),value)
            a = warm_control.commit_forecast(old,warm_history,i+1,obs['candidate_id'],{'temperature':.5},.8,.6,use='tentative_test')
            b = control.commit_forecast(new,cold_history,i+2,obs['candidate_id'],{'temperature':.5},.8,.6,use='tentative_test')
            old,_ = warm_control.feedback(old,a,warm_history,obs)
            new,_ = control.feedback(new,b,cold_history,obs)
            warm_history.append(obs)
            cold_history.append(obs)
            self.assertEqual(old['calibration'],new['calibration'])
            self.assertEqual(old['recommendation']['permission'],new['recommendation']['permission'])
        self.assertEqual(new['recommendation']['permission'],'allowed')

    def test_failure_never_reopens_and_stagnation_needs_anchor(self):
        history = [row('a',10.)]
        note = control.new_note('H001',1,'promote',{},['temperature'],created_query=2)
        packet = control.commit_forecast(note,history,2,'b',{'temperature':.5},.1,.8,use='tentative_test')
        note,event = control.feedback(note,packet,history,row('b',5.))
        self.assertTrue(event['accurate_failure_forecast'])
        self.assertEqual(note['recommendation']['permission'],'suppressed')
        history.append(row('b',5.))
        _,receipt = control.open_review(note,history,3,conditions={'temperature':.5},thresholds={'temperature':.25})
        self.assertFalse(receipt['opened'])
        history += [row('c',4.),row('d',3.)]
        opened,receipt = control.open_review(note,history,5,conditions={'temperature':.5},thresholds={'temperature':.25})
        self.assertTrue(receipt['opened'])
        packet = control.commit_forecast(opened,history,5,'e',{'temperature':.5},.1,.8,use='review_test')
        after,_ = control.feedback(opened,packet,history,row('e',2.))
        self.assertEqual(after['recommendation']['permission'],'suppressed')

    def test_all_twenty_paid_queries_in_metric(self):
        state = dict(history=[row('p'+str(i),float(i+1)) for i in range(20)],decisions=[{}]*20)
        m = metrics(state)
        self.assertEqual(m['SearchAUC'],10.5)
        self.assertEqual(m['first_final_best_query'],20)
        self.assertEqual(m['initial'],0)
        with self.assertRaises(ValueError):
            metrics(dict(history=state['history'][4:],decisions=[{}]*16))


if __name__=='__main__':
    unittest.main()
