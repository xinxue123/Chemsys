from copy import deepcopy
import unittest
import numpy as np
from audit_knowledge_rank_v9 import independent_kf,rank
from knowledge_rank_v9 import kf,engine


class IndependentAuditTests(unittest.TestCase):
    def test_random_trajectories_match_independent_kf_and_lifecycle(self):
        rng=np.random.default_rng(37)
        for _ in range(80):
            state=kf.initial()
            for t in range(1,17):
                if state['status']=='retracted':break
                p,g=map(float,rng.uniform(0,1,2));success=bool(rng.integers(2))
                actual,event=kf.update(state,p,g,success,t);expected,e=independent_kf(state,p,g,success,t)
                for key in actual:
                    if type(actual[key]) is float:self.assertAlmostEqual(actual[key],expected[key],places=12)
                    else:self.assertEqual(actual[key],expected[key])
                self.assertEqual(event['reason'],e['reason']);state=actual

    def test_bad_kf_equation_is_detectable(self):
        state=kf.initial();expected,_=independent_kf(state,.9,.5,False,1)
        wrong=deepcopy(expected);wrong['variance']=state['variance']
        self.assertNotAlmostEqual(wrong['variance'],expected['variance'])


if __name__=='__main__':unittest.main()
