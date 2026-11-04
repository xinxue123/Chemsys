"""Checks the supplemental interventions, not just their registration text."""
from pathlib import Path
from copy import deepcopy
import sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12 import study as s

class Interventions(unittest.TestCase):
    def test_budget_matrix_and_missing_cells(self):
        self.assertEqual(len(s.JOBS),71);self.assertEqual(len(set(s.JOBS)),71)
        self.assertEqual(sum(a=='bo' for t,z,a in s.JOBS),15)
        self.assertEqual(sum(a=='reasoning_bo' for t,z,a in s.JOBS),24)
        self.assertFalse(any((t,0,'memory') in s.JOBS for t in s.FOCUS))
    def test_fixed_filter_does_not_pause_on_adverse_outcomes(self):
        st=s.v12.engine.kf.initial();native=deepcopy(st)
        for t in range(1,6):
            st,e=s.no_kf_update(st,.95,.2,False,t)
            if native['status']!='retracted':native,_=s.NATIVE_KF(native,.95,.2,False,t)
            self.assertLess(e['Brier_gain'],0);self.assertEqual(e['after'],st)
        self.assertEqual(native['status'],'retracted')
        self.assertEqual((st['weight'],st['mean'],st['variance'],st['status']),(.4,0.,.25,'active'))
        self.assertEqual(st['local_updates'],5)
    def test_refresh_only_initial(self):
        self.assertTrue(s.no_refresh_event(None,{'edit_calls':[]},None,1)['allowed'])
        for t in (2,4,11,16):self.assertFalse(s.no_refresh_event(None,{'edit_calls':[1]},None,t)['allowed'])
    def test_initial_draws_and_native_function_unchanged(self):
        for task in s.TASKS:
            p=s.NATIVE_LOAD(ROOT,task,seed=0)
            for seed in (1,2):self.assertEqual(s.initialization(p,seed),s.NATIVE_LOAD(ROOT,task,seed=seed).initial_ids)
            self.assertEqual(len(set(s.initialization(p,3))),4)
        self.assertIs(s.v12.engine.kf.update,s.NATIVE_KF)
    def test_namespace_separation(self):
        self.assertEqual(len({s.base(ROOT,*j) for j in s.JOBS}),71)
        with self.assertRaises(ValueError):s.base(ROOT,'HPLC',0,'memory')

if __name__=='__main__':unittest.main()
