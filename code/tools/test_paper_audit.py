"""Exercise audit adapters on real frozen traces and removal transitions offline."""
from copy import deepcopy
import unittest
from paper_audit import (ROOT,study,audit_llm,control_run,request_and_prefix_evidence,
                        independent_fixed_kf,independent_initial_edit)

class AuditAdapters(unittest.TestCase):
    def test_native_v12_and_reasoning_bo_replay(self):
        task='Reizman_Suzuki_Case_4';pool=study.NATIVE_LOAD(ROOT,task,seed=0)
        control=control_run(ROOT,task,0)
        for arm,module in [('memory',study.v12),('reasoning_bo',study.v9)]:
            p=module.check(ROOT);run=module.directory(ROOT)/'runs'/f'{task}_{arm}'
            r=audit_llm(ROOT,run,p,pool,control,arm)
            self.assertEqual(r['rounds'],16)
        # Mutation of the registered frozen algorithm is neither necessary nor allowed.
        self.assertIs(study.v12.engine.kf.update,study.NATIVE_KF)

    def test_existing_alternative_initialization_control_routing(self):
        task='HPLC';seed=3;p=study.seeds.check(ROOT)
        pool=study.seeds.registered_pool(ROOT,task,seed,p)
        run=study.seeds.pair_base(ROOT,task,seed)/'runs'/f'{task}_memory'
        r=audit_llm(ROOT,run,p,pool,control_run(ROOT,task,seed),'memory')
        self.assertEqual(r['rounds'],16)

    def test_independent_removal_rule_matches_observed_transition(self):
        state=study.v12.engine.kf.initial()
        for t in range(1,6):
            expected,measurement=independent_fixed_kf(state,.95,.2,False,t)
            actual,event=study.no_kf_update(state,.95,.2,False,t)
            self.assertEqual(actual,expected)
            for key,value in measurement.items():
                if isinstance(value,float):self.assertAlmostEqual(value,event[key],places=12)
                else:self.assertEqual(value,event[key])
            self.assertEqual((expected['status'],expected['mean'],expected['variance'],expected['weight']),('active',0.,.25,.4))
            state=expected
        # A native filter transition must be rejected for this removal arm.
        changed,_=study.NATIVE_KF(deepcopy(state),.95,.2,False,6)
        frozen,_=independent_fixed_kf(state,.95,.2,False,6)
        self.assertNotEqual(changed,frozen)

    def test_no_refresh_and_original_control_without_request_scanning(self):
        self.assertTrue(independent_initial_edit(None,{'edit_calls':[]},None,1)['allowed'])
        for t in (4,11,16):
            self.assertEqual(independent_initial_edit(None,{'edit_calls':[1]},None,t),
                             dict(allowed=False,reasons=['refresh_disabled_ablation']))
        contexts,prefixes,responses=request_and_prefix_evidence(ROOT,control_run(ROOT,'HPLC',0),'bo')
        self.assertEqual((len(contexts),len(prefixes),len(responses)),(0,16,0))

if __name__=='__main__':unittest.main()
