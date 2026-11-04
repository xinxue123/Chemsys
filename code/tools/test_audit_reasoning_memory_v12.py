"""Boundary checks for the independent audit, without data or model calls."""
from copy import deepcopy
import unittest
import numpy as np
from audit_reasoning_memory_v12 import independent_event,independent_menu,audit_birth
from knowledge_information_v10.tests.test_pipeline import fixture,context
from reasoning_memory_v12.tests.test_pipeline import planned,ROOT
from reasoning_memory_v12 import engine


class AuditTests(unittest.TestCase):
    def test_reserved_late_call_requires_actual_evidence(self):
        pool,state=fixture();state['edit_calls']=[1,4,7]
        self.assertFalse(independent_event(pool,state,[],11)['allowed'])
        state['decisions']=[dict(selected_id='p5',incumbent_before_y=.4,incumbent_after_y=.4,KF_feedback=[],memory_references=[]) for _ in range(3)]
        self.assertEqual(independent_event(pool,state,[],11),dict(allowed=True,reasons=['stagnation']))

    def test_ablation_does_not_change_native_top_choice(self):
        pool,state=fixture();state.update(arm='category',category_menu_insertions=0)
        keys=['p0','p1','p3','p5','p7','p9','p11'];state['history']=[dict(candidate_id=k,y=.2,raw_value=20.) for k in keys]
        state['decisions']=[dict(selected_id=k) for k in keys[-3:]];ctx=context(pool,state)
        ctx['base_logei'][np.array([f[1]=='B' for f in pool.features])]-=2
        a={k:ctx[k] for k in ('available','base_logei','tie_priority')}
        original,menu,ids,cat=independent_menu(pool,state,a,4)
        self.assertTrue(cat['inserted']);self.assertEqual(original[0],menu[0]);self.assertEqual(len(set(ids)),5)
        self.assertEqual(len(set(original)^set(menu)),2)

    def test_birth_audit_rejects_negative_numeric_replay(self):
        pool,state,ctx,ids,m,r=planned();c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        self.assertEqual(audit_birth(pool,state,c['hypotheses_at_commit'],1),0)
        hs=deepcopy(c['hypotheses_at_commit']);hs[0]['kf']['mean']=-.01
        with self.assertRaises(AssertionError):audit_birth(pool,state,hs,1)


if __name__=='__main__':unittest.main()
