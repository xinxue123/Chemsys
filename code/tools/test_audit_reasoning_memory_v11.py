from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import audit_reasoning_memory_v11 as audit
from reasoning_memory_v11.tests.test_pipeline import planned,action,ROOT
from reasoning_memory_v11 import llm,engine


class AuditorTests(unittest.TestCase):
    def test_confidence_priority_then_original_order_not_candidate_id(self):
        insight=dict(hypotheses=[dict(confidence='low',parameter_sets=[dict(candidate_id='a')]),
            dict(confidence='HIGH',parameter_sets=[dict(candidate_id='z'),dict(candidate_id='b')]),
            dict(confidence='high',parameter_sets=[dict(candidate_id='c')])])
        self.assertEqual(audit.official_winner(insight),('z',1))

    def test_inheritance_audit_rejects_reset_negative_evidence(self):
        pool,state,ctx,ids,m,r=planned();c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.1,raw_value=10.))
        revise=action();revise.update(op='revise',claim='revision with inherited caution')
        hs=llm.validate_memory(dict(summary='revision',actions=[revise]),pool,state,ids,4)['hypotheses']
        self.assertEqual(audit.audit_birth(pool,state,hs,4),1)
        forged=deepcopy(hs);forged[-1]['inherited_negative']=[]
        with self.assertRaises(AssertionError):audit.audit_birth(pool,state,forged,4)
        forged=deepcopy(hs);forged[-1]['kf']['status']='active'
        with self.assertRaises(AssertionError):audit.audit_birth(pool,state,forged,4)

    def test_request_audit_rejects_future_observation_even_if_rehashed(self):
        protocol=dict(model='test',reasoning_effort='medium');public=dict(own_observations=[dict(candidate_id='known',objective=10)])
        malicious=deepcopy(public);malicious['own_observations'].append(dict(candidate_id='future',objective=99))
        request=dict(system='test',public=malicious,**protocol);key=audit.digest(request)
        with patch.object(audit.study.old,'read',side_effect=[request,dict(context_sha256=key,raw={})]):
            with self.assertRaises(AssertionError):audit.check_request(Path('unused'),key,'test',public,{},protocol)


if __name__=='__main__':unittest.main()
