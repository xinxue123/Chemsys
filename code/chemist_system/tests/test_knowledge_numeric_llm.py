"""V8 public prompt/schema/cache checks with a fake client only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chem.benchmarks import knowledge_numeric_llm as llm
from chem.benchmarks import knowledge_actionable_llm as previous
from chem.benchmarks import knowledge_openai_client as api
from test_knowledge_numeric_policy import action,create,proposal


def context(pool):
    values=np.array([-.04,-2.,0.,-3.,-.2,-3.,-.01])
    return dict(bo_index=2,epsilon=1.,eligible=np.array([1,0,1,0,1,0,1],dtype=bool),base_logei=values)


class NumericLLMTests(unittest.TestCase):
    def test_new_schema_requires_real_numerical_prior_or_explicit_null(self):
        sample=proposal(action());self.assertEqual(llm.validate(sample),sample)
        sample['actions'][0]['numeric_prior']=None;llm.validate(sample)
        del sample['actions'][0]['numeric_prior']
        with self.assertRaises(ValueError):llm.validate(sample)
        for bad in (0.,True,float('nan'),float('inf')):
            sample=proposal(action());sample['actions'][0]['numeric_prior']['delta_sd_raw']=bad
            with self.assertRaises(ValueError):llm.validate(sample)

    def test_retain_and_retract_cannot_silently_change_prior(self):
        for op in ('retain','retract'):
            a=action(op=op,id='H001',plus_id=None,minus_id=None,focus={},background={},numeric_prior=None)
            llm.validate(proposal(a));a['numeric_prior']=dict(delta_mean_raw=1.,delta_sd_raw=1.)
            with self.assertRaises(ValueError):llm.validate(proposal(a))

    def test_payload_whitelists_history_ledger_and_budget_flags_only(self):
        class Forbidden:
            def __getitem__(self,key):raise AssertionError('Private prediction/oracle accessed')
            def __deepcopy__(self,memo):raise AssertionError('Private prediction/oracle copied')
        pool,past,menu,hs=create();ctx=context(pool);ctx['stats']=Forbidden()
        hs[0]['birth']=Forbidden()
        hs[0]['ledger']['events']=[dict(event_id='seen',query_round=1,point_id='a_plus',reference_id='a_ref',
            gp_gain=.2,null_gain=.1,mean_gain=.05,null_mean_gain=.02,prediction=Forbidden())]
        state=dict(history=past,hypotheses=hs,verification_used=0,unqueried_targets=Forbidden())
        safe_domain=dict(description='Synthetic domain',source_references=['private/path'])
        with patch.object(llm,'task_context',return_value=safe_domain):
            payload=llm.public_payload(pool,state,2,menu,[],context=ctx)
        serialized=json.dumps(payload,allow_nan=False)
        for forbidden in ('source_references','unqueried_targets','gp_mean','beta_mean','birth_prefix','GP_fit'):
            self.assertNotIn(forbidden,serialized)
        self.assertEqual(payload['knowledge_ledger'][0]['numeric_prior'],dict(delta_mean_raw=15.,delta_sd_raw=8.))
        self.assertEqual(payload['knowledge_ledger'][0]['tested_predictions'][0]['null_mean_gain'],.02)
        flags={r['point_id']:r['passes_narrow_cost_gate'] for r in payload['proposal_support']['candidate_support']}
        self.assertEqual(flags,{'a_plus':True,'b_plus':True,'c_plus':False,'extra':True})
        self.assertTrue(payload['proposal_support']['optional_window_open'])

    def test_window_and_shared_cost_budget_match_declared_settings(self):
        pool,past,menu,hs=create();ctx=context(pool);ctx['epsilon']=.02
        state=dict(history=past,hypotheses=hs,verification_used=1)
        with patch.object(llm,'task_context',return_value={}):
            payload=llm.public_payload(pool,state,11,menu,[],context=ctx)
        flags={r['point_id']:r['passes_narrow_cost_gate'] for r in payload['proposal_support']['candidate_support']}
        self.assertFalse(flags['a_plus']);self.assertTrue(flags['extra'])
        self.assertFalse(payload['proposal_support']['optional_window_open'])

    def test_new_prompt_and_full_context_have_separate_cache_identity(self):
        class Client:
            calls=0
            def complete(self,system,public,**kwargs):
                self.calls+=1
                return dict(output_text=json.dumps(proposal()),resolved_model=api.MODEL)
        client=Client();workspace=Path(__file__).resolve().parents[3]
        temporary_root=(workspace/'runtime/knowledge_loop/test_temp').resolve()
        if not temporary_root.is_relative_to(workspace):raise ValueError('Test path escaped workspace')
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            old=previous.request(client,{'own_observations':[]},folder)
            new=llm.request(client,{'own_observations':[]},folder)
            repeated=llm.request(client,{'own_observations':[]},folder)
        self.assertNotEqual(old['context_sha256'],new['context_sha256'])
        self.assertEqual(new,repeated);self.assertEqual(client.calls,2)
        self.assertIs(llm.make_menu,previous.make_menu)
        self.assertIn('NEGATIVE residual',llm.SYSTEM)
        self.assertIn('not experimental noise',llm.SYSTEM)
        self.assertIn('four separate safety accounts',llm.SYSTEM)


if __name__=='__main__':unittest.main()
