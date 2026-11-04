"""Proposal-support interface tests: synthetic inputs, no model or oracle."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import knowledge_actionable_llm as new
from chem.benchmarks import knowledge_transfer_llm as old
from chem.benchmarks import knowledge_openai_client as api


def fixture():
    ids=['known','base','near','outside']
    pool=SimpleNamespace(ids=ids,ix={k:i for i,k in enumerate(ids)},
        axes=['temperature','catalyst'],kinds=['continuous','categorical'],
        features=np.array([[.2,'A'],[.4,'A'],[.5,'A'],[.6,'B']],dtype=object))
    state=dict(history=[dict(candidate_id='known',y=.2,raw_value=20.)],verification_used=0)
    context=dict(bo_index=1,epsilon=1.,eligible=np.array([False,True,True,True]),
        base_logei=np.array([-.2,0.,-.04,-.2]))
    return pool,state,context


def payload(pool,state,context,t=1):
    with patch.object(old,'public_payload',return_value={'existing_payload':'unchanged'}):
        return new.public_payload(pool,state,t,['base','near','outside'],[],context=context)


class ActionablePromptTests(unittest.TestCase):
    def test_narrow_gate_marks_base_without_identifying_it_as_bo(self):
        pool,state,context=fixture()
        p=payload(pool,state,context)
        s=p['proposal_support'];rows=s['candidate_support']
        self.assertEqual([r['passes_narrow_cost_gate'] for r in rows],[True,True,False])
        self.assertTrue(s['optional_window_open'])
        self.assertEqual(p['existing_payload'],'unchanged')
        self.assertFalse(any('bo_index' in r or 'base_logei' in r or 'GP_mean' in r for r in rows))
        self.assertIn('required_focus_axes',rows[-1]['references'][0])

    def test_time_and_quota_are_separate_from_candidate_cost(self):
        pool,state,context=fixture()
        late=payload(pool,state,context,t=11)['proposal_support']
        self.assertFalse(late['optional_window_open'])
        self.assertTrue(late['candidate_support'][0]['passes_narrow_cost_gate'])
        state['verification_used']=1
        used=payload(pool,state,context,t=1)['proposal_support']
        self.assertFalse(used['optional_window_open'])
        self.assertTrue(used['candidate_support'][1]['passes_narrow_cost_gate'])

    def test_actual_smaller_shared_epsilon_is_respected(self):
        pool,state,context=fixture();context['epsilon']=.02
        rows=payload(pool,state,context)['proposal_support']['candidate_support']
        self.assertEqual([r['passes_narrow_cost_gate'] for r in rows],[True,False,False])

    def test_no_gp_posterior_or_external_fields_are_inspected_or_forwarded(self):
        class Forbidden:
            def __getitem__(self,key):
                raise AssertionError('Unnecessary posterior access')
        pool,state,context=fixture()
        context['stats']=Forbidden();state['unqueried_targets']=Forbidden()
        snapshot=deepcopy(state['history'])
        value=payload(pool,state,context)
        self.assertEqual(state['history'],snapshot)
        self.assertNotIn('stats',value)
        self.assertNotIn('unqueried_targets',value)
        json.dumps(value,allow_nan=False)

    def test_missing_context_is_an_explicit_error(self):
        pool,state,context=fixture()
        with self.assertRaises(TypeError):
            new.public_payload(pool,state,1,['base'],[])

    def test_unchanged_action_schema_and_prompt_disclaimer(self):
        sample=dict(actions=[],summary='No defensible local hypothesis.')
        self.assertEqual(new.validate(sample),old.validate(sample))
        self.assertTrue(new.SYSTEM.startswith(old.SYSTEM))
        self.assertIn('NOT a chemistry-quality label',new.SYSTEM)
        self.assertIn('not a forced experiment',new.SYSTEM)
        self.assertIs(new.make_menu,old.make_menu)

    def test_cache_separates_old_prompt_and_reuses_identical_new_context(self):
        class Client:
            calls=0
            def complete(self,system,public,**kwargs):
                self.calls+=1
                return dict(output_text=json.dumps(dict(actions=[],summary='Abstain.')),
                    resolved_model=api.MODEL)
        client=Client()
        # The desktop Windows sandbox can deny atomic replace in its redirected
        # package TEMP. The real cache lives in the workspace; exercise it here.
        workspace=Path(__file__).resolve().parents[3]
        temporary_root=(workspace/'runtime/knowledge_loop/test_temp').resolve()
        if not temporary_root.is_relative_to(workspace):
            raise ValueError('Test temporary directory escaped the workspace')
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            a=old.request(client,{'own_observations':[]},directory)
            b=new.request(client,{'own_observations':[]},directory)
            c=new.request(client,{'own_observations':[]},directory)
        self.assertNotEqual(a['context_sha256'],b['context_sha256'])
        self.assertEqual(b,c)
        self.assertEqual(client.calls,2)


if __name__=='__main__':
    unittest.main()
