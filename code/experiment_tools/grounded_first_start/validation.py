"""Input/chronology qualification, never substitute experimental LLM replies."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest
from contextlib import redirect_stdout
import numpy as np
from . import proposal,study as s


def synthetic(conditions):
    return dict(summary='Synthetic QA fixture, not an experimental response.',first_choice=dict(
        conditions=deepcopy(conditions),rationale='Synthetic input/chronology test only.',
        key_assumptions=['No scientific claim is made by this synthetic fixture.'],
        forecast=dict(mean_raw=50.,sd_raw=20.),falsification='This is only a schema fixture.'))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s.initialize();cls.pools={t:s.pool_for(t) for t in s.native.TASKS}
        cls.control={t:s.read(s.COLD/'runs'/(s.SHORT[t]+'_bo')/'state.json') for t in cls.pools}

    def test_zero_data_background_and_MTBD(self):
        for task,pool in self.pools.items():
            public=proposal.public(pool)
            self.assertEqual(public['own_observations'],[]);self.assertEqual(public['experimental_budget']['observed'],0)
            self.assertFalse(public['GP_available']);self.assertFalse(public['qLogEI_available']);self.assertIsNone(public['incumbent'])
            self.assertEqual(pool.initial_ids,())
            self.assertNotIn('source_references',public['background'])
            for row in public['input_only_examples']:
                self.assertEqual(row['conditions'],pool.points[pool.ix[row['candidate_id']]])
                self.assertEqual(set(row),{'candidate_id','conditions'})
            if task=='Doyle_Buchwald_Hartwig':
                base=next(v for v in public['background']['inputs'] if v['name']=='Base')
                identity=base['chemical_identities']['CN1CCCN2C1=NCCC2']
                self.assertIn('MTBD',identity['name']);self.assertNotIn('DBU',identity['name'])
                self.assertEqual(identity['graph']['formula'],'C8H15N3')

    def test_exact_inputs_and_forecasts(self):
        for task,pool in self.pools.items():
            row=proposal.examples(pool)[0];raw=synthetic(row['conditions'])
            plan=proposal.validate_proposal(raw,pool)
            self.assertEqual(plan['exact_candidate_id'],row['candidate_id'])
            bad=deepcopy(raw);bad['first_choice']['forecast']['sd_raw']=0.
            with self.assertRaises(ValueError):proposal.validate_proposal(bad,pool)
            bad=deepcopy(raw);bad['first_choice']['conditions']['made_up_axis']=4
            with self.assertRaises(ValueError):proposal.validate_proposal(bad,pool)

    def test_physical_projection_and_actual_confirmation(self):
        for task,pool in self.pools.items():
            point=deepcopy(proposal.examples(pool)[0]['conditions'])
            continuous=[a for a,k in zip(pool.axes,pool.kinds) if k=='continuous']
            if not continuous:continue
            axis=continuous[0];bounds=next(v['declared_input_bounds'] for v in proposal.background(pool)['inputs'] if v['name']==axis)
            point[axis]=bounds[0]+.4321987*(bounds[1]-bounds[0])
            plan=proposal.validate_proposal(synthetic(point),pool);self.assertIsNone(plan['exact_candidate_id'])
            chosen=plan['neighbors'][0]
            for name,kind in zip(pool.axes,pool.kinds):
                if kind=='categorical':self.assertEqual(chosen['conditions'][name],point[name])
            raw=synthetic(point);raw['first_choice'].pop('conditions');raw['first_choice']['candidate_id']=chosen['candidate_id']
            valid=proposal.validate_confirmation(raw,pool,plan['neighbors'])
            self.assertEqual(valid['choice']['candidate_id'],chosen['candidate_id'])
            raw['first_choice']['candidate_id']='unlisted_candidate'
            with self.assertRaises(ValueError):proposal.validate_confirmation(raw,pool,plan['neighbors'])

    def test_no_GP_before_LLM_first_and_outcome_commit(self):
        pool=self.pools[s.native.TASKS[0]];point=pool.points[0];events=[]
        response=dict(raw=synthetic(point),plan=proposal.validate_proposal(synthetic(point),pool),request='SYNTHETIC_QA_ONLY')
        class DummyGP:
            def __init__(self,*args,**kwargs):
                self.counters=dict(operator_compiles_in_loop=0)
                if events[:2]!=['LLM','committed_observation']:raise AssertionError('GP constructed before LLM first point')
                events.append('GP_constructed')
            def context(self,history,q):
                js=[j for j,k in enumerate(pool.ids) if k not in {h['candidate_id'] for h in history}]
                n=len(pool.ids)
                return dict(bo_index=js[0],stats=dict(mu_y=np.zeros(n),sd_y=np.ones(n),sd_predictive_y=np.ones(n),fit_parameters={},fit_key='SYNTHETIC_QA'),
                    base_logei=np.zeros(n),available=~np.isin(pool.ids,[h['candidate_id'] for h in history]),tie_priority=np.arange(n),
                    stage='SYNTHETIC_QA',acquisition_meta=dict(synthetic=True))
            def release(self):pass
        mock_root=s.BASE/'preflight_mock';mock_root.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=mock_root) as tmp:
            folder=Path(tmp).resolve();self.assertTrue(folder.is_relative_to(s.BASE.resolve()))
            def reply(*args,**kwargs):
                self.assertFalse((folder/'rounds/01_feedback.json').exists());events.append('LLM');return deepcopy(response)
            def oracle(root,actual_pool,candidate):
                q=len([v for v in events if v=='committed_observation'])+1
                commit=s.read(folder/'rounds'/f'{q:02d}_commit.json')
                self.assertEqual(commit['decision']['selected_id'],candidate)
                events.append('committed_observation')
                return dict(candidate_id=candidate,y=.1,raw_value=10.)
            with (patch.object(s,'check',return_value=dict(protocol_sha256='SYNTHETIC_QA_ONLY')),
                patch.object(s,'folder_for',return_value=folder),patch.object(s.transport,'request',side_effect=reply),
                patch.object(s.native,'read_observation',side_effect=oracle),patch.object(s,'ColdGP',DummyGP),
                redirect_stdout(io.StringIO())):
                result=s.run_one(pool.task)
            self.assertEqual(result['status'],'COMPLETE')
            final=s.read(folder/'state.json');self.assertEqual(len(final['history']),20)
            self.assertFalse(final['decisions'][0]['decision']['GP_fitted'])
            self.assertEqual(events[:3],['LLM','committed_observation','GP_constructed'])

    def test_native_cold_recipe_matches_existing_BO_prefixes(self):
        diagnostics=[]
        for task,pool in self.pools.items():
            control=self.control[task]
            self.assertEqual(len(control['history']),20)
            gp=s.ColdGP(pool,s.BASE/'preflight_cache'/s.SHORT[task],seed=0)
            zero=gp.context([],1)
            self.assertFalse(zero['fitted_GP']);self.assertFalse(zero['acquisition_meta']['qLogEI_defined'])
            self.assertEqual(pool.ids[zero['bo_index']],control['history'][0]['candidate_id'])
            for n in (1,4):
                ctx=gp.context(control['history'][:n],n+1)
                self.assertEqual(pool.ids[ctx['bo_index']],control['history'][n]['candidate_id'])
                self.assertTrue(np.isfinite(ctx['stats']['sd_y']).all())
                self.assertEqual(gp.counters['operator_compiles_in_loop'],0)
                diagnostics.append(dict(task=task,observed_count=n,next_choice_matches_frozen_control=True,stage=ctx['stage']))
                gp.release()
        s.write(s.BASE/'preflight_numerics.json',diagnostics,replace=True)


def preflight():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value=dict(status='PASS_FIRST_CHOICE_PREFLIGHT' if result.wasSuccessful() else 'FAIL_FIRST_CHOICE_PREFLIGHT',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),new_experimental_queries=0,
        actual_LLM_calls=0,test_output=stream.getvalue(),source_sha256=s.sha(__file__))
    s.write(s.BASE/'preflight.json',value,replace=True)
    if not result.wasSuccessful():print(stream.getvalue());raise RuntimeError('Preflight failed')
    return {k:v for k,v in value.items() if k!='test_output'}
