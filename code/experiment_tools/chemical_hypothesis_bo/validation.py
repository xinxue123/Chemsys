"""Numerical preflight using input-only fixtures and frozen initial observations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import math
import unittest
import numpy as np
from . import background,hypotheses,experts,policy,prompts


def fixture_note(pool):
    """Synthetic scope fixture, never passed off as an LLM answer."""
    axis=next((a for a,k in zip(pool.axes,pool.kinds) if k=='continuous'),pool.axes[0])
    if pool.kinds[pool.axes.index(axis)]=='continuous':
        domain=next(a['declared_input_bounds'] for a in background.context(pool)['inputs'] if a['name']==axis)
        mid=float(np.mean(domain));preferred={axis:dict(bounds=[mid,domain[1]])};comparison={axis:dict(bounds=[domain[0],mid])}
    else:
        levels=sorted({p[axis] for p in pool.points});mid=len(levels)//2
        preferred={axis:dict(allowed=levels[:mid])};comparison={axis:dict(allowed=levels[mid:])}
    return dict(claim='Synthetic preflight contrast, not a scientific prior.',rationale='Numerical boundary fixture only.',
        basis='mechanistic_prior',focus=[axis],preferred=preferred,comparison=comparison,
        supporting_observation_ids=[],falsification='A preflight fixture has no claimed scientific validity.')


class GroundedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from . import study
        cls.s=study
        study.initialize()
        cls.prior=study.previous.check(True)
        cls.pools={t:study.native.load_task(study.ROOT,t,seed=0) for t in study.native.TASKS}
        cls.history={t:study.read(study.ROOT/cls.prior['controls'][t]['path'])['history'][:4] for t in cls.pools}

    def test_background_has_real_identities_and_units(self):
        totals=[]
        for task,pool in self.pools.items():
            card=background.context(pool);self.assertNotIn('task',card);self.assertNotIn('source_references',card)
            self.assertEqual(len(card['inputs']),len(pool.axes))
            if pool.group=='P0':
                self.assertEqual(len(card['fixed_reactants']),2)
                catalyst=next(i for i in card['inputs'] if i['name']=='catalyst')
                self.assertEqual(set(catalyst['chemical_identities']),set(catalyst['levels']))
                self.assertEqual(card['fixed_conditions']['base'],'DBU')
                self.assertEqual(card['fixed_conditions']['medium'],'THF/water 5:1 by volume')
            elif pool.group=='P1':
                for axis in card['inputs']:
                    for value,identity in axis['chemical_identities'].items():
                        self.assertEqual(value,identity['SMILES'])
                        self.assertTrue(identity['graph']['canonical_SMILES'])
            if task=='Doyle_Buchwald_Hartwig':self.assertEqual(card['fixed_reactants'][0]['name'],'4-methylaniline (p-toluidine)')
            if task=='SnAr':self.assertIn('STY',card['process_interpretation'])
            if task=='HPLC':self.assertIn('peak area',card['process_interpretation'])
            totals.append(dict(task=task,background_bytes=len(json.dumps(card)),axes=list(pool.axes),unit=pool.unit))
        self.s.write(self.s.BASE/'preflight_backgrounds.json',totals,replace=True)

    def test_structural_identity_and_physical_units(self):
        for task,pool in self.pools.items():
            note=fixture_note(pool)
            raw=dict(summary='Synthetic preflight only.',hypotheses=[note])
            plan=hypotheses.validate(raw,pool,self.history[task]);self.assertEqual(len(plan['notes']),1)
            changed=deepcopy(note);changed['claim']='Different wording with unchanged scopes.'
            for scope in (changed['preferred'],changed['comparison']):
                for v in scope.values():
                    if 'allowed' in v:v['allowed'].reverse()
            self.assertEqual(hypotheses.structural_key(note),hypotheses.structural_key(changed))
            bad=deepcopy(raw);bad['hypotheses'][0]['supporting_observation_ids']=['not_observed']
            with self.assertRaises(ValueError):hypotheses.validate(bad,pool,self.history[task])
            if pool.group=='P0':
                with self.assertRaises(ValueError):hypotheses.region(pool,{'temperature':dict(bounds=[0.,1.])})
                with self.assertRaises(ValueError):hypotheses.region(pool,{'catalyst':dict(allowed=[0])})
            with self.assertRaises(ValueError):hypotheses.region(pool,{pool.axes[0]:[]})

    def test_prompt_only_supplies_own_raw_observations(self):
        for task,pool in self.pools.items():
            packet=prompts.public(pool,self.history[task],credit=dict(base=0.))
            self.assertEqual(len(packet['own_observations']),4)
            for row,observation in zip(packet['own_observations'],self.history[task]):
                self.assertEqual(row['objective'],observation['raw_value'])
                self.assertEqual(row['objective_unit'],pool.unit)
                self.assertEqual(row['conditions'],pool.points[pool.ix[observation['candidate_id']]])
            encoded=json.dumps(packet)
            for text in ('base_logei','mu_y','sd_y','BO_reference_id','fullTextXML','source_image','outcome_tables','kinetic_constants'):
                self.assertNotIn(text,encoded)

    def test_prediction_credit_and_structural_retention(self):
        decision=dict(query=1,arm='adaptive',log_weights_before={'base':math.log(.5),'chem_x':math.log(.5)},
            predictions=[dict(id='base',mu_y=1.,sd_observation_y=1.,mean_shift_y=0.),
                dict(id='chem_x',mu_y=-1.,sd_observation_y=1.,mean_shift_y=-2.)])
        frozen=deepcopy(decision);obs=dict(candidate_id='observed',y=-1.,raw_value=-1.)
        fb=policy.feedback(decision,obs,dict(temperature=75.))
        self.assertEqual(decision,frozen);self.assertGreater(fb['log_weights_after']['chem_x'],fb['log_weights_after']['base'])
        self.assertTrue(fb['ledger'][0]['informative'])
        same=policy.edit_weights(fb['log_weights_after'],[dict(id='chem_x')])
        for k in same:self.assertAlmostEqual(same[k],fb['log_weights_after'][k])
        updated=policy.edit_weights(fb['log_weights_after'],[dict(id='chem_new')])
        self.assertAlmostEqual(math.exp(updated['chem_new']),.2)
        self.assertAlmostEqual(math.exp(updated['base']),.8)
        decision['arm']='fixed'
        self.assertEqual(policy.feedback(decision,obs,{})['log_weights_after'],decision['log_weights_before'])

    def test_mixture_uses_expected_improvement_and_native_budget(self):
        context=dict(bo_index=0,eligible=np.array([True,True,False]),
            base_logei=np.array([0.,-.1,-20.]),tie_priority=np.array([2,1,0]))
        values=dict(mixture=np.array([0.,1.,100.]))
        self.assertEqual(policy.choose(context,values,1),1)
        self.assertEqual(policy.choose(context,values,16),0)
        values['mixture']=context['base_logei']
        self.assertEqual(policy.choose(context,values,1),0)
        context['eligible'][1]=False
        self.assertEqual(policy.choose(context,dict(mixture=np.array([0.,100.,1000.])),1),0)

    def test_all_native_families_replay_and_field_bound(self):
        diagnostics=[]
        for task,pool in self.pools.items():
            history=self.history[task]
            gp=self.s.native.GPAdapter(pool,self.s.BASE/'preflight_cache'/self.s.SHORT[task],seed=0)
            context=gp.context(history,1);statehash=self.s.previous.models.state_hash(context['stats']['model'])
            native_mu=context['stats']['mu_y'].copy();native_sd=context['stats']['sd_y'].copy()
            empty=experts.evaluate(gp,context,[],dict(base=0.))
            np.testing.assert_array_equal(empty['mixture'],context['base_logei'])
            self.assertEqual(policy.choose(context,empty,1),context['bo_index'])
            notes=hypotheses.validate(dict(summary='Preflight fixture.',hypotheses=[fixture_note(pool)]),pool,history)['notes']
            values=experts.evaluate(gp,context,notes,policy.initial(notes))
            self.assertLessEqual(values['native_replay_error'],2e-8)
            shift=values['fields'][notes[0]['id']]
            self.assertLessEqual(float(np.max(np.abs(shift)-.5*native_sd)),1e-10)
            np.testing.assert_array_equal(context['stats']['mu_y'],native_mu)
            np.testing.assert_array_equal(context['stats']['sd_y'],native_sd)
            self.assertEqual(self.s.previous.models.state_hash(context['stats']['model']),statehash)
            j=policy.choose(context,values,1);self.assertTrue(context['eligible'][j])
            self.assertEqual(gp.counters['operator_compiles_in_loop'],0)
            diagnostics.append(dict(task=task,native_qLogEI_error=values['native_replay_error'],
                native_model_unchanged=True,max_abs_shift_y=float(abs(shift).max()),
                bounded_by_half_latent_SD=True,operator_compiles_in_loop=0))
            gp.release()
        self.s.write(self.s.BASE/'preflight_numerics.json',diagnostics,replace=True)


def preflight():
    from . import study
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(GroundedChecks)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    value=dict(status='PASS_GROUNDED_PREFLIGHT' if result.wasSuccessful() else 'FAIL_GROUNDED_PREFLIGHT',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        new_experimental_queries=0,new_LLM_calls=0,test_output=stream.getvalue(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    study.write(study.BASE/'preflight.json',value,replace=True)
    if not result.wasSuccessful():
        print(stream.getvalue());raise RuntimeError('Grounded chemical knowledge preflight failed')
    return {k:v for k,v in value.items() if k!='test_output'}
