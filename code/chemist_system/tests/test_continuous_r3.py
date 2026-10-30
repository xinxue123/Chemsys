"""R3 guards: independent GP review, local support, causal timing and recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from chem.benchmarks import continuous_r3_engine as e
from chem.benchmarks import continuous_r3_study as s
from chem.benchmarks import continuous_r3_llm as llm


def fixture():
    z=np.linspace(0,1,64).reshape(-1,1)
    pool={'task':'fixture','ids':[f'p{i}' for i in range(64)],'axes':['x'],'z':z.tolist(),
        'points':[{'x':float(x[0])} for x in z],'objective_scale':1.,'objective_unit':'unit',
        'bounds':{'x':[0.,1.]},'description':'fixture','objective_name':'objective'}
    state={'task':'fixture','protocol_sha256':'test','initial_knowledge':{},'legacy_context':{},
           'history':[{'candidate_id':'p0','y':.2}],'decisions':[],
           'hypotheses':{},'authorization_events':[]}
    stats={'mu_y':np.full(64,.2),'sd_y':np.full(64,.2),'sd_predictive_y':np.full(64,.2),
           'mu_std':np.full(64,.2),'sd_std':np.full(64,.2),'std':1.,'mean':0.,'best_std':.4}
    class FakeCache:
        def __init__(self):
            self.pool=pool;self.ids=pool['ids'];self.ix={key:j for j,key in enumerate(self.ids)}
            self.X=e.base.torch.tensor(pool['z'],dtype=e.base.torch.double)
        def score(self,history,shift=None):
            shift=np.zeros(64) if shift is None else shift
            return e.base.acquisition(stats['mu_std'],stats['sd_std'],shift,.4,0),{'mc_seed':0}
    cache=FakeCache();base,_=cache.score(state['history'])
    d={'base':base,'base_meta':{'mc_seed':0},'stats':stats,'available':np.ones(64,dtype=bool),
       'eligible':np.ones(64,dtype=bool),'reference':0,'information_proxy':np.zeros(64),
       'tie_priority':np.arange(64),'budget':2.}
    plan={'action':'propose','candidate_id':'p31','prediction':.6,
          'hypothesis':{'statement':'approximately constant local response','focus_axes':['x'],
                        'falsification':'a lower measured local response'},
          'withdraw_ids':[],'reason':'test a local response hypothesis','support':[1]}
    return pool,state,cache,d,plan


def observe(state,cache,d,plan,t,y):
    decision=e.decide(cache,state,d,t,{'plan':plan},'context_'+str(t))
    # A physical point has already been chosen; this helper tests frozen feedback.
    outcome=e.feedback(state,d,decision,t,y,cache.pool['ids'][decision['selected']])
    return decision,outcome


class ContinuousR3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):e.base.warm()

    def test_unvalidated_model_cannot_change_any_GP_mean(self):
        pool,state,cache,d,plan=fixture()
        decision=e.decide(cache,state,d,1,{'plan':plan},'context')
        self.assertEqual(decision['mode'],'GP_reviewed_LLM_trial')
        self.assertTrue(np.all(decision['guidance']['shift_y']==0.))
        self.assertTrue(np.all(decision['guidance']['owner']==-1))

    def test_base_GP_rejects_high_LLM_forecast_outside_gate(self):
        pool,state,cache,d,plan=fixture();d['eligible'][31]=False
        plan['prediction']=2.
        decision=e.decide(cache,state,d,1,{'plan':plan},'context')
        self.assertFalse(decision['GP_review_accepted']);self.assertEqual(decision['selected'],0)
        self.assertEqual(decision['mode'],'BO_rejected_LLM_proposal')

    def test_positive_feedback_authorizes_only_the_following_round(self):
        pool,state,cache,d,plan=fixture();decision,outcome=observe(state,cache,d,plan,1,.5)
        self.assertIsNone(outcome['actual_local_update'])
        self.assertEqual(outcome['actual_paired_SE_gain'],0.)
        key=decision['registered_hypothesis'];self.assertEqual(state['hypotheses'][key]['status'],'unconfirmed')
        with self.assertRaises(AssertionError):e.authorize(state,1)
        events=e.authorize(state,2)
        self.assertEqual(events[0]['hypothesis_id'],key);self.assertEqual(state['hypotheses'][key]['status'],'probation')
        self.assertLessEqual(state['hypotheses'][key]['actual_bank']['mean'],.25)
        guided=e.guidance(cache,state,d,2)
        self.assertTrue(np.any(guided['shift_y']!=0.))
        self.assertLessEqual(float(np.max(abs(guided['shift_y']))),.25*.2+1e-12)

    def test_single_failure_continues_shadow_and_recovers_without_global_pause(self):
        pool,state,cache,d,plan=fixture();decision,outcome=observe(state,cache,d,plan,1,.5)
        key=decision['registered_hypothesis'];e.authorize(state,2)
        decision,outcome=observe(state,cache,d,plan,2,.05)
        self.assertIsNotNone(outcome['actual_local_update'])
        self.assertLess(outcome['actual_paired_SE_gain'],0.)
        self.assertEqual(state['hypotheses'][key]['status'],'quarantined')
        failed_mean=state['hypotheses'][key]['actual_bank']['mean']
        self.assertLessEqual(failed_mean,.25)
        e.authorize(state,3);observe(state,cache,d,plan,3,.5)
        self.assertEqual(len(state['hypotheses'][key]['shadow_records']),3)
        self.assertEqual(e.authorize(state,4),[])
        observe(state,cache,d,plan,4,.5);events=e.authorize(state,5)
        self.assertTrue(events);self.assertEqual(state['hypotheses'][key]['status'],'supported')
        self.assertNotIn('active_suspended',state)

    def test_renaming_same_model_does_not_reset_negative_records(self):
        pool,state,cache,d,plan=fixture();a=e.register(pool,state,1,plan,'a')
        state['hypotheses'][a]['last_actual_failure_query']=1
        state['hypotheses'][a]['shadow_records'].append({'query':1,'informative':False})
        changed=deepcopy(plan);changed['hypothesis']['statement']='new wording'
        b=e.register(pool,state,2,changed,'b')
        self.assertEqual(a,b);self.assertEqual(state['hypotheses'][b]['last_actual_failure_query'],1)
        self.assertEqual(len(state['hypotheses'][b]['shadow_records']),1)

    def test_local_effect_is_zero_outside_scope_and_id_independent(self):
        pool,state,cache,d,plan=fixture();key=e.register(pool,state,1,plan,'context')
        h=state['hypotheses'][key];feature,weight=e.features(pool,h,d['stats'])
        center=np.array(h['scope']['center']);r=h['scope']['radius_RMS_normalized']
        distance=np.sqrt(np.mean((np.array(pool['z'])-center)**2,axis=1))
        self.assertTrue(np.all(feature[distance>=r]==0.))
        renamed=deepcopy(pool);renamed['ids']=list(reversed(pool['ids']))
        self.assertTrue(np.array_equal(e.scope(pool,31)[1],e.scope(renamed,31)[1]))

    def test_weak_or_out_of_scope_checks_do_not_create_negative_veto(self):
        pool,state,cache,d,plan=fixture();key=e.register(pool,state,1,plan,'context')
        g=e.guidance(cache,state,d,1)
        d['stats']['mu_y'][:]=.6
        g=e.guidance(cache,state,d,1);decision={'guidance':g,'selected':31}
        result=e.feedback(state,d,decision,1,.1,'p31')
        self.assertFalse(result['shadow_checks'][0]['informative'])
        self.assertEqual(state['hypotheses'][key]['last_actual_failure_query'],-1)

    def test_withdrawal_immediately_disables_only_named_model(self):
        pool,state,cache,d,plan=fixture();decision,_=observe(state,cache,d,plan,1,.5)
        key=decision['registered_hypothesis'];e.authorize(state,2)
        abstain={'action':'abstain','candidate_id':None,'prediction':None,'hypothesis':{},
                 'withdraw_ids':[key],'reason':'withdraw this unsupported model','support':[1]}
        decision=e.decide(cache,state,d,2,{'plan':abstain},'withdraw')
        self.assertEqual(state['hypotheses'][key]['status'],'withdrawn')
        self.assertTrue(np.all(decision['guidance']['shift_y']==0.))

    def test_card_contains_BO_and_schema_rejects_undisplayed_candidate(self):
        pool,state,cache,d,plan=fixture();card=e.candidate_card(cache,state,d)
        self.assertIn(pool['ids'][0],{r['candidate_id'] for r in card})
        bad=deepcopy(plan);bad['candidate_id']='not-a-menu-point'
        with self.assertRaises(ValueError):llm.validate(bad,pool,state,card)

    def test_request_cache_roundtrip_is_canonical_and_has_no_network_call(self):
        pool,state,cache,d,plan=fixture();pool['task']='SnAr'
        card=e.candidate_card(cache,state,d)
        abstain={'action':'abstain','candidate_id':None,'prediction':None,'hypothesis':{},
                 'withdraw_ids':[],'reason':'no additional local information','support':[1]}
        self.assertEqual(llm.validate(abstain,pool,state,card),abstain)
        root=Path('outputs/r3_behavioral_tmp').resolve();root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(root))
            with patch.object(e,'ROOT',Path(tmp)),patch.object(llm.transport,'get',return_value={'plan':abstain}) as mocked:
                first=llm.request(None,pool,state,d,5,card);second=llm.request(None,pool,state,d,5,card)
                self.assertEqual(first,second);self.assertEqual(mocked.call_count,2)

    def test_real_protected_prefix_fresh_plan_feedback_and_next_prompt_serialize(self):
        task='SnAr';state=s.initial(task,{'protocol_sha256':'test'})
        root=Path('outputs/r3_behavioral_tmp').resolve();root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp,redirect_stdout(io.StringIO()):
            self.assertTrue(Path(tmp).resolve().is_relative_to(root))
            for t in range(1,5):s.bootstrap(state,Path(tmp)/'state.json',t)
        pool=s.original.pool(task);cache=e.Cache(pool)
        raw={'action':'abstain','candidate_id':None,'prediction':None,'hypothesis':{},
             'withdraw_ids':[],'reason':'pipeline fixture without a new experiment','support':[8]}
        prepared=s.plan(state,cache,5,lambda own,d,t,card:({'raw':raw,'plan':llm.validate(raw,pool,own,card)},'fixture_context'))
        j=prepared['decision']['selected']
        # Synthetic feedback is confined to this in-memory behavioral fixture;
        # it is neither revealed from the benchmark nor used to fit a GP.
        observation={'candidate_id':pool['ids'][j],'y':.4,'raw_value':.4*pool['objective_scale']}
        event=s.apply_feedback(state,cache,prepared,5,observation)
        state['history'].append(observation);state['decisions'].append(event)
        text=__import__('json').dumps(llm.payload(pool,state,prepared['d'],6,prepared['card']),allow_nan=False)
        self.assertTrue(text);self.assertEqual(cache.counters['fits'],0)

    def test_full_R2_prefix_observations_forecasts_and_legacy_state_preserved(self):
        root=Path('outputs/r3_behavioral_tmp').resolve();root.mkdir(parents=True,exist_ok=True)
        for task in s.TASKS:
            state=s.initial(task,{'protocol_sha256':'test'})
            source=s.data.read(s.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
            with tempfile.TemporaryDirectory(dir=root) as tmp,redirect_stdout(io.StringIO()):
                self.assertTrue(Path(tmp).resolve().is_relative_to(root))
                for t in range(1,e.BOOTSTRAP[task]+1):s.bootstrap(state,Path(tmp)/'state.json',t)
            n=e.BOOTSTRAP[task]
            self.assertEqual(state['history'],source['history'][:4+n])
            self.assertEqual(state['protected_legacy_state']['history'],state['history'])
            self.assertEqual(len(state['protected_legacy_state']['decisions']),n)
            for a,b in zip(state['decisions'],source['decisions'][:n]):
                for key in ['selected_id','measurement','raw_value','incumbent_after']:
                    self.assertEqual(a[key],b[key])
            self.assertEqual(state['hypotheses'],{})


if __name__=='__main__':unittest.main()
