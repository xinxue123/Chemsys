"""Scientific invariants of blind proposal, validation timing and inheritance."""
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io
import json
import tempfile
import unittest
import numpy as np
from chem.benchmarks import continuous_r4_engine as e
from chem.benchmarks import continuous_r4_llm as llm
from chem.benchmarks import continuous_r4_study as s
from chem.benchmarks.suzuki_domain import digest


def fixture():
    z=np.linspace(0,1,64).reshape(-1,1)
    pool={'task':'fixture','ids':[f'p{i}' for i in range(64)],'axes':['x'],'z':z.tolist(),
        'points':[{'x':float(v[0])} for v in z],'bounds':{'x':[0.,1.]},'objective_scale':1.,
        'objective_unit':'unit','description':'fixture','objective_name':'objective'}
    state={'task':'fixture','history':[{'candidate_id':f'p{i}','y':.2,'raw_value':.2} for i in range(4)],
        'decisions':[],'initial_knowledge':{},'hypotheses':{},'calls':[],'authorization_events':[],
        'inheritance_events':[],'evidence_ledger':[],'scheduled_validations':[],'legacy_context':{},'protocol_sha256':'test'}
    stats={'mu_y':np.full(64,.2),'sd_y':np.full(64,.2),'sd_predictive_y':np.full(64,.2),
        'mu_std':np.full(64,.2),'sd_std':np.full(64,.2),'std':1.,'mean':0.,'best_std':.4}
    class FakeCache:
        def __init__(self):
            self.pool=pool;self.ids=pool['ids'];self.ix={key:j for j,key in enumerate(self.ids)}
            self.X=e.base.torch.tensor(z,dtype=e.base.torch.double)
        def score(self,history,shift=None):
            return e.base.acquisition(stats['mu_std'],stats['sd_std'],np.zeros(64) if shift is None else shift,.4,0),{'mc_seed':0}
    cache=FakeCache();score,_=cache.score(state['history']);mask=np.arange(64)>=4
    d={'base':score,'base_meta':{'mc_seed':0},'stats':stats,'available':mask.copy(),'eligible':mask.copy(),
        'reference':4,'information_proxy':np.zeros(64),'tie_priority':np.arange(64),'budget':2.}
    plan={'action':'propose','point':{'x':float(z[31,0])},'prediction':.6,'hypothesis':{'reference_index':1,
        'focus_axes':['x'],'statement':'local response contrast','falsification':'a low response'},
        'pause_ids':[],'reason':'test the contrast','support':[1]}
    return pool,state,cache,d,plan


def check_point(pool,state,cache,d,t,j,y):
    g=e.guidance(cache,state,d,t);decision={'selected':j};out=e.feedback(pool,state,d,g,decision,t,y)
    d['available'][j]=False;d['eligible'][j]=False
    return out


class ContinuousR4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):e.base.warm()

    def test_blind_payload_is_invariant_to_any_GP_or_legacy_forecast(self):
        pool,state,cache,d,plan=fixture();a=llm.payload(pool,state,1)
        state['decisions']=[{'measurement':{'prequery_mu_y':999,'prequery_predictive_sd_y':999}}]
        state['legacy_context']={'BO_reference':'secret','base_qLogEI':[999]};state['GP_mean']=999
        self.assertEqual(a,llm.payload(pool,state,1))
        self.assertEqual(set(a),{'description','domain','objective_name','objective_unit','objective_scale','query_round',
            'remaining_queries','observations','initial_uncertain_preferences_normalized_0_to_1','own_past_hypothesis_declarations'})
        self.assertTrue(all(set(o)=={'index','point','objective'} for o in a['observations']))

    def test_proposal_and_receipt_precede_current_GP_computation(self):
        pool,state,cache,d,plan=fixture();order=[]
        def provider(own,t):order.append('LLM');return {'raw':plan,'plan':plan},'context'
        def setup(*args):self.assertEqual(order,['LLM','receipt']);order.append('GP');return d
        with patch.object(e,'setup',setup):
            s.plan(state,cache,1,provider,lambda r:order.append('receipt'))
        self.assertEqual(order,['LLM','receipt','GP'])

    def test_physical_projection_is_nearest_unqueried_and_has_no_GP(self):
        pool,state,cache,d,plan=fixture();p=e.projection(pool,state,1,{'x':.0})
        self.assertEqual(p['candidate_id'],'p4');self.assertFalse(p['uses_GP'])
        self.assertAlmostEqual(p['normalized_Euclidean_distance'],4/63)

    def test_schema_rejects_future_reference_and_out_of_domain_point(self):
        pool,state,cache,d,plan=fixture();self.assertEqual(llm.validate(plan,pool,state),plan)
        invalid=deepcopy(plan);invalid['hypothesis']['reference_index']=5
        with self.assertRaises(ValueError):llm.validate(invalid,pool,state)
        invalid=deepcopy(plan);invalid['point']['x']=2
        with self.assertRaises(ValueError):llm.validate(invalid,pool,state)

    def test_no_mean_correction_after_only_one_positive(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        before=e.guidance(cache,state,d,1);self.assertFalse(np.any(before['shift_y']))
        check_point(pool,state,cache,d,1,31,.6);e.authorize(pool,state,2)
        self.assertEqual(state['hypotheses'][key]['evidence']['positive_count'],1)
        self.assertFalse(np.any(e.guidance(cache,state,d,2)['shift_y']))

    def test_two_distinct_prospective_points_authorize_next_round_only(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        check_point(pool,state,cache,d,1,31,.6);e.authorize(pool,state,2)
        second=check_point(pool,state,cache,d,2,30,.6)
        self.assertIsNone(second['actual_local_update'])
        with self.assertRaises(AssertionError):e.authorize(pool,state,2)
        events,_=e.authorize(pool,state,3);self.assertTrue(events)
        self.assertEqual(state['hypotheses'][key]['active_cap'],.25)
        decision=e.decide(cache,state,d,e.guidance(cache,state,d,3),3,None)
        self.assertTrue(decision['local_ranking_changed']);json.dumps(decision,allow_nan=False)

    def test_scheduler_chooses_informative_alternative_inside_original_gate(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        check_point(pool,state,cache,d,1,31,.6);e.authorize(pool,state,2)
        g=e.guidance(cache,state,d,2);v=e.validation_schedule(state,d,g,2)
        self.assertIsNotNone(v['selected_index']);self.assertNotEqual(v['selected_index'],d['reference'])
        self.assertTrue(d['eligible'][v['selected_index']]);self.assertEqual(v['hypothesis_id'],key)
        d['eligible'][:]=False;d['eligible'][4]=True
        self.assertIsNone(e.validation_schedule(state,d,g,2)['selected_index'])

    def test_high_forecast_does_not_bypass_GP_gate(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c');d['eligible'][p['index']]=False
        decision=e.decide(cache,state,d,e.guidance(cache,state,d,1),1,p)
        self.assertFalse(decision['GP_review_accepted']);self.assertEqual(decision['selected'],4)

    def test_existing_and_new_related_models_share_failure_clock(self):
        pool,state,cache,d,plan=fixture();source,_=e.register(pool,state,1,plan,'a')
        other=deepcopy(plan);other['prediction']=.61;older,_=e.register(pool,state,1,other,'b')
        h=state['hypotheses'][source]
        event={'query':2,'candidate_id':'p31','source_hypothesis_id':source,'kind':'actual_failure',
            'source_model':{k:deepcopy(h[k]) for k in ['scope','basis','prediction_y']},
            'source_deployed_or_probe_shift_y':.05,'GP_mean_y':.2,'GP_predictive_SD_y':.2,'observed_y':.05,'event_id':'failure'}
        state['evidence_ledger'].append(event);e.synchronize(pool,state,3)
        other['prediction']=.62;newer,_=e.register(pool,state,3,other,'c');e.synchronize(pool,state,3)
        for name in [older,newer]:
            self.assertEqual(state['hypotheses'][name]['inherited_actual_failure_query'],2)
            self.assertEqual(state['hypotheses'][name]['inherited_failure_ids'],['failure'])
        # A materially corrected forecast pointing toward the observation does
        # not inherit the failure just because its geometric support overlaps.
        other['prediction']=0.;corrected,_=e.register(pool,state,3,other,'d');e.synchronize(pool,state,3)
        self.assertEqual(state['hypotheses'][corrected]['inherited_failure_ids'],[])

    def test_pause_keeps_monitoring_and_recovers_after_new_feedback(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        check_point(pool,state,cache,d,1,31,.6);e.authorize(pool,state,2)
        check_point(pool,state,cache,d,2,30,.6);e.authorize(pool,state,3)
        stop={'action':'abstain','pause_ids':[key],'reason':'pause','support':[1]}
        e.register(pool,state,3,stop,'pause');check_point(pool,state,cache,d,3,29,.6)
        e.authorize(pool,state,4);self.assertEqual(state['hypotheses'][key]['status'],'paused')
        check_point(pool,state,cache,d,4,28,.6);e.authorize(pool,state,5)
        self.assertEqual(state['hypotheses'][key]['status'],'paused')
        check_point(pool,state,cache,d,5,27,.6);events,_=e.authorize(pool,state,6)
        self.assertTrue(events);self.assertFalse(state['hypotheses'][key]['paused_by_LLM'])
        self.assertEqual(len(state['hypotheses'][key]['shadow_records']),5)

    def test_actual_failure_decreases_KF_and_can_recover(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        check_point(pool,state,cache,d,1,31,.6);e.authorize(pool,state,2)
        check_point(pool,state,cache,d,2,30,.6);e.authorize(pool,state,3)
        bad=check_point(pool,state,cache,d,3,29,.05);m=bad['actual_local_update']['measurement']
        self.assertLess(m['actual_paired_SE_gain'],0);self.assertLessEqual(m['coefficient_after'],m['coefficient_before'])
        e.authorize(pool,state,4);check_point(pool,state,cache,d,4,28,.6)
        e.authorize(pool,state,5);check_point(pool,state,cache,d,5,27,.6)
        events,_=e.authorize(pool,state,6);self.assertEqual(events[0]['from'],'quarantined')

    def test_weak_positive_and_negative_records_both_retained(self):
        pool,state,cache,d,plan=fixture();plan['prediction']=.201;key,p=e.register(pool,state,1,plan,'c')
        good=check_point(pool,state,cache,d,1,31,.6)['shadow_checks'][0]
        bad=check_point(pool,state,cache,d,2,30,.05)['shadow_checks'][0]
        self.assertFalse(good['informative']);self.assertFalse(bad['informative'])
        self.assertGreater(good['weighted_standardized_gain'],0);self.assertLess(bad['weighted_standardized_gain'],0)
        self.assertEqual(len(state['hypotheses'][key]['shadow_records']),2)

    def test_renaming_cannot_reset_model_evidence(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c')
        check_point(pool,state,cache,d,1,31,.05);changed=deepcopy(plan);changed['hypothesis']['statement']='another name'
        again,_=e.register(pool,state,2,changed,'d');self.assertEqual(key,again)
        self.assertEqual(len(state['hypotheses'][key]['shadow_records']),1)

    def test_affine_focus_and_conditional_background_are_executable(self):
        pool,state,cache,d,plan=fixture();key,p=e.register(pool,state,1,plan,'c');h=state['hypotheses'][key]
        forecast=e.local_forecast(pool,h)
        self.assertAlmostEqual(forecast[31],.6);self.assertAlmostEqual(forecast[0],.2)
        self.assertNotEqual(forecast[30],forecast[31]);self.assertTrue(np.all(e.scope_weight(pool,h)[-4:]==0))
        two=deepcopy(pool);two['axes']=['x','b'];two['z']=[[v[0],0.] for v in pool['z']];two['z'][31][1]=.5
        h2=deepcopy(h);h2['scope']['center']=[h['scope']['center'][0],0.]
        self.assertEqual(e.scope_weight(two,h2)[31],0.)

    def test_call_budget_cooldown_and_late_reserve(self):
        pool,state,cache,d,plan=fixture();self.assertTrue(e.call_permission(state,5)['allowed'])
        state['calls']=[{'query':1},{'query':3},{'query':5}]
        self.assertEqual(e.call_permission(state,8)['reason'],'reserve_late_call')
        self.assertTrue(e.call_permission(state,9)['allowed']);state['calls'].append({'query':9})
        self.assertFalse(e.call_permission(state,11)['allowed']);state['calls']=[]
        self.assertFalse(e.call_permission(state,15)['allowed'])

    def test_protected_prefix_preserves_full_state_and_predictions(self):
        for task,n in e.BOOTSTRAP.items():
            state=s.initial(task,{'protocol_sha256':'test'})
            scratch=Path(__file__).resolve().parents[1]/'outputs'/'continuous_r4_test_temp'
            scratch.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=scratch) as tmp,redirect_stdout(io.StringIO()):
                self.assertIn(scratch.resolve(),Path(tmp).resolve().parents)
                path=Path(tmp)/'state.json'
                for t in range(1,n+1):s.parent.bootstrap(state,path,t)
                old=s.data.read(s.parent.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
                self.assertEqual(state['history'],old['history'][:n+4])
                self.assertEqual(state['protected_legacy_state']['history'],state['history'])
                self.assertEqual(state['evidence_ledger'],[])
                for t in range(1,n+1):
                    source=s.parent.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'/f'{t:02d}_candidates.npz'
                    self.assertEqual((path.parent/'rounds'/source.name).read_bytes(),source.read_bytes())

    def test_real_prefix_plan_synthetic_feedback_and_next_payload_serialize(self):
        def refuse(*args,**kwargs):raise RuntimeError('unit test prohibits API/fitting')
        for task,prefix in e.BOOTSTRAP.items():
            pool=s.original.pool(task);e.seed_cache(pool);cache=e.Cache(pool)
            source=s.parent.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json'
            old=s.data.read(source);state=s.initial(task,{'protocol_sha256':'unit_fixture'})
            state['history']=deepcopy(old['history'][:4+prefix]);state['decisions']=deepcopy(old['decisions'][:prefix])
            reference=max(range(len(state['history'])),key=lambda j:state['history'][j]['y'])+1
            h=state['history'][reference-1];point=deepcopy(pool['points'][pool['ids'].index(h['candidate_id'])])
            axis=pool['axes'][0];lo,hi=pool['bounds'][axis];point[axis]=(lo+hi)/2
            raw={'action':'propose','point':point,'prediction':h['raw_value']*.95,
                'hypothesis':{'reference_index':reference,'focus_axes':[axis],'statement':'unit synthetic contrast','falsification':'unit low response'},
                'pause_ids':[],'reason':'unit fixture only, never benchmark evidence','support':[reference]}
            def provider(own,t):return {'raw':raw,'plan':llm.validate(raw,pool,own)},'unit_fixture'
            with patch.object(e.base,'fit_gpytorch_mll',refuse),patch.object(llm.transport,'get',refuse):
                p=s.plan(state,cache,prefix+1,provider)
                key=pool['ids'][p['decision']['selected']]
                synthetic={'candidate_id':key,'y':h['y'],'raw_value':h['raw_value']}
                event=s.apply_feedback(state,cache,p,prefix+1,synthetic)
                state['history'].append(synthetic);state['decisions'].append(event)
                e.authorize(pool,state,prefix+2)
                json.dumps(event,allow_nan=False);json.dumps(state,allow_nan=False)
                json.dumps(llm.payload(pool,state,prefix+2),allow_nan=False)
            self.assertEqual(cache.counters['fits'],0);cache.release()


if __name__=='__main__':unittest.main()
