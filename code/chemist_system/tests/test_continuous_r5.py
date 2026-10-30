"""R5 semantic, scheduling, point-evidence and recovery regression tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import io
import json
import tempfile
import unittest
import numpy as np
from tests.test_continuous_r4 import fixture as r4_fixture
from chem.benchmarks import continuous_r5_engine as e
from chem.benchmarks import continuous_r5_llm as llm
from chem.benchmarks import continuous_r5_study as s


def fixture():
    pool,state,cache,d,unused=r4_fixture()
    menu=e.candidate_menu(pool,state,1);row=min(menu,key=lambda r:abs(r['point']['x']-.5))
    plan={'action':'propose','candidate_id':row['candidate_id'],'prediction':.6,
        'hypothesis':{'reference_index':1,'focus_axes':['x'],'directions':{'x':'increase'},
            'statement':'local increasing-input response contrast','falsification':'a low observed response'},
        'pause_ids':[],'reason':'compare physical reference regions','support':[1]}
    return pool,state,cache,d,plan


def observe(pool,state,cache,d,t,j,y):
    g=e.guidance(cache,state,d,t);out=e.feedback(pool,state,d,g,{'selected':j},t,y)
    d['available'][j]=False;d['eligible'][j]=False
    return out


def raw_model(pool,state,j,prediction,t,key):
    raw={'action':'propose','point':pool['points'][j],'prediction':prediction,
        'hypothesis':{'reference_index':1,'focus_axes':['x'],'statement':'synthetic model','falsification':'synthetic low'},
        'pause_ids':[],'reason':'synthetic fixture only','support':[1]}
    name,p=e.previous.register(pool,state,t,raw,key);h=state['hypotheses'][name]
    h['semantic_contract']={'directions':{'x':'increase'},'reference_index':1,'actual_candidate_id':p['candidate_id'],
        'background_max_change':.25,'projection_distance':0.}
    return name


class ContinuousR5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):e.base.warm()

    def test_menu_contains_only_actual_unqueried_inputs_without_GP(self):
        pool,state,cache,d,plan=fixture();menu=e.candidate_menu(pool,state,1)
        self.assertLessEqual(len(menu),48);self.assertEqual(len({r['candidate_id'] for r in menu}),len(menu))
        for row in menu:
            self.assertNotIn(row['candidate_id'],{h['candidate_id'] for h in state['history']})
            self.assertEqual(row['point'],pool['points'][pool['ids'].index(row['candidate_id'])])
        before=llm.payload(pool,state,1);state['legacy_context']={'BO_reference':'secret','GP_mean':999}
        state['decisions']=[{'measurement':{'prequery_mu_y':999}}];state['GP_mean']=999
        self.assertEqual(before,llm.payload(pool,state,1))
        self.assertTrue(all(set(r)=={'index','point','objective'} for r in before['observations']))

    def test_identity_keeps_exact_inputs_and_prediction_anchor(self):
        pool,state,cache,d,plan=fixture();llm.validate(plan,pool,state,1)
        name,p=e.register(pool,state,1,plan,'c');self.assertTrue(p['semantic_review']['admitted'])
        self.assertEqual(p['normalized_Euclidean_distance'],0.);self.assertEqual(p['candidate_id'],plan['candidate_id'])
        self.assertEqual(p['point'],p['proposed_physical_point'])
        self.assertAlmostEqual(e.local_forecast(pool,state['hypotheses'][name])[p['index']],.6)

    def test_semantic_mismatch_is_rejected_without_repair_or_registration(self):
        pool,state,cache,d,plan=fixture();plan['hypothesis']['directions']['x']='decrease'
        self.assertEqual(llm.validate(plan,pool,state,1),plan)
        name,p=e.register(pool,state,1,plan,'c');self.assertIsNone(name);self.assertEqual(state['hypotheses'],{})
        decision=e.decide(cache,state,d,e.guidance(cache,state,d,1),1,p)
        self.assertEqual(decision['selected'],d['reference']);self.assertEqual(decision['mode'],'BO_semantically_rejected_proposal')

    def test_R4_HPLC_reversed_interventions_are_rejected(self):
        pool=s.original.pool('HPLC');reference=pool['points'][pool['ids'].index('HPLC_00448')]
        for key,focus,directions in [('HPLC_00272',['sample_loop','sample_flow'],{'sample_loop':'increase','sample_flow':'increase'}),
            ('HPLC_00308',['wait_time'],{'wait_time':'decrease'}),('HPLC_00416',['wait_time'],{'wait_time':'decrease'})]:
            result=e.direction_check(pool,reference,pool['points'][pool['ids'].index(key)],focus,directions)
            self.assertFalse(result['admitted']);self.assertTrue(any(v.startswith('focus_direction_mismatch:') for v in result['violations']))

    def test_scope_obeys_declared_direction_and_background(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');h=state['hypotheses'][name]
        self.assertEqual(e.scope_weight(pool,h)[0],0.)
        two=deepcopy(pool);two['axes']=['x','background'];two['z']=[[v[0],0.] for v in pool['z']]
        h2=deepcopy(h);h2['scope']['center']=[h['scope']['center'][0],0.];two['z'][p['index']][1]=.5
        self.assertEqual(e.scope_weight(two,h2)[p['index']],0.)

    def test_two_distinct_experiments_required_before_mean_deployment(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');j=p['index']
        observe(pool,state,cache,d,1,j,.6);e.authorize(pool,state,2)
        self.assertFalse(np.any(e.guidance(cache,state,d,2)['shift_y']))
        out=observe(pool,state,cache,d,2,j-1,.6);self.assertIsNone(out['actual_local_update'])
        with self.assertRaises(AssertionError):e.authorize(pool,state,2)
        events,_=e.authorize(pool,state,3);self.assertTrue(events)
        decision=e.decide(cache,state,d,e.guidance(cache,state,d,3),3,None)
        self.assertTrue(decision['local_ranking_changed']);json.dumps(decision,allow_nan=False)

    def test_duplicate_positive_record_is_not_a_second_experiment(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c')
        observe(pool,state,cache,d,1,p['index'],.6);h=state['hypotheses'][name]
        row=deepcopy(h['shadow_records'][0]);row['query']=2;h['shadow_records'].append(row)
        self.assertEqual(e.evidence(h,3)['positive_count'],1);self.assertFalse(e.evidence(h,3)['eligible'])

    def test_unconfirmed_initial_probe_can_execute_but_cannot_change_mean(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');e.authorize(pool,state,2)
        g=e.guidance(cache,state,d,2);v=e.validation_schedule(state,d,g,2)
        self.assertIsNotNone(v['selected_index']);self.assertEqual(v['stage'],'initial_probe')
        permission=e.call_permission(pool,state,d,g,v,2);self.assertEqual(permission['reason'],'validation_executable_now')
        self.assertFalse(np.any(g['shift_y']))
        decision=e.decide(cache,state,d,g,2,None);self.assertEqual(decision['selected'],v['selected_index'])
        self.assertEqual(state['scheduled_validations'][0]['stage'],'initial_probe')
        self.assertIsNone(e.validation_schedule(state,d,g,3)['selected_index'])

    def test_unexecutable_pending_model_does_not_block_new_LLM(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c')
        observe(pool,state,cache,d,1,p['index'],.6);e.authorize(pool,state,2)
        d['eligible'][:]=False;d['eligible'][4:6]=True;g=e.guidance(cache,state,d,2);v=e.validation_schedule(state,d,g,2)
        self.assertIsNone(v['selected_index'])
        permission=e.call_permission(pool,state,d,g,v,2);self.assertTrue(permission['allowed'])
        self.assertIn(name,permission['untestable_pending_models_released'])

    def test_singleton_budget_skips_proposal_and_validation(self):
        pool,state,cache,d,plan=fixture();d['eligible'][:]=False;d['eligible'][4]=True
        g=e.guidance(cache,state,d,1);v=e.validation_schedule(state,d,g,1)
        self.assertIsNone(v['selected_index']);self.assertFalse(e.call_permission(pool,state,d,g,v,1)['allowed'])

    def test_controller_uses_GP_but_proposer_payload_is_blind(self):
        pool,state,cache,d,plan=fixture();order=[]
        def setup(*args):order.append('controller_GP');return d
        def provider(own,t):order.append('blind_LLM');return {'raw':plan,'plan':plan},'c'
        with patch.object(e,'setup',setup):
            prepared=s.plan(state,cache,1,provider,lambda r:order.append('proposal_receipt'))
        self.assertEqual(order,['controller_GP','blind_LLM','proposal_receipt'])
        receipt=prepared['blind_proposal_receipt'];self.assertTrue(receipt['controller_GP_computed'])
        self.assertFalse(receipt['LLM_received_GP_information']);self.assertFalse(receipt['GP_review_completed'])

    def test_executable_validation_wait_is_followed_by_that_experiment(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c')
        def refuse(*a,**k):raise RuntimeError('LLM must not be called when validation is executable')
        with patch.object(e,'setup',lambda *args:d):prepared=s.plan(state,cache,2,refuse)
        self.assertEqual(prepared['call_permission']['reason'],'validation_executable_now')
        self.assertEqual(prepared['decision']['mode'],'scheduled_local_validation')

    def test_point_negative_transfers_below_half_overlap_for_old_and_new_models(self):
        pool,state,cache,d,plan=fixture();source=raw_model(pool,state,31,.6,1,'source')
        older=raw_model(pool,state,49,.65,1,'older');h=state['hypotheses'][source]
        event={'query':2,'candidate_id':'p40','source_hypothesis_id':source,'kind':'actual_failure',
            'source_model':{k:deepcopy(h[k]) for k in ['scope','basis','prediction_y','semantic_contract']},
            'source_deployed_or_probe_shift_y':.05,'GP_mean_y':.2,'GP_predictive_SD_y':.2,'observed_y':.05,'event_id':'failed_point'}
        state['evidence_ledger'].append(event);e.synchronize(pool,state,3)
        newer=raw_model(pool,state,49,.66,3,'newer');e.synchronize(pool,state,3)
        w1=e.scope_weight(pool,h)>1e-12;w2=e.scope_weight(pool,state['hypotheses'][older])>1e-12
        overlap=np.sum(w1&w2)/min(w1.sum(),w2.sum());self.assertLess(overlap,.5)
        for name in [older,newer]:
            target=state['hypotheses'][name];self.assertEqual(len(target['inherited_evidence']),1)
            self.assertEqual(target['inherited_actual_failure_query'],-1)
            self.assertLess(target['inherited_evidence'][0]['weighted_standardized_gain'],0.)

    def test_own_check_deduplicates_inherited_point_penalty(self):
        pool,state,cache,d,plan=fixture();source=raw_model(pool,state,31,.6,1,'a');target=raw_model(pool,state,49,.65,1,'b')
        h=state['hypotheses'][source];state['evidence_ledger'].append({'query':2,'candidate_id':'p40','source_hypothesis_id':source,
            'kind':'shadow_failure','source_model':{k:deepcopy(h[k]) for k in ['scope','basis','prediction_y','semantic_contract']},
            'GP_mean_y':.2,'GP_predictive_SD_y':.2,'observed_y':.05,'event_id':'one'})
        state['hypotheses'][target]['shadow_records'].append({'query':2,'candidate_id':'p40'})
        e.synchronize(pool,state,3);self.assertEqual(state['hypotheses'][target]['inherited_evidence'],[])

    def test_repeated_pause_preserves_epoch_and_first_recovery_positive(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');j=p['index']
        stop={'action':'abstain','pause_ids':[name],'reason':'pause','support':[1]}
        e.register(pool,state,2,stop,'p1');observe(pool,state,cache,d,3,j,.6)
        e.register(pool,state,4,stop,'p2');h=state['hypotheses'][name]
        self.assertEqual(h['last_pause_query'],2);self.assertFalse(h['pause_records'][-1]['new_pause_epoch'])
        e.authorize(pool,state,4);observe(pool,state,cache,d,4,j-1,.6)
        events,_=e.authorize(pool,state,5);self.assertTrue(events);self.assertFalse(h['paused_by_LLM'])

    def test_actual_negative_KF_gain_cannot_raise_coefficient_and_recovery_works(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');j=p['index']
        observe(pool,state,cache,d,1,j,.6);e.authorize(pool,state,2);observe(pool,state,cache,d,2,j-1,.6);e.authorize(pool,state,3)
        bad=observe(pool,state,cache,d,3,j-2,.05)['actual_local_update']['measurement']
        self.assertLess(bad['actual_paired_SE_gain'],0);self.assertLessEqual(bad['coefficient_after'],bad['coefficient_before'])
        e.authorize(pool,state,4);observe(pool,state,cache,d,4,j-3,.6);e.authorize(pool,state,5);observe(pool,state,cache,d,5,j-4,.6)
        events,_=e.authorize(pool,state,6);self.assertEqual(events[0]['from'],'quarantined')

    def test_late_calls_reserved_and_multiple_high_references_visible(self):
        pool,state,cache,d,plan=fixture();state['calls']=[{'query':1},{'query':3}]
        g=e.guidance(cache,state,d,8);v=e.validation_schedule(state,d,g,8)
        self.assertEqual(e.call_permission(pool,state,d,g,v,8)['reason'],'reserve_two_late_calls')
        state['history'].append({'candidate_id':'p50','y':.19,'raw_value':.19})
        state['history'].extend([{'candidate_id':'p20','y':.1,'raw_value':.1},{'candidate_id':'p40','y':.1,'raw_value':.1}])
        refs=e.high_references(pool,state);self.assertGreaterEqual(len(refs),2)
        self.assertTrue(e.call_permission(pool,state,d,g,v,11)['allowed'])
        public=llm.payload(pool,state,11);self.assertEqual(public['high_references'],refs)

    def test_high_forecast_cannot_override_original_GP_gate(self):
        pool,state,cache,d,plan=fixture();plan['prediction']=2.;name,p=e.register(pool,state,1,plan,'c')
        d['eligible'][p['index']]=False;decision=e.decide(cache,state,d,e.guidance(cache,state,d,1),1,p)
        self.assertFalse(decision['GP_review_accepted']);self.assertEqual(decision['selected'],d['reference'])

    def test_renaming_does_not_reset_evidence(self):
        pool,state,cache,d,plan=fixture();name,p=e.register(pool,state,1,plan,'c');observe(pool,state,cache,d,1,p['index'],.05)
        plan['hypothesis']['statement']='renamed';same,p=e.register(pool,state,2,plan,'d')
        self.assertEqual(same,name);self.assertEqual(len(state['hypotheses'][same]['shadow_records']),1)

    def test_full_protected_prefix_preserves_states_predictions_and_budget(self):
        for task,n in e.BOOTSTRAP.items():
            state=s.initial(task,{'protocol_sha256':'fixture'});scratch=Path(__file__).resolve().parents[1]/'outputs'/'continuous_r5_test_temp'
            scratch.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=scratch) as tmp,redirect_stdout(io.StringIO()):
                self.assertIn(scratch.resolve(),Path(tmp).resolve().parents);path=Path(tmp)/'state.json'
                for t in range(1,n+1):s.prefix_study.bootstrap(state,path,t)
                original=s.data.read(s.prefix_study.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
                self.assertEqual(state['history'],original['history'][:4+n]);self.assertEqual(state['protected_legacy_state']['history'],state['history'])
                for t in range(1,n+1):
                    source=s.prefix_study.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'/f'{t:02d}_candidates.npz'
                    self.assertEqual((path.parent/'rounds'/source.name).read_bytes(),source.read_bytes())

    def test_real_prefix_new_plan_feedback_next_payload_serialize_without_API_or_refit(self):
        def refuse(*a,**k):raise RuntimeError('unit fixture forbids API and fitting')
        for task,prefix in e.BOOTSTRAP.items():
            pool=s.original.pool(task);e.seed_cache(pool);cache=e.Cache(pool)
            src=s.data.read(s.prefix_study.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
            state=s.initial(task,{'protocol_sha256':'synthetic_unit'});state['history']=deepcopy(src['history'][:4+prefix]);state['decisions']=deepcopy(src['decisions'][:prefix])
            def provider(own,t):
                menu=e.candidate_menu(pool,own,t);by={r['index']:r for r in e.high_references(pool,own)}
                for row in menu:
                    ref=by[row['nearest_reference_index']];focus=row['largest_changed_axes']
                    directions={a:'increase' if row['point'][a]>ref['point'][a] else 'decrease' for a in focus}
                    if not e.direction_check(pool,ref['point'],row['point'],focus,directions)['admitted']:continue
                    raw={'action':'propose','candidate_id':row['candidate_id'],'prediction':ref['objective']*.95,
                        'hypothesis':{'reference_index':ref['index'],'focus_axes':focus,'directions':directions,
                            'statement':'synthetic unit only','falsification':'synthetic low'},'pause_ids':[],
                        'reason':'synthetic fixture, never benchmark evidence','support':[ref['index']]}
                    return {'raw':raw,'plan':llm.validate(raw,pool,own,t)},'unit_fixture'
                raise AssertionError('no semantically admissible physical menu fixture')
            with patch.object(e.base,'fit_gpytorch_mll',refuse),patch.object(llm.transport,'get',refuse):
                prepared=s.plan(state,cache,prefix+1,provider);j=prepared['decision']['selected'];h=max(state['history'],key=lambda v:v['y'])
                synthetic={'candidate_id':pool['ids'][j],'y':h['y'],'raw_value':h['raw_value']}
                event=s.apply_feedback(state,cache,prepared,prefix+1,synthetic);state['history'].append(synthetic);state['decisions'].append(event)
                e.authorize(pool,state,prefix+2);json.dumps(state,allow_nan=False);json.dumps(event,allow_nan=False)
                json.dumps(llm.payload(pool,state,prefix+2),allow_nan=False)
            self.assertEqual(cache.counters['fits'],0);cache.release()


if __name__=='__main__':unittest.main()
