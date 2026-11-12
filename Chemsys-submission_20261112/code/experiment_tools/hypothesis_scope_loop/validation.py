"""Meaningful scope, feedback, inheritance and chronology checks; no new experiments."""
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import unittest
import numpy as np
from . import study as s,knowledge as k,policy,prompts


def draft(pool,point=None):
    point=point or pool.points[0]
    cat=next((a for a,t in zip(pool.axes,pool.kinds) if t=='categorical'),None)
    if cat:scope={cat:dict(allowed=[point[cat]])}
    else:
        axis=pool.axes[0];bounds=next(v['declared_input_bounds'] for v in prompts.first.background(pool)['inputs'] if v['name']==axis)
        middle=sum(bounds)/2;scope={axis:dict(bounds=[bounds[0],middle] if point[axis]<=middle else [middle,bounds[1]])}
    return dict(id='SYNTHETIC_QA',claim='Synthetic scope fixture, not an experimental response.',direction='favour',
        scope=scope,forecast=dict(mean_raw=80.,sd_raw=20.),falsification='Synthetic validation fixture.',
        supporting_observation_ids=[],review_action='new')


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s.initialize();cls.pools={t:s.pool_for(t) for t in s.native.TASKS};cls.pool=next(iter(cls.pools.values()))

    def notes(self):
        scopes=k.Scopes(self.pool);notes,_=k.install([draft(self.pool)],[],[],scopes,1,False);return scopes,notes

    def test_scope_selection_leaves_native_predictions_unchanged(self):
        scopes,notes=self.notes();inside=scopes.mask(notes[0]['scope']);n=len(inside)
        scores=np.arange(n,dtype=float);bo=int(np.flatnonzero(~inside)[-1]);scores[bo]=n+1
        ctx=dict(available=np.ones(n,bool),base_logei=scores,tie_priority=np.arange(n),bo_index=bo)
        before=scores.copy();choice=policy.select(ctx,scopes,notes,40.)
        self.assertTrue(inside[choice['index']]);self.assertNotEqual(choice['index'],bo)
        self.assertEqual(choice['index'],int(np.flatnonzero(inside)[-1]));np.testing.assert_array_equal(scores,before)
        self.assertTrue(choice['causes'][notes[0]['semantic_key']])
        notes[0]['permission']='paused';choice=policy.select(ctx,scopes,notes,40.)
        self.assertEqual(choice['index'],bo);self.assertTrue(choice['fallback'])
        notes[0]['permission']='active';ctx['available']=~inside
        choice=policy.select(ctx,scopes,notes,40.);self.assertEqual(choice['index'],bo);self.assertTrue(choice['fallback'])

    def test_same_region_renaming_preserves_failed_evidence(self):
        scopes,notes=self.notes();notes[0].update(permission='paused',negative_streak=2,retests_used=1)
        notes[0]['KF'].update(reliability=.1,updates=3);note=deepcopy(notes[0]);d=draft(self.pool);d.update(id='renamed',review_action='retest')
        after,edits=k.install([d],notes,[],scopes,8,True)
        self.assertEqual(after[0]['permission'],'paused');self.assertEqual(after[0]['KF'],note['KF'])
        self.assertEqual(after[0]['negative_streak'],2);self.assertEqual(after[0]['retests_used'],1)
        after,_=k.install([d],[],notes,scopes,12,True);self.assertEqual(after[0]['permission'],'paused')

    def test_accurate_failure_never_restores_positive_recommendation(self):
        scopes,notes=self.notes();notes[0]['forecast']=dict(mean_raw=20.,sd_raw=5.)
        notes,events=k.pause_for_recommendation(notes,80.)
        self.assertEqual(notes[0]['permission'],'paused');self.assertFalse(events[0]['chemical_claim_declared_false'])
        j=int(np.flatnonzero(scopes.mask(notes[0]['scope']))[0])
        ctx=dict(stats=dict(mu_y=np.full(len(self.pool.ids),.8),sd_predictive_y=np.full(len(self.pool.ids),.2)))
        forecasts=k.freeze_forecasts(self.pool,scopes,notes,j,4,80.,ctx,{})
        after,events=k.feedback(notes,forecasts,dict(candidate_id=self.pool.ids[j],raw_value=20.),4)
        self.assertGreater(events[0]['KF']['Brier_gain'],0);self.assertEqual(after[0]['permission'],'paused')
        self.assertFalse(k.eligible(after[0],80.))

    def test_one_successful_explicit_retest_has_reachable_recovery(self):
        scopes,notes=self.notes();notes[0].update(permission='paused',negative_streak=2,deployed_nonimprovements=2)
        d=draft(self.pool);d['review_action']='retest'
        held,_=k.install([d],notes,[],scopes,8,False);self.assertEqual(held[0]['permission'],'paused')
        after,_=k.install([d],notes,[],scopes,8,True);self.assertEqual(after[0]['permission'],'probation')
        self.assertEqual(after[0]['retests_used'],1)
        j=int(np.flatnonzero(scopes.mask(after[0]['scope']))[0])
        ctx=dict(stats=dict(mu_y=np.full(len(self.pool.ids),.1),sd_predictive_y=np.full(len(self.pool.ids),.1)))
        forecasts=k.freeze_forecasts(self.pool,scopes,after,j,8,50.,ctx,{after[0]['semantic_key']:True})
        recovered,events=k.feedback(after,forecasts,dict(candidate_id=self.pool.ids[j],raw_value=90.),8)
        self.assertGreater(events[0]['KF']['Brier_gain'],0);self.assertEqual(recovered[0]['permission'],'active')
        self.assertEqual(recovered[0]['negative_streak'],0);self.assertEqual(recovered[0]['retests_used'],1)
        failed,_=k.feedback(after,forecasts,dict(candidate_id=self.pool.ids[j],raw_value=40.),8)
        self.assertEqual(failed[0]['permission'],'paused')

    def test_guided_failures_only_when_rule_changes_choice(self):
        scopes,notes=self.notes();j=int(np.flatnonzero(scopes.mask(notes[0]['scope']))[0])
        ctx=dict(stats=dict(mu_y=np.full(len(self.pool.ids),.8),sd_predictive_y=np.full(len(self.pool.ids),.2)))
        for causes,expected in (({},0),({notes[0]['semantic_key']:True},1)):
            f=k.freeze_forecasts(self.pool,scopes,notes,j,3,70.,ctx,causes)
            after,_=k.feedback(notes,f,dict(candidate_id=self.pool.ids[j],raw_value=60.),3)
            self.assertEqual(after[0]['deployed_nonimprovements'],expected)

    def test_initial_and_full_history_public_inputs_and_validation(self):
        for task,pool in self.pools.items():
            locked=s.locked_first(task);state=dict(history=[],decisions=[],notes=[],archive=[],calls=[],permission_events=[])
            event=dict(call=True,reasons=['initial'],stagnated=False);public=prompts.public(pool,state,event,locked)
            self.assertEqual(public['own_observations'],[]);self.assertFalse(public['current_GP_predictions_or_ranking_supplied'])
            self.assertIsNone(public['incumbent_raw']);self.assertNotIn('raw_value',str(public['locked_initial_preferred_experiment']))
            note=draft(pool,locked['conditions']);raw=dict(summary='Synthetic QA only',hypotheses=[note],preferred_experiment=dict(
                conditions=locked['conditions'],basis_ids=[note['id']],rationale='Synthetic QA only'))
            self.assertEqual(prompts.validate(raw,pool,state,locked)['projection']['candidate_id'],locked['candidate_id'])
            old=s.read(s.first.folder_for(task)/'state.json')
            for row in old['decisions'][:7]:
                row=deepcopy(row);row['decision']['incumbent_before']=max((h['raw_value'] for h in state['history']),default=None)
                state['history'].append({a:row[a] for a in ('candidate_id','y','raw_value')});state['decisions'].append(row)
            public=prompts.public(pool,state,event,locked)
            self.assertEqual(len(public['own_observations']),7);self.assertIsNone(public['locked_initial_preferred_experiment'])
            self.assertEqual([r['candidate_id'] for r in public['own_observations']],[r['candidate_id'] for r in state['history']])

    def test_refresh_schedule_reserves_late_call(self):
        scopes,notes=self.notes();notes[0]['permission']='paused'
        state=dict(notes=notes,calls=[dict(query=1),dict(query=4),dict(query=7)],decisions=[])
        available=np.ones(len(self.pool.ids),bool)
        self.assertEqual(policy.event(state,scopes,available,50.,10)['hold'],'reserved_late_refresh')
        self.assertTrue(policy.event(state,scopes,available,50.,15)['call'])
        state['calls'].append(dict(query=15));self.assertEqual(policy.event(state,scopes,available,50.,18)['hold'],'LLM_budget_exhausted')

    def test_first_commit_precedes_oracle_and_GP_construction(self):
        pool=self.pool;events=[];locked=s.locked_first(pool.task);note=draft(pool,locked['conditions'])
        raw=dict(summary='Synthetic QA only',hypotheses=[note],preferred_experiment=dict(conditions=locked['conditions'],basis_ids=[note['id']],rationale='Synthetic QA only'))
        response=dict(raw=raw,plan=prompts.validate(raw,pool,dict(history=[]),locked),request='SYNTHETIC_QA_ONLY')
        class DummyGP:
            def __init__(self,*args,**kw):
                if events[:2]!=['LLM','committed_observation']:raise AssertionError('Premature GP construction')
                self.counters=dict(operator_compiles_in_loop=0)
            def context(self,history,q):
                n=len(pool.ids);available=~np.isin(pool.ids,[h['candidate_id'] for h in history]);j=int(np.flatnonzero(available)[0])
                return dict(bo_index=j,stats=dict(mu_y=np.zeros(n),sd_y=np.ones(n),sd_predictive_y=np.ones(n),fit_parameters={},fit_key='SYNTHETIC_QA'),
                    base_logei=-np.arange(n,dtype=float),available=available,tie_priority=np.arange(n),stage='SYNTHETIC_QA',acquisition_meta={})
            def release(self):pass
        mock=s.BASE/'preflight_mock';mock.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=mock) as tmp:
            folder=Path(tmp)
            def reply(*args,**kw):
                self.assertFalse((folder/'rounds/01_feedback.json').exists());events.append('LLM');return deepcopy(response)
            def oracle(root,p,candidate):
                q=events.count('committed_observation')+1;commit=s.read(folder/'rounds'/f'{q:02d}_commit.json')
                self.assertEqual(commit['decision']['selected_id'],candidate);events.append('committed_observation')
                return dict(candidate_id=candidate,y=.9,raw_value=90.)
            original=policy.event
            def schedule(state,*args):return original(state,*args) if not state['calls'] else dict(call=False,reasons=[],hold='SYNTHETIC_QA',stagnated=False)
            with (patch.object(s,'check',return_value=dict(protocol_sha256='SYNTHETIC_QA_ONLY')),
                patch.object(s,'folder_for',return_value=folder),patch.object(s.transport,'request',side_effect=reply),
                patch.object(s.native,'read_observation',side_effect=oracle),patch.object(s,'ColdGP',DummyGP),
                patch.object(policy,'event',side_effect=schedule),redirect_stdout(io.StringIO())):
                self.assertEqual(s.run_one(pool.task)['status'],'COMPLETE')
            self.assertEqual(events[:2],['LLM','committed_observation']);self.assertEqual(events.count('committed_observation'),20)

    def test_native_GP_replays_frozen_one_and_four_observation_prefixes(self):
        diagnostics=[]
        for task,pool in self.pools.items():
            old=s.read(s.first.folder_for(task)/'state.json');gp=s.ColdGP(pool,s.BASE/'preflight_cache'/s.SHORT[task],seed=0)
            for n in (1,4):
                ctx=gp.context(old['history'][:n],n+1)
                self.assertEqual(pool.ids[ctx['bo_index']],old['history'][n]['candidate_id'])
                self.assertTrue(np.isfinite(ctx['stats']['sd_y']).all());self.assertEqual(gp.counters['operator_compiles_in_loop'],0)
                diagnostics.append(dict(task=task,prefix_length=n,native_choice_matches_frozen=True));gp.release()
        s.write(s.BASE/'preflight_numerics.json',diagnostics,replace=True)


def preflight():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value=dict(status='PASS_SCOPE_LOOP_PREFLIGHT' if result.wasSuccessful() else 'FAIL_SCOPE_LOOP_PREFLIGHT',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),actual_LLM_calls=0,new_experimental_queries=0,test_output=stream.getvalue())
    s.write(s.BASE/'preflight.json',value,replace=True)
    if not result.wasSuccessful():print(stream.getvalue());raise RuntimeError('Preflight failed')
    return {a:b for a,b in value.items() if a!='test_output'}
