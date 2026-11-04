"""Integration invariants for prospective selection; no live API or real oracle."""
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chem.benchmarks import knowledge_engine as engine
from chem.benchmarks import knowledge_llm as llm
from chem.benchmarks import knowledge_policy as policy
from chem.benchmarks import knowledge_study as study

ROOT = Path(__file__).resolve().parents[3]

def make_pool():
    ids = tuple(f'experiment_{39-i:02d}' for i in range(40))
    features = tuple((i / 39.,) for i in range(40))
    return SimpleNamespace(task='FakeTask', group='P2', ids=ids,
        ix={key:i for i,key in enumerate(ids)}, axes=('temperature',), kinds=('continuous',),
        features=features, points=tuple({'temperature':20 + 80*x[0]} for x in features),
        initial_ids=ids[:4], scale=100., unit='test units')


def observation(pool, key):
    index = pool.ix[key]
    return dict(candidate_id=key, y=.01*index, raw_value=float(index))


def make_state(pool, hypotheses=(), history_indices=None):
    indices = list(range(4)) if history_indices is None else history_indices
    return dict(task=pool.task, mode='deployment', condition='agent_initial_terra', seed=0,
        history=[observation(pool,pool.ids[i]) for i in indices], hypotheses=deepcopy(list(hypotheses)),
        calls=[], decisions=[dict(query=i+1) for i in range(max(0,len(indices)-4))],
        verification_used=0, status='RUNNING', protocol_sha256='test-protocol')


def add_claim(pool, hypotheses, plus, minus, bounds):
    request = {'actions':[dict(op='add',id=None,plus_id=pool.ids[plus],minus_id=pool.ids[minus],
        focus={'temperature':{'direction':'increase','bounds':list(bounds)}},reason='Local temperature contrast.')],
        'summary':'Prospective claim.'}
    result, reviews = policy.review_and_apply(pool, hypotheses, request, list(pool.ids), 1)
    assert reviews[0]['accepted'], reviews
    return result


class FakeGP:
    def __init__(self, pool, *args, **kwargs):
        self.pool = pool
        self.counters = {'fits':0,'operator_compiles_in_loop':0}

    def context(self, history, t):
        n = len(self.pool.ids)
        available = ~np.isin(self.pool.ids,[h['candidate_id'] for h in history])
        scores = -np.arange(n,dtype=float)*.01
        bo = int(np.flatnonzero(available)[0])
        return dict(stats=dict(mu_y=np.full(n,.3),sd_y=np.full(n,.2),
            sd_predictive_y=np.full(n,.25),mean=0.,std=1.,fit_key='test',fit_parameters={}),
            base_logei=scores,available=available,eligible=available.copy(),
            bo_index=bo,epsilon=2.,tie_priority=np.arange(n),acquisition_meta={})

    def contrast(self, stats, plus, minus, indices):
        return .1,.2,np.zeros(len(indices))

    def release(self):
        pass


class KnowledgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        domain = patch.object(llm,'task_context',return_value={'description':'Synthetic integration fixture'})
        domain.start()
        self.addCleanup(domain.stop)

    def test_prompt_state_recovers_and_suspends_using_current_coverage_without_mutation(self):
        pool = make_pool()
        hs = add_claim(pool, [], 20, 12, (.25, .6))
        pred = policy.shadow(hs[0],pool.ids[15],1,0.,1.,0.,1.,.7)
        policy.apply_evidence(hs[0],pred,1.,True)
        hs[0]['status']='dormant';hs[0]['ledger']['weight']=0.
        state = make_state(pool,hs,[0,1,2,3,15])
        original=deepcopy(state);context=FakeGP(pool).context(state['history'],2)
        prepared,events=study.prepare_prompt_state(pool,state,context,2)
        self.assertEqual(state,original)
        self.assertEqual(prepared['hypotheses'][0]['status'],'active')
        self.assertGreater(prepared['hypotheses'][0]['ledger']['weight'],0)
        self.assertEqual(events[0]['to_status'],'active')
        self.assertEqual(prepared['hypotheses'][0]['ledger']['events'],hs[0]['ledger']['events'])
        context['eligible'][:]=False;context['eligible'][30:]=True
        sleeping,events=study.prepare_prompt_state(pool,prepared,context,2)
        self.assertEqual(sleeping['hypotheses'][0]['status'],'dormant')
        self.assertEqual(sleeping['hypotheses'][0]['ledger']['weight'],0.)
        self.assertEqual(sleeping['hypotheses'][0]['ledger']['score'],hs[0]['ledger']['score'])

    def test_zero_optional_budget_cannot_spend_verification(self):
        pool=make_pool();hs=add_claim(pool,[],20,12,(.25,.6))
        state=make_state(pool,hs);gp=FakeGP(pool);context=gp.context(state['history'],1)
        context['base_logei'][:]=0.
        gp.contrast=lambda stats,plus,minus,indices: (.1,.2,np.array([0. if j==context['bo_index'] else .08 for j in indices]))
        rows=[dict(id=hs[0]['id'],version=1,weight=0.,eligible_inside_scope=2,can_change_rank_at_cap=True)]
        calculations={(hs[0]['id'],1):(np.array([20,21]),context['base_logei'],(.1,.2,np.array([.08,.08])))}
        with patch.object(engine,'coverage',return_value=(rows,calculations)):
            with patch.dict(engine.PARAMETERS,maximum_optional_verifications=1):
                enabled=engine.plan(pool,gp,state,1,context,mode='full')
            with patch.dict(engine.PARAMETERS,maximum_optional_verifications=0):
                selected=engine.plan(pool,gp,state,1,context,mode='full')
        self.assertEqual(enabled['mode'],'optional_verification')
        self.assertEqual(selected['mode'],'base_bo')
        self.assertIsNone(selected['verification'])

    def test_empty_scope_high_weight_does_not_block_supported_hypothesis(self):
        pool = make_pool()
        hypotheses = add_claim(pool,[],8,4,(0.,.25))
        hypotheses = add_claim(pool,hypotheses,32,16,(.4,.9))
        for h, point, t, y in [(hypotheses[0],9,1,2.),(hypotheses[1],23,2,.5)]:
            pred = policy.shadow(h,pool.ids[point],t,0.,1.,0.,1.,.7)
            policy.apply_evidence(h,pred,y,True)
        self.assertGreater(policy.weight(hypotheses[0]),policy.weight(hypotheses[1]))
        state = make_state(pool,hypotheses,[0,1,2,3,9,23])
        gp = FakeGP(pool)
        context = gp.context(state['history'],3)
        context['eligible'] &= np.arange(40)>=20
        context['bo_index'] = 20
        def scores(gp,ctx,h,js,history):
            value = ctx['base_logei'].copy()
            if h['id']==hypotheses[1]['id']:
                value[22] += 1.
            return value,(0.,1.,np.zeros(len(js)))
        with patch.object(engine,'conditional_scores',side_effect=scores):
            commit = engine.plan(pool,gp,state,3,context,mode='deployment')
        self.assertEqual(commit['selected_id'],pool.ids[22])
        self.assertEqual(commit['mode'],'knowledge_guided')
        self.assertEqual(commit['coverage'][0]['eligible_inside_scope'],0)
        self.assertEqual([h['id'] for h in commit['deployed_versions']],[hypotheses[1]['id']])

    def test_zero_covariance_and_noop_mixture_preserve_exact_tie_choice(self):
        pool = make_pool()
        hypotheses = add_claim(pool,[],32,16,(0.,1.))
        pred = policy.shadow(hypotheses[0],pool.ids[9],1,0.,1.,0.,1.,.7)
        policy.apply_evidence(hypotheses[0],pred,1.,True)
        state = make_state(pool,hypotheses,[0,1,2,3,9])
        gp = FakeGP(pool)
        context = gp.context(state['history'],2)
        context['base_logei'][:] = -2.
        context['stats']['mu_y'] = np.linspace(-2,2,40)
        context['stats']['sd_y'] = np.linspace(.03,2,40)
        context['tie_priority'] = np.arange(40)[::-1]
        context['bo_index'] = 39
        scores,_ = engine.conditional_scores(gp,context,hypotheses[0],np.arange(40),state['history'])
        np.testing.assert_array_equal(scores,context['base_logei'])
        commit = engine.plan(pool,gp,state,2,context,mode='deployment')
        self.assertEqual(commit['selected_id'],pool.ids[39])
        self.assertEqual(commit['mode'],'base_bo')
        self.assertEqual(commit['combined_logei'],commit['reference_logei'])

    def test_wrong_version_feedback_is_atomic(self):
        pool = make_pool()
        hypotheses = add_claim(pool,[],8,4,(0.,.3))
        hypotheses = add_claim(pool,hypotheses,32,16,(.4,.9))
        state = make_state(pool,hypotheses)
        gp = FakeGP(pool)
        commit = engine.plan(pool,gp,state,1,gp.context(state['history'],1),mode='shadow')
        commit['shadows'][1]['prediction']['version'] += 1
        before = deepcopy(state)
        with self.assertRaises(ValueError):
            engine.feedback(state,commit,observation(pool,commit['selected_id']))
        self.assertEqual(state,before)

    def test_zero_effect_coverage_does_not_collapse_adjacent_gp_scores(self):
        pool = make_pool()
        hypotheses = add_claim(pool,[],32,16,(0.,1.))
        state = make_state(pool,hypotheses)
        gp = FakeGP(pool)
        context = gp.context(state['history'],1)
        context['eligible'][:] = False
        context['eligible'][[4,5]] = True
        context['base_logei'][4] = -1.
        context['base_logei'][5] = np.nextafter(-1.,-np.inf)
        context['bo_index'] = 4
        context['tie_priority'][4] = 1
        context['tie_priority'][5] = 0
        rows,_ = engine.coverage(pool,gp,context,hypotheses,state['history'])
        self.assertFalse(rows[0]['can_change_rank_at_cap'])

    def test_out_of_sequence_and_duplicate_feedback_rejected(self):
        pool = make_pool()
        state = make_state(pool)
        gp = FakeGP(pool)
        commit = engine.plan(pool,gp,state,1,gp.context(state['history'],1),mode='bo')
        bad = dict(commit,query=99)
        before = deepcopy(state)
        with self.assertRaises(ValueError):
            engine.feedback(state,bad,observation(pool,bad['selected_id']))
        self.assertEqual(state,before)
        outcome = observation(pool,commit['selected_id'])
        engine.feedback(state,commit,outcome)
        with self.assertRaises(ValueError):
            engine.feedback(state,commit,outcome)
        self.assertEqual(len(state['history']),5)

    def test_public_prompt_contains_feedback_but_no_gp_prediction_internals(self):
        pool = make_pool()
        hypotheses = add_claim(pool,[],32,16,(0.,1.))
        pred = policy.shadow(hypotheses[0],pool.ids[9],1,.123,1.,.25,1.,.7)
        policy.apply_evidence(hypotheses[0],pred,1.,True)
        state = make_state(pool,hypotheses,[0,1,2,3,9])
        payload = llm.public_payload(pool,state,2,[pool.ids[16],pool.ids[32]],[])
        encoded = json.dumps(payload)
        for forbidden in ('gp_mean_y','gp_var_y','cov_y_contrast','contrast_mean','prediction_sha256','BO_reference_id','base_logei'):
            self.assertNotIn(forbidden,encoded)
        self.assertIn('delta_log_score',encoded)
        self.assertIn('observed_y',encoded)
        self.assertEqual(len(payload['own_observations']),5)
        self.assertTrue(all(set(x)=={'id','point','normalized_continuous'} for x in payload['candidate_menu']))

    def test_identical_public_requests_reuse_reply_without_cross_prefix_reuse(self):
        calls=[]
        class Client:
            def complete(self,*args,**kwargs):
                calls.append(args)
                return dict(output_text=json.dumps({'actions':[],'summary':'No justified claim.'}),
                            resolved_model='gpt-6.1-terra')
        with tempfile.TemporaryDirectory(prefix='knowledge_integration_',dir=ROOT/'runtime') as temp:
            a=llm.request(Client(),{'observations':[1]},temp)
            b=llm.request(Client(),{'observations':[1]},temp)
            c=llm.request(Client(),{'observations':[1,2]},temp)
        self.assertEqual(len(calls),2)
        self.assertEqual(a,b)
        self.assertNotEqual(a['context_sha256'],c['context_sha256'])

    def test_durable_commit_precedes_observation_and_resume_never_recalls_llm(self):
        pool = make_pool()
        protocol = {'seed':0,'protocol_sha256':'test-protocol','supported_conditions':['agent_initial_terra']}
        requested, queried = [], []
        crash = {'armed':True}
        class Client:
            def close(self):
                pass
        with tempfile.TemporaryDirectory(prefix='knowledge_integration_',dir=ROOT/'runtime') as temp, ExitStack() as stack:
            root=Path(temp)
            pool.source_path=root/'synthetic_inputs.json'
            pool.source_path.write_text('{"synthetic":true}',encoding='utf-8')
            directory=study.paths(root,'test')/'runs'/'FakeTask_full_agent_initial_terra'
            statefile=directory/'state.json'
            def oracle(root_arg,actual_pool,key):
                if key not in pool.initial_ids:
                    state=study.read(statefile)
                    t=len(state['decisions'])+1
                    path=directory/'rounds'/f'{t:02d}_commit.json'
                    self.assertTrue(path.exists(),'oracle called before a durable commit')
                    commit=study.read(path)
                    self.assertEqual(commit['selected_id'],key)
                    self.assertEqual(commit['history_sha256'],llm.digest(state['history']))
                    if crash['armed']:
                        crash['armed']=False
                        raise RuntimeError('simulated interruption after commitment')
                    queried.append(key)
                return observation(pool,key)
            def request(client,payload,folder):
                requested.append(payload['query'])
                return {'proposal':{'actions':[],'summary':'No supported preference.'},
                        'context_sha256':llm.digest(payload),'attempt':1,'resolved_model':'gpt-6.1-terra'}
            for target,replacement in [(study,'check'),(study.gpmod,'load_task')]:
                stack.enter_context(patch.object(target,replacement,return_value=protocol if replacement=='check' else pool))
            stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.api,'APIClient',return_value=Client()))
            stack.enter_context(patch.object(study.llm,'request',side_effect=request))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):
                    study.run_one(root,'test',pool.task,'full')
                committed=study.read(directory/'rounds/01_commit.json')
                self.assertEqual(requested,[1])
                self.assertEqual(study.read(statefile)['status'],'INTERRUPTED')
                result=study.run_one(root,'test',pool.task,'full')
            self.assertEqual(requested.count(1),1)
            self.assertEqual(study.read(directory/'rounds/01_commit.json'),committed)
            self.assertEqual(len(queried),16)
            self.assertEqual(len(set(queried)),16)
            final=study.read(statefile)
            self.assertEqual(len(final['history']),20)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(len(final['calls']),len(requested))
            for t,decision in enumerate(final['decisions'],1):
                commit=study.read(directory/'rounds'/f'{t:02d}_commit.json')
                self.assertEqual(decision['commit_sha256'],llm.digest(commit))


if __name__=='__main__':
    unittest.main()
