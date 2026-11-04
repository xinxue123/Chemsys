from copy import deepcopy
from contextlib import ExitStack,redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_information_v10 import proposals,llm,engine,study
from knowledge_rank_v9 import kf

ROOT=Path(__file__).resolve().parents[4]


def fixture():
    ids=tuple(f'p{i}' for i in range(40));features=tuple((i/39,'A' if i%2 else 'B') for i in range(40))
    pool=SimpleNamespace(ids=ids,ix={k:i for i,k in enumerate(ids)},axes=('x','category'),kinds=('continuous','categorical'),
        features=features,points=[dict(x=x,category=c) for x,c in features],task='synthetic',unit='test',scale=100.,initial_ids=ids[:4],
        raw_pool={'ids':ids})
    state=dict(task='synthetic',arm='information',seed=0,protocol_sha256='registered',initial_commit_sha256='init',
        history=[dict(candidate_id=ids[j],y=.1*(j+1),raw_value=10.*(j+1)) for j in range(4)],hypotheses=[],decisions=[],
        insights=[],edit_calls=[],information_used=0,dissent_used=False,tested_probe_signatures=[],status='READY')
    return pool,state


def context(pool,state):
    n=len(pool.ids);avail=~np.isin(pool.ids,[r['candidate_id'] for r in state['history']])
    return dict(stats=dict(mu_y=np.full(n,.4),sd_y=np.full(n,.2),sd_predictive_y=np.full(n,.25),fit_key='synthetic',fit_parameters={}),
        base_logei=-np.linspace(0,.5,n),available=avail,tie_priority=np.arange(n),eligible=avail,epsilon=.2)


def action():
    return dict(op='add',id='H001',claim='x may improve the objective conditional on category A',focus=['x'],effect='promote',scope={},reason='contrast',
        probe=dict(target={'x':.95,'category':'A'},why_informative='Tests a new x regime',if_above='Explore that regime',if_below='Return to measured region'))


def planned(t=1):
    pool,state=fixture();ctx=context(pool,state)
    raw=dict(summary='hypothesis',actions=[action()]);pp=llm.validate_proposal(raw,pool,state,t,ctx['available'],ctx['tie_priority'])
    proposal=dict(plan=pp,raw=raw,context_sha256='proposal')
    _,menu,_=engine.menu(pool,state,pp['hypotheses'],ctx,t);ids=[pool.ids[j] for j in menu]
    rawf=dict(summary='forecast',assessments=[dict(id='H001',probabilities={key:.8 for key in ids})])
    forecast=dict(plan=llm.validate_forecast(rawf,pp['hypotheses'],ids),raw=rawf,context_sha256='forecast')
    return pool,state,ctx,proposal,forecast


def fake_information(gp,ctx,indices):return {j:(.01 if j<10 else .5) for j in indices},None


class PipelineTests(unittest.TestCase):
    def test_full_pool_projection_exact_categories_and_highest_known_reference(self):
        pool,state,ctx,p,f=planned();h=p['plan']['hypotheses'][0]
        mapped=proposals.project(pool,h,ctx['available'],ctx['tie_priority'])
        self.assertGreater(mapped['index'],8);self.assertEqual(pool.features[mapped['index']][1],'A')
        self.assertEqual(h['reference_id'],'p3');proposals.validate(h)
        self.assertEqual(state['hypotheses'],[])

    def test_category_ids_do_not_define_distance_or_projection_order(self):
        pool,state,ctx,p,f=planned();h=p['plan']['hypotheses'][0]
        j=proposals.project(pool,h,ctx['available'],ctx['tie_priority'])['index']
        permutation=np.arange(40)[::-1];renamed=deepcopy(pool)
        renamed.features=tuple(pool.features[i] for i in permutation);renamed.ids=tuple(f'new{i}' for i in range(40))
        selected=proposals.project(renamed,h,ctx['available'][permutation],ctx['tie_priority'][permutation])['index']
        self.assertEqual(renamed.features[selected],pool.features[j])

    def test_information_can_select_outside_top5_without_modifying_GP(self):
        pool,state,ctx,p,f=planned();before=engine.previous.gp_hash(ctx)
        with patch.object(engine.information,'evaluate',side_effect=fake_information):
            c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True,reasons=['initial']),None)
        self.assertEqual(c['mode'],'information_KG');self.assertNotIn(c['selected_id'],c['ordered_ids'])
        self.assertEqual(before,engine.previous.gp_hash(ctx))
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.2,raw_value=20.))
        self.assertEqual(state['information_used'],1);self.assertEqual(state['edit_calls'],[1])
        self.assertLess(state['hypotheses'][0]['kf']['weight'],.4)
        unchanged=deepcopy(state)
        with self.assertRaises(ValueError):engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.2,raw_value=20.))
        self.assertEqual(state,unchanged)

    def test_dissent_once_and_exhausted_budget_no_override(self):
        pool,state,ctx,p,f=planned()
        with patch.object(engine.information,'evaluate',side_effect=lambda gp,c,ids:({j:.1 for j in ids},None)):
            c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
            self.assertEqual(c['mode'],'information_disagreement')
            state['dissent_used']=True
            c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
            self.assertFalse(c['information_override'])
        state['information_used']=3
        c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
        self.assertFalse(c['information_override'])

    def test_paused_rules_and_repeated_probe_cannot_use_active_budget(self):
        pool,state,ctx,p,f=planned();h=p['plan']['hypotheses'][0]
        h['kf'].update(status='paused',paused_round=1)
        with patch.object(engine.information,'evaluate',side_effect=fake_information):
            c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
            self.assertFalse(c['information_override'])
            h['kf']=kf.initial();state['tested_probe_signatures']=[h['probe_signature']]
            c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
            self.assertFalse(c['information_override'])

    def test_proposal_prompt_contains_no_GP_shortlist_and_forecast_validates_menu(self):
        pool,state,ctx,p,f=planned()
        with patch.object(llm.previous,'task_context',return_value={'objective':'test'}):
            public=llm.proposal_public(pool,state,1,dict(allowed=True),ctx['tie_priority'])
        self.assertNotIn('candidate_menu',public);self.assertEqual(len(public['input_coverage_examples']),24)
        raw=deepcopy(f['raw']);raw['assessments'][0]['probabilities']['unlisted']=.9
        with self.assertRaises(ValueError):llm.validate_forecast(raw,p['plan']['hypotheses'],list(f['plan']['probabilities']['H001']))

    def test_probe_tamper_is_atomic(self):
        pool,state,ctx,p,f=planned()
        with patch.object(engine.information,'evaluate',side_effect=fake_information):c,_=engine.plan(pool,state,ctx,1,p,f,dict(allowed=True),None)
        c['hypotheses_at_commit'][0]['probe']['if_above']='changed';unchanged=deepcopy(state)
        with self.assertRaises(ValueError):engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.5,raw_value=50.))
        self.assertEqual(state,unchanged)

    def test_complete16_round_crash_resume_commits_before_query(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v10_test_') as directory,ExitStack() as stack:
            root=Path(directory);pool,_=fixture();pool.source_path=root/'pool.json';pool.source_path.write_text('{}')
            run=study.directory(root)/'runs'/'synthetic_information';requests=[];queried=[];initial=[];crash=[True]
            protocol=dict(seed=0,protocol_sha256='registered',tasks=['synthetic'],initial=4,queries=16)
            class FakeGP:
                def __init__(self,*a,**kw):self.counters={'fits':0,'operator_compiles_in_loop':0}
                def context(self,history,t):return context(pool,dict(history=history))
                def release(self):pass
            class FakeClient:
                def close(self):pass
            def oracle(r,p,key):
                self.assertTrue((run/'initial_commit.json').exists())
                if key in pool.initial_ids:initial.append(key)
                else:
                    st=study.old.read(run/'state.json');t=len(st['decisions'])+1
                    c=study.old.read(run/'rounds'/f'{t:02d}_commit.json');self.assertEqual(c['selected_id'],key)
                    if crash[0]:crash[0]=False;raise RuntimeError('Synthetic crash after commitment')
                    queried.append(key)
                j=pool.ix[key];return dict(candidate_id=key,y=.1+.01*j,raw_value=10.+j)
            def request(client,system,public,directory,validator):
                requests.append((public['query'],system))
                if system==llm.PROPOSAL_SYSTEM:raw=dict(summary='test',actions=[action()] if public['query']==1 else [])
                else:
                    ids=[p['candidate_id'] for p in public['candidate_menu']]
                    raw=dict(summary='test',assessments=[dict(id=h['id'],probabilities={key:.8 for key in ids})
                        for h in public['knowledge'] if h['kf']['status']!='retracted'])
                return dict(raw=raw,plan=validator(raw),context_sha256=digest(public))
            stack.enter_context(patch.object(study,'check',return_value=protocol))
            stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
            stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.api,'APIClient',return_value=FakeClient()))
            stack.enter_context(patch.object(llm.previous,'task_context',return_value={'objective':'test'}))
            stack.enter_context(patch.object(llm,'request',side_effect=request))
            stack.enter_context(patch.object(engine.information,'evaluate',side_effect=fake_information))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):study.run_one(root,'synthetic')
                committed=study.old.read(run/'rounds/01_commit.json')
                result=study.run_one(root,'synthetic')
            self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(set(queried)),16)
            self.assertEqual(len(initial),4);self.assertEqual(sum(t==1 for t,system in requests),2)
            self.assertEqual(study.old.read(run/'rounds/01_commit.json'),committed)
            self.assertLessEqual(result['information_used'],3)


if __name__=='__main__':unittest.main()
