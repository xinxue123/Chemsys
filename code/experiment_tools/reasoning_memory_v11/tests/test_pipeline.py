from copy import deepcopy
from contextlib import ExitStack,redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil,unittest
from unittest.mock import patch
from chem.benchmarks.knowledge_llm import digest
from knowledge_information_v10.tests.test_pipeline import fixture,context
from reasoning_memory_v11 import llm,engine,study
from knowledge_rank_v9 import policy,kf

ROOT=Path(__file__).resolve().parents[4]


def action(reference='p3'):
    return dict(op='add',id='H001',claim='x may improve objective',focus=['x'],effect='promote',scope={},reference_id=reference,reason='prospective contrast')


def response(ids,hs):
    return dict(insight=dict(comment='own observations only',keywords='contrast',hypotheses=[
        dict(strategy='a',rationale='test trend',confidence=conf,parameter_sets=[dict(candidate_id=key)])
        for conf,key in zip(['medium','high','high'],[ids[0],ids[2],ids[1]])]),
        assessments=[dict(id=h['id'],probabilities={key:.8 for key in ids}) for h in hs if h['kf']['status']!='retracted'],memory_references=[])


def planned():
    pool,state=fixture();state['arm']='rbo_memory';ctx=context(pool,state)
    _,display=engine.previous.candidates(pool,ctx,1);ids=[pool.ids[j] for j in display]
    raw=dict(summary='new note',actions=[action()]);memory=dict(raw=raw,plan=llm.validate_memory(raw,pool,state,ids,1),context_sha256='memory')
    hs=memory['plan']['hypotheses'];raw=response(ids,hs)
    reasoning=dict(raw=raw,plan=llm.validate_reasoning(ROOT,raw,hs,ids,llm.retrieve(pool,hs,ids)),context_sha256='reasoning')
    return pool,state,ctx,ids,memory,reasoning


class PipelineTests(unittest.TestCase):
    def test_official_stable_sort_and_compile_once(self):
        llm.compiled_extractor.cache_clear();raw=response(['a','b','c'],[])['insight']
        self.assertEqual(llm.official_extract(ROOT,raw),'c')
        raw['hypotheses'][1]['parameter_sets'].append(dict(candidate_id='a'))
        self.assertEqual(llm.official_extract(ROOT,raw),'c')
        self.assertEqual(llm.compiled_extractor.cache_info().misses,1)

    def test_same_official_insight_cannot_be_overridden_by_weights_or_forecasts(self):
        pool,state,ctx,ids,m,r=planned();before=engine.previous.gp_hash(ctx)
        baseline=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        for status,weight,p in [('active',0.,.0),('paused',.8,1.)]:
            m['plan']['hypotheses'][0]['kf'].update(status=status,weight=weight,paused_round=1 if status=='paused' else None)
            r['plan']['probabilities']['H001']={key:p for key in ids}
            c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
            self.assertEqual(c['selected_id'],baseline['selected_id']);self.assertEqual(c['GP_sha256'],before)
        self.assertEqual(engine.previous.gp_hash(ctx),before)

    def test_note_scope_and_caution_survive(self):
        pool,state,ctx,ids,m,r=planned();h=m['plan']['hypotheses'][0]
        h['kf'].update(status='retracted',weight=0.)
        found=llm.retrieve(pool,[h],ids)
        self.assertEqual(len(found['cautionary_notes']),1);self.assertEqual(found['supporting_notes'],[])
        h['scope']={'x':{'bounds':[.9,1.]}};h['semantic_sha256']=policy.semantics(h);h['structure_sha256']=policy.structural(h)
        self.assertEqual(llm.retrieve(pool,[h],ids)['cautionary_notes'],[])

    def test_new_reference_must_be_observed_incumbent(self):
        pool,state,ctx,ids,m,r=planned()
        bad=llm.validate_memory(dict(summary='test',actions=[action('p1')]),pool,state,ids,1)
        self.assertEqual(bad['hypotheses'],[]);self.assertFalse(bad['reviews'][0]['accepted'])
        self.assertEqual(m['plan']['hypotheses'][0]['reference_id'],'p3')

    def test_no_outside_menu_or_missing_forecast(self):
        pool,state,ctx,ids,m,r=planned();hs=m['plan']['hypotheses'];retrieved=llm.retrieve(pool,hs,ids)
        raw=deepcopy(r['raw']);raw['insight']['hypotheses'][0]['parameter_sets'][0]['candidate_id']='unlisted'
        with self.assertRaises(ValueError):llm.validate_reasoning(ROOT,raw,hs,ids,retrieved)
        raw=deepcopy(r['raw']);raw['assessments']=[]
        with self.assertRaises(ValueError):llm.validate_reasoning(ROOT,raw,hs,ids,retrieved)

    def test_visible_forecast_note_citation_cannot_veto_official_choice(self):
        pool,state,ctx,ids,m,r=planned();hs=m['plan']['hypotheses'];raw=deepcopy(r['raw'])
        empty=dict(supporting_notes=[],cautionary_notes=[])
        raw['memory_references']=[dict(id='H001',version=1,use='counterevidence')]
        result=llm.validate_reasoning(ROOT,raw,hs,ids,empty)
        self.assertEqual(result['selected_id'],r['plan']['selected_id'])
        raw['memory_references'][0]['version']=999
        with self.assertRaises(ValueError):llm.validate_reasoning(ROOT,raw,hs,ids,empty)

    def test_prompt_uses_only_own_prefix_and_advisory_memory(self):
        pool,state,ctx,ids,m,r=planned()
        with patch.object(llm.previous,'task_context',return_value={'objective':'test'}):
            pub=llm.reasoning_public(ROOT,pool,state,m['plan']['hypotheses'],ids,1)
        self.assertEqual([row['candidate_id'] for row in pub['own_observations']],list(pool.initial_ids))
        self.assertEqual([row['candidate_id'] for row in pub['candidate_menu']],ids)
        self.assertNotIn('mu_y',pub);self.assertNotIn('base_logei',pub)
        self.assertIn('4 completed experiments',pub['official_optimization_prompt'])

    def test_feedback_once_atomic_and_negative_inheritance(self):
        pool,state,ctx,ids,m,r=planned();c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.1,raw_value=10.))
        self.assertEqual(state['edit_calls'],[1]);h=state['hypotheses'][0]
        self.assertLess(h['kf']['weight'],.4);unchanged=deepcopy(state)
        with self.assertRaises(ValueError):engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.1,raw_value=10.))
        self.assertEqual(state,unchanged)
        revise=action();revise.update(op='revise',claim='revised prospective explanation')
        notes=llm.validate_memory(dict(summary='revise',actions=[revise]),pool,state,ids,4)['hypotheses']
        self.assertEqual(notes[-1]['kf']['status'],'paused');self.assertEqual(len(notes[-1]['inherited_negative']),1)
        self.assertEqual(notes[0]['kf']['status'],'retracted')

    def test_pause_resume_and_retraction_remain_available_in_memory(self):
        value=kf.initial()
        for t in (1,2):value,_=kf.update(value,.9,.1,False,t)
        self.assertEqual(value['status'],'paused')
        for t in (3,4,5):value,_=kf.update(value,.9,.1,True,t)
        self.assertEqual(value['status'],'active')
        for t in (6,7,8,9):value,_=kf.update(value,.9,.1,False,t)
        self.assertEqual(value['status'],'retracted')

    def test_complete16_round_crash_resume_commits_before_query(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v11_test_') as directory,ExitStack() as stack:
            root=Path(directory);pool,_=fixture();pool.source_path=root/'pool.json';pool.source_path.write_text('{}')
            rel=Path('code/chemist_system/third_party/reasoning_bo_official/src')
            for name in ('bo/reasoner/base.py','prompts/test_prompts.json'):
                dest=root/rel/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel/name,dest)
            run=study.directory(root)/'runs'/'synthetic_rbo_memory';requests=[];queried=[];initial=[];crash=[True]
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
                requests.append((public['query'],system));ids=[p['candidate_id'] for p in public['candidate_menu']]
                if system==llm.MEMORY_SYSTEM:
                    raw=dict(summary='test',actions=[action(public['incumbent_id'])] if public['query']==1 else [])
                else:raw=response(ids,public['forecast_notes'])
                return dict(raw=raw,plan=validator(raw),context_sha256=digest(public))
            # Public Notes contain lifecycle directly, while response() accepts internal notes.
            original=response
            def public_response(ids,notes):
                converted=[dict(h,kf={'status':h.get('lifecycle','active')}) for h in notes]
                return original(ids,converted)
            stack.enter_context(patch(__name__+'.response',side_effect=public_response))
            stack.enter_context(patch.object(study,'check',return_value=protocol))
            stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
            stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.api,'APIClient',return_value=FakeClient()))
            stack.enter_context(patch.object(llm.previous,'task_context',return_value={'objective':'test'}))
            stack.enter_context(patch.object(llm,'request',side_effect=request))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):study.run_one(root,'synthetic')
                committed=study.old.read(run/'rounds/01_commit.json');result=study.run_one(root,'synthetic')
            self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(set(queried)),16)
            self.assertEqual(len(initial),4);self.assertEqual(sum(t==1 for t,system in requests),2)
            self.assertEqual(study.old.read(run/'rounds/01_commit.json'),committed)
            self.assertLessEqual(result['LLM_calls'],20)


if __name__=='__main__':unittest.main()
