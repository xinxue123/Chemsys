from copy import deepcopy
from contextlib import ExitStack,redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil,unittest
from unittest.mock import patch
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_information_v10.tests.test_pipeline import fixture,context
from reasoning_memory_v12 import llm,engine,policy,study
from knowledge_rank_v9 import kf

ROOT=Path(__file__).resolve().parents[4]


def action(reference='p3'):
    return dict(op='add',id='H001',claim='x may improve objective',focus=['x'],effect='promote',scope={},
        reference_id=reference,reason='prospective contrast',evidence_queries=[])


def response(ids,notes):
    return dict(insight=dict(comment='own observations only',keywords='contrast',hypotheses=[
        dict(strategy='a',rationale='test trend',confidence=conf,parameter_sets=[dict(candidate_id=key)])
        for conf,key in zip(['medium','high','high'],[ids[0],ids[2],ids[1]])]),
        assessments=[dict(id=h['id'],version=h['version'],p_improve_incumbent=.8) for h in notes if h.get('lifecycle',h.get('kf',{}).get('status'))!='retracted'],
        memory_references=[],decision_purpose=dict(purpose='learn',uncertainty='test x',if_improved='test neighboring x',if_not='test alternatives'))


def planned(state=None,t=1):
    pool,initial=fixture();state=state if state is not None else initial;state['arm']='memory'
    state.setdefault('category_menu_insertions',0);state.setdefault('category_selected_queries',0)
    ctx=context(pool,state);_,_,display,_=engine.menu(pool,state,ctx,t,'memory');ids=[pool.ids[int(j)] for j in display]
    memory=None
    if t==1:
        raw=dict(summary='new note',actions=[action()]);memory=dict(raw=raw,plan=llm.validate_memory(raw,pool,state,ids,t),context_sha256='memory')
    hs=memory['plan']['hypotheses'] if memory else state['hypotheses'];raw=response(ids,hs)
    reasoning=dict(raw=raw,plan=llm.validate_reasoning(ROOT,raw,pool,hs,ids,llm.retrieve(pool,hs,ids)),context_sha256='reasoning')
    return pool,state,ctx,ids,memory,reasoning


class PipelineTests(unittest.TestCase):
    def test_official_extractor_compiles_once_and_weights_do_not_override(self):
        llm.compiled_extractor.cache_clear();pool,state,ctx,ids,m,r=planned();before=engine.previous.gp_hash(ctx)
        c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        for status,weight in [('active',0.),('paused',.8)]:
            m['plan']['hypotheses'][0]['kf'].update(status=status,weight=weight)
            r['plan']['forecasts']['H001']['p']=.01
            other=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
            self.assertEqual(c['selected_id'],other['selected_id']);self.assertEqual(other['GP_sha256'],before)
        self.assertEqual(llm.compiled_extractor.cache_info().misses,1)

    def test_sidecar_errors_never_repair_a_valid_insight(self):
        pool,state,ctx,ids,m,r=planned();hs=m['plan']['hypotheses'];raw=deepcopy(r['raw'])
        raw.update(assessments=[dict(id='H001',version=1,p_improve_incumbent=float('nan'))],memory_references=[dict(id='unknown',version=9,use='support')],decision_purpose=None)
        result=llm.validate_reasoning(ROOT,raw,pool,hs,ids,llm.retrieve(pool,hs,ids))
        self.assertEqual(result['selected_id'],r['plan']['selected_id']);self.assertEqual(result['forecasts'],{})
        self.assertEqual(result['memory_references'][0]['use'],'unsupported');self.assertGreater(len(result['warnings']),0)
        raw['insight']['hypotheses'][0]['parameter_sets'][0]['candidate_id']='not-in-menu'
        with self.assertRaises(ValueError):llm.validate_reasoning(ROOT,raw,pool,hs,ids,llm.retrieve(pool,hs,ids))

    def test_immature_and_paused_support_becomes_exploration(self):
        pool,state,ctx,ids,m,r=planned();hs=m['plan']['hypotheses'];raw=deepcopy(r['raw'])
        raw['memory_references']=[dict(id='H001',version=1,use='support')]
        for status in ('active','paused'):
            hs[0]['kf']['status']=status
            result=llm.validate_reasoning(ROOT,raw,pool,hs,ids,llm.retrieve(pool,hs,ids))
            self.assertEqual(result['memory_references'][0]['use'],'exploratory_transfer')
            self.assertEqual(result['selected_id'],r['plan']['selected_id'])
        hs[0]['kf']['status']='active';hs[0]['events']=[dict(success=True,Brier_gain=.1),dict(success=True,Brier_gain=.1)]
        self.assertTrue(policy.support_eligible(hs[0]))
        hs[0]['events'][1]['success']=False;self.assertFalse(policy.support_eligible(hs[0]))

    def test_only_relevant_notes_visible_and_out_of_scope_choice_remains_valid(self):
        pool,state,ctx,ids,m,r=planned();h=m['plan']['hypotheses'][0]
        h['scope']={'x':{'bounds':[.9,1.]}};h['semantic_sha256']=policy.base.semantics(h);h['structure_sha256']=policy.base.structural(h)
        found=llm.retrieve(pool,[h],ids);self.assertEqual(found['notes'],[])
        raw=response(ids,[]);raw['memory_references']=[dict(id='H001',version=1,use='support')]
        result=llm.validate_reasoning(ROOT,raw,pool,[h],ids,found)
        self.assertEqual(result['selected_id'],r['plan']['selected_id']);self.assertEqual(result['expected_forecasts'],0)
        with patch.object(llm.previous,'task_context',return_value={'objective':'test'}):pub=llm.reasoning_public(ROOT,pool,state,[h],ids,1)
        self.assertNotIn('forecast_notes',pub);self.assertNotIn('mu_y',pub);self.assertEqual(pub['retrieved_notes']['notes'],[])

    def test_current_incumbent_changes_but_birth_reference_does_not(self):
        pool,state,ctx,ids,m,r=planned();c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.6,raw_value=60.))
        self.assertEqual(state['hypotheses'][0]['reference_y'],.4)
        pool,state,ctx,ids,m,r=planned(state,2);c2=engine.plan(ROOT,pool,state,ctx,2,m,r,dict(allowed=False))
        self.assertEqual(c2['predictions'][0]['prediction']['reference_y'],.6)
        self.assertNotEqual(c2['predictions'][0]['prediction']['reference_id'],state['hypotheses'][0]['reference_id'])
        engine.feedback(state,c2,dict(candidate_id=c2['selected_id'],y=.5,raw_value=50.))
        self.assertFalse(state['decisions'][-1]['KF_feedback'][0]['success']);self.assertEqual(state['hypotheses'][0]['reference_y'],.4)
        snapshot=deepcopy(state)
        with self.assertRaises(ValueError):engine.feedback(state,c2,dict(candidate_id=c2['selected_id'],y=.5,raw_value=50.))
        self.assertEqual(snapshot,state)

    def test_changed_focus_inherits_facts_without_numeric_replay(self):
        pool,state,ctx,ids,m,r=planned();c=engine.plan(ROOT,pool,state,ctx,1,m,r,dict(allowed=True))
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.1,raw_value=10.))
        a=action();a.update(op='revise',focus=['category'],claim='test an alternative categorical explanation',evidence_queries=[1])
        notes,reviews=policy.apply(pool,state['hypotheses'],[a],state['history'],4)
        self.assertTrue(reviews[0]['accepted']);self.assertEqual(len(notes[-1]['carried_fact_ids']),1)
        self.assertEqual(notes[-1]['kf'],kf.initial());self.assertEqual(notes[-1]['inherited_negative'],[])
        self.assertEqual(notes[-1]['events'],[]);self.assertFalse(policy.support_eligible(notes[-1]))

    def test_scopes_need_pool_support_not_current_menu_and_no_coverage_retraction(self):
        pool,state,ctx,ids,m,r=planned();a=action();a['scope']={'x':{'bounds':[.9,1.]}}
        hs,reviews=policy.apply(pool,[],[a],state['history'],1)
        self.assertTrue(reviews[0]['accepted']);self.assertEqual(len(policy.scope_indices(pool,hs[0],[pool.ix[k] for k in ids])),0)
        a.update(op='retract',claim=None,effect=None,reference_id=None,focus=[],scope={},reason='not in current menu')
        new,rr=policy.apply(pool,hs,[a],state['history'],4)
        self.assertFalse(rr[0]['accepted']);self.assertEqual(new,hs)

    def test_events_do_not_force_out_of_scope_or_empty_late_refresh(self):
        pool,state,ctx,ids,m,r=planned();state['edit_calls']=[1];state['hypotheses']=m['plan']['hypotheses']
        self.assertEqual(engine.edit_event(pool,state,[],4),dict(allowed=False,reasons=[]))
        state['edit_calls']=[1,4,7]
        self.assertEqual(engine.edit_event(pool,state,[],11),dict(allowed=False,reasons=[]))
        state['decisions']=[dict(selected_id='p5',incumbent_before_y=.4,incumbent_after_y=.4,KF_feedback=[]) for _ in range(3)]
        self.assertTrue(engine.edit_event(pool,state,[],11)['allowed'])
        self.assertFalse(engine.edit_event(pool,state,[],10)['allowed'])

    def test_category_ablation_only_changes_one_menu_member_and_never_late(self):
        pool,state=fixture();state['arm']='category';state['category_menu_insertions']=0
        keys=['p0','p1','p3','p5','p7','p9','p11'];state['history']=[dict(candidate_id=k,y=.2,raw_value=20.) for k in keys]
        state['decisions']=[dict(selected_id=k) for k in keys[-3:]];ctx=context(pool,state)
        ctx['base_logei'][np.array([f[1]=='B' for f in pool.features])]-=2
        original,menu,display,audit=engine.menu(pool,state,ctx,4,'category')
        self.assertTrue(audit['inserted']);self.assertEqual(list(original[:4]),list(menu[:4]));self.assertEqual(len(set(display)),5)
        self.assertEqual(pool.features[pool.ix[audit['challenger_id']]][1],'B')
        self.assertFalse(engine.menu(pool,state,ctx,4,'memory')[-1]['inserted'])
        self.assertFalse(engine.menu(pool,state,ctx,9,'category')[-1]['inserted'])
        state['category_menu_insertions']=2;self.assertFalse(engine.menu(pool,state,ctx,4,'category')[-1]['inserted'])
        state['category_menu_insertions']=0
        order=np.arange(len(pool.ids))[::-1];renamed=deepcopy(pool)
        renamed.features=tuple(pool.features[i] for i in order);renamed.ids=tuple(f'renamed{i}' for i in range(len(order)));renamed.ix={k:i for i,k in enumerate(renamed.ids)}
        mapping={pool.ids[old]:renamed.ids[j] for j,old in enumerate(order)};s2=deepcopy(state)
        for row in s2['history']:row['candidate_id']=mapping[row['candidate_id']]
        for row in s2['decisions']:row['selected_id']=mapping[row['selected_id']]
        c2=deepcopy(ctx)
        for key in ('base_logei','available','tie_priority'):c2[key]=ctx[key][order]
        a2=engine.menu(renamed,s2,c2,4,'category')[-1]
        self.assertEqual(pool.features[pool.ix[audit['challenger_id']]],renamed.features[renamed.ix[a2['challenger_id']]])

    def test_complete16_round_crash_resume_commits_before_query(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_test_') as directory,ExitStack() as stack:
            root=Path(directory);pool,_=fixture();pool.source_path=root/'pool.json';pool.source_path.write_text('{}')
            rel=Path('code/chemist_system/third_party/reasoning_bo_official/src')
            for name in ('bo/reasoner/base.py','prompts/test_prompts.json'):
                dest=root/rel/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel/name,dest)
            run=study.directory(root)/'runs'/'synthetic_memory';requests=[];queried=[];initial=[];crash=[True]
            protocol=dict(seed=0,protocol_sha256='registered',tasks=['synthetic'],runs=[['synthetic','memory']],initial=4,queries=16)
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
                raw=(dict(summary='test',actions=[action(public['incumbent_id'])] if public['query']==1 else []) if system==llm.MEMORY_SYSTEM
                    else response(ids,public['retrieved_notes']['notes']))
                return dict(raw=raw,plan=validator(raw),context_sha256=digest(public))
            stack.enter_context(patch.object(study,'check',return_value=protocol))
            stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
            stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.api,'APIClient',return_value=FakeClient()))
            stack.enter_context(patch.object(llm.previous,'task_context',return_value={'objective':'test'}))
            stack.enter_context(patch.object(llm,'request',side_effect=request))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):study.run_one(root,'synthetic','memory')
                committed=study.old.read(run/'rounds/01_commit.json');result=study.run_one(root,'synthetic','memory')
            self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(set(queried)),16)
            self.assertEqual(len(initial),4);self.assertEqual(sum(t==1 for t,system in requests),2)
            self.assertEqual(study.old.read(run/'rounds/01_commit.json'),committed);self.assertLessEqual(result['LLM_calls'],20)


if __name__=='__main__':unittest.main()
