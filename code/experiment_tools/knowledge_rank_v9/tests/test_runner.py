"""Commit-before-reveal and crash-resume under all registered arms, no API/fit."""
from contextlib import ExitStack,redirect_stdout
from io import StringIO
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from chem.benchmarks.knowledge_llm import write
from knowledge_rank_v9 import study,llm
from test_core import setup,context,action

ROOT=Path(__file__).resolve().parents[4]


class RunnerTests(unittest.TestCase):
    def test_all_arms_complete_and_resume_without_repeating_selection_call(self):
        for arm in study.ARMS:
            with self.subTest(arm=arm),TemporaryDirectory(dir=ROOT/'runtime',prefix='rank_v9_test_') as temporary,ExitStack() as stack:
                root=Path(temporary);pool,_=setup();pool.initial_ids=tuple(pool.ids[:4]);pool.source_path=root/'pool.json'
                pool.source_path.write_text('{}',encoding='utf-8')
                for name in ('src/prompts/test_prompts.json','src/bo/reasoner/base.py'):
                    relative=Path('code/chemist_system/third_party/reasoning_bo_official')/name
                    (root/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,root/relative)
                protocol=dict(seed=0,protocol_sha256='registered',tasks=['synthetic'],arms=list(study.ARMS),initial=4,queries=16)
                run=study.directory(root)/'runs'/f'synthetic_{arm}'
                initial_count=[];queried=[];requests=[];crash=[True]
                class FakeGP:
                    def __init__(self,*args,**kwargs):self.counters={'fits':0,'operator_compiles_in_loop':0}
                    def context(self,history,t):
                        c=context(pool,dict(history=history));c['stats']['fit_parameters']={'fake':True};return c
                    def release(self):pass
                class FakeClient:
                    def close(self):pass
                def oracle(_root,_pool,key):
                    init=run/'initial_commit.json';self.assertTrue(init.exists())
                    if key in pool.initial_ids:initial_count.append(key)
                    else:
                        state=study.old.read(run/'state.json');t=len(state['decisions'])+1
                        cp=run/'rounds'/f'{t:02d}_commit.json';self.assertTrue(cp.exists())
                        self.assertEqual(study.old.read(cp)['selected_id'],key)
                        if crash[0]:crash[0]=False;raise RuntimeError('Synthetic interrupted revelation')
                        queried.append(key)
                    j=pool.ix[key];return dict(candidate_id=key,y=.1+.001*j,raw_value=10.+.1*j)
                def fake_request(client,system,public,directory,validator):
                    requests.append(public['query']);ids=[row['candidate_id'] for row in public['candidate_menu']]
                    if system==llm.KF_SYSTEM:
                        live={h['id']:h for h in public['knowledge'] if h['kf']['status']!='retracted'}
                        actions=[action()] if public['query']==1 else []
                        if actions:live['H001']={}
                        raw=dict(summary='test',actions=actions,assessments=[dict(id=key,probabilities={x:.5 for x in ids}) for key in live])
                    else:
                        raw=dict(comment='test',keywords='test',hypotheses=[dict(strategy='test',rationale='test',confidence='high',
                            parameter_sets=[dict(candidate_id=key)]) for key in ids[:3]])
                    return dict(context_sha256='context',raw=raw,plan=validator(raw))
                stack.enter_context(patch.object(study,'check',return_value=protocol))
                stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
                stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
                stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
                stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
                stack.enter_context(patch.object(study.api,'APIClient',return_value=FakeClient()))
                stack.enter_context(patch.object(llm,'request',side_effect=fake_request))
                stack.enter_context(patch.object(llm,'task_context',return_value={'objective':'synthetic'}))
                with redirect_stdout(StringIO()):
                    with self.assertRaises(RuntimeError):study.run_one(root,'synthetic',arm)
                    first=study.old.read(run/'rounds/01_commit.json')
                    result=study.run_one(root,'synthetic',arm)
                self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(queried),16);self.assertEqual(len(set(queried)),16)
                self.assertEqual(len(initial_count),4);self.assertEqual(first,study.old.read(run/'rounds/01_commit.json'))
                self.assertEqual(requests.count(1),0 if arm=='bo' else 1)
                self.assertEqual(result['queries'],16)
                saved=study.old.read(run/'state.json');bad=dict(saved);bad.pop('initial_commit_sha256')
                write(run/'state.json',bad)
                with self.assertRaises(KeyError):study.run_one(root,'synthetic',arm)


if __name__=='__main__':unittest.main()
