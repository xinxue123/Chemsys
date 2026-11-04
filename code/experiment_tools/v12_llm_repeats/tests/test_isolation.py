"""Exercise real request caching and the unchanged V12 loop with synthetic oracles."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,shutil,unittest
from chem.benchmarks.knowledge_llm import digest
from knowledge_information_v10.tests.test_pipeline import fixture,context
from reasoning_memory_v12.tests.test_pipeline import action,response
from v12_llm_repeats import study

ROOT=Path(__file__).resolve().parents[4]


class IsolationTests(unittest.TestCase):
    def test_three_fresh_response_namespaces_and_crash_resume_keep_exact_prompts(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_repeats_test_') as directory,ExitStack() as stack:
            root=Path(directory);pool,_=fixture();pool.source_path=root/'pool.json';pool.source_path.write_text('{}')
            rel=Path('code/chemist_system/third_party/reasoning_bo_official/src')
            for name in ('bo/reasoner/base.py','prompts/test_prompts.json'):
                dest=root/rel/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel/name,dest)
            calls=[];queried=[];initial=[];gp_paths=[];crash=[True];active=[None]
            p=dict(seed=0,protocol_sha256='replicate_registered',tasks=['synthetic'],runs=[['synthetic','memory']],
                replication_runs=[['synthetic',r] for r in study.REPLICATES],initial=4,queries=16)
            class FakeGP:
                def __init__(self,pool,path,seed=0):
                    gp_paths.append((path,seed));self.counters={'fits':0,'operator_compiles_in_loop':0}
                def context(self,history,t):return context(pool,dict(history=history))
                def release(self):pass
            class FakeClient:
                def complete(self,system,public,**kwargs):
                    self_test.assertNotIn('replicate',public);self_test.assertNotIn('protocol_sha256',public)
                    ids=[x['candidate_id'] for x in public['candidate_menu']]
                    if system==study.v12.llm.MEMORY_SYSTEM:
                        raw=dict(summary='test',actions=[action(public['incumbent_id'])] if public['query']==1 else [])
                    else:raw=response(ids,public['retrieved_notes']['notes'])
                    calls.append(dict(rep=active[0],query=public['query'],system=system,public=deepcopy(public)))
                    return dict(output_text=json.dumps(raw),id='fresh_response_'+str(len(calls)),usage=dict(input_tokens=1,output_tokens=1,total_tokens=2))
                def close(self):pass
            self_test=self
            def oracle(r,pool,key):
                run=study.run_path(root,'synthetic',active[0]);self.assertTrue((run/'initial_commit.json').exists())
                if key in pool.initial_ids:initial.append((active[0],key))
                else:
                    state=study.old.read(run/'state.json');t=len(state['decisions'])+1;c=study.old.read(run/'rounds'/f'{t:02d}_commit.json')
                    self.assertEqual(c['selected_id'],key)
                    if crash[0]:crash[0]=False;raise RuntimeError('crash after commitment')
                    queried.append((active[0],key))
                j=pool.ix[key];return dict(candidate_id=key,y=.1+.01*j,raw_value=10.+j)
            stack.enter_context(patch.object(study,'check',return_value=p))
            stack.enter_context(patch.object(study.v12.gpmod,'load_task',return_value=pool))
            stack.enter_context(patch.object(study.v12.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.v12.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.v12.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.v12.api,'APIClient',side_effect=lambda config:FakeClient()))
            stack.enter_context(patch.object(study.v12.llm.previous,'task_context',return_value={'objective':'test'}))
            native_run=study.v12.run_one;native_directory=study.v12.directory;native_check=study.v12.check
            with redirect_stdout(StringIO()):
                for rep in study.REPLICATES:
                    active[0]=rep
                    if rep=='r01':
                        with self.assertRaises(RuntimeError):study.run_one(root,'synthetic',rep)
                    result=study.run_one(root,'synthetic',rep);self.assertEqual(result['status'],'COMPLETE')
                count=len(calls);study.run_one(root,'synthetic','r03');self.assertEqual(len(calls),count)
            self.assertIs(study.v12.run_one,native_run);self.assertIs(study.v12.directory,native_directory);self.assertIs(study.v12.check,native_check)
            self.assertTrue(all(path==study.directory(root)/'cache'/'synthetic' and seed==0 for path,seed in gp_paths))
            self.assertEqual(len(initial),12);self.assertEqual(len(queried),48)
            for rep in study.REPLICATES:self.assertEqual(len({key for r,key in queried if r==rep}),16)
            first=[c for c in calls if c['query']==1 and c['system']==study.v12.llm.MEMORY_SYSTEM]
            self.assertEqual(len(first),3);self.assertEqual(len({digest(c['public']) for c in first}),1)
            hashes=[];ids=[]
            for rep in study.REPLICATES:
                run=study.run_path(root,'synthetic',rep);c=study.old.read(run/'rounds/01_commit.json');hashes.append(c['LLM_context_sha256'])
                saved=study.old.read(study.replicate_base(root,'synthetic',rep)/'requests'/c['LLM_context_sha256']/'validated.json');ids.append(saved['response_id'])
            self.assertEqual(len(set(hashes)),1);self.assertEqual(len(set(ids)),3)
            self.assertFalse((root/'experiments/runs/knowledge_loop_20261101/v12_reasoning_memory').exists())

    def test_unregistered_replica_rejected_before_any_write(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_repeats_guard_') as folder:
            root=Path(folder)
            with patch.object(study,'check',return_value={'replication_runs':[['synthetic','r01']]}):
                with self.assertRaises(ValueError):study.run_one(root,'synthetic','r02')
            self.assertFalse(study.directory(root).exists())

    def test_fresh_launch_rejects_populated_response_namespace(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_repeats_cache_') as folder:
            root=Path(folder);base=study.replicate_base(root,'synthetic','r01');(base/'requests').mkdir(parents=True)
            p=dict(protocol_sha256='frozen',replication_runs=[['synthetic','r01']])
            with patch.object(study,'check',return_value=p),patch.object(study.v12,'run_one') as run:
                with self.assertRaises(ValueError):study.run_one(root,'synthetic','r01')
                run.assert_not_called()
            self.assertFalse((base/'launch.json').exists())

    def test_replica_tags_never_become_paths_outside_workspace(self):
        for task,rep in [('../history','r01'),('HPLC','../r01'),('HPLC','r00')]:
            with self.assertRaises(ValueError):study.replicate_base(ROOT,task,rep)


if __name__=='__main__':unittest.main()
