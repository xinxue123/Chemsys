"""Exercise changed initialization routing without substituting the native loops."""
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
from v12_initialization_seeds import study

ROOT=Path(__file__).resolve().parents[4]


class SeedAdapterTests(unittest.TestCase):
    def test_initialization_draw_depends_on_features_not_ids_or_storage_order(self):
        pool,_=fixture();renamed=deepcopy(pool)
        renamed.ids=tuple('new'+str(j) for j in range(len(pool.ids)))
        renamed.features=tuple(reversed(pool.features));renamed.ix={k:i for i,k in enumerate(renamed.ids)}
        draws=[]
        for seed in study.SEEDS:
            a,_=study.draw_initial_ids(pool,seed);b,_=study.draw_initial_ids(renamed,seed)
            self.assertEqual([pool.features[pool.ix[k]] for k in a],[renamed.features[renamed.ix[k]] for k in b])
            self.assertEqual(len(set(a)),4);draws.append(a)
        self.assertEqual(len(set(draws)),3)

    def test_native_paired_loops_use_matching_inputs_and_isolated_fresh_responses(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_seeds_test_') as folder,ExitStack() as stack:
            root=Path(folder);pool,_=fixture();pool.source_path=root/'pool.json';pool.source_path.write_text('{}')
            rel=Path('code/chemist_system/third_party/reasoning_bo_official/src')
            for name in ('bo/reasoner/base.py','prompts/test_prompts.json'):
                dest=root/rel/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel/name,dest)
            p=dict(seed=0,protocol_sha256='registered',tasks=['synthetic'],runs=[['synthetic',a] for a in study.ARMS],
                paired_runs=[['synthetic',s] for s in study.SEEDS],initial=4,queries=16,initial_registry_sha256='input_only')
            pools={};queried=[];calls=[];gp_paths=[];crash=[True];active=[None];self_test=self
            for seed in study.SEEDS:
                pp=deepcopy(pool);pp.initial_ids=study.draw_initial_ids(pp,seed)[0];pools[seed]=pp
                study.write(study.directory(root)/'preflight/synthetic'/f'seed_{seed}/receipt.json',
                    dict(status='PASS_SHARED_INITIAL_GP_AND_MENU',initial_ids=list(pp.initial_ids)))
            class FakeGP:
                def __init__(self,pp,path,seed=0):
                    self.pool=pp;gp_paths.append((active[0],path,seed));self.counters={'fits':0,'operator_compiles_in_loop':0}
                def context(self,history,t):return context(self.pool,dict(history=history))
                def release(self):pass
            class FakeClient:
                def complete(self,system,public,**kwargs):
                    self_test.assertNotIn('initialization_seed',public);self_test.assertNotIn('protocol_sha256',public)
                    ids=[x['candidate_id'] for x in public['candidate_menu']]
                    if system==study.v12.llm.MEMORY_SYSTEM:
                        raw=dict(summary='test',actions=[action(public['incumbent_id'])] if public['query']==1 else [])
                    else:raw=response(ids,public['retrieved_notes']['notes'])
                    calls.append((active[0],public['query']))
                    return dict(output_text=json.dumps(raw),id='fresh_'+str(len(calls)),usage={})
                def close(self):pass
            def oracle(r,pp,key):
                seed=active[0];self.assertEqual(pp.initial_ids,pools[seed].initial_ids)
                if key not in pp.initial_ids:
                    matches=[]
                    for arm in study.ARMS:
                        run=study.run_path(root,'synthetic',seed,arm);sf=run/'state.json'
                        if not sf.exists():continue
                        state=study.old.read(sf);t=len(state['decisions'])+1;cp=run/'rounds'/f'{t:02d}_commit.json'
                        if cp.exists() and study.old.read(cp)['selected_id']==key:matches.append(arm)
                    self.assertEqual(len(matches),1);arm=matches[0]
                    if seed==1 and arm=='memory' and crash[0]:crash[0]=False;raise RuntimeError('crash after committed LLM choice')
                    queried.append((seed,arm,key))
                j=pp.ix[key];return dict(candidate_id=key,y=.1+.01*j,raw_value=10.+j)
            stack.enter_context(patch.object(study,'check',return_value=p))
            stack.enter_context(patch.object(study,'registered_pool',side_effect=lambda r,t,s,p:pools[s]))
            stack.enter_context(patch.object(study.v12.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.v12.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.v12.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.v12.api,'APIClient',side_effect=lambda config:FakeClient()))
            stack.enter_context(patch.object(study.v12.llm.previous,'task_context',return_value={'objective':'test'}))
            native=(study.v12.run_one,study.v9.run_one,study.v12.directory,study.v9.directory,study.v12.gpmod.load_task)
            with redirect_stdout(StringIO()):
                for seed in study.SEEDS:
                    active[0]=seed
                    if seed==1:
                        with self.assertRaises(RuntimeError):study.run_pair(root,'synthetic',seed)
                    result=study.run_pair(root,'synthetic',seed);self.assertTrue(all(r['status']=='COMPLETE' for r in result))
                count=len(calls);study.run_pair(root,'synthetic',3);self.assertEqual(len(calls),count)
            self.assertEqual(native,(study.v12.run_one,study.v9.run_one,study.v12.directory,study.v9.directory,study.v12.gpmod.load_task))
            self.assertTrue(all(path==study.directory(root)/'cache/synthetic' and s==0 for _,path,s in gp_paths))
            for seed in study.SEEDS:
                states=[study.old.read(study.run_path(root,'synthetic',seed,a)/'state.json') for a in study.ARMS]
                self.assertEqual(states[0]['history'][:4],states[1]['history'][:4])
                self.assertEqual([r['candidate_id'] for r in states[0]['history'][:4]],list(pools[seed].initial_ids))
                for arm in study.ARMS:self.assertEqual(sum(s==seed and a==arm for s,a,k in queried),16)
                self.assertEqual(sum(s==seed for s,t in calls),len(states[1]['edit_calls'])+16)
            ids=[]
            for path in study.directory(root).glob('s*/*/requests/*/validated.json'):ids.append(study.old.read(path)['response_id'])
            self.assertEqual(len(ids),len(calls));self.assertEqual(len(set(ids)),len(ids))
            self.assertFalse((root/'experiments/runs/knowledge_loop_20261101/v12_reasoning_memory').exists())

    def test_unregistered_pairs_and_unsafe_paths_rejected(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_seeds_guard_') as folder:
            with patch.object(study,'check',return_value={'paired_runs':[['HPLC',1]]}):
                with self.assertRaises(ValueError):study.run_pair(Path(folder),'HPLC',2)
            self.assertFalse(study.directory(folder).exists())
        for task,seed in [('HPLC',0),('HPLC',True),('../history',1)]:
            with self.assertRaises(ValueError):study.pair_base(ROOT,task,seed)

    def test_fresh_pair_rejects_populated_LLM_cache(self):
        with TemporaryDirectory(dir=ROOT/'runtime',prefix='v12_seeds_cache_') as folder:
            root=Path(folder);base=study.pair_base(root,'HPLC',1);(base/'requests').mkdir(parents=True)
            study.write(study.directory(root)/'preflight/HPLC/seed_1/receipt.json',dict(status='PASS_SHARED_INITIAL_GP_AND_MENU'))
            with patch.object(study,'check',return_value=dict(paired_runs=[['HPLC',1]],protocol_sha256='p')),patch.object(study.v12,'run_one') as runner:
                with self.assertRaises(ValueError):study.run_pair(root,'HPLC',1)
                runner.assert_not_called()
            self.assertFalse((base/'launch.json').exists())


if __name__=='__main__':unittest.main()
