"""V8 runner registration and commit wiring; all inputs/clients/GPs synthetic."""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from io import StringIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from chem.benchmarks import knowledge_study as study
from chem.benchmarks import knowledge_numeric_engine as engine
from chem.benchmarks import knowledge_numeric_llm as llm
from chem.benchmarks import knowledge_numeric_policy as policy
from chem.benchmarks import knowledge_birth_gp as birth
from test_knowledge_study import FakeGP,make_pool,observation

ROOT=Path(__file__).resolve().parents[3]


def temporary_workspace():
    return tempfile.TemporaryDirectory(prefix='numeric_runner_test_',dir=ROOT/'runtime')


class NumericStudyTests(unittest.TestCase):
    def test_components_and_frozen_protocol_register_only_new_v8_arms(self):
        self.assertEqual(study.components('v8'),(engine,llm,policy))
        with temporary_workspace() as temporary:
            root=Path(temporary)
            files={
                'docs/knowledge_birth_v8_design_20261101.txt':'synthetic preregistration only',
                'code/resources/knowledge_loop_goal_20261101_majority.json':'{}',
                'experiments/reports/knowledge_loop_20261101/baseline_inventory.json':json.dumps({'tasks':[]})}
            for name,text in files.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
            with patch.object(study,'source_hashes',return_value={'synthetic_source.py':'hash-one'}):
                first=study.prepare(root,'v8');path=study.paths(root,'v8')/'protocol.json'
                contents=path.read_bytes();second=study.prepare(root,'v8')
                self.assertEqual(first,second);self.assertEqual(contents,path.read_bytes())
            self.assertEqual(first['mechanism'],'frozen_birth_numeric_v8')
            self.assertEqual(first['proposal_interface'],'numeric_birth_contrast_v1')
            self.assertEqual(first['supported_modes'],['bo','full','verification','no_calibration'])
            self.assertEqual(first['supported_conditions'],['agent_initial_terra'])
            self.assertEqual(first['stage1_arms'],['bo','full'])
            self.assertEqual(first['stage2_arms'],['verification','no_calibration'])
            self.assertEqual(first['seed'],0);self.assertEqual((first['initial'],first['queries']),(4,16))
            self.assertEqual(first['design_document_sha256'],study.sha(root/'docs/knowledge_birth_v8_design_20261101.txt'))
            self.assertEqual(first['prompt_sha256'],llm.digest(llm.SYSTEM))
            with patch.object(study,'source_hashes',return_value={'synthetic_source.py':'hash-two'}):
                with self.assertRaises(AssertionError):study.prepare(root,'v8')
            self.assertEqual(contents,path.read_bytes())
            for seed in (1,2):
                with self.assertRaises(ValueError):study.prepare(root,'v8',seed=seed)

    def test_stage2_requires_all_matched_tasks_majority_and_weighted_selection(self):
        with temporary_workspace() as temporary:
            root=Path(temporary);tasks=[f'synthetic{i}' for i in range(8)]
            protocol=dict(tasks=tasks,protocol_sha256='synthetic-stage')
            with self.assertRaises(ValueError):study._v8_stage2_gate(root,'v8','verification',protocol)
            for index,task in enumerate(tasks):
                for mode in ('bo','full'):
                    values=[0.]*4+[2. if mode=='full' and index<5 else 1.]*16
                    state=dict(task=task,mode=mode,condition='agent_initial_terra',seed=0,status='COMPLETE',
                        protocol_sha256='synthetic-stage',verification_used=0,calls=[],
                        history=[dict(candidate_id=f'x{j}',raw_value=value,y=value) for j,value in enumerate(values)],
                        decisions=[dict(mode='base_bo') for _ in range(16)])
                    study.llm.write(study.paths(root,'v8')/'runs'/f'{task}_{mode}_agent_initial_terra'/'state.json',state)
            with self.assertRaises(ValueError):study._v8_stage2_gate(root,'v8','no_calibration',protocol)
            path=study.paths(root,'v8')/'runs'/f'{tasks[0]}_full_agent_initial_terra'/'state.json'
            state=study.read(path);state['decisions'][0]['mode']='knowledge_guided';study.llm.write(path,state)
            gate=study._v8_stage2_gate(root,'v8','verification',protocol)
            self.assertEqual(gate,dict(matched_tasks=8,strict_SearchAUC_wins=5,weighted_choices=1))
            # Four victories cannot launch an ablation, even with learned credit.
            path=study.paths(root,'v8')/'runs'/f'{tasks[4]}_full_agent_initial_terra'/'state.json'
            state=study.read(path)
            for row in state['history'][4:]:row.update(raw_value=1.,y=1.)
            study.llm.write(path,state)
            with self.assertRaises(ValueError):study._v8_stage2_gate(root,'v8','verification',protocol)
            self.assertIsNone(study._v8_stage2_gate(root,'v7','verification',protocol))

    def test_full_and_cut_feedback_wiring_preserve_commit_before_revelation(self):
        for mode in ('full','no_calibration'):
            with self.subTest(mode=mode),temporary_workspace() as temporary,ExitStack() as stack:
                root=Path(temporary);pool=make_pool()
                pool.source_path=root/'synthetic_inputs.json';pool.source_path.write_text('{}',encoding='utf-8')
                protocol=dict(seed=0,protocol_sha256='synthetic-v8',supported_conditions=['agent_initial_terra'],
                              supported_modes=['bo','full','verification','no_calibration'])
                directory=study.paths(root,'v8')/'runs'/f'{pool.task}_{mode}_agent_initial_terra'
                statefile=directory/'state.json';requested=[];queried=[];attached=[];crash={'armed':True}
                class Client:
                    def close(self):pass
                class Store:
                    def __init__(self,actual_pool,cache,destination):
                        self_test.assertIs(actual_pool,pool)
                        attached.append((cache.numeric_calibrate,Path(destination)))
                    def birth(self,*args):raise AssertionError('No experts in this transport fixture')
                self_test=self
                def fake_request(client,payload,folder):
                    self.assertEqual(payload['proposal_interface'],'numeric_birth_contrast_v1')
                    self.assertIn('proposal_support',payload);requested.append(payload['query'])
                    return dict(proposal=dict(actions=[],summary='No justified numeric hypothesis.'),
                                context_sha256=llm.digest(payload),attempt=1,resolved_model='gpt-6.1-terra')
                def oracle(actual_root,actual_pool,key):
                    if key in pool.initial_ids:
                        self.assertTrue((directory/'initial_commit.json').exists())
                    else:
                        state=study.read(statefile);t=len(state['decisions'])+1
                        commit_path=directory/'rounds'/f'{t:02d}_commit.json'
                        self.assertTrue(commit_path.exists())
                        commit=study.read(commit_path)
                        self.assertEqual(commit['selected_id'],key)
                        self.assertEqual(commit['history_sha256'],llm.digest(state['history']))
                        if crash['armed']:
                            crash['armed']=False;raise RuntimeError('Synthetic interruption after commitment')
                        queried.append(key)
                    return observation(pool,key)
                stack.enter_context(patch.object(study,'check',return_value=protocol))
                stack.enter_context(patch.object(study,'_v8_stage2_gate',return_value={'synthetic_gate':True}))
                stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
                stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
                stack.enter_context(patch.object(birth,'BirthGPStore',Store))
                stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
                stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
                stack.enter_context(patch.object(study.api,'APIClient',return_value=Client()))
                stack.enter_context(patch.object(llm,'request',side_effect=fake_request))
                stack.enter_context(patch.object(llm,'task_context',return_value={'description':'Synthetic domain'}))
                with redirect_stdout(StringIO()):
                    with self.assertRaises(RuntimeError):study.run_one(root,'v8',pool.task,mode)
                    first_commit=study.read(directory/'rounds/01_commit.json')
                    saved=study.read(statefile);missing=deepcopy(saved);missing.pop('initial_commit_sha256')
                    llm.write(statefile,missing)
                    with self.assertRaises(ValueError):study.run_one(root,'v8',pool.task,mode)
                    llm.write(statefile,saved)
                    result=study.run_one(root,'v8',pool.task,mode)
                self.assertEqual(result['status'],'COMPLETE');self.assertEqual(len(queried),16)
                self.assertEqual(len(set(queried)),16);self.assertEqual(requested.count(1),1)
                self.assertEqual(first_commit,study.read(directory/'rounds/01_commit.json'))
                self.assertEqual(attached,[(mode!='no_calibration',directory/'births')]*2)
                state=study.read(statefile);self.assertEqual(len(state['history']),20)
                for t,decision in enumerate(state['decisions'],1):
                    self.assertEqual(decision['commit_sha256'],llm.digest(study.read(directory/'rounds'/f'{t:02d}_commit.json')))

    def test_unregistered_conditions_and_modes_are_rejected_before_gp_creation(self):
        protocol=dict(seed=0,protocol_sha256='synthetic-v8',supported_conditions=['agent_initial_terra'],
                      supported_modes=['bo','full','verification','no_calibration'])
        with patch.object(study,'check',return_value=protocol),patch.object(study.gpmod,'load_task',return_value=make_pool()),\
                patch.object(study.gpmod,'GPAdapter',side_effect=AssertionError('GP must not be created')):
            with self.assertRaises(ValueError):study.run_one(ROOT,'v8','FakeTask','deployment')
            with self.assertRaises(ValueError):study.run_one(ROOT,'v8','FakeTask','full','correct')

    def test_cli_accepts_no_calibration_without_starting_a_run(self):
        spec=importlib.util.spec_from_file_location('numeric_test_entry',ROOT/'code/chemist_system/knowledge_loop.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(sys,'argv',['knowledge_loop.py','run','--variant','v8','--mode','no_calibration','--task','SnAr']),\
                patch.object(module.os,'chdir'),patch.object(study,'batch') as batch:
            module.main()
        self.assertEqual(batch.call_args.args[1:4],('v8',['SnAr'],'no_calibration'))


if __name__=='__main__':unittest.main()
