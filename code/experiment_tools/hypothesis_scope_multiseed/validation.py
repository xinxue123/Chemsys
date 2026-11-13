"""Pre-register the seed adapter; never substitute synthetic experimental LLMs."""
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import unittest
import numpy as np
from . import study as s


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pools={t:s.frozen.pool_for(t) for t in s.TASKS}

    def test_seed0_saved_choices_replayed_without_policy_changes(self):
        count=0
        for task,pool in self.pools.items():
            folder=s.frozen.folder_for(task);state=s.read(folder/'state.json');scopes=s.knowledge.Scopes(pool)
            for q in range(2,21):
                c=s.read(folder/'rounds'/f'{q:02d}_commit.json')
                with np.load(folder/'rounds'/f'{q:02d}_GP.npz') as a:
                    ctx={key:a[key].copy() for key in ('base_logei','available','tie_priority')}
                    ctx['bo_index']=s.policy.argmax(ctx,ctx['available']);seed0=s.retie(ctx,pool,q,0)
                    np.testing.assert_array_equal(seed0['tie_priority'],ctx['tie_priority'])
                    choice=s.policy.select(seed0,scopes,c['working_knowledge']['notes'],c['decision']['incumbent_before'])
                    self.assertEqual(pool.ids[choice['index']],state['history'][q-1]['candidate_id']);count+=1
        self.assertEqual(count,152)

    def test_seed_changes_only_priority_not_GP_scores(self):
        for task,pool in self.pools.items():
            folder=s.frozen.folder_for(task)
            with np.load(folder/'rounds/02_GP.npz') as a:
                ctx={key:a[key].copy() for key in ('base_logei','available','tie_priority','mu_y','sd_y')}
                ctx.update(bo_index=s.policy.argmax(ctx,ctx['available']),epsilon=.5)
                for seed in s.SEEDS:
                    out=s.retie(ctx,pool,2,seed)
                    for key in ('base_logei','mu_y','sd_y','available'):np.testing.assert_array_equal(out[key],ctx[key])
                    np.testing.assert_array_equal(out['tie_priority'],s.frozen.native.semantic_priorities(pool,2,seed))
                    self.assertEqual(out['base_logei'][out['bo_index']],ctx['base_logei'][ctx['bo_index']])

    def test_all_initial_priors_match_blinded_original(self):
        for task,pool in self.pools.items():
            prior=s.initial_prior(task);old=s.read(s.frozen.folder_for(task)/'state.json');c=s.read(s.frozen.folder_for(task)/'rounds/01_commit.json')
            for seed in s.SEEDS:
                state=s.initial_state(task,seed,'hyp',dict(protocol_sha256='SYNTHETIC_QA_ONLY'))
                valid=s.prompts.validate(prior['raw'],pool,state,s.frozen.locked_first(task))
                self.assertEqual(valid,prior['plan']);self.assertEqual(valid['projection']['candidate_id'],old['history'][0]['candidate_id'])
                self.assertEqual(state['seed'],seed);self.assertEqual(state['numerical_seed'],0)
                notes,_=s.knowledge.install(valid['drafts'],[],[],s.knowledge.Scopes(pool),1,False)
                self.assertEqual(notes,c['working_knowledge']['notes'])

    def test_feature_prior_does_not_depend_on_candidate_ID_text(self):
        pool=next(iter(self.pools.values()));copy=deepcopy(pool)
        copy.ids=['arbitrary_'+str(i) for i in range(len(pool.ids))]
        for q in (1,2,7):
            for seed in s.SEEDS:np.testing.assert_array_equal(s.frozen.native.semantic_priorities(pool,q,seed),s.frozen.native.semantic_priorities(copy,q,seed))

    def test_completion_is_required_before_sealing(self):
        mock=s.BASE/'preflight_mock';mock.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=mock) as tmp:
            af=Path(tmp).resolve();io=af/'output.json'
            for name,value in (('request.json',dict(request_sha256='SYNTHETIC_QA_ONLY')),('assignment.json',dict(agent_id='/root/SYNTHETIC_QA')),
                ('export.json',dict(output=io.relative_to(s.ROOT).as_posix(),input='SYNTHETIC_QA_UNUSED')),('output.json',dict(SYNTHETIC_QA_ONLY=True))):s.write(af/name,value)
            with self.assertRaises(FileNotFoundError):s.transport.seal(af/'request.json')


def preflight():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value=dict(status='PASS_MULTISEED_ADAPTER_PREFLIGHT' if result.wasSuccessful() else 'FAIL_MULTISEED_ADAPTER_PREFLIGHT',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),new_experimental_queries=0,new_LLM_calls=0,
        seed0_saved_decisions_replayed=152,test_output=stream.getvalue())
    s.write(s.BASE/'preflight.json',value,replace=True)
    if not result.wasSuccessful():print(stream.getvalue());raise RuntimeError('Preflight failed')
    return {a:b for a,b in value.items() if a!='test_output'}
