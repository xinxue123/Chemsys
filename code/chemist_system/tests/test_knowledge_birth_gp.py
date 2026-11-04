"""Frozen-GP custody and independent Gaussian-block checks; never fit models.

Run --real-cache for the eight archived initial fits. That mode reads only
candidate inputs, already queried history/commits, and existing cache weights.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from botorch.models import SingleTaskGP

from chem.benchmarks import knowledge_gp as gpmod
from chem.benchmarks import knowledge_birth_gp as m


def synthetic(root,n=9):
    source=root/'inputs';source.mkdir();(source/'pool.json').write_text('{}',encoding='utf-8')
    cache=root/'cache';cache.mkdir()
    ids=tuple(f'p{j}' for j in range(n));X=torch.linspace(0,1,n,dtype=torch.double).reshape(-1,1)
    pool=gpmod.Pool('synthetic','P2',ids,('x',),('continuous',),tuple({'x':float(x)} for x in X.reshape(-1)),
        tuple((float(x),) for x in X.reshape(-1)),ids[:4],'unit',100.,{},root,source/'pool.json')
    history=[dict(candidate_id=ids[j],y=y,raw_value=y*100) for j,y in enumerate((.1,.3,.25,.5))]
    ordered=[dict(candidate_id=x['candidate_id'],y=x['y']) for x in history]
    y=torch.tensor([[x['y']] for x in history],dtype=torch.double);mean=y.mean();std=y.std(unbiased=False)
    model=SingleTaskGP(X[:4],(y-mean)/std);model.eval()
    key=m.digest(dict(history=ordered,recipe='synthetic_no_fit'))
    torch.save(dict(key=key,history=ordered,state=model.state_dict()),cache/(key+'.pt'))
    adapter=SimpleNamespace(pool=pool,X=X,directory=cache,cache=SimpleNamespace(recipe='synthetic_no_fit'),counters={'fits':0})
    context={'stats':dict(model=model,fit_key=key,mean=float(mean),std=float(std))}
    h=dict(id='H1',version=1,created_round=1,focus={'x':{'direction':'increase','bounds':[0.,1.]}},
        scope={'axes':{'x':{'kind':'continuous','bounds':[0.,1.]}},'background_radius':.25},plus_id='p4',minus_id='p3',
        numeric_prior={'delta_mean_raw':15.,'delta_sd_raw':10.},feature_vector=[0.,0.,0.,0.,1.,.8,.6,.3,0.]+[0.]*(n-9),birth_prefix=history)
    return pool,adapter,context,h,history


def native_blocks(store,h,indices):
    s=store._load(h);meta=s['meta'];std=meta['normalization']['outer_std'];mean=meta['normalization']['outer_mean']
    with torch.no_grad():
        p=s['model'].posterior(store.gp.X[indices]);mu=p.mean.reshape(-1).numpy()*std+mean
        cov=p.mvn.covariance_matrix.numpy()*std**2
    return mu,cov


class BirthGPTests(unittest.TestCase):
    def setUp(self):
        workspace=next(p for p in Path(__file__).resolve().parents if (p/'code/chemist_system').is_dir())
        temporary=workspace/'runtime/knowledge_loop/birth_adapter_review/tests';temporary.mkdir(parents=True,exist_ok=True)
        self.tmp=TemporaryDirectory(dir=temporary);self.root=Path(self.tmp.name)
        self.pool,self.gp,self.context,self.h,self.history=synthetic(self.root)
        self.store=m.BirthGPStore(self.pool,self.gp,self.root/'births')
        self.h['birth']=self.store.birth(self.h,self.history,self.context)
    def tearDown(self):self.tmp.cleanup()

    def test_birth_prior_and_four_distribution_contract(self):
        meta=self.h['birth'];prediction=self.store.predict(self.h,self.history,[4,5,6])
        mu,cov=native_blocks(self.store,self.h,[4,5,6])
        self.assertAlmostEqual(meta['prior_mean'],.15-(mu[0]-.5),places=13)
        self.assertAlmostEqual(meta['prior_variance'],.01,places=14)
        np.testing.assert_allclose(prediction['null']['mean'],mu,atol=1e-12)
        np.testing.assert_allclose(prediction['null']['latent_variance'],np.diag(cov),atol=1e-12)
        for name in ('full','null','cut_feedback','variance_only'):
            self.assertEqual(set(prediction[name]),{'mean','latent_variance','observation_variance'})
        np.testing.assert_array_equal(prediction['full']['mean'],prediction['cut_feedback']['mean'])
        self.assertEqual(self.gp.counters['fits'],0)

    def test_two_feedback_points_match_independent_joint_gaussian(self):
        history=self.history+[dict(candidate_id='p4',y=.7,raw_value=70.),dict(candidate_id='p5',y=.4,raw_value=40.)]
        result=self.store.predict(self.h,history,[6,7,8]);meta=self.h['birth']
        mu,C=native_blocks(self.store,self.h,[4,5,6,7,8]);phi=np.array(self.h['feature_vector'])[4:]
        K=C+meta['prior_variance']*np.outer(phi,phi);mean=mu+phi*meta['prior_mean']
        A=K[:2,:2]+meta['normalization']['noise_variance_y']*np.eye(2)
        expected=mean[2:]+K[2:,:2]@np.linalg.solve(A,np.array([.7,.4])-mean[:2])
        variance=np.diag(K[2:,2:]-K[2:,:2]@np.linalg.solve(A,K[:2,2:]))
        np.testing.assert_allclose(result['full']['mean'],expected,atol=2e-12)
        np.testing.assert_allclose(result['full']['latent_variance'],variance,atol=2e-12)
        cut=self.store.predict(self.h,history,[6,7,8],calibrate=False)
        np.testing.assert_array_equal(cut['mean'],cut['cut_feedback']['mean'])
        np.testing.assert_array_equal(cut['full']['mean'],result['full']['mean'])
        self.assertNotEqual(result['beta_mean'],meta['prior_mean'])
        self.assertEqual(self.gp.counters['fits'],0)

    def test_restart_identical_and_chronological_feedback_is_immutable(self):
        history=self.history+[dict(candidate_id='p4',y=.7,raw_value=70.)]
        first=self.store.predict(self.h,history,[5,6])
        restored=m.BirthGPStore(self.pool,self.gp,self.root/'births')
        second=restored.predict(self.h,history,[5,6])
        np.testing.assert_array_equal(first['mean'],second['mean'])
        self.assertEqual(first['block_receipt']['receipt_sha256'],second['block_receipt']['receipt_sha256'])
        with self.assertRaises(ValueError):restored.predict(self.h,self.history,[5])
        changed=deepcopy(history);changed[-1]['y']+=.1
        with self.assertRaises(ValueError):restored.predict(self.h,changed,[5])
        with self.assertRaises(ValueError):restored.predict(self.h,history,[4])

    def test_fit_metadata_vectors_and_encoding_tamper_are_rejected(self):
        altered=deepcopy(self.h);altered['birth']['prior_mean']+=1
        with self.assertRaises(ValueError):self.store.predict(altered,self.history,[4])
        altered=deepcopy(self.h);altered['numeric_prior']['delta_mean_raw']+=1
        with self.assertRaises(ValueError):self.store.predict(altered,self.history,[4])
        self.gp.X[8,0]=.99
        with self.assertRaises(ValueError):self.store.predict(self.h,self.history,[4])

    def test_no_fit_on_missing_cache_and_feature_scale_is_explicit(self):
        altered=deepcopy(self.h);altered['feature_vector'][4]=2
        with self.assertRaises(ValueError):self.store.birth(altered,self.history,self.context)
        path=self.gp.directory/(self.context['stats']['fit_key']+'.pt');path.rename(path.with_suffix('.missing'))
        with self.assertRaises(FileNotFoundError):self.store.birth(self.h,self.history,self.context)
        self.assertEqual(self.gp.counters['fits'],0)

    def test_current_gp_mutation_does_not_change_frozen_birth(self):
        first=self.store.predict(self.h,self.history,[4,5])
        with torch.no_grad():self.context['stats']['model'].mean_module.raw_constant.fill_(100.)
        second=self.store.predict(self.h,self.history,[4,5])
        np.testing.assert_array_equal(first['mean'],second['mean'])

    def test_roundoff_normalization_refuses_material_covariance_repair(self):
        out,receipt=m._roundoff_replace(1.+np.finfo(float).eps,1.,2.,'synthetic')
        self.assertEqual(float(out),1.);self.assertEqual(receipt['changed_entries'],1)
        with self.assertRaises(ValueError):m._roundoff_replace(.9,1.,2.,'synthetic')

    def test_diagonal_kernel_never_receives_the_entire_large_pool(self):
        sub=self.root/'large';sub.mkdir()
        pool,adapter,context,h,history=synthetic(sub,n=600)
        store=m.BirthGPStore(pool,adapter,sub/'births');model=context['stats']['model']
        original=model.covar_module.forward;seen=[]
        def guarded(*args,**kwargs):
            if kwargs.get('diag',False):
                seen.append(len(args[0]));self.assertLessEqual(len(args[0]),256)
            return original(*args,**kwargs)
        with patch.object(model.covar_module,'forward',side_effect=guarded):
            store._basis(model,adapter.X[:4],context['stats']['mean'],context['stats']['std'])
        self.assertEqual(seen,[256,256,88])


def real_cache_audit():
    root=next(p for p in Path(__file__).resolve().parents if (p/'experiments/runs/knowledge_loop_20261101/v7').exists())
    archive=root/'experiments/runs/knowledge_loop_20261101/v7';out=root/'runtime/knowledge_loop/birth_adapter_review';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in gpmod.TASKS:
        pool=gpmod.load_task(root,task);adapter=gpmod.GPAdapter(pool,archive/'cache'/task)
        run=archive/'runs'/f'{task}_full_agent_initial_terra';state=json.loads((run/'state.json').read_text(encoding='utf-8'))
        commit=json.loads((run/'rounds/01_commit.json').read_text(encoding='utf-8'));history=state['history'][:4]
        original=next(h for h in commit['hypotheses_at_commit'] if h['status']!='retracted')
        h={k:deepcopy(original[k]) for k in ('id','version','created_round','focus','scope','plus_id','minus_id')}
        h['feature_vector']=[0.]*len(pool.ids);h['feature_vector'][pool.ix[h['plus_id']]]=1.
        h['numeric_prior']=dict(delta_mean_raw=.1*pool.scale,delta_sd_raw=.1*pool.scale);h['birth_prefix']=history
        with np.load(run/'rounds/01_GP.npz',allow_pickle=False) as z:mu=z['mu_y'].copy()
        store=m.BirthGPStore(pool,adapter,out/task)
        h['birth']=store.birth(h,history,{'stats':dict(fit_key=commit['GP_fit_key'],mu_y=mu)})
        post_history=state['history'][:6];observed={x['candidate_id'] for x in post_history}
        queries=[j for j,key in enumerate(pool.ids) if key not in observed][:5]
        result=store.predict(h,post_history,queries)
        zids=[pool.ix[x['candidate_id']] for x in post_history[4:]]
        native_mu,native_cov=native_blocks(store,h,zids+queries)
        with np.load(result['block_receipt']['file'],allow_pickle=False) as z:
            cross_error=float(np.max(np.abs(z['cov_xz']-native_cov[2:,:2])));zz_error=float(np.max(np.abs(z['cov_zz']-native_cov[:2,:2])))
        restored=m.BirthGPStore(pool,adapter,out/task).predict(h,post_history,queries)
        assert np.array_equal(restored['mean'],result['mean'])
        assert max(cross_error,zz_error)<1e-9 and adapter.counters['fits']==0
        rows.append(dict(task=task,pool_candidates=len(pool.ids),snapshot_sha256=h['birth']['snapshot_sha256'],
            native_posterior_check=h['birth']['native_posterior_check'],cross_error=cross_error,zz_error=zz_error,
            GP_fits=adapter.counters['fits'],covariance_block_shape=[len(queries),2],block_receipt=result['block_receipt'],
            statement='Synthetic feature/prior for transport validation; already queried targets only; not an LLM experiment.'))
        print(task,'PASS fits=0',cross_error,zz_error,flush=True)
    receipt=dict(status='PASS_EIGHT_INITIAL_CACHED_GP_TRANSPORT',tasks=rows,GP_fits=0,LLM_calls=0,oracle_reads=0,
        new_experiments=0,source_sha256=m.filehash(m.__file__),tests_sha256=m.filehash(__file__))
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    if '--real-cache' in sys.argv:real_cache_audit()
    else:unittest.main()
