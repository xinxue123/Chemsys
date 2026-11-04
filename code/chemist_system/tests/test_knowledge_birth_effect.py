"""Gaussian identity and failure-mode checks, not a benchmark policy result."""
from math import log
import unittest

import numpy as np

from chem.benchmarks import knowledge_birth_effect as m


def fixture(prior_mean=.7,prior_variance=.8,noise=.02,feature=None):
    x=np.linspace(-1.,1.,12)
    cov=.2*np.exp(-np.square(x[:,None]-x[None,:])/.4)+.01*np.eye(len(x))
    mean=.1+.2*x
    phi=np.sin(2*x) if feature is None else np.asarray(feature)
    ids=tuple(f'p{i}' for i in range(len(x)))
    model=m.BirthEffect(ids,ids[:2],mean,np.diag(cov),phi,
                        prior_mean,prior_variance,noise,'synthetic-birth-fixed')
    return model,ids,mean,cov,phi


class BirthEffectTests(unittest.TestCase):
    def test_independent_joint_gaussian_conditioning_equivalence(self):
        for m0,p0 in [(.7,.8),(-.4,.3),(.7,0.),(0.,0.)]:
            model,ids,mean,cov,phi=fixture(m0,p0)
            z=np.array([2,5,8]);q=np.array([3,4,6,7,9,10,11]);y=np.array([.4,-.2,.8])
            got=model.predict([ids[j] for j in z],y,cov[np.ix_(z,z)],
                              [ids[j] for j in q],cov[np.ix_(q,z)])
            # Independent formulation: condition the single joint Gaussian,
            # without using beta's posterior or residual-feature formula.
            joint_mean=mean+phi*m0;joint_cov=cov+p0*np.outer(phi,phi)
            a=joint_cov[np.ix_(z,z)]+.02*np.eye(len(z))
            expected_mean=joint_mean[q]+joint_cov[np.ix_(q,z)]@np.linalg.solve(a,y-joint_mean[z])
            expected_var=np.diag(joint_cov)[q]-np.einsum('ij,ji->i',joint_cov[np.ix_(q,z)],
                np.linalg.solve(a,joint_cov[np.ix_(z,q)]))
            np.testing.assert_allclose(got['expert_mean'],expected_mean,atol=5e-14,rtol=2e-13)
            np.testing.assert_allclose(got['expert_latent_variance'],expected_var,atol=5e-14,rtol=2e-13)
            np.testing.assert_allclose(got['expert_observation_variance'],expected_var+.02,atol=5e-14)

    def test_birth_duplicate_changed_and_removed_feedback_are_rejected(self):
        model,ids,mean,cov,phi=fixture()
        with self.assertRaises(ValueError):
            model.predict([ids[0]],[.1],cov[:1,:1],[ids[3]],cov[3:4,:1])
        with self.assertRaises(ValueError):
            model.predict([ids[2],ids[2]],[.1,.1],np.ones((2,2)),[ids[3]],np.ones((1,2)))
        first=model.predict([ids[2]],[.4],cov[2:3,2:3],[ids[3]],cov[3:4,2:3])
        repeated=model.predict([ids[2]],[.4],cov[2:3,2:3],[ids[3]],cov[3:4,2:3])
        self.assertEqual(first['beta_mean'],repeated['beta_mean'])
        self.assertEqual(model.cache_info()['post_birth_observations'],1)
        self.assertEqual(model.cache_info()['cholesky'],1)
        with self.assertRaises(ValueError):
            model.predict([ids[2]],[.41],cov[2:3,2:3],[ids[3]],cov[3:4,2:3])
        with self.assertRaises(ValueError):
            model.predict([],[],np.empty((0,0)),[ids[3]],np.empty((1,0)))

    def test_zero_feature_and_zero_prior_variance_limits(self):
        for m0,p0,feature in [(3.,2.,np.zeros(12)),(0.,0.,None)]:
            model,ids,mean,cov,phi=fixture(m0,p0,feature=feature)
            z=[2,5];q=[3,4,7]
            got=model.predict([ids[j] for j in z],[.3,.7],cov[np.ix_(z,z)],
                              [ids[j] for j in q],cov[np.ix_(q,z)])
            np.testing.assert_array_equal(got['expert_mean'],got['null_mean'])
            np.testing.assert_array_equal(got['expert_latent_variance'],got['null_latent_variance'])

    def test_no_post_birth_observations_preserves_the_prior(self):
        model,ids,mean,cov,phi=fixture()
        q=[2,3,5]
        got=model.predict([],[],np.empty((0,0)),[ids[j] for j in q],np.empty((len(q),0)))
        np.testing.assert_allclose(got['expert_mean'],mean[q]+.7*phi[q])
        np.testing.assert_allclose(got['expert_latent_variance'],np.diag(cov)[q]+.8*phi[q]**2)
        self.assertEqual(got['beta_mean'],.7);self.assertEqual(got['beta_variance'],.8)

    def test_gp_known_component_has_no_remaining_feature_effect(self):
        # phi_x=C_xz A^-1 phi_z makes the effect already explained by the
        # GP conditioning. A nonzero learned beta must not double-count it.
        model=m.BirthEffect(['birth','z','q'],['birth'],[0,0,0],[1,1,.5],[0,1,.5],2.,1.,0.,'known')
        got=model.predict(['z'],[2.],[[1.]],['q'],[[.5]])
        self.assertEqual(float(got['residual_feature'][0]),0.)
        np.testing.assert_array_equal(got['expert_mean'],got['null_mean'])
        np.testing.assert_array_equal(got['expert_latent_variance'],got['null_latent_variance'])

    def test_zero_query_feature_does_not_erase_joint_correlation(self):
        model=m.BirthEffect(['birth','z','q'],['birth'],[0,0,0],[1,1,1],[0,1,0],1.,1.,0.,'cross')
        got=model.predict(['z'],[2.],[[1.]],['q'],[[.5]])
        self.assertEqual(float(got['residual_feature'][0]),-.5)
        self.assertNotEqual(float(got['expert_mean'][0]),float(got['null_mean'][0]))

    def test_fixed_synthetic_holdout_after_wrong_direction_calibration(self):
        # Fixed synthetic mechanism and held-out points, not a hyperparameter
        # search: beta_truth=.8, independent GP residual variance=.01.
        phi=np.array([0.,-1.,-.5,.5,1.,-.8,-.2,.3,.9])
        ids=tuple(f'x{i}' for i in range(len(phi)));cov=.01*np.eye(len(phi))
        model=m.BirthEffect(ids,[ids[0]],np.zeros(len(phi)),np.diag(cov),phi,-.5,1.,.01,'fixed-holdout')
        z=np.array([1,2,3,4]);q=np.array([5,6,7,8])
        train=.8*phi[z]+np.array([.02,-.01,.03,-.02])
        heldout=.8*phi[q]+np.array([.01,-.02,.02,-.01])
        before=model.predict([],[],np.empty((0,0)),[ids[j] for j in q],np.empty((len(q),0)))
        after=model.predict([ids[j] for j in z],train,cov[np.ix_(z,z)],
                            [ids[j] for j in q],cov[np.ix_(q,z)])
        self.assertLess(before['beta_mean'],0);self.assertGreater(after['beta_mean'],0)
        self.assertLess(abs(after['beta_mean']-.8),abs(before['beta_mean']-.8))
        rmse=lambda pred:float(np.sqrt(np.mean((pred-heldout)**2)))
        self.assertLess(rmse(after['expert_mean']),rmse(after['null_mean']))
        self.assertLess(rmse(after['expert_mean']),rmse(before['expert_mean']))
        def logscore(mu,var):return -.5*np.mean(np.log(2*np.pi*var)+(heldout-mu)**2/var)
        self.assertGreater(logscore(after['expert_mean'],after['expert_observation_variance']),
                           logscore(after['null_mean'],after['null_observation_variance']))

    def test_unmodeled_background_interaction_is_not_automatically_solved(self):
        # Same feature has opposite true directions in two backgrounds. This
        # one-coefficient model averages them; it cannot identify the missing
        # interaction or promise transfer into either background.
        ids=['birth','positive_background','negative_background','future_positive']
        cov=.01*np.eye(4)
        model=m.BirthEffect(ids,['birth'],[0]*4,np.diag(cov),[0,1,1,1],0.,1.,.01,'interaction')
        first=model.predict([ids[1]],[1.],cov[1:2,1:2],[ids[2]],cov[2:3,1:2])
        true_negative=-1.
        expert_loss=(true_negative-first['expert_mean'][0])**2
        null_loss=(true_negative-first['null_mean'][0])**2
        self.assertGreater(expert_loss,null_loss)
        second=model.predict(ids[1:3],[1.,-1.],cov[1:3,1:3],[ids[3]],cov[3:4,1:3])
        self.assertAlmostEqual(second['beta_mean'],0.,places=14)
        self.assertLess(second['beta_variance'],first['beta_variance'])
        self.assertGreater(abs(second['expert_mean'][0]-1.),.9)

    def test_snapshot_and_seen_covariance_are_frozen_without_n_squared_registry(self):
        mu=np.zeros(5);var=np.ones(5);phi=np.arange(5,dtype=float)
        model=m.BirthEffect(list('abcde'),['a'],mu,var,phi,0.,1.,.1,'freeze')
        mu[:]=999.;var[:]=999.;phi[:]=999.
        first=model.predict(['b'],[1.],[[1.]],['c','d'],[[.2],[.3]])
        self.assertLess(np.max(first['expert_mean']),10.)
        first['expert_mean'][:]=999.
        repeat=model.predict(['b'],[1.],[[1.]],['c','d'],[[.2],[.3]])
        self.assertLess(np.max(repeat['expert_mean']),10.)
        with self.assertRaises(ValueError):model.predict(['b'],[1.],[[1.]],['c'],[[.21]])
        self.assertEqual(model.cache_info()['covariance_registry_bytes'],5*8)
        self.assertEqual(model.cache_info()['covariance_columns'],1)
        self.assertEqual(model.cache_info()['operator_compiles'],0)

    def test_invalid_covariance_is_not_repaired_with_jitter(self):
        model=m.BirthEffect(list('abcd'),['a'],[0]*4,[1]*4,[0,1,1,1],0.,1.,.1,'bad')
        with self.assertRaises(ValueError):
            model.predict(['b','c'],[0,0],[[1,2],[2,1]],['d'],[[0,0]])
        with self.assertRaises(ValueError):
            model.predict(['b'],[0],[[1]],['d'],[[2.]])
        self.assertEqual(model.cache_info()['post_birth_observations'],0)
        self.assertEqual(model.cache_info()['cholesky'],0)

    def test_high_noise_cannot_hide_invalid_latent_cross_covariance(self):
        model=m.BirthEffect(['birth','z','q'],['birth'],[0]*3,[1]*3,[0,1,1],0.,1.,10.,'high-noise')
        # A=11 would give a positive observation-conditioned variance 1-4/11,
        # but the latent block [[1,2],[2,1]] has eigenvalues -1 and 3.
        with self.assertRaisesRegex(ValueError,'latent joint covariance'):
            model.predict(['z'],[.2],[[1.]],['q'],[[2.]])
        self.assertEqual(model.cache_info()['post_birth_observations'],0)
        self.assertEqual(model.cache_info()['covariance_columns'],0)
        self.assertEqual(model.cache_info()['cholesky'],0)

    def test_singular_legal_latent_block_with_noise_is_accepted(self):
        model=m.BirthEffect(['birth','z1','z2','q'],['birth'],[0]*4,[1]*4,[0,1,1,1],0.,1.,.1,'singular')
        got=model.predict(['z1','z2'],[.2,.3],[[1.,1.],[1.,1.]],['q'],[[1.,1.]])
        self.assertGreaterEqual(got['null_latent_variance'][0],0.)
        self.assertEqual(got['diagnostics']['latent_joint_psd_queries'],1)
        self.assertEqual(got['diagnostics']['latent_covariance_repairs'],0)
        self.assertEqual(got['diagnostics']['jitter_added'],0.)
        self.assertLessEqual(abs(got['diagnostics']['minimum_query_latent_joint_eigenvalue']),
                             got['diagnostics']['maximum_latent_joint_psd_tolerance'])

    def test_near_singular_legal_small_eigen_direction_is_preserved(self):
        delta=1e-14
        # Gram vectors z1=(1,0), z2=(1,sqrt(delta)), q=(0,sqrt(delta)).
        # A small positive eigen-direction must not be dropped as a nullspace.
        model=m.BirthEffect(['birth','z1','z2','q'],['birth'],[0]*4,[1,1,1+delta,delta],
                            [0,0,1,1],0.,1.,.1,'near-singular')
        got=model.predict(['z1','z2'],[.2,.3],[[1.,1.],[1.,1+delta]],['q'],[[0.,delta]])
        self.assertGreater(got['null_latent_variance'][0],0.)
        self.assertLessEqual(got['null_latent_variance'][0],delta)
        self.assertEqual(got['diagnostics']['latent_covariance_repairs'],0)
        self.assertEqual(got['diagnostics']['jitter_added'],0.)

    def test_invalid_last_query_in_second_psd_batch_does_not_commit_feedback(self):
        queries=[f'q{i}' for i in range(m.LATENT_PSD_BATCH_SIZE+1)]
        ids=['birth','z']+queries
        model=m.BirthEffect(ids,['birth'],np.zeros(len(ids)),np.ones(len(ids)),
                            np.ones(len(ids)),0.,1.,10.,'batch-check')
        cross=np.zeros((len(queries),1));cross[-1,0]=2.
        with self.assertRaisesRegex(ValueError,'query index 256'):
            model.predict(['z'],[.2],[[1.]],queries,cross)
        self.assertEqual(model.cache_info()['post_birth_observations'],0)
        self.assertEqual(model.cache_info()['fit_entries'],0)


if __name__=='__main__':unittest.main()
